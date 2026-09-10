import os
import time
from datetime import date, timedelta
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from taa.util.logger import ExpLogger
from taa.util.metrics import TAAMetrics
from taa.model.simple_baseline import SimpleBaselineModel, TAATemporal
from taa.configs.config import Config

def setup(rank: int, world_size: int):
    """DDP 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # GPU 프로세스 그룹 초기화
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=50))

def cleanup():
    """DDP 정리"""
    dist.destroy_process_group()

def train_epoch(
    model: nn.Module,
    train_loader: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: ExpLogger,
    rank: int
) -> Dict[str, float]:
    """한 epoch 학습"""
    model.train()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'total_loss': 0.}

    if rank == 0:
        train_loader = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(train_loader):
        # 데이터 준비
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        is_positive = batch['is_positive'].to(device)  # (B,)

        # Forward pass
        predictions = model(frames)
        losses = criterion(predictions['risk_score'], is_positive)

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Loss 기록 (rank 0만)
        if rank == 0:
            for k, v in losses.items():
                total_losses[k] += v.item()

            if isinstance(train_loader, tqdm):
                current_avg_losses = {k: v / (batch_idx + 1) for k, v in total_losses.items()}
                train_loader.set_postfix({
                    **{f"batch_{k}": f"{v.item():.4f}" for k, v in losses.items()},
                    **{f"avg_{k}": f"{v:.4f}" for k, v in current_avg_losses.items()}
                })

            # 로그 기록
            lr = optimizer.param_groups[0]['lr']
            logger.log_train_step(epoch, batch_idx, {k: v.item() for k, v in losses.items()}, lr)

    # Epoch 평균 loss 계산 (rank 0만)
    if rank == 0:
        for k in total_losses:
            total_losses[k] /= len(train_loader)

    return total_losses

@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: Any,
    criterion: nn.Module,
    metrics_calculator: TAAMetrics,
    device: torch.device,
    epoch: int,
    rank: int
) -> Dict[str, float]:
    """검증 수행"""
    model.eval()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'total_loss': 0.}

    all_preds = []
    all_targets = []

    if rank == 0:
        val_loader = tqdm(val_loader, desc='Validation')

    for batch in val_loader:
        # 데이터 준비
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        is_positive = batch['is_positive'].to(device)  # (B,)

        # Forward pass
        predictions = model(frames)
        losses = criterion(predictions['risk_score'], is_positive)

        # Loss 기록 (rank 0만)
        if rank == 0:
            for k, v in losses.items():
                total_losses[k] += v.item()

        # anomaly probability 계산 softmax
        anomaly_probs = torch.stack(predictions['risk_score'], dim=0)[:, 0, :]  # (T, 1)
        anomaly_probs = torch.softmax(anomaly_probs, dim=1)[:, 1:]

        # 예측값 저장
        all_preds.append(anomaly_probs.cpu())
        all_targets.append(is_positive.cpu())

    # 모든 GPU의 예측값 수집
    T = all_preds[0].size(0)  # T
    all_preds = torch.cat(all_preds, dim=0)  # List[Tensor([T, 1]), Tensor([T, 1]), ..., N] -> Tensor([N*T, 1])
    all_targets = torch.cat(all_targets, dim=0)  # List[Tensor([True]), Tensor([False]), ..., N] -> Tensor([N])
    all_targets = all_targets.unsqueeze(1).repeat(1, T).view(-1)  # Tensor([N, 1]) -> Tensor([N, T])

    # 각 GPU의 예측값 크기
    local_size = torch.tensor([all_preds.size(0)], device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)

    # 모든 GPU의 예측값을 수집하기 위한 텐서 준비
    max_size = max(sizes).item()
    pred_gathered = [torch.zeros((max_size, all_preds.size(1)), device=device) for _ in range(dist.get_world_size())]
    target_gathered = [torch.zeros(max_size, device=device) for _ in range(dist.get_world_size())]

    # 예측값 수집
    pred_padded = torch.zeros((max_size, all_preds.size(1)), device=device)
    pred_padded[:all_preds.size(0)] = all_preds
    target_padded = torch.zeros(max_size, device=device)
    target_padded[:all_targets.size(0)] = all_targets

    dist.all_gather(pred_gathered, pred_padded)
    dist.all_gather(target_gathered, target_padded)

    # 모든 GPU의 결과 합치기
    if rank == 0:
        all_preds_combined = []
        all_targets_combined = []
        
        for i, size in enumerate(sizes):
            all_preds_combined.append(pred_gathered[i][:size])
            all_targets_combined.append(target_gathered[i][:size])
        
        all_preds_combined = torch.cat(all_preds_combined, dim=0)
        all_targets_combined = torch.cat(all_targets_combined, dim=0)

        # 메트릭 계산
        metrics = metrics_calculator.calculate_metrics(
            predictions=all_preds_combined,
            targets=all_targets_combined,
            epoch=epoch
        )
        
        return total_losses, metrics

    return total_losses, {}

def train(rank: int, world_size: int, config: Dict[str, Any]):
    """단일 GPU에서의 학습 프로세스"""
    # DDP 설정
    setup(rank, world_size)

    # 로거 초기화 (rank 0만)
    logger = ExpLogger() if rank == 0 else None
    if rank == 0:
        logger.save_exp_info(config)

    # 디바이스 설정
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

    # 모델 생성
    model = SimpleBaselineModel(
        backbone_type=config['model']['backbone_type'],
        pretrained=config['model']['pretrained']
    ).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    
    # Loss 함수와 optimizer 설정
    criterion = TAATemporal(
        lambda_temporal=config['loss']['lambda_temporal'],
        fps=config['dataset']['fps'],
        toa=config['dataset']['toa']
    ).to(device)
    
    if config['train']['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=float(config['train']['learning_rate'])
        )
    elif config['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            model.parameters(),
            lr=float(config['train']['learning_rate']),
            momentum=0.9
        )
    else:
        raise ValueError(f"Invalid optimizer: {config['train']['optimizer']}")
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=(rank==0)
    )

    # 데이터 로더 생성
    if config['dataset']['name'] == 'DAD':
        if config['dataset']['hdf5']:
            from taa.dataset.DAD_hdf5 import create_dad_loader_hdf5 as create_data_loader
        else:
            from taa.dataset.DAD import create_dad_loader as create_data_loader
    elif config['dataset']['name'] == 'ROL':
        from taa.dataset.ROL import create_rol_loader as create_data_loader
    else:
        raise ValueError(f"Invalid dataset: {config['dataset']['name']}")
    
    train_sampler = DistributedSampler(
        dataset=create_data_loader(
            root_path=config['dataset']['root_path'],
            split='training',
            batch_size=1,
            num_workers=config['train']['num_workers'],
            pin_memory=False,
            drop_last=False
        ),
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    train_loader = create_data_loader(
        root_path=config['dataset']['root_path'],
        split='training',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        img_size=config['dataset']['img_size'],
        sampler=train_sampler
    )

    val_sampler = DistributedSampler(
        dataset=create_data_loader(
            root_path=config['dataset']['root_path'],
            split='testing',
            batch_size=1,
            num_workers=config['train']['num_workers'],
            pin_memory=False,
            drop_last=False
        ),
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    val_loader = create_data_loader(
        root_path=config['dataset']['root_path'],
        split='testing',
        batch_size=1,
        num_workers=config['train']['num_workers'],
        img_size=config['dataset']['img_size'],
        sampler=val_sampler
    )

    # 메트릭 계산기 초기화 (rank 0만)
    metrics_calculator = TAAMetrics(fps=config['dataset']['fps']) if rank == 0 else None

    # 학습 루프
    best_val_auc = 0
    for epoch in range(int(config['train']['num_epochs'])):
        # train sampler의 epoch 설정
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)

        # 학습
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, logger, rank
        )

        # 검증
        val_losses, val_metrics = validate(
            model, val_loader, criterion, metrics_calculator,
            device, epoch, rank
        )

        # rank 0만 로깅 및 체크포인트 저장
        if rank == 0:
            # 로그 기록
            logger.log_eval(epoch, val_losses, val_metrics)

            # 학습률 조정
            scheduler.step(val_losses['total_loss'])

            # 모델 저장
            if val_metrics['auc_roc'] > best_val_auc:
                best_val_auc = val_metrics['auc_roc']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_auc': best_val_auc,
                }
                torch.save(checkpoint, logger.get_checkpoint_path('best_model'))
                print(f"Best model saved! AUC-ROC: {best_val_auc:.4f}")

            # 현재 epoch 결과 출력
            print(f"\nEpoch {epoch}")
            print("Train:", {k: f"{v:.4f}" for k, v in train_losses.items()})
            print("Val losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})
            print("Val metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
            print("-" * 50)

        # 모든 프로세스 동기화
        dist.barrier()

    # 정리
    cleanup()

def main():
    # 설정 파일 로드
    config = Config.load_config('taa/configs/taa_DAD_simplebaseline.yaml')

    # DDP 설정
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size, config),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main() 