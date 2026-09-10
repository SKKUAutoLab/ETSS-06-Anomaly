import os
import time
from datetime import date
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from taa.dataset.DAD import create_dad_loader
from taa.dataset.DAD_hdf5 import create_dad_loader_hdf5
# from taa.dataset.ROL import create_rol_loader
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from taa.util.logger import ExpLogger
from taa.util.metrics import TAAMetrics
from taa.util.visualization import visualize_patch_attention
from taa.model.taa_yolov10 import YOLOv10TAADetectionModel, TAATemporal, AdaLEA
from taa.configs.config import Config
import cv2
from datetime import timedelta

def setup(rank: int, world_size: int):
    """DDP 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29557'

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
    """한 epoch 학습 (전체 프레임 입력)"""
    model.train()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'attn_loss': 0., 'total_loss': 0.}

    if rank == 0:
        train_loader = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(train_loader):
        # 데이터 준비
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        
        # Forward pass (전체 프레임)
        predictions = model(frames)
        
        # Loss 계산 전에 predictions의 모든 텐서를 device로 이동
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
                
        losses = criterion(predictions, batch)

        if rank == 0:
            print("Predictions:", torch.softmax(torch.stack(predictions['risk_score'], dim=0)[:, 0, :], dim=1)[30:40, :])

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # Gradient clipping
        optimizer.step()

        # 동기화 지점 추가
        torch.cuda.synchronize()
        
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
            
            # if batch_idx % 25 == 0:
            #     if is_positive:
            #         annotations = batch['annotations']
                
            #     toa = 85
            #     try:
            #         attn_in_each_obj_at_toa_largest_obj = predictions['attns_in_each_obj'][toa][largest_obj_idx]
            #     except:
            #         attn_in_each_obj_at_toa_largest_obj = None
            #     obj_attn_at_toa = predictions['obj_attns'][toa]
            #     detection_at_toa = predictions['detections'][toa][2]
                
            #     largest_obj_idx = obj_attn_at_toa.argmax().item()
            #     xyxy_largest_obj = detection_at_toa[largest_obj_idx]
                
            #     visualize_patch_attention(
            #         frame_at_toa=frames[0][toa].cpu().numpy(),
            #         bbox_xyxy=xyxy_largest_obj,
            #         epoch=epoch,
            #         batch_idx=batch_idx,
            #         save_dir="taa/_visualization",
            #         is_frame_in_0to1=True,
            #         attn_map_8x8=attn_in_each_obj_at_toa_largest_obj,
            #         plot_all_detections=detection_at_toa,
            #         obj_attn_at_toa=obj_attn_at_toa
            #     )

    # Epoch 평균 loss 계산 (rank 0만)
    if rank == 0:
        for k in total_losses:
            total_losses[k] /= len(train_loader)
        
        # 에포크 종료 시 요약 통계 저장
        logger.log_epoch_end(epoch)

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
    """검증 수행 (전체 프레임 입력)"""
    model.eval()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'attn_loss': 0., 'total_loss': 0.}

    all_preds = []
    all_targets = []

    if rank == 0:
        val_loader = tqdm(val_loader, desc='Validation')

    for batch in val_loader:
        # 데이터 준비
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        is_positive = batch['is_positive'].to(device)  # (B,)

        # Forward pass
        with torch.no_grad():
            predictions = model(frames)
            
        # Loss 계산
        losses = criterion(predictions, batch)

        # Loss 기록 (rank 0만)
        if rank == 0:
            for k, v in losses.items():
                total_losses[k] += v.item()

        # anomaly probability 계산
        risk_scores = predictions['risk_score']  # List[T] of (B, 2)
        B = risk_scores[0].shape[0]  # batch size
        T = len(risk_scores)  # sequence length
        
        # Stack and reshape predictions
        anomaly_probs = torch.stack(risk_scores, dim=1)  # (B, T, 2)
        anomaly_probs = torch.softmax(anomaly_probs, dim=2)  # Apply softmax
        anomaly_probs = anomaly_probs[:, :, 1]  # (B, T) - Keep positive class probability
        
        # Store predictions and targets
        all_preds.append(anomaly_probs)  # (B, T)
        all_targets.append(is_positive)  # (B,)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)  # (N, T)
    all_targets = torch.cat(all_targets, dim=0)  # (N,)

    # Reshape predictions for evaluation
    B, T = all_preds.shape
    all_preds = all_preds.reshape(-1)  # (N*T,)
    all_targets = all_targets.unsqueeze(1).repeat(1, T).reshape(-1)  # (N*T,)

    # 각 GPU의 예측값 크기
    local_size = torch.tensor([all_preds.size(0)], device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)

    # 모든 GPU의 예측값을 수집하기 위한 텐서 준비
    max_size = max(sizes).item()
    pred_gathered = [torch.zeros(max_size, device=device) for _ in range(dist.get_world_size())]
    target_gathered = [torch.zeros(max_size, device=device) for _ in range(dist.get_world_size())]

    # 예측값 수집을 위한 패딩
    pred_padded = torch.zeros(max_size, device=device)
    pred_padded[:all_preds.size(0)] = all_preds
    target_padded = torch.zeros(max_size, device=device)
    target_padded[:all_targets.size(0)] = all_targets

    # 예측값 수집
    dist.all_gather(pred_gathered, pred_padded)
    dist.all_gather(target_gathered, target_padded)

    # 모든 GPU의 결과 합치기 (rank 0만)
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
            targets=all_targets_combined
        )
        
        return total_losses, metrics

    return total_losses, {}


def train(rank: int, world_size: int, config: Dict[str, Any]):
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
    if config['model']['opticalflow'] == 'neuflow':
        from taa.model.taa_yolov10_neuflow import YOLOv10TAANeuFlowDetectionModel
        model = YOLOv10TAANeuFlowDetectionModel(yolo_id=config['model']['detector'], yolo_ft=config['model']['yolo_ft'], conf_thresh=config['model']['conf_thresh']).to(device)
    else:
        model = YOLOv10TAADetectionModel(yolo_id=config['model']['detector'], yolo_ft=config['model']['yolo_ft'], conf_thresh=config['model']['conf_thresh']).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)
    
    # Loss 함수와 optimizer 설정
    if config['loss']['name'] == 'AdaLEA':
        criterion = AdaLEA(
            lambda_temporal=config['loss']['lambda_temporal'],
            lambda_attn=config['loss']['lambda_attn'],
            fps=config['dataset']['fps'],
            iou_thresh=config['loss']['iou_thresh'],
            gamma=config['loss']['gamma']
        ).to(device)
    else:
        criterion = TAATemporal(
            lambda_temporal=config['loss']['lambda_temporal'],
            lambda_attn=config['loss']['lambda_attn'],
            fps=config['dataset']['fps'],
            iou_thresh=config['loss']['iou_thresh']
        ).to(device)
    
    # select paramters not in detector, flow_model and only requires_grad
    train_paramerters = [p for n, p in model.named_parameters() if 'detector' not in n and 'flow_model' not in n and p.requires_grad]
    
    if config['train']['optimizer'] == 'Adam':
        optimizer = optim.Adam( 
            train_paramerters,  # 학습 가능한 파라미터만 선택
            lr=float(config['train']['learning_rate'])
        )
    elif config['train']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            train_paramerters,  # 학습 가능한 파라미터만 선택
            lr=float(config['train']['learning_rate'])
        )
    elif config['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            train_paramerters,  # 학습 가능한 파라미터만 선택
            lr=float(config['train']['learning_rate'])
        )
    else:
        raise ValueError(f"Invalid optimizer: {config['train']['optimizer']}")
    
    if config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, threshold=0.2, verbose=(rank==0)
        )
    elif config['train']['scheduler'] == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=10, gamma=0.1
        )
    elif config['train']['scheduler'] == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=100, eta_min=1e-5
        )
    else:
        raise ValueError(f"Invalid scheduler: {config['train']['scheduler']}")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_ap = 0
    best_val_mtta = 0
    if config['train'].get('resume', False):
        if os.path.isfile(config['train']['resume_path']):
            if rank == 0:
                print(f"=> loading checkpoint '{config['train']['resume_path']}'")
            
            # Map model to be loaded to specified single gpu
            loc = f'cuda:{rank}'
            checkpoint = torch.load(config['train']['resume_path'], map_location=loc)
            start_epoch = checkpoint['epoch'] + 1
            best_val_ap = checkpoint.get('best_val_ap', 0)
            best_val_mtta = checkpoint.get('best_val_mtta', 0)
            # Load model state
            model.module.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Load scheduler state if it exists
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            if rank == 0:
                print(f"=> loaded checkpoint '{config['train']['resume_path']}' (epoch {checkpoint['epoch']})")
        else:
            if rank == 0:
                print(f"=> no checkpoint found at '{config['train']['resume_path']}'")

    # 데이터 로더 생성
    if config['dataset']['name'] == 'DAD' or config['dataset']['name'] == 'DAD_test':
        if config['dataset']['hdf5']:
            from taa.dataset.DAD_hdf5 import create_dad_loader_hdf5 as create_data_loader
            from taa.dataset.DAD_hdf5 import DistributedWeightedSampler
            from taa.dataset.DAD_hdf5 import DADDatasetHDF5 as Dataset
        else:
            from taa.dataset.DAD import create_dad_loader as create_data_loader
            from taa.dataset.DAD import DistributedWeightedSampler
            from taa.dataset.DAD import DADDataset as Dataset
    elif config['dataset']['name'] == 'ROL':
        from taa.dataset.ROL import create_rol_loader as create_data_loader
        from taa.dataset.ROL import DistributedWeightedSampler
    elif config['dataset']['name'] == 'CCD':  # Add CCD dataset support
        from taa.dataset.CCD import create_ccd_loader as create_data_loader
        from taa.dataset.CCD import DistributedWeightedSampler
        from taa.dataset.CCD import CCDDataset as Dataset
    else:
        raise ValueError(f"Invalid dataset: {config['dataset']['name']}")
    
    # Create train loader based on dataset type
    if config['dataset']['name'] == 'ROL':
        train_loader = create_data_loader(
            root_path=config['dataset']['root_path'],
            split='train',  # ROL uses 'train' instead of 'training'
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            img_size=config['dataset']['img_size'],
            world_size=world_size,
            rank=rank
        )
        
        val_loader = create_data_loader(
            root_path=config['dataset']['root_path'],
            split='val',  # ROL uses 'val' instead of 'testing'
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            img_size=config['dataset']['img_size'],
            world_size=world_size,
            rank=rank
        )
    elif config['dataset']['name'] == 'CCD':  # Add CCD dataset loading
        train_sampler = DistributedWeightedSampler(
            dataset=Dataset(
                root_path=config['dataset']['root_path'],
                split='training',
                img_size=config['dataset']['img_size']
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
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            img_size=config['dataset']['img_size'],
            sampler=val_sampler
        )
    else:
        # Existing DAD dataset loading code
        train_sampler = DistributedWeightedSampler(
            dataset=Dataset(
                root_path=config['dataset']['root_path'],
                split='training',
                transform=None,
                img_size=config['dataset']['img_size']
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
            batch_size=config['train']['batch_size'],
            num_workers=config['train']['num_workers'],
            img_size=config['dataset']['img_size'],
            sampler=val_sampler
        )

    # 메트릭 계산기 초기화 (rank 0만)
    metrics_calculator = TAAMetrics(dataset=config['dataset']['name']) if rank == 0 else None

    # 학습 루프
    for epoch in range(start_epoch, int(config['train']['num_epochs'])):
        # train sampler의 epoch 설정
        train_sampler.set_epoch(epoch)
        # validation sampler의 epoch 설정
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

        # Initialize mtta tensor on all ranks
        mtta_tensor = torch.zeros(1, device=device)
        
        # rank 0만 로깅 및 체크포인트 저장
        if rank == 0:
            mtta = val_metrics.get('mtta', 0.0)  # Default to 0.0 if not present
            mtta_tensor[0] = mtta
            
            # 로그 기록
            logger.log_eval(epoch, val_losses, val_metrics)

            # 학습률 조정
            scheduler.step(val_losses['total_loss'])

            # Save regular checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_ap': best_val_ap,
                'best_val_mtta': best_val_mtta
            }
            
            # Save latest checkpoint
            # torch.save(checkpoint, logger.get_checkpoint_path('latest'))
            
            # save every epoch
            checkpoint['current_ap'] = val_metrics['ap']
            checkpoint['current_mtta'] = val_metrics['mtta']
            torch.save(checkpoint, logger.get_checkpoint_path(f'epoch_{epoch}'))
            
            # Save best model checkpoint
            if val_metrics['ap'] > best_val_ap:
                best_val_ap = val_metrics['ap']
                best_val_mtta = val_metrics['mtta']
                torch.save(checkpoint, logger.get_checkpoint_path('best_model'))
                print(f"Best model saved! AP: {best_val_ap:.4f}, MTTA: {best_val_mtta:.4f}")

            # 현재 epoch 결과 출력
            print(f"\nEpoch {epoch}")
            print("Train:", {k: f"{v:.4f}" for k, v in train_losses.items()})
            print("Val losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})
            print("Val metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
            print("-" * 50)
        
        dist.broadcast(mtta_tensor, src=0)
        
        # 모든 프로세스 동기화
        dist.barrier()

        if config['loss']['name'] == 'AdaLEA':
            criterion.last_mtta = mtta_tensor.item()

    # 정리
    cleanup()

def main():
    # 디버깅 정보 활성화
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    # 설정 파일 로드
    config = Config.load_config('taa/configs/taa_DAD_yolov10.yaml')
    
    # DDP 설정
    world_size = torch.cuda.device_count()
    mp.spawn(train,
             args=(world_size, config),
             nprocs=world_size,
             join=True)

if __name__ == '__main__':
    main()
