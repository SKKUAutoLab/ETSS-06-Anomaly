import os
import time
from datetime import date
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from taa.dataset.DAD import create_dad_loader
from taa.dataset.DAD_hdf5 import create_dad_loader_hdf5
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from taa.util.logger import ExpLogger
from taa.util.metrics import TAAMetrics
from taa.util.visualization import visualize_patch_attention
from taa.model.taa_yolov10 import YOLOv10TAADetectionModel, TAATemporal
from taa.configs.config import Config
import cv2

def train_epoch(
    model: nn.Module,
    train_loader: Any,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: ExpLogger,
) -> Dict[str, float]:
    """한 epoch 학습 (전체 프레임 입력)"""
    model.train()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'attn_loss': 0., 'total_loss': 0.}

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

        print("Predictions:", torch.softmax(torch.stack(predictions['risk_score'], dim=0)[:, 0, :], dim=1)[80:90, :])

        # Backward pass
        optimizer.zero_grad()
        losses['total_loss'].backward()
        optimizer.step()

        # Loss 기록
        for k, v in losses.items():
            total_losses[k] += v.item()

        current_avg_losses = {k: v / (batch_idx + 1) for k, v in total_losses.items()}
        train_loader.set_postfix({
            **{f"batch_{k}": f"{v.item():.4f}" for k, v in losses.items()},
            **{f"avg_{k}": f"{v:.4f}" for k, v in current_avg_losses.items()}
        })

        # 로그 기록
        lr = optimizer.param_groups[0]['lr']
        logger.log_train_step(epoch, batch_idx, {k: v.item() for k, v in losses.items()}, lr)

    # Epoch 평균 loss 계산
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
) -> Dict[str, float]:
    """검증 수행 (전체 프레임 입력)"""
    model.eval()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'attn_loss': 0., 'total_loss': 0.}

    all_preds = []
    all_targets = []

    val_loader = tqdm(val_loader, desc='Validation')

    for batch in val_loader:
        # 데이터 준비
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        is_positive = batch['is_positive'].to(device)  # (B,)

        # Forward pass (전체 프레임)
        with torch.no_grad():
            predictions = model(frames)
        if is_positive:
            predictions['annotations'] = batch['annotations'][0]
        losses = criterion(predictions, is_positive)

        # Loss 기록
        for k, v in losses.items():
            total_losses[k] += v.item()

        # anomaly probability 계산 softmax
        anomaly_probs = torch.stack(predictions['risk_score'], dim=0)[:, 0, :]  # (T, 1)
        anomaly_probs = torch.softmax(anomaly_probs, dim=1)[:, 1:]

        # 예측값 저장
        all_preds.append(anomaly_probs.cpu())
        all_targets.append(is_positive.cpu())

    # 예측값 결합
    T = all_preds[0].size(0)  # T
    all_preds = torch.cat(all_preds, dim=0)  # List[Tensor([T, 1])] -> Tensor([N*T, 1])
    all_targets = torch.cat(all_targets, dim=0)  # List[Tensor([True])] -> Tensor([N])
    all_targets = all_targets.unsqueeze(1).repeat(1, T).view(-1)  # Tensor([N]) -> Tensor([N*T])

    # 메트릭 계산
    metrics = metrics_calculator.calculate_metrics(
        predictions=all_preds,
        targets=all_targets,
    )

    return total_losses, metrics

def train(config: Dict[str, Any]):
    # 로거 초기화
    logger = ExpLogger()
    logger.save_exp_info(config)

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 모델 생성
    if config['model']['opticalflow'] == 'neuflow':
        from taa.model.taa_yolov10_neuflow import YOLOv10TAANeuFlowDetectionModel
        model = YOLOv10TAANeuFlowDetectionModel(
            yolo_id=config['model']['detector'],
            yolo_ft=config['model']['yolo_ft'],
            conf_thresh=config['model']['conf_thresh']
        ).to(device)
    else:
        model = YOLOv10TAADetectionModel(
            yolo_id=config['model']['detector'],
            yolo_ft=config['model']['yolo_ft'],
            conf_thresh=config['model']['conf_thresh']
        ).to(device)

    # Loss 함수와 optimizer 설정
    criterion = TAATemporal(
        lambda_temporal=config['loss']['lambda_temporal'],
        lambda_attn=config['loss']['lambda_attn'],
        fps=config['dataset']['fps'],
        iou_thresh=config['loss']['iou_thresh']
    ).to(device)
    
    # select paramters not in detector, flow_model and only requires_grad
    train_paramerters = [p for n, p in model.named_parameters() if 'detector' not in n and 'flow_model' not in n and p.requires_grad]
    
    # Optimizer 설정
    if config['train']['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            train_paramerters,
            lr=float(config['train']['learning_rate'])
        )
    elif config['train']['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(
            train_paramerters,
            lr=float(config['train']['learning_rate'])
        )
    elif config['train']['optimizer'] == 'SGD':
        optimizer = optim.SGD(
            train_paramerters,
            lr=float(config['train']['learning_rate'])
        )
    else:
        raise ValueError(f"Invalid optimizer: {config['train']['optimizer']}")
    
    # Scheduler 설정
    if config['train']['scheduler'] == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=2, threshold=0.2, verbose=True
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
    if config['train'].get('resume', False):
        if os.path.isfile(config['train']['resume_path']):
            print(f"=> loading checkpoint '{config['train']['resume_path']}'")
            checkpoint = torch.load(config['train']['resume_path'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_ap = checkpoint.get('best_val_ap', 0)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'scheduler_state_dict' in checkpoint and scheduler is not None:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print(f"=> loaded checkpoint '{config['train']['resume_path']}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{config['train']['resume_path']}'")

    # 데이터 로더 생성
    if config['dataset']['name'] == 'DAD' or config['dataset']['name'] == 'DAD_test':
        if config['dataset']['hdf5']:
            from taa.dataset.DAD_hdf5 import create_dad_loader_hdf5 as create_data_loader
            from taa.dataset.DAD_hdf5 import DADDatasetHDF5 as Dataset
        else:
            from taa.dataset.DAD import create_dad_loader as create_data_loader
            from taa.dataset.DAD import Dataset
    elif config['dataset']['name'] == 'ROL':
        from taa.dataset.ROL import create_rol_loader as create_data_loader
        from taa.dataset.ROL import ROLDataset as Dataset
    else:
        raise ValueError(f"Invalid dataset: {config['dataset']['name']}")

    train_loader = create_data_loader(
        root_path=config['dataset']['root_path'],
        split='training',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        img_size=config['dataset']['img_size']
    )

    val_loader = create_data_loader(
        root_path=config['dataset']['root_path'],
        split='testing',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        img_size=config['dataset']['img_size']
    )

    # 메트릭 계산기 초기화
    metrics_calculator = TAAMetrics()

    # 학습 루프
    for epoch in range(start_epoch, int(config['train']['num_epochs'])):
        # 학습
        train_losses = train_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, logger
        )

        # 검증
        val_losses, val_metrics = validate(
            model, val_loader, criterion, metrics_calculator,
            device, epoch
        )

        # 로그 기록
        logger.log_eval(epoch, val_losses, val_metrics)

        # 학습률 조정
        scheduler.step(val_losses['total_loss'])

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_ap': best_val_ap,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, logger.get_checkpoint_path('latest'))
        
        # Save best model checkpoint
        if val_metrics['ap'] > best_val_ap:
            best_val_ap = val_metrics['ap']
            torch.save(checkpoint, logger.get_checkpoint_path('best_model'))
            print(f"Best model saved! AP: {best_val_ap:.4f}")

        # 현재 epoch 결과 출력
        print(f"\nEpoch {epoch}")
        print("Train:", {k: f"{v:.4f}" for k, v in train_losses.items()})
        print("Val losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})
        print("Val metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})
        print("-" * 50)

def main():
    # 설정 파일 로드
    config = Config.load_config('taa/configs/taa_DAD_yolov10.yaml')
    train(config)

if __name__ == '__main__':
    main() 
