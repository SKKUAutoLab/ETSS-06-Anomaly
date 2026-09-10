import os
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Any
from taa.util.metrics import TAAMetrics
from taa.model.taa_yolov10 import YOLOv10TAADetectionModel, TAATemporal, AdaLEA
from taa.configs.config import Config


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: Any,
    criterion: nn.Module,
    metrics_calculator: TAAMetrics,
    device: torch.device
) -> Dict[str, float]:
    """검증 수행 (전체 프레임 입력)"""
    model.eval()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'attn_loss': 0., 'total_loss': 0.}

    all_preds = []
    all_targets = []

    val_loader = tqdm(val_loader, desc='Evaluation')

    for batch in val_loader:
        # 데이터 준비
        frames = batch['frames'].to(device)  # (B, T, C, H, W)
        is_positive = batch['is_positive'].to(device)  # (B,)

        # Forward pass
        with torch.no_grad():
            predictions = model(frames)
            
        # Loss 계산
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
                
        losses = criterion(predictions, batch)

        # Loss 기록
        for k, v in losses.items():
            total_losses[k] += v.item()

        # anomaly probability 계산
        risk_scores = predictions['risk_score']  # List[T] of (B, 2)
        
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

    # 메트릭 계산
    metrics = metrics_calculator.calculate_metrics(
        predictions=all_preds,
        targets=all_targets
    )

    # Loss 평균 계산
    for k in total_losses:
        total_losses[k] /= len(val_loader)
        
    return total_losses, metrics


def evaluate(config: Dict[str, Any], checkpoint_path: str):
    """단일 GPU에서의 평가 프로세스"""
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
    
    # 체크포인트 로드
    if os.path.isfile(checkpoint_path):
        print(f"=> loading checkpoint '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # YOLOv10 모델의 가중치는 제외하고 나머지 state_dict만 로드
        state_dict = checkpoint['model_state_dict']
        # filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('detector.')}
        model.detector.fuse()
        model = model.train()
        model = model.eval()
        # strict=False로 설정하여 detector 부분을 제외하고 로드
        model.load_state_dict(state_dict, strict=True)
        print(f"=> loaded checkpoint '{checkpoint_path}'")
        
        if 'best_val_ap' in checkpoint:
            print(f"=> This checkpoint AP {checkpoint['best_val_ap']:.4f}")
        if 'best_val_mtta' in checkpoint:
            print(f"=> This checkpoint MTTA {checkpoint['best_val_mtta']:.4f}")
    else:
        print(f"=> no checkpoint found at '{checkpoint_path}'")
        return

    # Loss 함수 설정
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

    # 데이터 로더 생성
    if config['dataset']['name'] == 'DAD' or config['dataset']['name'] == 'DAD_test':
        if config['dataset']['hdf5']:
            from taa.dataset.DAD_hdf5 import create_dad_loader_hdf5 as create_data_loader
        else:
            from taa.dataset.DAD import create_dad_loader as create_data_loader
    elif config['dataset']['name'] == 'ROL':
        from taa.dataset.ROL import create_rol_loader as create_data_loader
    elif config['dataset']['name'] == 'CCD':
        from taa.dataset.CCD import create_ccd_loader as create_data_loader
    else:
        raise ValueError(f"Invalid dataset: {config['dataset']['name']}")

    val_loader = create_data_loader(
        root_path=config['dataset']['root_path'],
        split='testing',
        batch_size=config['train']['batch_size'],
        num_workers=config['train']['num_workers'],
        img_size=config['dataset']['img_size']
    )

    # 메트릭 계산기 초기화
    metrics_calculator = TAAMetrics(dataset=config['dataset']['name'])

    # 평가 수행
    val_losses, val_metrics = validate(
        model, val_loader, criterion, metrics_calculator, device
    )

    # 결과 출력
    print("\nEvaluation Results:")
    print("Losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})
    print("Metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})


def main():
    # 설정 파일 로드
    config = Config.load_config('taa/configs/taa_DAD_yolov10.yaml')
    
    # 체크포인트 경로 설정
    checkpoint_path = 'taa/_experiments/20251102_153919/checkpoints/best_model.pth'

    # 평가 실행
    evaluate(config, checkpoint_path)


if __name__ == '__main__':
    main() 
