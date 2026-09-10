import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from typing import Dict, Any
from datetime import timedelta
from taa.util.metrics import TAAMetrics
from taa.model.taa_yolov10 import YOLOv10TAADetectionModel, TAATemporal, AdaLEA
from taa.configs.config import Config


def setup(rank: int, world_size: int):
    """DDP 설정"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29557'
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, timeout=timedelta(seconds=50))


def cleanup():
    """DDP 정리"""
    dist.destroy_process_group()


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: Any,
    criterion: nn.Module,
    metrics_calculator: TAAMetrics,
    device: torch.device,
    rank: int
) -> Dict[str, float]:
    """검증 수행 (전체 프레임 입력)"""
    model.eval()
    total_losses = {'ce_loss': 0., 'temporal_loss': 0., 'attn_loss': 0., 'total_loss': 0.}

    all_preds = []
    all_targets = []

    if rank == 0:
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


def evaluate(rank: int, world_size: int, config: Dict[str, Any], checkpoint_path: str):
    """단일 GPU에서의 평가 프로세스"""
    # DDP 설정
    setup(rank, world_size)

    # 디바이스 설정
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)

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
        
    model.detector.fuse()
    model = model.train()
    model = model.eval()
    # 체크포인트 로드
    if os.path.isfile(checkpoint_path):
        if rank == 0:
            print(f"=> loading checkpoint '{checkpoint_path}'")
            
        loc = f'cuda:{rank}'

        checkpoint = torch.load(checkpoint_path, map_location=loc)
        # Load the full state dictionary from the checkpoint
        full_state_dict = checkpoint['model_state_dict']
        model.load_state_dict(full_state_dict, strict=True)
        # model.detector.fuse()
        
        # filtered_state_dict = {k: v for k, v in full_state_dict.items() if not (k.startswith('detector') or k.startswith('flow_model'))}
        # model.load_state_dict(filtered_state_dict, strict=False)
        if rank == 0:
            print(f"=> loaded checkpoint '{checkpoint_path}'")
            if 'current_ap' in checkpoint:
                print(f"=> This checkpoint AP {checkpoint['current_ap']:.4f}")
            if 'current_mtta' in checkpoint:
                print(f"=> This checkpoint MTTA {checkpoint['current_mtta']:.4f}")
    else:
        if rank == 0:
            print(f"=> no checkpoint found at '{checkpoint_path}'")
            return

    model = DDP(model, device_ids=[rank], find_unused_parameters=False, broadcast_buffers=False)

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
            from taa.dataset.DAD_hdf5 import DADDatasetHDF5 as Dataset
        else:
            from taa.dataset.DAD import create_dad_loader as create_data_loader
            from taa.dataset.DAD import DADDataset as Dataset
    elif config['dataset']['name'] == 'ROL':
        from taa.dataset.ROL import create_rol_loader as create_data_loader
    elif config['dataset']['name'] == 'CCD':
        from taa.dataset.CCD import create_ccd_loader as create_data_loader
        from taa.dataset.CCD import CCDDataset as Dataset
    else:
        raise ValueError(f"Invalid dataset: {config['dataset']['name']}")

    # Create validation sampler and loader
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

    # 평가 수행
    val_losses, val_metrics = validate(
        model, val_loader, criterion, metrics_calculator,
        device, rank
    )

    # rank 0만 결과 출력
    if rank == 0:
        print("\nEvaluation Results:")
        print("Losses:", {k: f"{v:.4f}" for k, v in val_losses.items()})
        print("Metrics:", {k: f"{v:.4f}" for k, v in val_metrics.items()})

    # 정리
    cleanup()
    
    dist.barrier()  # 모든 프로세스 동기화
    dist.destroy_process_group()


def main():
    # 디버깅 정보 활성화
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    
    # 설정 파일 로드
    config = Config.load_config('taa/configs/taa_DAD_yolov10.yaml')
    
    # 체크포인트 경로 설정
    checkpoint_path = 'taa/_experiments/20250204_063847/checkpoints/epoch_46.pth'  # Update with actual checkpoint path

    # DDP 설정
    world_size = torch.cuda.device_count()
    mp.spawn(evaluate,
             args=(world_size, config, checkpoint_path),
             nprocs=world_size,
             join=True)


if __name__ == '__main__':
    main() 