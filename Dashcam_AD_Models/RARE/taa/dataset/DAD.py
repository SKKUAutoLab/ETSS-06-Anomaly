import os
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler
from typing import List, Tuple, Dict
from taa.dataset.transforms import VideoTransformTrain, VideoTransformVal
import time
import matplotlib.pyplot as plt
import math


class DADDataset(Dataset):
    """DAD (Driver Accident Dataset) 데이터셋을 위한 PyTorch Dataset 클래스"""
    
    def __init__(self, root_path: str, split: str = 'training', transform=None, img_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            root_path (str): 데이터셋의 루트 경로 (e.g., ./data/DAD/)
            split (str): 'training' 또는 'testing'
            transform: 커스텀 transform이 필요한 경우 지정
            img_size (tuple): 이미지 크기 (height, width)
            detection (bool): use or not the pre-extracted detection from YOLOv10
        """
        self.root_path = root_path
        self.split = split
        self.n_frames = 100  # 각 비디오는 100 프레임으로 구성
        self.toa = 90.  # Changed to float to match HDF5 version
        
        # Transform 설정
        self.transform = transform
        if transform is None:
            self.transform = VideoTransformTrain(size=img_size) if split == 'training' else VideoTransformVal(size=img_size)
        
        # 비디오 경로 리스트 생성
        self.videos: List[Tuple[str, bool]] = []  # (video_id, is_positive)
        self._load_video_list()
    
        # annotation 캐시
        self.annotations: Dict[str, List] = {}
        self._load_annotations()
    
    def _load_video_list(self):
        """positive와 negative 비디오 리스트를 로드"""
        split_path = self.split
        
        # Positive 비디오 리스트
        pos_path = os.path.join(self.root_path, 'frames', split_path, 'positive')
        for video_id in sorted(os.listdir(pos_path)):
            self.videos.append((video_id, True))
        
        # number of positive videos
        self.n_pos_videos = len(self.videos)
        
        # Negative 비디오 리스트
        neg_path = os.path.join(self.root_path, 'frames', split_path, 'negative')
        for video_id in sorted(os.listdir(neg_path)):
            self.videos.append((video_id, False))

        # number of negative videos
        self.n_videos = len(self.videos)
        self.n_neg_videos = self.n_videos - self.n_pos_videos
    
    def _load_annotations(self):
        """Positive 비디오의 annotation 파일들을 프레임 단위로 로드하여 저장"""
        self.annotations = {}  # 비디오별 annotation을 저장할 딕셔너리
        
        for video_id, is_positive in self.videos:
            if is_positive:
                anno_path = os.path.join(self.root_path, 'annotation', f'{video_id}.txt')
                if os.path.exists(anno_path):
                    # 비디오별로 프레임 단위 딕셔너리 초기화
                    frame_annotations = {}
                    
                    with open(anno_path, 'r') as f:
                        lines = f.readlines()
                    
                    # 각 줄을 읽어서 프레임별로 정리
                    for line in lines:
                        parts = line.strip().split()
                        frame_idx = int(parts[0])
                        
                        # annotation 정보 구성
                        obj_info = {
                            'track_id': int(parts[1]),
                            'class': parts[2],
                            'bbox': [float(x) for x in parts[3:7]],  # [x1, y1, x2, y2]
                            'is_related': int(parts[7])
                        }
                        
                        # 해당 프레임이 처음 나오는 경우 리스트 초기화
                        if frame_idx not in frame_annotations:
                            frame_annotations[frame_idx] = []
                        
                        # 프레임에 객체 정보 추가
                        frame_annotations[frame_idx].append(obj_info)
                    
                    self.annotations[video_id] = frame_annotations
    
    def __len__(self) -> int:
        """데이터셋의 비디오 개수를 반환"""
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        주어진 인덱스에 해당하는 비디오의 프레임들과 annotation을 반환
        
        Args:
            idx (int): 비디오 인덱스
            
        Returns:
            dict: {
                'frames': torch.Tensor (T, C, H, W),
                'is_positive': bool,
                'video_id': str,
                'annotations': List[dict] (positive 비디오인 경우)
            }
        """
        video_id, is_positive = self.videos[idx]
        category = 'positive' if is_positive else 'negative'
        
        # 프레임 로드
        frames = []
        frame_dir = os.path.join(self.root_path, 'frames', self.split, category, video_id)
        
        for i in range(1, self.n_frames + 1):
            frame_path = os.path.join(frame_dir, f'{i:06d}.jpg')
            frame = Image.open(frame_path).convert('RGB')
            frames.append(frame)
        
        # Transform 적용 (모든 프레임에 동일한 augmentation)
        frames = self.transform(frames)  # (T, C, H, W)
        
        result = {
            'frames': frames,
            'video_id': video_id,
            'toa': 90. if is_positive else 101.,  # Changed to float to match HDF5 version
            'is_positive': is_positive,
        }
        
        # Positive 비디오의 경우 annotation 추가
        if is_positive and video_id in self.annotations:
            result['annotations'] = self.annotations[video_id]
        
        return result

def create_weighted_sampler(dataset: DADDataset) -> WeightedRandomSampler:
    """클래스 불균형을 해결하기 위한 WeightedRandomSampler를 생성합니다.
    
    각 샘플의 가중치는 해당 클래스의 샘플 수에 반비례하도록 설정됩니다.
    이를 통해 각 배치에서 positive와 negative 샘플이 균형있게 선택됩니다.
    
    Args:
        dataset: DAD 데이터셋 인스턴스
        
    Returns:
        WeightedRandomSampler: 가중치가 적용된 sampler
    """
    # 각 클래스의 가중치 계산 (샘플 수의 역수)
    pos_weight = 1.0 / dataset.n_pos_videos
    neg_weight = 1.0 / dataset.n_neg_videos
    
    # 각 비디오의 가중치 리스트 생성
    weights = []
    for _, is_positive in dataset.videos:
        if is_positive:
            weights.append(pos_weight)
        else:
            weights.append(neg_weight)
            
    # torch tensor로 변환
    weights = torch.DoubleTensor(weights)
    
    # Sampler 생성 (전체 데이터셋 크기만큼 샘플링)
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True  # 복원 추출을 통해 클래스 균형 유지
    )
    
    return sampler

class DistributedWeightedSampler(DistributedSampler):
    """클래스 불균형을 해결하면서 분산 학습을 지원하는 Sampler입니다."""
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        
        # 클래스별 가중치 계산
        self.weights = self._get_weights(dataset)
        
        # 각 GPU당 샘플 수 계산
        self.num_samples_per_replica = self._get_num_samples_per_replica()
    
    def _get_weights(self, dataset):
        """클래스별 가중치를 계산하고 정규화합니다."""
        n_samples = len(dataset)
        n_pos = dataset.n_pos_videos
        n_neg = dataset.n_neg_videos
        
        # 클래스 비율의 역수로 가중치 계산
        pos_weight = n_samples / (2.0 * n_pos) if n_pos > 0 else 0
        neg_weight = n_samples / (2.0 * n_neg) if n_neg > 0 else 0
        
        weights = []
        for _, is_positive in dataset.videos:
            weights.append(pos_weight if is_positive else neg_weight)
        
        # 가중치를 텐서로 변환하고 정규화
        weights = torch.DoubleTensor(weights)
        weights = weights / weights.sum()  # 정규화
        
        return weights
    
    def _get_num_samples_per_replica(self):
        """각 GPU당 샘플 수를 계산합니다."""
        total_size = len(self.dataset)
        num_samples = math.ceil(total_size / self.num_replicas)
        return num_samples
    
    def __iter__(self):
        if self.shuffle:
            # epoch별로 동일한 셔플을 보장하기 위한 시드 설정
            g = torch.Generator()
            g.manual_seed(self.epoch)
            
            # 전체 데이터셋에 대한 weighted sampling
            indices = torch.multinomial(
                self.weights,
                len(self.dataset),
                replacement=True,
                generator=g
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        # 전체 샘플 수를 num_replicas로 나누어 떨어지게 조정
        total_size = self.num_samples_per_replica * self.num_replicas
        
        # 부족한 샘플을 채우기 위해 마지막 샘플들을 복제
        if len(indices) < total_size:
            indices += indices[:(total_size - len(indices))]
        
        # 현재 GPU(rank)에 해당하는 샘플만 선택
        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_replica
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples_per_replica

def create_dad_loader(
    root_path: str,
    split: str = 'training',
    batch_size: int = 1,
    num_workers: int = 4,
    img_size: tuple = (224, 224),
    pin_memory: bool = True,
    drop_last: bool = True,
    sampler = None,
    world_size = None,
    rank = None
) -> DataLoader:
    """분산 학습 환경에서 클래스 불균형을 해결하는 DataLoader를 생성합니다."""
    
    dataset = DADDataset(
        root_path=root_path,
        split=split,
        img_size=img_size
    )
    
    # sampler가 직접 전달되지 않은 경우
    if sampler is None and split == 'training':
        if world_size is not None and rank is not None:
            # 분산 학습 환경에서는 DistributedWeightedSampler 사용
            sampler = DistributedWeightedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
        else:
            # 단일 GPU 환경에서는 일반 WeightedRandomSampler 사용
            sampler = create_weighted_sampler(dataset)
    
    shuffle = sampler is None  # sampler를 사용할 때는 shuffle=False
    if split == 'testing' or 'val':
        shuffle = False
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        sampler=sampler
    )
    
    return loader


def collate_fn(batch: list) -> Dict[str, torch.Tensor]:
    """배치 데이터를 처리하는 함수

    Args:
        batch (list): DADDataset.__getitem__()의 반환값 리스트

    Returns:
        Dict[str, torch.Tensor]: {
            'frames': (B, T, C, H, W),
            'video_ids': list of str,
            'is_positive': (B,),
            'annotations': list of annotations (optional)
        }
    """
    # 배치 크기
    batch_size = len(batch)
    
    # 프레임 스택
    frames = torch.stack([item['frames'] for item in batch])  # (B, T, C, H, W)
    
    # 비디오 ID 리스트
    video_ids = [item['video_id'] for item in batch]
    
    # Positive/Negative 레이블
    is_positive = torch.tensor([item['is_positive'] for item in batch], dtype=torch.bool)
    toas = torch.tensor([item['toa'] for item in batch], dtype=torch.float)  # Changed to float
    result = {
        'frames': frames,
        'video_ids': video_ids,
        'is_positive': is_positive,
        'toa': toas,  # Changed name to match HDF5 version
    }
    
    # Annotation이 있는 경우 추가
    if 'annotations' in batch[0]:
        annotations = [item.get('annotations', []) for item in batch]
        result['annotations'] = annotations
    
    return result


if __name__ == '__main__':
    import time
    from collections import defaultdict
    import matplotlib.pyplot as plt
    
    def validate_dataset(root_path, split, batch_size=1):
        print(f"\n{'='*20} Validating {split} dataset {'='*20}")
        
        # 1. 데이터 로더 생성 및 기본 정보 출력
        start_time = time.time()
        loader = create_dad_loader(
            root_path=root_path,
            split=split,
            batch_size=batch_size,
            num_workers=4
        )
        print(f"Dataset size: {len(loader.dataset)} videos")
        print(f"Number of batches: {len(loader)}")
        
        # 2. Positive/Negative 비디오 분포 확인
        print(f"\nClass distribution:")
        print(f"- Positive videos: {loader.dataset.n_pos_videos} ({loader.dataset.n_pos_videos/loader.dataset.n_videos*100:.2f}%)")
        print(f"- Negative videos: {loader.dataset.n_neg_videos} ({loader.dataset.n_neg_videos/loader.dataset.n_videos*100:.2f}%)")
        
        # 3. 첫 번째 배치 상세 검증
        print("\nValidating first batch...")
        batch = next(iter(loader))
        
        # 기본 데이터 형식 및 shape 확인
        print("\nBatch content verification:")
        print(f"- frames shape: {batch['frames'].shape}")
        print(f"- video_ids: {batch['video_ids']}")
        print(f"- is_positive shape: {batch['is_positive'].shape}")
        print(f"- is_positive: {batch['is_positive']}")
        
        # 데이터 범위 확인
        frames_min = batch['frames'].min().item()
        frames_max = batch['frames'].max().item()
        print(f"- frames value range: [{frames_min:.3f}, {frames_max:.3f}]")
        
        # value could be in range of [0, 1]
        assert frames_min >= 0 and frames_max <= 1, "Frames value should be in range of [0, 1]"
        
        
        # 4. Annotation 검증 (training split의 경우)
        if split == 'training' and 'annotations' in batch:
            print("\nAnnotation verification:")
            for i, video_annos in enumerate(batch['annotations']):
                if video_annos:  # positive 비디오인 경우
                    print(f"\nVideo {i} ({batch['video_ids'][i]}):")
                    
                    # Annotation 통계
                    frame_counts = defaultdict(int)
                    object_classes = defaultdict(int)
                    related_objects = 0
                    
                    for anno in video_annos:
                        frame_counts[anno['frame_idx']] += 1
                        object_classes[anno['class']] += 1
                        if anno['is_related']:
                            related_objects += 1
                    
                    print(f"- Total annotations: {len(video_annos)}")
                    print(f"- Unique frames with annotations: {len(frame_counts)}")
                    print(f"- Object classes distribution: {dict(object_classes)}")
                    print(f"- Related objects: {related_objects}")
                    
                    # BBox 범위 검증
                    invalid_bbox = 0
                    for anno in video_annos:
                        bbox = anno['bbox']
                        if not (0 <= bbox[0] <= 1 and 0 <= bbox[1] <= 1 and 
                               0 <= bbox[2] <= 1 and 0 <= bbox[3] <= 1 and
                               bbox[2] > bbox[0] and bbox[3] > bbox[1]):
                            invalid_bbox += 1
                    if invalid_bbox > 0:
                        print(f"- Warning: {invalid_bbox} invalid bounding boxes found!")
        
        # 5. 데이터 로딩 성능 측정
        print("\nPerformance test:")
        start_time = time.time()
        num_batches = min(10, len(loader))  # 최대 10개 배치로 테스트
        
        for i, batch in enumerate(loader):
            if i >= num_batches - 1:
                break
        
        total_time = time.time() - start_time
        fps = (num_batches * batch_size * 100) / total_time  # 100은 프레임 수
        print(f"- Loading speed: {fps:.2f} frames/second")
        print(f"- Average batch loading time: {total_time/num_batches:.3f} seconds")
        
        # 6. 메모리 사용량 확인 (GPU 사용 시)
        if torch.cuda.is_available():
            print("\nGPU Memory usage:")
            print(f"- Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            print(f"- Cached: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
        
        # 7. 샘플 시각화
        print("\nSaving sample visualization...")
        sample_video_idx = 0
        sample_frame_idx = 49  # 중간 프레임
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        frame = batch['frames'][sample_video_idx, sample_frame_idx].permute(1, 2, 0)
        frame = (frame - frame.min()) / (frame.max() - frame.min())  # Normalize for visualization
        plt.imshow(frame)
        plt.title(f"Video: {batch['video_ids'][sample_video_idx]}\nFrame: {sample_frame_idx}")
        
        if 'annotations' in batch and batch['annotations'][sample_video_idx]:
            # 해당 프레임의 annotation 표시
            frame_annos = [
                anno for anno in batch['annotations'][sample_video_idx] 
                if anno['frame_idx'] == sample_frame_idx
            ]
            
            for anno in frame_annos:
                bbox = anno['bbox']
                x1, y1, x2, y2 = bbox
                h, w = frame.shape[:2]
                rect = plt.Rectangle(
                    (x1 * w, y1 * h), 
                    (x2 - x1) * w, 
                    (y2 - y1) * h,
                    fill=False,
                    color='red' if anno['is_related'] else 'blue'
                )
                plt.gca().add_patch(rect)
        
        plt.subplot(1, 2, 2)
        frame = batch['frames'][sample_video_idx, -1].permute(1, 2, 0)
        frame = (frame - frame.min()) / (frame.max() - frame.min())
        plt.imshow(frame)
        plt.title(f"Last frame")
        
        plt.savefig(f'taa/dataset/DAD_dataset_sample_{split}.png')
        plt.close()
        
        print(f"\n{'='*20} Validation completed {'='*20}\n")
    
    # 메인 실행부
    root_path = "./taa/data/DAD"  # 데이터셋 경로를 적절히 수정해주세요
    
    # Training set 검증
    validate_dataset(root_path, 'training', batch_size=1)  # Changed batch_size to 1
    
    # Testing set 검증
    validate_dataset(root_path, 'testing', batch_size=1)   # Changed batch_size to 1
    
    print("Dataset validation completed!")