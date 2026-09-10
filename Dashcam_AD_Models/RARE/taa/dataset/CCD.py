import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler, DistributedSampler
from typing import List, Tuple, Dict
from taa.dataset.transforms import VideoTransformTrain, VideoTransformVal
import math


class CCDDataset(Dataset):
    """CCD (Car Crash Dataset) 데이터셋을 위한 PyTorch Dataset 클래스"""
    
    def __init__(self, root_path: str, split: str = 'training', transform=None, img_size: Tuple[int, int] = (640, 640)):
        """
        Args:
            root_path (str): 데이터셋의 루트 경로 (e.g., ./data/CCD/)
            split (str): 'training' 또는 'testing'
            transform: 커스텀 transform이 필요한 경우 지정
            img_size (tuple): 이미지 크기 (height, width)
        """
        self.root_path = root_path
        self.split = split
        self.n_frames = 50  # 각 비디오는 50 프레임으로 구성
        
        # Transform 설정
        self.transform = transform
        if transform is None:
            self.transform = VideoTransformTrain(size=img_size) if split == 'training' else VideoTransformVal(size=img_size)
        
        # ToA 정보를 저장할 딕셔너리
        self.toa_dict = {}
        self._load_toa_info()
        
        # 비디오 경로 리스트 생성
        self.videos: List[Tuple[str, bool]] = []  # (video_id, is_positive)
        self._load_video_list()
    
    def _load_toa_info(self):
        """Crash-1500.txt 파일에서 ToA 정보를 로드"""
        annotation_path = os.path.join(self.root_path, 'Crash-1500.txt')
        
        with open(annotation_path, 'r') as f:
            for line in f:
                try:
                    # 쉼표로 분리하되, 대괄호 내부의 쉼표는 보존
                    parts = []
                    in_brackets = False
                    current_part = ''
                    
                    for char in line.strip():
                        if char == '[':
                            in_brackets = True
                        elif char == ']':
                            in_brackets = False
                        elif char == ',' and not in_brackets:
                            parts.append(current_part.strip())
                            current_part = ''
                        else:
                            current_part += char
                    
                    if current_part:
                        parts.append(current_part.strip())
                    
                    video_id = parts[0]
                    bin_labels_str = parts[1]
                    
                    # 문자열을 리스트로 변환
                    bin_labels = [int(x.strip()) for x in bin_labels_str.strip('[]').split(',')]
                    
                    # 첫 번째 1이 나타나는 인덱스 + 1을 ToA로 사용
                    try:
                        toa = bin_labels.index(1) + 1
                        self.toa_dict[video_id] = float(toa)
                    except ValueError:
                        print(f"Warning: No accident frame found for video {video_id}")
                        self.toa_dict[video_id] = 51.  # 기본값으로 51 설정
                
                except Exception as e:
                    print(f"Error processing line: {line.strip()}")
                    print(f"Error details: {str(e)}")
                    continue
    
    def _load_video_list(self):
        """positive와 negative 비디오 리스트를 로드"""
        # Positive 비디오 리스트
        pos_path = os.path.join(self.root_path, 'frames', self.split, 'positive')
        for video_id in sorted(os.listdir(pos_path)):
            self.videos.append((video_id, True))
        
        # number of positive videos
        self.n_pos_videos = len(self.videos)
        
        # Negative 비디오 리스트
        neg_path = os.path.join(self.root_path, 'frames', self.split, 'negative')
        for video_id in sorted(os.listdir(neg_path)):
            self.videos.append((video_id, False))

        # number of negative videos
        self.n_videos = len(self.videos)
        self.n_neg_videos = self.n_videos - self.n_pos_videos
    
    def __len__(self) -> int:
        """데이터셋의 비디오 개수를 반환"""
        return len(self.videos)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        주어진 인덱스에 해당하는 비디오의 프레임들을 반환
        
        Args:
            idx (int): 비디오 인덱스
            
        Returns:
            dict: {
                'frames': torch.Tensor (T, C, H, W),
                'is_positive': bool,
                'video_id': str,
                'toa': float
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
        
        # Transform 적용
        frames = self.transform(frames)
        
        # ToA 값 가져오기
        toa = self.toa_dict.get(video_id, 51.) if is_positive else 51.
        
        return {
            'frames': frames,
            'video_id': video_id,
            'toa': toa,
            'is_positive': is_positive,
        }


def create_weighted_sampler(dataset: CCDDataset) -> WeightedRandomSampler:
    """클래스 불균형을 해결하기 위한 WeightedRandomSampler를 생성합니다."""
    # 각 클래스의 가중치 계산 (샘플 수의 역수)
    pos_weight = 1.0 / dataset.n_pos_videos
    neg_weight = 1.0 / dataset.n_neg_videos
    
    # 각 비디오의 가중치 리스트 생성
    weights = []
    for _, is_positive in dataset.videos:
        weights.append(pos_weight if is_positive else neg_weight)
            
    # torch tensor로 변환
    weights = torch.DoubleTensor(weights)
    
    # Sampler 생성
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(dataset),
        replacement=True
    )
    
    return sampler


class DistributedWeightedSampler(DistributedSampler):
    """클래스 불균형을 해결하면서 분산 학습을 지원하는 Sampler입니다."""
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.weights = self._get_weights(dataset)
        self.num_samples_per_replica = self._get_num_samples_per_replica()
    
    def _get_weights(self, dataset):
        """클래스별 가중치를 계산하고 정규화합니다."""
        n_samples = len(dataset)
        n_pos = dataset.n_pos_videos
        n_neg = dataset.n_neg_videos
        
        pos_weight = n_samples / (2.0 * n_pos) if n_pos > 0 else 0
        neg_weight = n_samples / (2.0 * n_neg) if n_neg > 0 else 0
        
        weights = []
        for _, is_positive in dataset.videos:
            weights.append(pos_weight if is_positive else neg_weight)
        
        weights = torch.DoubleTensor(weights)
        weights = weights / weights.sum()
        
        return weights
    
    def _get_num_samples_per_replica(self):
        """각 GPU당 샘플 수를 계산합니다."""
        total_size = len(self.dataset)
        num_samples = math.ceil(total_size / self.num_replicas)
        return num_samples
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.multinomial(
                self.weights,
                len(self.dataset),
                replacement=True,
                generator=g
            ).tolist()
        else:
            indices = list(range(len(self.dataset)))
        
        total_size = self.num_samples_per_replica * self.num_replicas
        
        if len(indices) < total_size:
            indices += indices[:(total_size - len(indices))]
        
        indices = indices[self.rank:total_size:self.num_replicas]
        assert len(indices) == self.num_samples_per_replica
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples_per_replica


def create_ccd_loader(
    root_path: str,
    split: str = 'training',
    batch_size: int = 1,
    num_workers: int = 4,
    img_size: tuple = (640, 640),
    pin_memory: bool = True,
    drop_last: bool = True,
    sampler = None,
    world_size = None,
    rank = None
) -> DataLoader:
    """분산 학습 환경에서 클래스 불균형을 해결하는 DataLoader를 생성합니다."""
    
    dataset = CCDDataset(
        root_path=root_path,
        split=split,
        img_size=img_size
    )
    
    if sampler is None and split == 'training':
        if world_size is not None and rank is not None:
            sampler = DistributedWeightedSampler(
                dataset=dataset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
        else:
            sampler = create_weighted_sampler(dataset)
    
    shuffle = sampler is None
    if split == 'testing' or split == 'val':
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
    """배치 데이터를 처리하는 함수"""
    frames = torch.stack([item['frames'] for item in batch])
    video_ids = [item['video_id'] for item in batch]
    is_positive = torch.tensor([item['is_positive'] for item in batch], dtype=torch.bool)
    toas = torch.tensor([item['toa'] for item in batch], dtype=torch.float)
    
    return {
        'frames': frames,
        'video_ids': video_ids,
        'is_positive': is_positive,
        'toa': toas,
    }
