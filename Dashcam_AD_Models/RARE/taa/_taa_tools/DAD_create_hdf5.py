"""
Requirements:
    pip install torch torchvision
    pip install h5py
    pip install Pillow
    pip install tqdm

Example usage:
    python create_hdf5.py \
        --root_path ./data/DAD \
        --split training \
        --output_hdf5 ./data/DAD_training_640x640.h5
"""

import os
import argparse
import h5py
import torch
import numpy as np

from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms.functional as F

from typing import Tuple, List, Dict

# (실제로는 user가 작성하신 DADDataset, VideoTransform를 import)
class VideoTransform:
    """
    예시용: 비디오 데이터를 (640,640)으로만 Resize + ToTensor() 변환
    (ColorJitter, Normalize 등은 적용하지 않음)
    """
    def __init__(self, size=(640, 640), is_train=False):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.is_train = is_train

    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        transformed_frames = []
        for frame in frames:
            frame = F.resize(frame, self.size, interpolation=F.InterpolationMode.BILINEAR)
            frame = F.to_tensor(frame)
            transformed_frames.append(frame)
        return torch.stack(transformed_frames)  # (T, C, H, W)


class DADDataset(torch.utils.data.Dataset):
    """
    간단한 예시용 DAD Dataset (기존 사용자 코드와 유사)
    """
    def __init__(
        self,
        root_path: str,
        split: str = 'training',
        transform=None,
        img_size: Tuple[int, int] = (224, 224)
    ):
        self.root_path = root_path
        self.split = split
        self.n_frames = 100  # 각 비디오는 100 프레임
        
        # Transform 설정
        self.transform = transform
        if self.transform is None:
            self.transform = VideoTransform(size=img_size, is_train=(split == 'training'))

        # positive, negative 폴더 스캔
        self.videos = []  # (video_id, is_positive)
        pos_path = os.path.join(self.root_path, 'frames', split, 'positive')
        neg_path = os.path.join(self.root_path, 'frames', split, 'negative')
        if os.path.exists(pos_path):
            for video_id in sorted(os.listdir(pos_path)):
                self.videos.append((video_id, True))
        if os.path.exists(neg_path):
            for video_id in sorted(os.listdir(neg_path)):
                self.videos.append((video_id, False))

    def __len__(self) -> int:
        return len(self.videos)

    def __getitem__(self, idx: int) -> Dict:
        video_id, is_positive = self.videos[idx]
        category = 'positive' if is_positive else 'negative'
        
        # 프레임 로드
        frames = []
        frame_dir = os.path.join(self.root_path, 'frames', self.split, category, video_id)
        for i in range(1, self.n_frames + 1):
            frame_path = os.path.join(frame_dir, f'{i:06d}.jpg')
            img = Image.open(frame_path).convert('RGB')
            frames.append(img)

        # transform -> (T, C, H, W) 텐서
        frames_tensor = self.transform(frames)

        return {
            'frames': frames_tensor,  # (T, C, H, W)
            'video_id': video_id,
            'is_positive': is_positive
        }


def create_hdf5_from_dad(
    root_path: str,
    split: str,
    output_hdf5: str,
    transform_size: Tuple[int, int] = (640, 640),
    num_workers: int = 4
):
    """
    DADDataset을 이용해 HDF5 파일 생성
    - (transform_size)로 resize, to_tensor() 적용
    - HDF5 내부에 'frames/positive', 'frames/negative'로 그룹을 나눈 뒤 video_id 저장
    - 마찬가지로 'labels/positive', 'labels/negative'에도 레이블 저장
    
    Args:
        root_path (str): DAD 데이터셋 루트 (예: ./data/DAD)
        split (str): 'training' 또는 'testing'
        output_hdf5 (str): 생성할 HDF5 파일 경로
        transform_size (tuple): (height, width)
        num_workers (int): DataLoader num_workers
    """
    dataset = DADDataset(
        root_path=root_path,
        split=split,
        transform=VideoTransform(size=transform_size, is_train=False)
    )
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    with h5py.File(output_hdf5, 'w') as h5f:
        # 상위 그룹
        frames_group = h5f.create_group('frames')
        labels_group = h5f.create_group('labels')
        
        # 하위 subgroup: frames/positive, frames/negative
        frames_pos_group = frames_group.create_group('positive')
        frames_neg_group = frames_group.create_group('negative')
        
        # labels/positive, labels/negative
        labels_pos_group = labels_group.create_group('positive')
        labels_neg_group = labels_group.create_group('negative')
        
        print(f"[INFO] Creating HDF5: {output_hdf5}")
        
        for batch in tqdm(loader, total=len(loader)):
            frames_tensor = batch['frames'].squeeze(0)  # (T, C, H, W)
            video_id = batch['video_id'][0]
            is_positive = batch['is_positive'].item()
            
            # float32 변환
            frames_np = frames_tensor.cpu().numpy().astype(np.float32)
            
            # positive, negative에 따라 다른 하위 그룹에 저장
            if is_positive:
                frames_pos_group.create_dataset(
                    video_id, data=frames_np, compression='gzip'
                )
                labels_pos_group.create_dataset(
                    video_id, data=int(is_positive)
                )
            else:
                frames_neg_group.create_dataset(
                    video_id, data=frames_np, compression='gzip'
                )
                labels_neg_group.create_dataset(
                    video_id, data=int(is_positive)
                )

        print(f"[INFO] HDF5 saved to {output_hdf5}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, required=True,
                        help="Path to DAD dataset root, e.g. ./data/DAD")
    parser.add_argument('--split', type=str, default='training',
                        choices=['training', 'testing'], help="Dataset split")
    parser.add_argument('--output_hdf5', type=str, required=True,
                        help="Output HDF5 file path")
    parser.add_argument('--resize', nargs='+', type=int, default=[640, 640],
                        help="Resize height width (default: 640 640)")
    parser.add_argument('--num_workers', type=int, default=4,
                        help="DataLoader num_workers")
    args = parser.parse_args()

    resize_tuple = tuple(args.resize) if len(args.resize) == 2 else (640, 640)
    
    create_hdf5_from_dad(
        root_path=args.root_path,
        split=args.split,
        output_hdf5=args.output_hdf5,
        transform_size=resize_tuple,
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    main()
