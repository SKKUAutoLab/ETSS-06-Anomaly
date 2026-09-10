import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import numbers
import random
import numpy as np
from typing import List, Tuple, Union


class VideoTransform:
    """
    비디오 데이터를 위한 Transform 클래스
    
    한 비디오의 모든 프레임에 동일한 augmentation을 적용합니다.
    이벤트 소실을 방지하기 위해 spatial information을 보존하는 augmentation만 사용합니다.
    """
    
    def __init__(
        self,
        size: Union[int, Tuple[int, int], List[int]] = (640, 640),
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
        is_train: bool = True
    ):
        """
        Args:
            size: 출력 이미지 크기 (height, width)
            mean: 정규화를 위한 RGB 평균값
            std: 정규화를 위한 RGB 표준편차
            is_train: 학습 모드 여부 (augmentation 적용)
        """
        self.size = tuple(size) if isinstance(size, list) else size
        self.size = self.size if isinstance(self.size, tuple) else (self.size, self.size)
        self.mean = mean
        self.std = std
        self.is_train = is_train
        
        # Random augmentation을 위한 확률
        self.flip_prob = 0.5
        self.color_jitter_prob = 0.8
        
        # Color jittering 범위
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.1
        
    def _get_params(self) -> dict:
        """비디오의 모든 프레임에 적용할 augmentation 파라미터 생성"""
        params = {}
        
        if self.is_train:
            # Horizontal flip
            params['flip'] = random.random() < self.flip_prob
            
            # Color jittering
            params['color_jitter'] = random.random() < self.color_jitter_prob
            if params['color_jitter']:
                params['brightness_factor'] = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                params['contrast_factor'] = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                params['saturation_factor'] = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
                params['hue_factor'] = random.uniform(-self.hue, self.hue)
        
        return params
    
    def __call__(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """
        비디오의 모든 프레임에 동일한 transform 적용
        
        Args:
            frames: PIL Image 리스트 [T]
            
        Returns:
            torch.Tensor: 변환된 프레임 텐서 (T, C, H, W)
        """
        # 모든 프레임에 적용할 augmentation 파라미터 생성
        params = self._get_params()
        
        # Transform 적용
        transformed_frames = []
        for frame in frames:
            # Resize
            frame = F.resize(frame, self.size, interpolation=F.InterpolationMode.BILINEAR)
            
            if self.is_train:
                # Horizontal flip
                if params['flip']:
                    frame = F.hflip(frame)
                
                # Color jittering
                if params['color_jitter']:
                    frame = F.adjust_brightness(frame, params['brightness_factor'])
                    frame = F.adjust_contrast(frame, params['contrast_factor'])
                    frame = F.adjust_saturation(frame, params['saturation_factor'])
                    frame = F.adjust_hue(frame, params['hue_factor'])
            
            # ToTensor (if input is PIL Image)
            frame = F.to_tensor(frame)
            
            # Normalize
            # frame = F.normalize(frame, mean=self.mean, std=self.std)
            
            transformed_frames.append(frame)
        
        # Stack frames
        return torch.stack(transformed_frames)


class VideoTransformTrain(VideoTransform):
    """학습용 Transform"""
    def __init__(self, size=(640, 640)):
        super().__init__(size=size, is_train=True)


class VideoTransformVal(VideoTransform):
    """검증용 Transform"""
    def __init__(self, size=(640, 640)):
        super().__init__(size=size, is_train=False)


class VideoTransformHDF5:
    """
    HDF5 데이터를 위한 Transform 클래스
    
    이미 640x640 크기로 정규화된 텐서 상태의 데이터를 처리합니다.
    resize, to_tensor, normalize 과정을 건너뛰고 augmentation만 적용합니다.
    """
    
    def __init__(self, is_train: bool = True):
        """
        Args:
            is_train: 학습 모드 여부 (augmentation 적용)
        """
        self.is_train = is_train
        
        # Random augmentation을 위한 확률
        self.flip_prob = 0.5
        self.color_jitter_prob = 0.8
        
        # Color jittering 범위
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        
    def _get_params(self) -> dict:
        """비디오의 모든 프레임에 적용할 augmentation 파라미터 생성"""
        params = {}
        
        if self.is_train:
            # Horizontal flip
            params['flip'] = random.random() < self.flip_prob
            
            # Color jittering
            params['color_jitter'] = random.random() < self.color_jitter_prob
            if params['color_jitter']:
                params['brightness_factor'] = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
                params['contrast_factor'] = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
                params['saturation_factor'] = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
        
        return params
    
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        """
        비디오의 모든 프레임에 동일한 transform 적용
        
        Args:
            frames: 정규화된 프레임 텐서 (T, C, H, W)
            
        Returns:
            torch.Tensor: augmentation이 적용된 프레임 텐서 (T, C, H, W)
        """
            
        # 모든 프레임에 적용할 augmentation 파라미터 생성
        params = self._get_params()
        
        if self.is_train:
            # Horizontal flip
            if params['flip']:
                frames = torch.flip(frames, dims=[-1])  # W 차원으로 flip
            
            # Color jittering
            if params['color_jitter']:
                # Brightness
                if params['brightness_factor'] != 1:
                    frames = frames * params['brightness_factor']
                
                # Contrast
                if params['contrast_factor'] != 1:
                    mean = torch.mean(frames, dim=(1, 2, 3), keepdim=True)
                    frames = (frames - mean) * params['contrast_factor'] + mean
                
                # Saturation
                if params['saturation_factor'] != 1:
                    # RGB -> 그레이스케일 변환 (BT.601 공식)
                    gray = frames[:, 0] * 0.299 + frames[:, 1] * 0.587 + frames[:, 2] * 0.114
                    gray = gray.unsqueeze(1).expand_as(frames)
                    frames = torch.lerp(gray, frames, params['saturation_factor'])
                
        # 값 범위를 [0, 1]로 클리핑
        frames = torch.clamp(frames, 0, 1)
        
        return frames


class VideoTransformTrainHDF5(VideoTransformHDF5):
    """HDF5 데이터용 학습 Transform"""
    def __init__(self):
        super().__init__(is_train=True)


class VideoTransformValHDF5(VideoTransformHDF5):
    """HDF5 데이터용 검증 Transform"""
    def __init__(self):
        super().__init__(is_train=False)