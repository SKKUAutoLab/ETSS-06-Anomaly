import torch.nn as nn
import re
import math
import torch
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SElayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig

class DenseMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)

    
class VideoTokenReducer(nn.Module):
    def __init__(self, target_size):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(target_size)
    def forward(self, x):
        # Assuming x is of shape (B, C, T, H, W)
        return self.pool(x)

class VideoPosEncoder(nn.Module):
    def __init__(self, in_dim, out_dim, stride=1):
        super().__init__()
        self.peg = nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=(3, 3), stride=stride, padding=1, groups=out_dim))
    def forward(self, x):
        # x is of shape (B, C, T, H, W)
        shortcut = x
        x = self.peg(x) + shortcut
        return x

class ETProjector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.mlp = DenseMLP(inc, ouc)
        if inc == 576: # VideoMamba inc_features
            self.dwn = VideoTokenReducer((7,7)) # Reduce spatial size from 14x14 to 7x7 for video frames
            self.is_image=False
        else: 
            self.dwn = VideoTokenReducer((12,12)) #Reduce spatial size from 14x14 to 12x12 for images
            self.is_image=True
        self.peg = VideoPosEncoder(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)
        B, num_tokens, C = x.shape
        H = W = int(math.sqrt(num_tokens))
        # Step-by-step reshaping
        x = x.view(B, H, W, C) #[16, 8, 14, 14, 896]
        x = x.permute(0, 3, 1, 2)
        x = self.dwn(x)
        B, C, H, W = x.shape
        x = self.peg(x)
        x = x.permute(0, 2, 3, 1) #The shape of x after temporal pooling  torch.Size([2, 896, 4, 7, 7])
        if self.is_image:
            x = x.reshape(B, H * W, C)
        else:
            x = x.reshape(B, H * W, C)
        return x

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


def build_vision_projector(config, **kwargs):
    """
        mm_hidden_size = 576 for VideoMamba-m16_25M_f8_res224
        mm_hidden_size = 768 for Clip-vit-base-patch32
    """
    image_mm_projector = kwargs['image_mm_projector']
    if image_mm_projector:
        projector_type = getattr(config, 'image_mm_projector_type', 'linear')
        config.mm_hidden_size = 768
    else:
        if "VideoMamba" in config.mm_vision_tower:
            config.mm_hidden_size=576
        else:
            raise ValueError(f"Not implemented: {config.mm_vision_tower}")
        projector_type = getattr(config, 'mm_projector_type', 'linear')
    print(f"Building {projector_type}")

    # Linear Projector
    if projector_type == 'linear':
        projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
        if "VideoMamba" in config.mm_vision_tower:
            config.mm_hidden_size=576
        else:
            raise ValueError(f"Not implemented: {config.mm_vision_tower}")
        config.image_mm_hidden_size = 768
        return projector

    # ETP Projector
    elif projector_type.startswith('etp'):
        # Image ETP Projector
        if image_mm_projector:
            config.mm_hidden_size = 768
        # Video ETP Projector
        elif "VideoMamba" in config.mm_vision_tower:
            config.mm_hidden_size=576
        else:
            raise ValueError(f"Not implemented: {config.mm_vision_tower}")
        
        config.image_mm_hidden_size = 768
        return ETProjector(config)
    # MLP_GELU Projector
    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            config.mm_hidden_size = 576
            config.image_mm_hidden_size = 768
            return nn.Sequential(*modules)
    elif projector_type == 'identity':
        projector = IdentityMap()
        config.mm_hidden_size = 576
        config.image_mm_hidden_size = 768
        return projector

    raise ValueError(f'Unknown projector type: {projector_type}')
