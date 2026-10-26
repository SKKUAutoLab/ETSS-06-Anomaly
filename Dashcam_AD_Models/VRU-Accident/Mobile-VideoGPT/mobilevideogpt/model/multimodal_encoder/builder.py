import json
from .clip_encoder import CLIPVisionTower
from mobilevideogpt.model.videomamba.build_videomamba import build_videomamba

def build_vision_tower(vision_tower_cfg, **kwargs):
    image_vision_tower = kwargs['image_vision_tower']
    if image_vision_tower:
        vision_tower = getattr(vision_tower_cfg, 'image_mm_vision_tower', getattr(vision_tower_cfg, 'image_vision_tower', None))
        kwargs.pop('image_vision_tower', None)
    else:
        vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))

    print(f"Building {vision_tower}")
    if vision_tower.startswith("openai") or vision_tower.startswith("laion"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    elif 'InternVideo2' in vision_tower:
        InternVideoTower = build_internvideo(vision_tower)
        InternVideoTower.requires_grad_(False)
        return InternVideoTower
    elif 'VideoMamba' in vision_tower:
        VideoMambaTower = build_videomamba(vision_tower)
        VideoMambaTower.requires_grad_(False)
        return VideoMambaTower

    raise ValueError(f'Unknown vision tower: {vision_tower}')
