from mobilevideogpt.model.videomamba import build_videomamba
import numpy as np
import cv2
import os
import torch
from torch import nn
import json
from huggingface_hub import hf_hub_download

v_mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
v_std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
class ConfigWrapper:
    def __init__(self, config_dict):
        self._config_dict = config_dict
        self._create_attributes(self._config_dict)

    def _create_attributes(self, d, parent_attr=""):
        for key, value in d.items():
            if isinstance(value, dict):
                value = ConfigWrapper(value)
            attr_name = f"{parent_attr}.{key}" if parent_attr else key
            object.__setattr__(self, key, value)
            object.__setattr__(self, attr_name, value)

    def __getattr__(self, attr):
        return self.__dict__.get(attr)

def get_device_map() -> str:
        return 'cuda' if torch.cuda.is_available() else 'cpu'

def setup_videomamba(config:dict):
    model = VideoMamba_Stage2(config)

    if config.compile_model:
        torch.set_float32_matmul_precision('high')
        model = torch.compile(model)
    device = get_device_map()
    model_without_ddp = model
    
    # Rename the name of the parameters to align with Mobile-VideoGPT
    if config.vision_encoder.pretrained.startswith("hf://"):
        repo_id = "OpenGVLab/VideoMamba"
        filename = "videomamba_m16_25M_f8_res224.pth"
        local_checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
    else:
        local_checkpoint_path = config.vision_encoder.pretrained
    if os.path.isfile(local_checkpoint_path):
        checkpoint = torch.load(local_checkpoint_path, map_location="cpu")
        try:
            if "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint["module"]  # This is a deepspeed stage 1 model
        except:
            state_dict = checkpoint
       
        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith('vision_encoder'):
                new_key = f'vision_encoder.{k}'
            else:
                new_key = k
            new_state_dict[new_key] = v
        msg = model_without_ddp.load_state_dict(new_state_dict, strict=False)
        
    if config.use_bf16:
        model_without_ddp = model_without_ddp.to(torch.bfloat16)
    elif config.use_half_precision:
        model_without_ddp = model_without_ddp.to(torch.float16)
    else:
        model_without_ddp = model_without_ddp.to(torch.float32)

    model_without_ddp.eval()
    return (model_without_ddp)


def normalize(data):
    return (data / 255.0 - v_mean) / v_std


def _frame_from_video(video):
    while video.isOpened():
        success, frame = video.read()
        if success:
            yield frame
        else:
            break


def frames2tensor(vid_list, fnum=8, target_size=(224, 224), device=torch.device('cuda')):
    assert (len(vid_list) >= fnum)
    step = len(vid_list) // fnum
    vid_list = vid_list[::step][:fnum]
    vid_list = [cv2.resize(x[:, :, ::-1], target_size) for x in vid_list]
    vid_tube = [np.expand_dims(normalize(x), axis=(0, 1)) for x in vid_list]
    vid_tube = np.concatenate(vid_tube, axis=1)
    vid_tube = np.transpose(vid_tube, (0, 1, 4, 2, 3))
    vid_tube = torch.from_numpy(vid_tube).to(device, non_blocking=True).float()
    return vid_tube


def get_vid_feat(frames, vlm):
    return vlm.get_vid_features(frames)


def retrieve_vision(frames, model, topk: int = 4, config: dict = {}, device=torch.device('cuda')):
    vlm = model
    vlm = vlm.to(device)

    if config.num_frames is not None:
        fn = config.num_frames
    else:
        fn = 8

    if config.img_size is not None:
        size_t = config.img_size
    else:
        size_t = 224

    frames_tensor = frames2tensor(frames, fnum=fn, target_size=(size_t, size_t), device=device)
    vision_embeds, pooled_vision_embeds = vlm.get_vid_feat(frames_tensor)

    return vision_embeds, pooled_vision_embeds



class VideoTrainProcessor():
    def __init__(self, image_size=(224, 224), mean=None, std=None, num_frames=8):
        super().__init__()

        if mean is None:
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
        if std is None:
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
        self.mean = mean
        self.std = std
        self.num_frames = num_frames
        self.image_size = image_size

    def normalize(self, data):
        return (data / 255.0 - self.mean) / self.std

    def frames2tensor(self, vid_list, target_size=(224, 224), use_image=False):
        # Process each frame
        vid_list = [cv2.resize(x, target_size) for x in vid_list]
        vid_tube = [normalize(x) for x in vid_list]
        vid_tube = [np.transpose(x, (2, 0, 1)) for x in vid_tube]
        vid_tube = [torch.from_numpy(x) for x in vid_tube]

        return vid_tube

    def preprocess(self, vid_list, use_image=False):
        return {'pixel_values': self.frames2tensor(vid_list, use_image=use_image)}
    

    def save_pretrained(self, save_directory):
        os.makedirs(save_directory, exist_ok=True)
        config = {
                "image_size": self.image_size,
                "mean": self.mean.tolist() if isinstance(self.mean, np.ndarray) else self.mean,
                "std": self.std.tolist() if isinstance(self.std, np.ndarray) else self.std,
                "num_frames": self.num_frames,
                }
        with open(os.path.join(save_directory, "video_processor_config.json"), "w") as f:
            json.dump(config, f)
    
    @classmethod
    def from_pretrained(cls, load_directory):
        config_path = os.path.join(load_directory, "video_processor_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found in {load_directory}")
        with open(config_path, "r") as f:
            config = json.load(f)

        config["mean"] = np.array(config["mean"])
        config["std"] = np.array(config["std"])

        return cls(**config)

class VideoMamba_Stage2(nn.Module):
    """docstring for InternVideo2_Stage2"""

    def __init__(self, config, is_pretrain: bool = True):
        super(VideoMamba_Stage2, self).__init__()

        self.config = config

        self.is_pretrain = is_pretrain
        #self.vision_width = config.model.vision_encoder.clip_embed_dim
        self.embed_dim = config.vision_encoder.embed_dim

        # create modules.
        self.vision_encoder = self.build_vision_encoder()
        self.freeze_vision()
        #self.learnable_vision()

        self.image_processor = VideoTrainProcessor(num_frames=self.config.vision_encoder.num_frames)

    def load_model(self, path):
        if path == "OpenGVLab/VideoMamba/videomamba_m16_25M_f8_res224.pth":
            repo_id = "OpenGVLab/VideoMamba"
            filename = "videomamba_m16_25M_f8_res224.pth"
            local_checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
            checkpoint = torch.load(local_checkpoint_path, map_location="cpu")
        else: 
            checkpoint = torch.load(path, map_location="cpu")
        try :
            if "model" in checkpoint.keys():
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint["module"]  # This is a deepspeed stage 1 model
        except:
            state_dict = checkpoint
        self.load_state_dict(state_dict, strict=False)

    def freeze_vision(self):
        total_params = sum(p.numel() for p in self.vision_encoder.parameters())
        """freeze vision encoder"""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

    def learnable_vision(self):
        total_params = sum(p.numel() for p in self.vision_encoder.parameters())
        for p in self.vision_encoder.parameters():
            p.requires_grad = True

    @property
    def dtype(self):
        return self.vision_encoder.patch_embed.proj.weight.dtype

    def encode_vision(self, image: torch.Tensor, test: bool = False):
        """encode image / videos as features.

        Args:
            image (torch.Tensor): The input images.
            test (bool): Whether testing.

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].
            - student_output (torch.Tensor): The features of alignment. Shape: [K,B,N,C].
            - clip_output (torch.Tensor): The features of clip. Shape: [K,B,N,C].

        """

        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
        # whether save temporal dimension
        # keep_temporal=self.config.model.vision_encoder.keep_temporal
        if test:
            vision_embeds, pooled_vision_embeds, _, _ = self.vision_encoder(
                image, None, use_image
            )
            return vision_embeds, pooled_vision_embeds
        else:
            mask, targets_clip_middle_vis, targets_clip_final_vis = self.encode_teacher(image)
            # if mask is not None and (self.video_mask_type != 'tube' or self.image_mask_type != 'tube'):
            #     keep_temporal = False
            # print(f"\033[31mmask is {type(mask)}\033[0m")
            vision_embeds, pooled_vision_embeds, student_output, student_output_final = self.vision_encoder(
                image, mask, use_image
            )
            return vision_embeds, pooled_vision_embeds, student_output, student_output_final, targets_clip_middle_vis, targets_clip_final_vis

    def forward(self, image):
        T = image.shape[1]
        use_image = True if T == 1 else False
        image = image.permute(0, 2, 1, 3, 4).to(self.dtype)  # [B,T,C,H,W] -> [B,C,T,H,W]
        vision_embeds = self.vision_encoder(
            image, None, use_image)

        return vision_embeds

    def build_vision_encoder(self):
        """build vision encoder
        Returns: (vision_encoder, clip_teacher). Each is a `nn.Module`.

        """
        encoder_name = self.config.vision_encoder.name

        if encoder_name == 'videomamba_middle':
            vision_encoder = build_videomamba(self.config)
        else:
            raise ValueError(f"Not implemented: {encoder_name}")

        # parameters for mask
        img_size = self.config.vision_encoder.img_size
        num_frames = self.config.vision_encoder.num_frames
        tublet_size = self.config.vision_encoder.tubelet_size
        patch_size = self.config.vision_encoder.patch_size
        self.clip_img_size = self.config.vision_encoder.img_size
        self.video_mask_type = self.config.vision_encoder.video_mask_type
        self.video_window_size = (num_frames // tublet_size, img_size // patch_size, img_size // patch_size)
        self.video_mask_ratio = self.config.vision_encoder.video_mask_ratio
        self.image_mask_type = self.config.vision_encoder.image_mask_type
        self.image_window_size = (1, img_size // patch_size, img_size // patch_size)
        self.image_mask_ratio = self.config.vision_encoder.image_mask_ratio
        self.num_frames = self.config.vision_encoder.num_frames
        return vision_encoder

    def get_vid_feat(self, frames: torch.Tensor):
        """get the video features for the given frames.

        Args:
            frames (torch.Tensor): The input frames. Shape: [B,T,C,H,W].

        Returns: tuple.
            - vision_embeds (torch.Tensor): The output features. Shape: [B,N,C].
            - pooled_vision_embeds (torch.Tensor): The pooled output features. Shape: [B,1,C].

        """
        with torch.no_grad():
            vision_embeds, pooled_vision_embeds = self.encode_vision(
                frames, test=True
            )  # vfeat = self.vision_proj(vfeat)  # vfeat /= vfeat.norm(dim=-1, keepdim=True)
        return vision_embeds, pooled_vision_embeds

    @property
    def hidden_size(self):
        return self.vision_encoder.embed_dim

    @property
    def num_patches(self):
        return self.config.model.vision_encoder.patch_size
