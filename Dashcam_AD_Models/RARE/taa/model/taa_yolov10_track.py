import time
import math
from typing import Dict, List

from ultralytics import YOLOv10
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import (
    AIFI, CBAM, CIB, C2f, C2fCIB, PSA, LightConv, MLP, TransformerLayer
)
from ultralytics.nn.tasks import YOLOv10DetectionModel

from torchvision.ops import ps_roi_align, roi_align, RoIAlign, PSRoIAlign

from taa.model.module import AttentionFusion, Conv

from ultralytics.trackers.byte_tracker import BYTETracker, STrack
from ultralytics.trackers.utils.kalman_filter import KalmanFilterXYAH
from ultralytics.trackers.utils import matching
import numpy as np

YOLOv10_feature_mapping = {
    # backbone_feat, p3_feat, p4_feat, p5_feat, ROI Aligned feat
    'yolov10n': [[256, 7, 7], [64, 80, 80], [128, 40, 40], [256, 20, 20], [1, 8, 8], [2, 8, 8], [4, 8, 8]],
    'yolov10s': [[512, 20, 20], [128, 80, 80], [256, 40, 40], [512, 20, 20]],
    'yolov10m': [[576, 20, 20], [192, 80, 80], [384, 40, 40], [576, 20, 20]],
    'yolov10b': [[512, 20, 20], [256, 80, 80], [512, 40, 40], [512, 20, 20]],
    'yolov10l': [[512, 20, 20], [256, 80, 80], [512, 40, 40], [512, 20, 20]],
    'yolov10x': [[640, 20, 20], [320, 80, 80], [640, 40, 40], [640, 20, 20], [5, 8, 8], [10, 8, 8], [10, 8, 8]],
}

class YOLOv10TAATrackModel(nn.Module):
    """
    YOLOv10n based Traffic Accident Anticipation Model with improved attention and efficiency.
    """

    def __init__(self, yolo_id='yolov10x', yolo_ft='None', conf_thresh=0.15):
        super().__init__()

        # 1. Base detector initialization
        if yolo_ft == 'None':
            self.detector = YOLOv10.from_pretrained(f'jameslahm/{yolo_id}')
            # COCO class list
            '''
            0: person
            1: bicycle
            2: car
            3: motorcycle
            5: bus
            7: truck
            '''
            self.cls_list = [0, 1, 2, 3, 5, 7]
        else:
            self.detector = YOLOv10(yolo_ft)
            self.cls_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] # use all classes
            
        self.feature_mapping = YOLOv10_feature_mapping[yolo_id]
        # Freeze the base detector
        for param in self.detector.model.parameters():
            param.requires_grad = False
            
        self.conf_thresh = conf_thresh
        self.tracker = self._init_tracker()

        self.mean_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.scene_temporal = nn.GRU(input_size=self.feature_mapping[0][0], hidden_size=self.feature_mapping[0][0], num_layers=2, dropout=0.1, batch_first=True)
        # self.no_obj_head = nn.Sequential(nn.Linear(self.feature_mapping[0][0], self.feature_mapping[0][0]//4), nn.SiLU(inplace=True), nn.Linear(self.feature_mapping[0][0]//4, 2))
        
        self.roi_align_backbone = RoIAlign(output_size=8, spatial_scale=20/640, sampling_ratio=-1)
        self.roi_align_p3 = RoIAlign(output_size=8, spatial_scale=80/640, sampling_ratio=-1)
        self.roi_align_p4 = RoIAlign(output_size=8, spatial_scale=40/640, sampling_ratio=-1)
        self.roi_align_p5 = RoIAlign(output_size=8, spatial_scale=20/640, sampling_ratio=-1)
        
        self.backbone_channel_reduction = nn.Sequential(
            Conv(c1=self.feature_mapping[0][0], c2=self.feature_mapping[0][0], k=3, s=1, p=None, g=self.feature_mapping[0][0], d=1, act=True),
            Conv(c1=self.feature_mapping[0][0], c2=self.feature_mapping[0][0]//2, k=3, s=1, p=None, g=1, d=1, act=True),
            Conv(c1=self.feature_mapping[0][0]//2, c2=self.feature_mapping[0][0]//2, k=3, s=1, p=None, g=self.feature_mapping[0][0]//2, d=1, act=True),
            Conv(c1=self.feature_mapping[0][0]//2, c2=self.feature_mapping[0][0]//4, k=3, s=1, p=None, g=1, d=1, act=True),
        )
        
        self.p3_channel_reduction = nn.Sequential(
            Conv(c1=self.feature_mapping[1][0], c2=self.feature_mapping[1][0], k=3, s=1, p=None, g=self.feature_mapping[1][0], d=1, act=True),
            Conv(c1=self.feature_mapping[1][0], c2=self.feature_mapping[1][0]//2, k=3, s=1, p=None, g=1, d=1, act=True),
            Conv(c1=self.feature_mapping[1][0]//2, c2=self.feature_mapping[1][0]//2, k=3, s=1, p=None, g=self.feature_mapping[1][0]//2, d=1, act=True),
            Conv(c1=self.feature_mapping[1][0]//2, c2=self.feature_mapping[1][0]//4, k=3, s=1, p=None, g=1, d=1, act=True),
        )
        
        self.p4_channel_reduction = nn.Sequential(
            Conv(c1=self.feature_mapping[2][0], c2=self.feature_mapping[2][0], k=3, s=1, p=None, g=self.feature_mapping[2][0], d=1, act=True),
            Conv(c1=self.feature_mapping[2][0], c2=self.feature_mapping[2][0]//2, k=3, s=1, p=None, g=1, d=1, act=True),
            Conv(c1=self.feature_mapping[2][0]//2, c2=self.feature_mapping[2][0]//2, k=3, s=1, p=None, g=self.feature_mapping[2][0]//2, d=1, act=True),
            Conv(c1=self.feature_mapping[2][0]//2, c2=self.feature_mapping[2][0]//4, k=3, s=1, p=None, g=1, d=1, act=True),
        )
        
        self.p5_channel_reduction = nn.Sequential(
            Conv(c1=self.feature_mapping[3][0], c2=self.feature_mapping[3][0], k=3, s=1, p=None, g=self.feature_mapping[3][0], d=1, act=True),
            Conv(c1=self.feature_mapping[3][0], c2=self.feature_mapping[3][0]//2, k=3, s=1, p=None, g=1, d=1, act=True),
            Conv(c1=self.feature_mapping[3][0]//2, c2=self.feature_mapping[3][0]//2, k=3, s=1, p=None, g=self.feature_mapping[3][0]//2, d=1, act=True),
            Conv(c1=self.feature_mapping[3][0]//2, c2=self.feature_mapping[3][0]//4, k=3, s=1, p=None, g=1, d=1, act=True),
        )

        self.fuse_backbone_p3_p4_p5 = nn.Sequential(
            CBAM(c1=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, kernel_size=3),
            CBAM(c1=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, kernel_size=3),
            # CBAM(c1=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, kernel_size=3),
            # CBAM(c1=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, kernel_size=3),
        )

        self.coord_embed = nn.Sequential(
            nn.Linear(4, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 512),
            nn.SiLU(inplace=True),
            nn.Linear(512, 512),
            nn.SiLU(inplace=True),
        )
        
        self.fuse_coord_feat = nn.Sequential(
            nn.Linear(in_features=512+self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4,
                      out_features=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4
                      ),
            nn.InstanceNorm1d(self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4),
            nn.SiLU(inplace=True),

            nn.Linear(in_features=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4,
                      out_features=(self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//2
                      ),
            nn.InstanceNorm1d((self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//2),
            nn.SiLU(inplace=True),

            TransformerLayer(c=(self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//2, num_heads=8)
        )
        
        self.AttFusion = AttentionFusion(
            scene_dim=self.feature_mapping[0][0], 
            obj_dim=(self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//2, 
            ff_dim=128, num_heads=8, layer_norm=True
            )
        
        # self.fuse_temporal = nn.GRU(input_size=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, hidden_size=self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, num_layers=2, dropout=0.1, batch_first=True)
        self.risk_head = nn.Sequential(
            nn.Linear((self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//2, (self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//4),
            nn.SiLU(inplace=True),
            nn.Linear((self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//4, 2)
        )

        self._features = {}
        self._register_hooks()

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor (B, T, C, H, W)
                B: batch size
                T: temporal length
                C: channels
                H, W: height, width
        """
        B, T = x.shape[:2]

        risk_scores = []
        filtered_detections = []
        obj_attns = []

        scene_hidden = None
        for t in range(T):
            # if x.device key is in _features, clear the features
            if x.device in self._features:
                self._features[x.device].clear()

            # Modify the feature extraction process
            results = self.detector.predict(source=x[:, t], imgsz=640, conf=self.conf_thresh, verbose=False)
            
            detections = results[0].boxes # each sample of detections have det.cls, det.conf, det.xywh, det.xywhn, det.xyxy, det.xyxyn
            clss, confs, xyxy, xywhn = detections.cls, detections.conf, detections.xyxy, detections.xywhn # (N,), (N,), (N,4), (N,4)

            conf_mask = confs > self.conf_thresh
            cls_mask = torch.isin(clss, torch.tensor(self.cls_list, device=clss.device))
            mask = conf_mask & cls_mask
            
            # filter out detections with low confidence and not in cls_list 
            clss, confs, xyxy, xywhn = clss[mask], confs[mask], xyxy[mask], xywhn[mask]
            detections = detections[mask]
            
            tracks = self.tracker.update(detections.cpu().numpy())

            # Extract features using hooks
            backbone_feat, p3_feat, p4_feat, p5_feat = self._features[x.device]['model.8'].clone(), self._features[x.device]['model.16'].clone(), self._features[x.device]['model.19'].clone(), self._features[x.device]['model.22'].clone()
            
            # global avg pooling on backbone feature
            scene_feat = self.mean_pool(backbone_feat)
            scene_feat = scene_feat.squeeze().unsqueeze(0).unsqueeze(0) # n: (1, 1, 256), x: (1, 1, 640)
            scene_feat, scene_hidden = self.scene_temporal(scene_feat, scene_hidden) # n: (1, 1, 256), x: (1, 1, 640)
            scene_feat = scene_feat.squeeze(0) # n: (1, 256), x: (1, 640)
            # scene_risk = self.no_obj_head(scene_feat)

            if len(tracks) > 0:
                track_boxes_coords = torch.from_numpy(tracks[:, :4]).to(x.device)  # xyxy format
                track_boxes = torch.cat([torch.zeros_like(track_boxes_coords[:, :1]), track_boxes_coords], dim=1)
                track_boxes_coord = torch.cat([track_boxes_coords[:, :2] - track_boxes_coords[:, 2:4] / 2, track_boxes_coords[:, 2:4]], dim=1)
       
                # Extract object features
                obj_backbone_feats = self.roi_align_backbone(backbone_feat, track_boxes)
                obj_p3_feats = self.roi_align_p3(p3_feat, track_boxes)
                obj_p4_feats = self.roi_align_p4(p4_feat, track_boxes)
                obj_p5_feats = self.roi_align_p5(p5_feat, track_boxes)
                
                # Process features
                obj_backbone_feats = self.backbone_channel_reduction(obj_backbone_feats)
                obj_p3_feats = self.p3_channel_reduction(obj_p3_feats)
                obj_p4_feats = self.p4_channel_reduction(obj_p4_feats)
                obj_p5_feats = self.p5_channel_reduction(obj_p5_feats)
                
                # Combine and fuse features
                obj_feats = torch.cat([obj_backbone_feats, obj_p3_feats, obj_p4_feats, obj_p5_feats], dim=1)
                obj_feats = self.fuse_backbone_p3_p4_p5(obj_feats)
                obj_feats = self.mean_pool(obj_feats).squeeze(2).squeeze(2)
                obj_coord = self.coord_embed(track_boxes_coord) # n:(N, 512), x:(N, 512)
                obj_feats = torch.cat([obj_feats, obj_coord], dim=-1) # n:(N, 640), x:(N, 1024)
                obj_feats = self.fuse_coord_feat(obj_feats) # n:(N, 640), x:(N, 1024)
                
                # Fuse with scene features and predict risk
                fused_feat, obj_attn = self.AttFusion(scene_feat, obj_feats)
                obj_attns.append(obj_attn)
                risk_score = self.risk_head(fused_feat)
            
            else:
                risk_score = torch.tensor([0.0, 0.0], device=x.device)
            
            risk_scores.append(risk_score)
            filtered_detections.append(tracks)
            obj_attns.append(obj_attn)
            

        return {
            'risk_score': risk_scores,
            'detections': filtered_detections,
            'obj_attns': obj_attns
        }

    def _init_tracker(self):
        """Initialize ByteTracker with custom parameters"""
        from types import SimpleNamespace
        args = SimpleNamespace(
            track_high_thresh=self.conf_thresh,  # 높은 신뢰도 임계값
            track_low_thresh=0.1,                # 낮은 신뢰도 임계값
            new_track_thresh=self.conf_thresh,   # 새로운 트랙 생성 임계값
            track_buffer=10,                     # 트랙 버퍼 크기
            match_thresh=0.8                     # 매칭 임계값
        )
        return BYTETracker(args, frame_rate=20)

    def _register_hooks(self):
        """
        Register forward hooks to extract feature maps from intermediate layers.
        """
        def hook_fn(module, input, output, name):
            # input tensor의 device를 기준으로 동기화
            if input[0].device not in self._features:
                self._features[input[0].device] = {}
            # type_as를 사용하여 device 동기화
            self._features[input[0].device][name] = output.type_as(input[0])

        for name, module in self.detector.model.named_modules():
            if name in ['model.8', 'model.16', 'model.19', 'model.22']:
                module.register_forward_hook(
                    lambda m, i, o, n=name: hook_fn(m, i, o, n)
                )

    # def _register_hooks(self):
    #     """
    #     Register forward hooks to extract feature maps from intermediate layers.
    #     """

    #     def hook_fn(module, input, output, name):
    #         self._features[name] = output

    #     for name, module in self.detector.model.named_modules():
    #         # Adjusted layers to extract features from for YOLOv8n
    #         if name in ['model.8', 'model.16', 'model.19', 'model.22']:
    #             module.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n))

    def train(self, mode=True):
        """
        Override the train method to ensure detector stays in eval mode
        while other modules can be trained.
        """
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")

        self.training = mode

        # Set train/eval mode for modules other than the detector
        for name, module in self.named_children():
            if name != 'detector':
                module.train(mode)
                for param in module.parameters():
                    param.requires_grad = True
            elif name == 'detector':
                for param in module.parameters():
                    param.requires_grad = False

        return self

    def eval(self):
        """
        Set the model to evaluation mode.
        """
        self.train(False)
        return self
    
    def get_features_for_objects(self, feat_map, centers):
        """
        Extract features for all objects at once
        """
        H, W = feat_map.shape[-2:]
        # Scale coordinates to feature map size
        scaled_centers = centers.clone()
        scaled_centers[:, 0] *= (W - 1)
        scaled_centers[:, 1] *= (H - 1)
        scaled_centers = scaled_centers.long()
        
        # Extract features for all objects at once
        return feat_map[0, :, scaled_centers[:, 1], scaled_centers[:, 0]].t()  # (N, C)

class TAATemporal(nn.Module):
    """TAA를 위한 Loss 함수
    
    원본 _exp_loss와 동일한 가중치를 적용한 버전
    - Positive examples: exp(-(toa-t-1)/fps) * CE_loss
    - Negative examples: CE_loss
    - Temporal consistency: lambda_temporal * |pred_t - pred_{t-1}|
    """
    def __init__(self, lambda_temporal: float = 0.01, fps: float = 10.0, toa: int = 90):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.lambda_temporal = lambda_temporal
        self.fps = fps
        self.toa = toa
        
    def forward(self, pred, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        T = len(pred)
        pred_batch = torch.cat(pred, dim=0)  # (T, 2)
        time_steps = torch.arange(T, device=pred[0].device).float()
        
        # Cross Entropy Loss 계산
        target_expanded = torch.full((T,), target.item(), device=pred[0].device).long()
        ce_losses = self.ce(pred_batch, target_expanded)
        
        if target.item() == 1:  # Positive example
            # CE Loss with exponential penalty
            penalty = -torch.max(
                torch.tensor(0., device=pred[0].device),
                (torch.tensor(self.toa, device=pred[0].device) - time_steps - 1) / self.fps
            )
            penalty = torch.exp(penalty)
            ratio = T / penalty.sum()
            ce_loss = torch.mean(ratio * penalty * ce_losses)
            
            # Temporal loss for positive samples:
            # 단조 증가 제약만 적용 (한번 증가한 위험도가 감소하지 않도록)
            accident_probs = pred_batch[:, 1]  # (T,)
            if T > 1:
                temporal_diffs = accident_probs[1:] - accident_probs[:-1]  # (T-1,)
                temporal_loss = torch.relu(-temporal_diffs).mean()  # 감소하는 구간에 대해서만 패널티
            else:
                temporal_loss = torch.tensor(0., device=pred[0].device)
            
        else:  # Negative example
            ce_loss = torch.mean(ce_losses)
            
            # Temporal loss for negative samples:
            # 부드러운 변화 유지 (급격한 변동 방지)
            accident_probs = pred_batch[:, 1]  # (T,)
            if T > 2:
                first_order_diffs = accident_probs[1:] - accident_probs[:-1]  # (T-1,)
                second_order_diffs = first_order_diffs[1:] - first_order_diffs[:-1]  # (T-2,)
                temporal_loss = torch.mean(torch.abs(second_order_diffs))
            else:
                temporal_loss = torch.tensor(0., device=pred[0].device)
        
        total_loss = ce_loss + self.lambda_temporal * temporal_loss
        
        return {
            'ce_loss': ce_loss,
            'temporal_loss': temporal_loss,
            'total_loss': total_loss
        }


if __name__ == '__main__':
    import cv2
    import os
    from torchvision.io import read_image
    from torchvision.transforms import Resize
    
    def visualize_tracks(image, tracks):
        """시각화 함수: 트래킹 결과를 이미지에 그림"""
        img = image.cpu().numpy().transpose(1, 2, 0)
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        for track in tracks:
            # print(track)
            xyxy = track[:4]
            track_id = int(track[4])
            score = float(track[5])
            class_id = int(track[6])
            
            # 박스 그리기
            cv2.rectangle(img, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            
            # 텍스트 정보 추가
            text = f'ID:{track_id} C:{class_id} {score:.2f}'
            cv2.putText(img, text, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return img

    # Test code
    model = YOLOv10TAATrackModel().to('cuda')
    model.eval()
    
    # Load test video frames
    folder_path = "./taa/data/DAD/frames/training/positive/000001"
    images = sorted([f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))])
    images = [os.path.join(folder_path, img) for img in images]
    images = [read_image(img) for img in images]
    images = [Resize((640, 640))(img) for img in images]
    images = [img / 255.0 for img in images]
    images = torch.stack(images).unsqueeze(0).to('cuda')
    
    # Run inference
    with torch.no_grad():
        outputs = model(images)
    
    # Visualize results
    save_dir = './taa/_visualization/tracking'
    os.makedirs(save_dir, exist_ok=True)
    
    for t, (image, tracks) in enumerate(zip(images[0], outputs['detections'])):
        if len(tracks) > 0:
            vis_img = visualize_tracks(image, tracks)
            cv2.imwrite(f'{save_dir}/frame_{t:04d}.jpg', vis_img)