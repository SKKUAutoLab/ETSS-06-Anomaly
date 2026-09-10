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
from ultralytics.utils.metrics import box_iou
from ultralytics.nn.tasks import YOLOv10DetectionModel

from torchvision.ops import ps_roi_align, roi_align, RoIAlign, PSRoIAlign

from taa.model.module import AttentionFusion, Conv, TemporalAttentionRefinement

YOLOv10_feature_mapping = {
    # backbone_feat, p3_feat, p4_feat, p5_feat, ROI Aligned feat
    'yolov10n': [[256, 7, 7], [64, 80, 80], [128, 40, 40], [256, 20, 20], [1, 8, 8], [2, 8, 8], [4, 8, 8]],
    'yolov10s': [[512, 20, 20], [128, 80, 80], [256, 40, 40], [512, 20, 20]],
    'yolov10m': [[576, 20, 20], [192, 80, 80], [384, 40, 40], [576, 20, 20]],
    'yolov10b': [[512, 20, 20], [256, 80, 80], [512, 40, 40], [512, 20, 20]],
    'yolov10l': [[512, 20, 20], [256, 80, 80], [512, 40, 40], [512, 20, 20]],
    'yolov10x': [[640, 20, 20], [320, 80, 80], [640, 40, 40], [640, 20, 20], [5, 8, 8], [10, 8, 8], [10, 8, 8]],
}

class YOLOv10TAADetectionModel(nn.Module):
    """
    YOLOv10 기반 TAA 모델  
    옵션에 따라, obj_feats 구성을 위한 feature branch를 선택할 수 있습니다.
    """
    def __init__(self, yolo_id='yolov10x', yolo_ft='None', conf_thresh=0.10, buffer_size=10, obj_feats_mode='backbone'):
        super().__init__()

        # 1. Base detector initialization
        if yolo_ft == 'None':
            self.detector = YOLOv10.from_pretrained(f'jameslahm/{yolo_id}')
            self.cls_list = [0, 1, 2, 3, 5, 7]
        else:
            self.detector = YOLOv10(yolo_ft)
            self.cls_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        self.feature_mapping = YOLOv10_feature_mapping[yolo_id]
        for param in self.detector.model.parameters():
            param.requires_grad = False
            
        self.conf_thresh = conf_thresh
        self.mean_pool = nn.AdaptiveAvgPool2d(1)
        self.scene_temporal = nn.GRU(
            input_size=self.feature_mapping[0][0],
            hidden_size=self.feature_mapping[0][0],
            num_layers=2, dropout=0.1, batch_first=True
        )
        
        # --- Ablation Study: Feature Branch Selection ---
        if obj_feats_mode not in ['all', 'backbone', 'p3p4p5']:
            raise ValueError("obj_feats_mode는 'all', 'backbone', 'p3p4p5' 중 하나여야 합니다.")
        self.obj_feats_mode = obj_feats_mode
        if obj_feats_mode == 'all':
            self.used_indices = [0, 1, 2, 3]
        elif obj_feats_mode == 'backbone':
            self.used_indices = [0]
        elif obj_feats_mode == 'p3p4p5':
            self.used_indices = [1, 2, 3]
        # --------------------------------------------------
        
        input_size = 640  # 입력 이미지 크기
        
        # ROIAlign 및 채널 축소 모듈을 사용된 branch별로 동적 생성
        if 0 in self.used_indices:
            backbone_scale = self.feature_mapping[0][1] / input_size
            self.roi_align_backbone = RoIAlign(output_size=8, spatial_scale=backbone_scale, sampling_ratio=-1)
            self.backbone_channel_reduction = nn.Sequential(
                Conv(c1=self.feature_mapping[0][0], c2=self.feature_mapping[0][0], k=3, s=1, p=None,
                     g=self.feature_mapping[0][0], d=1, act=True),
                Conv(c1=self.feature_mapping[0][0], c2=self.feature_mapping[0][0]//2, k=3, s=1, p=None,
                     g=1, d=1, act=True),
                Conv(c1=self.feature_mapping[0][0]//2, c2=self.feature_mapping[0][0]//2, k=3, s=1, p=None,
                     g=self.feature_mapping[0][0]//2, d=1, act=True),
                Conv(c1=self.feature_mapping[0][0]//2, c2=self.feature_mapping[0][0]//4, k=3, s=1, p=None,
                     g=1, d=1, act=True),
            )
        if 1 in self.used_indices:
            p3_scale = self.feature_mapping[1][1] / input_size
            self.roi_align_p3 = RoIAlign(output_size=8, spatial_scale=p3_scale, sampling_ratio=-1)
            self.p3_channel_reduction = nn.Sequential(
                Conv(c1=self.feature_mapping[1][0], c2=self.feature_mapping[1][0], k=3, s=1, p=None,
                     g=self.feature_mapping[1][0], d=1, act=True),
                Conv(c1=self.feature_mapping[1][0], c2=self.feature_mapping[1][0]//2, k=3, s=1, p=None,
                     g=1, d=1, act=True),
                Conv(c1=self.feature_mapping[1][0]//2, c2=self.feature_mapping[1][0]//2, k=3, s=1, p=None,
                     g=self.feature_mapping[1][0]//2, d=1, act=True),
                Conv(c1=self.feature_mapping[1][0]//2, c2=self.feature_mapping[1][0]//4, k=3, s=1, p=None,
                     g=1, d=1, act=True),
            )
        if 2 in self.used_indices:
            p4_scale = self.feature_mapping[2][1] / input_size
            self.roi_align_p4 = RoIAlign(output_size=8, spatial_scale=p4_scale, sampling_ratio=-1)
            self.p4_channel_reduction = nn.Sequential(
                Conv(c1=self.feature_mapping[2][0], c2=self.feature_mapping[2][0], k=3, s=1, p=None,
                     g=self.feature_mapping[2][0], d=1, act=True),
                Conv(c1=self.feature_mapping[2][0], c2=self.feature_mapping[2][0]//2, k=3, s=1, p=None,
                     g=1, d=1, act=True),
                Conv(c1=self.feature_mapping[2][0]//2, c2=self.feature_mapping[2][0]//2, k=3, s=1, p=None,
                     g=self.feature_mapping[2][0]//2, d=1, act=True),
                Conv(c1=self.feature_mapping[2][0]//2, c2=self.feature_mapping[2][0]//4, k=3, s=1, p=None,
                     g=1, d=1, act=True),
            )
        if 3 in self.used_indices:
            p5_scale = self.feature_mapping[3][1] / input_size
            self.roi_align_p5 = RoIAlign(output_size=8, spatial_scale=p5_scale, sampling_ratio=-1)
            self.p5_channel_reduction = nn.Sequential(
                Conv(c1=self.feature_mapping[3][0], c2=self.feature_mapping[3][0], k=3, s=1, p=None,
                     g=self.feature_mapping[3][0], d=1, act=True),
                Conv(c1=self.feature_mapping[3][0], c2=self.feature_mapping[3][0]//2, k=3, s=1, p=None,
                     g=1, d=1, act=True),
                Conv(c1=self.feature_mapping[3][0]//2, c2=self.feature_mapping[3][0]//2, k=3, s=1, p=None,
                     g=self.feature_mapping[3][0]//2, d=1, act=True),
                Conv(c1=self.feature_mapping[3][0]//2, c2=self.feature_mapping[3][0]//4, k=3, s=1, p=None,
                     g=1, d=1, act=True),
            )
        
        # 선택된 branch들의 축소된 채널 수의 합을 fusion 모듈의 입력으로 사용
        self.fused_channel_sum = sum(self.feature_mapping[i][0] // 4 for i in self.used_indices)
        self.fuse_features = nn.Sequential(
            CBAM(c1=self.fused_channel_sum, kernel_size=3),
            CBAM(c1=self.fused_channel_sum, kernel_size=3),
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
            nn.Linear(in_features=512 + self.fused_channel_sum, out_features=self.fused_channel_sum),
            nn.LayerNorm(self.fused_channel_sum),
            nn.SiLU(inplace=True),
            nn.Linear(in_features=self.fused_channel_sum, out_features=self.fused_channel_sum // 2),
            nn.LayerNorm(self.fused_channel_sum // 2),
            nn.SiLU(inplace=True),
            TransformerLayer(c=self.fused_channel_sum // 2, num_heads=8)
        )
        
        self.AttFusion = AttentionFusion(
            scene_dim=self.feature_mapping[0][0],
            obj_dim=self.fused_channel_sum // 2,
            ff_dim=128,
            num_heads=8,
            layer_norm=True
        )
        
        self.buffer_size = buffer_size
        self.fused_feat_dim = self.fused_channel_sum // 2
        
        self.temporal_refinement = TemporalAttentionRefinement(
            input_dim=self.fused_feat_dim,
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            return_to_input_dim=True
        )
        
        self.risk_head = nn.Sequential(
            nn.Linear(self.fused_feat_dim, self.fused_feat_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(self.fused_feat_dim // 2, 2)
        )
        
        self._features = {}
        self._register_hooks()  # hooks 등록 (기존과 동일)

    def forward(self, x):
        """
        Forward pass  
        Args:
            x: Input tensor (B, T, C, H, W)
        """
        B, T = x.shape[:2]
        risk_scores = []
        filtered_detections = []
        obj_attns = []
        
        # feature buffer 초기화
        feat_buffer = torch.zeros(B, self.buffer_size, self.fused_feat_dim, device=x.device, dtype=x.dtype)
        scene_hidden = None
        
        for t in range(T):
            if x.device in self._features:
                self._features[x.device].clear()
            
            results = self.detector.predict(source=x[:, t], imgsz=640, conf=self.conf_thresh, verbose=False)
            
            MAX_DETECTIONS = 20
            batch_boxes = torch.zeros((B, MAX_DETECTIONS, 6), device=x.device)
            batch_boxes_normalized = torch.zeros((B, MAX_DETECTIONS, 4), device=x.device)
            valid_detections = torch.zeros(B, dtype=torch.long, device=x.device)
            
            # detection 후처리
            for batch_idx in range(B):
                detections = results[batch_idx].boxes
                clss, confs, xyxy, xywhn = detections.cls, detections.conf, detections.xyxy, detections.xywhn
                conf_mask = confs > self.conf_thresh
                cls_mask = torch.isin(clss, torch.tensor(self.cls_list, device=clss.device))
                mask = conf_mask & cls_mask
                clss = clss[mask]
                confs = confs[mask]
                xyxy = xyxy[mask]
                xywhn = xywhn[mask]
                if len(clss) > MAX_DETECTIONS:
                    top_k_indices = torch.topk(confs, k=MAX_DETECTIONS).indices
                    clss = clss[top_k_indices]
                    confs = confs[top_k_indices]
                    xyxy = xyxy[top_k_indices]
                    xywhn = xywhn[top_k_indices]
                num_dets = len(clss)
                batch_boxes[batch_idx, :num_dets, :4] = xyxy
                batch_boxes[batch_idx, :num_dets, 4] = clss
                batch_boxes[batch_idx, :num_dets, 5] = confs
                batch_boxes_normalized[batch_idx, :num_dets] = xywhn
                valid_detections[batch_idx] = num_dets
            filtered_detections.append((batch_boxes, valid_detections))
            
            # Hook을 통해 필요한 feature 추출 (사용하는 branch만)
            features = {}
            if 0 in self.used_indices:
                features[0] = self._features[x.device]['model.8'].clone()
            if 1 in self.used_indices:
                features[1] = self._features[x.device]['model.16'].clone()
            if 2 in self.used_indices:
                features[2] = self._features[x.device]['model.19'].clone()
            if 3 in self.used_indices:
                features[3] = self._features[x.device]['model.22'].clone()
            
            # scene feature 처리 (항상 backbone feature 사용)
            scene_feat = self.mean_pool(self._features[x.device]['model.8'])
            scene_feat = scene_feat.squeeze(-1).squeeze(-1).unsqueeze(1)
            scene_feat, scene_hidden = self.scene_temporal(scene_feat, scene_hidden)
            scene_feat = scene_feat.squeeze(1)
            
            # ROIAlign를 위한 rois 생성
            batch_indices = []
            roi_boxes = []
            for b in range(B):
                num_valid = valid_detections[b].item()
                batch_indices.extend([b] * num_valid)
                roi_boxes.append(batch_boxes[b, :num_valid, :4])
            if len(batch_indices) > 0:
                batch_indices = torch.tensor(batch_indices, device=x.device).view(-1, 1)
                roi_boxes = torch.cat(roi_boxes, dim=0)
                rois = torch.cat([batch_indices.float(), roi_boxes], dim=1)  # (total_boxes, 5)
            else:
                rois = torch.empty((0, 5), device=x.device)
            
            # 각 branch에 대해 ROIAlign 및 채널 축소 수행
            reduced_feats = []
            if 0 in self.used_indices:
                feat = self.roi_align_backbone(features[0], rois)
                feat = self.backbone_channel_reduction(feat)
                reduced_feats.append(feat)
            if 1 in self.used_indices:
                feat = self.roi_align_p3(features[1], rois)
                feat = self.p3_channel_reduction(feat)
                reduced_feats.append(feat)
            if 2 in self.used_indices:
                feat = self.roi_align_p4(features[2], rois)
                feat = self.p4_channel_reduction(feat)
                reduced_feats.append(feat)
            if 3 in self.used_indices:
                feat = self.roi_align_p5(features[3], rois)
                feat = self.p5_channel_reduction(feat)
                reduced_feats.append(feat)
            
            # Concatenate 후 fusion
            obj_feats = torch.cat(reduced_feats, dim=1)
            obj_feats = self.fuse_features(obj_feats)
            obj_feats = self.mean_pool(obj_feats).squeeze(-1).squeeze(-1)
            
            # 좌표 임베딩 및 feature fusion
            batch_coords = []
            for b in range(B):
                num_valid = valid_detections[b].item()
                batch_coords.append(batch_boxes_normalized[b, :num_valid])
            if batch_coords:
                obj_coords = torch.cat(batch_coords, dim=0)
            else:
                obj_coords = torch.empty((0, 4), device=x.device)
            obj_coord = self.coord_embed(obj_coords)
            obj_feats = torch.cat([obj_feats, obj_coord], dim=-1)
            obj_feats = self.fuse_coord_feat(obj_feats)
            
            # Detection별 feature를 배치 단위로 재구성 (padding 포함)
            start_idx = 0
            batch_obj_feats = torch.zeros(B, MAX_DETECTIONS, obj_feats.shape[-1], device=x.device)
            attention_mask = torch.zeros(B, MAX_DETECTIONS, dtype=torch.bool, device=x.device)
            for b in range(B):
                num_valid = valid_detections[b].item()
                if num_valid > 0:
                    end_idx = start_idx + num_valid
                    batch_obj_feats[b, :num_valid] = obj_feats[start_idx:end_idx]
                    attention_mask[b, :num_valid] = True
                    start_idx = end_idx
            
            fused_feat, obj_attn = self.AttFusion(
                scene_feat,           # (B, scene_dim)
                batch_obj_feats,      # (B, MAX_DETECTIONS, obj_dim)
                attention_mask        # (B, MAX_DETECTIONS)
            )
            obj_attns.append(obj_attn)
            
            # Feature buffer 업데이트 및 temporal refinement 적용
            feat_buffer = feat_buffer.roll(-1, dims=1)
            feat_buffer[:, -1, :] = fused_feat
            if t >= 2:
                current_refined = self.temporal_refinement(feat_buffer)
            else:
                current_refined = fused_feat
            
            risk_score = self.risk_head(current_refined)
            risk_scores.append(risk_score)
        
        return {
            'risk_score': risk_scores,
            'detections': filtered_detections,
            'obj_attns': obj_attns
        }

    def forward_single_batch(self, x):
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

        # Initialize feature buffer with zeros
        feat_buffer = torch.zeros(self.buffer_size, self.fused_feat_dim, device=x.device, dtype=x.dtype)
        current_buffer_idx = 0

        scene_hidden = None

        for t in range(T):
            # if x.device key is in _features, clear the features
            if x.device in self._features:
                self._features[x.device].clear()

            # Modify the feature extraction process
            results = self.detector.predict(source=x[:, t], imgsz=640, conf=self.conf_thresh, verbose=False)

            detections = results[
                0].boxes  # each sample of detections have det.cls, det.conf, det.xywh, det.xywhn, det.xyxy, det.xyxyn
            clss, confs, xyxy, xywhn = detections.cls, detections.conf, detections.xyxy, detections.xywhn  # (N,), (N,), (N,4), (N,4)

            conf_mask = confs > self.conf_thresh
            cls_mask = torch.isin(clss, torch.tensor(self.cls_list, device=clss.device))
            mask = conf_mask & cls_mask

            # mask = confs > self.conf_thresh

            # filter out detections with low confidence and not in cls_list
            clss, confs, xyxy, xywhn = clss[mask], confs[mask], xyxy[mask], xywhn[mask]

            if len(clss) > 20:
                # Get indices of top 20 confidence scores
                top_k_indices = torch.topk(confs, k=20).indices

                # Filter all tensors to keep only top 20
                clss = clss[top_k_indices]
                confs = confs[top_k_indices]
                xyxy = xyxy[top_k_indices]
                xywhn = xywhn[top_k_indices]

            # Extract features using hooks
            backbone_feat, p3_feat, p4_feat, p5_feat = self._features[x.device]['model.8'].clone(), \
            self._features[x.device]['model.16'].clone(), self._features[x.device]['model.19'].clone(), \
            self._features[x.device]['model.22'].clone()

            # global avg pooling on backbone feature
            scene_feat = self.mean_pool(backbone_feat)
            scene_feat = scene_feat.squeeze().unsqueeze(0)  # n: (1, 1, 256), x: (1, 1, 640)
            scene_feat, scene_hidden = self.scene_temporal(scene_feat, scene_hidden)  # n: (1, 1, 256), x: (1, 1, 640)
            scene_feat = scene_feat.squeeze(0)  # n: (1, 256), x: (1, 640)

            num_obj = len(clss)
            if num_obj == 0:
                xyxy = torch.tensor([[0, 0, 320, 320], [320, 320, 640, 640], [0, 320, 320, 640], [320, 0, 640, 320]],
                                    dtype=torch.float32).to(x.device)
                xywhn = torch.tensor(
                    [[160, 160, 160, 160], [480, 480, 160, 160], [160, 480, 160, 160], [480, 160, 160, 160]],
                    dtype=torch.float32).to(x.device)
                filtered_detections.append((torch.tensor([]), torch.tensor([]), xyxy))
            else:
                filtered_detections.append((clss, confs, xyxy))

            # xyxy: (N, 4) -> 0: x1, 1: y1, 2: x2, 3: y2 -> xyxy: (N, 5) 0: 0, 1: x1, 2: y1, 3: x2, 4: y2
            xyxy = torch.cat([torch.zeros_like(xyxy[:, :1]), xyxy], dim=1)

            # 각 객체에 대한 특징 추출
            obj_backbone_feats = self.roi_align_backbone(input=backbone_feat, rois=xyxy)  # n:(N, 256, 8, 8), x:(N, 640, 8, 8)
            obj_p3_feats = self.roi_align_p3(input=p3_feat, rois=xyxy)  # n:(N, 64, 8, 8), x:(N, 320, 8, 8)
            obj_p4_feats = self.roi_align_p4(input=p4_feat, rois=xyxy)  # n:(N, 128, 8, 8), x:(N, 640, 8, 8)
            obj_p5_feats = self.roi_align_p5(input=p5_feat, rois=xyxy)  # n:(N, 256, 8, 8), x:(N, 640, 8, 8)

            obj_backbone_feats = self.backbone_channel_reduction(obj_backbone_feats)  # n:(N, 64, 8, 8), x:(N, 160, 8, 8)
            obj_p3_feats = self.p3_channel_reduction(obj_p3_feats)  # n:(N, 16, 8, 8), x:(N, 80, 8, 8)
            obj_p4_feats = self.p4_channel_reduction(obj_p4_feats)  # n:(N, 32, 8, 8), x:(N, 160, 8, 8)
            obj_p5_feats = self.p5_channel_reduction(obj_p5_feats)  # n:(N, 64, 8, 8), x:(N, 160, 8, 8)

            obj_feats = torch.cat([obj_backbone_feats, obj_p3_feats, obj_p4_feats, obj_p5_feats], dim=1)  # n:(N, 448, 8, 8), x:(N, 1280, 8, 8)
            obj_feats = self.fuse_backbone_p3_p4_p5(obj_feats)  # n:(N, 448, 8, 8), x:(N, 1280, 8, 8)

            # global avg pooling
            obj_feats = self.mean_pool(obj_feats).squeeze(2).squeeze(2)  # n:(N, 112), x:(N, 400)
            obj_coord = self.coord_embed(xywhn)  # n:(N, 512), x:(N, 512)
            obj_feats = torch.cat([obj_feats, obj_coord], dim=-1)  # n:(N, 640), x:(N, 1024)
            obj_feats = self.fuse_coord_feat(obj_feats)  # n:(N, 640), x:(N, 1024)
            
            fused_feat, obj_attn = self.AttFusion(scene_feat, obj_feats.unsqueeze(0))  # (1, 64), (1, N)
            obj_attns.append(obj_attn)

            # Update feature buffer``
            feat_buffer = feat_buffer.roll(-1, dims=0)  # Roll buffer to make space for new feature
            feat_buffer[-1] = fused_feat  # Add new feature

            # Apply temporal refinement on the buffer
            if t >= 2:  # Wait for at least 3 frames of context
                # Process the buffer through temporal attention refinement
                current_refined = self.temporal_refinement(feat_buffer.unsqueeze(0))
            else:
                current_refined = fused_feat

            # Get risk prediction using refined features
            risk_score = self.risk_head(current_refined)
            risk_scores.append(risk_score)  # Add batch dimension back

        '''
        Return:
            risk_score: List[torch.Tensor] : length of T, each shape: (1, 2)
            detections: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] : length of T, each Tuple: (cls: N, conf: N, xyxy: N, 4)
        '''

        return {
            'risk_score': risk_scores,
            'detections': filtered_detections,
            'obj_attns': obj_attns
        }


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

class TAATemporal(nn.Module):
    """TAA를 위한 Loss 함수
    
    - Positive examples: exp(-(toa-t-1)/fps) * CE_loss
    - Negative examples: CE_loss
    - Temporal consistency: lambda_temporal * |pred_t - pred_{t-1}|
    - Attention supervision: lambda_attn * attention_loss
    """
    def __init__(self, lambda_temporal: float = 0.01, lambda_attn: float = 0.1, fps: float = 10.0, iou_thresh: float = 0.3):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.lambda_temporal = lambda_temporal
        self.lambda_attn = lambda_attn
        self.fps = fps
        self.iou_thresh = iou_thresh
        
    def forward(self, predictions: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        # Unpack predictions
        risk_scores = predictions['risk_score']  # List[T] of (B, 2)
        obj_attns = predictions['obj_attns']  # List[T] of List[B] of (N,)
        detections = predictions['detections']  # List[T] of Tuple(batch_boxes, valid_detections)

        '''
        Batch['frames'] : (B, T, C, H, W)
        Batch['frame_ids'] : list (B,)
        Batch['is_positive'] : (B,)
        Batch['toa'] : (B,)
        Batch['annotations'] : list (B,) if positive dict:100, or list:0
        '''
        
        T = len(risk_scores)
        B = batch['is_positive'].shape[0]
        
        # Stack risk scores: List[T] of (B, 2) -> (T, B, 2)
        pred_batch = torch.stack(risk_scores, dim=0)  # (T, B, 2)
        time_steps = torch.arange(T, device=pred_batch.device).float()
        
        # Expand time steps for broadcasting: (T,) -> (T, B)
        time_steps = time_steps.unsqueeze(-1).expand(-1, B)
        
        # Calculate CE Loss for all samples in batch
        pred_batch_reshaped = pred_batch.reshape(-1, 2)  # (T*B, 2)
        # Get targets: (B,) -> (T, B)
        target = batch['is_positive']  # (B,)
        target_expanded = target.unsqueeze(0).long().to(pred_batch_reshaped.device).expand(T, -1)  # (T, B)
        target_expanded_reshaped = target_expanded.reshape(-1)  # (T*B)
        ce_losses = self.ce(pred_batch_reshaped, target_expanded_reshaped)  # (T*B,)
        ce_losses = ce_losses.view(T, B)  # Reshape back to (T, B)
        
        # Initialize losses for batch
        ce_loss = torch.zeros(B, device=pred_batch.device)
        temporal_loss = torch.zeros(B, device=pred_batch.device)
        attn_loss = torch.zeros(B, device=pred_batch.device)
        
        # Process each sample in batch
        for b in range(B):
            if target[b].item() == 1:  # Positive example
                # Get ToA for current sample
                toa = batch['toa'][b]
                
                # CE Loss with exponential penalty
                penalty = -torch.max(
                    torch.tensor(0., device=pred_batch.device),
                    (toa - time_steps[:, b] - 1) / self.fps
                )
                penalty = torch.exp(penalty)
                ratio = T / penalty.sum()
                ce_loss[b] = torch.mean(ratio * penalty * ce_losses[:, b])
                
                # softmax on last dimension of pred_batch
                pred_batch[:, b, :] = torch.softmax(pred_batch[:, b, :], dim=-1)
                
                # Temporal loss for positive samples
                accident_probs = pred_batch[:, b, 1]  # (T,)
                if T > 1:
                    temporal_diffs = accident_probs[1:] - accident_probs[:-1]  # (T-1,)
                    temporal_loss[b] = torch.relu(-temporal_diffs).mean()
                
                # Attention loss for positive samples
                if 'annotations' in batch:
                    annotations = batch['annotations'][b]
                    if annotations != []:
                        frame_attn_losses = []
                        
                        for t in range(T):
                            ann_t = t + 1
                            if ann_t in annotations:  # 해당 프레임에 annotation이 있는 경우
                                # Get predicted boxes and attention scores for current frame
                                batch_boxes, valid_detections = detections[t]
                                pred_boxes = batch_boxes[b, :valid_detections[b], :4]  # Get boxes for current batch
                                curr_attn = obj_attns[t][b]  # Get attention for current batch
                                
                                if valid_detections[b] == 0:
                                    continue
                                    
                                # Get ground truth boxes and related flags for current batch
                                gt_boxes = []
                                for anno in annotations[ann_t]:
                                    if anno['is_related'] == 1:  # Check batch index
                                        gt_boxes.append(anno['bbox'])
                                
                                if len(gt_boxes) > 0:
                                    gt_boxes = torch.tensor(gt_boxes, device=pred_boxes.device)
                                    frame_attn_loss = accident_attention_loss(
                                        attn_weights=curr_attn,  # (N,)
                                        pred_boxes_xyxy=pred_boxes,  # (N, 4)
                                        gt_boxes_xyxy=gt_boxes,  # (M, 4)
                                        eps=self.iou_thresh
                                    )
                                    
                                    if frame_attn_loss.item() != 0.0:
                                        frame_attn_losses.append(frame_attn_loss)
                        
                        # Average attention loss over frames if any valid losses exist
                        if frame_attn_losses:
                            attn_loss[b] = torch.stack(frame_attn_losses).mean()
            
            else:  # Negative example
                # Simple mean for CE loss
                ce_loss[b] = torch.mean(ce_losses[:, b])
                
                pred_batch[:, b, :] = torch.softmax(pred_batch[:, b, :], dim=-1)
                
                # Temporal loss for negative samples
                accident_probs = pred_batch[:, b, 1]  # (T,)
                if T > 2:
                    first_order_diffs = accident_probs[1:] - accident_probs[:-1]  # (T-1,)
                    second_order_diffs = first_order_diffs[1:] - first_order_diffs[:-1]  # (T-2,)
                    temporal_loss[b] = torch.mean(torch.abs(second_order_diffs))
        
        # Average losses across batch
        ce_loss = ce_loss.mean()
        temporal_loss = temporal_loss.mean()
        attn_loss = attn_loss.mean()
        
        total_loss = ce_loss + self.lambda_temporal * temporal_loss + self.lambda_attn * attn_loss
        
        return {
            'ce_loss': ce_loss,
            'temporal_loss': temporal_loss,
            'attn_loss': attn_loss,
            'total_loss': total_loss
        }

class AdaLEA(nn.Module):
    """Loss for Traffic Accident Anticipation (TAA) with Adaptive Early Anticipation (AdaLEA)
    
    For positive samples:
        L_p^{AdaLEA} = ∑_{t=1}^{T} -α_t · log(r_t)
    where:
        - r_t is the predicted accident probability at time t.
        - d = toa - t - 1  (number of frames before the accident; here, toa is provided per sample)
        - α_t = exp(-max(0, d - fps * Φ(e-1) - γ))
          with Φ(e-1) given by batch['prev_attt'] (previous epoch’s ATTC),
          fps is the video frame rate, and γ is a hyperparameter.
    
    For negative samples:
        L_n^{AdaLEA} = ∑_{t=1}^{T} -log(1 - r_t)
    
    The loss also adds temporal consistency and attention supervision terms
    with weights lambda_temporal and lambda_attn, respectively.
    """
    def __init__(self, 
                 lambda_temporal: float = 0.01, 
                 lambda_attn: float = 0.1, 
                 fps: float = 10.0, 
                 iou_thresh: float = 0.3,
                 gamma: float = 5.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.lambda_temporal = lambda_temporal
        self.lambda_attn = lambda_attn
        self.fps = fps
        self.iou_thresh = iou_thresh
        self.gamma = gamma
        self.last_mtta = 0.0
        
    def forward(self, predictions: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        # Unpack predictions
        risk_scores = predictions['risk_score']  # List[T] of (B, 2)
        obj_attns = predictions['obj_attns']      # List[T] of List[B] of (N,)
        detections = predictions['detections']    # List[T] of Tuple(batch_boxes, valid_detections)

        """
        Batch entries:
            'frames': (B, T, C, H, W)
            'frame_ids': list (B,)
            'is_positive': (B,)       --> 1 for accident (positive), 0 for negative.
            'toa': (B,)               --> Time of accident for positive samples.
            'annotations': list (B,)  --> For positive samples, a dict of annotations; or empty list.
            Optionally, for AdaLEA:
            'prev_mtta': float       --> MTTA value from previous epoch.
        """
        
        T = len(risk_scores)
        B = batch['is_positive'].shape[0]
        
        # Stack risk scores: List[T] of (B, 2) -> (T, B, 2)
        pred_batch = torch.stack(risk_scores, dim=0)  # (T, B, 2)
        # Create a time-step tensor: (T, B)
        time_steps = torch.arange(T, device=pred_batch.device, dtype=torch.float32)
        time_steps = time_steps.unsqueeze(-1).expand(-1, B)
        
        eps = 1e-8  # to avoid log(0)
        
        # Initialize losses for the batch
        ce_loss = torch.zeros(B, device=pred_batch.device)
        temporal_loss = torch.zeros(B, device=pred_batch.device)
        attn_loss = torch.zeros(B, device=pred_batch.device)
        
        # Process each sample in the batch
        for b in range(B):
            # For positive samples (accident videos)
            if batch['is_positive'][b].item() == 1:
                toa = batch['toa'][b]  # Accident time (a scalar)
                # Retrieve previous MTTA (Φ(e-1)); default to 0.0 if not provided.
                mtta = batch.get('prev_mtta', 0.0)
                mtta = torch.tensor(mtta, device=pred_batch.device, dtype=torch.float32)
                
                # Compute probabilities from logits
                probs = torch.softmax(pred_batch[:, b, :], dim=-1)
                # Update pred_batch (used later for temporal loss)
                pred_batch[:, b, :] = probs
                # r: predicted risk probability for accident (class index 1)
                r = probs[:, 1]
                
                # Compute "time-to-accident" for each time step: d = toa - t - 1.
                # (Here, time_steps[:, b] runs from 0 to T-1.)
                d = toa - time_steps[:, b] - 1  # shape: (T,)
                # Compute adaptive penalty factor:
                #   α = exp( - max(0, d - (fps * prev_mtta + gamma)) )
                raw = d - (self.fps * mtta + self.gamma)
                alpha = torch.exp(-torch.clamp(raw, min=0))
                
                # AdaLEA loss for positive sample: -α · log(r)
                pos_loss = -alpha * torch.log(r + eps)
                ce_loss[b] = pos_loss.mean()
                
                # ---------------------------
                # Temporal loss (unchanged for positives)
                accident_probs = probs[:, 1]  # (T,)
                if T > 1:
                    temporal_diffs = accident_probs[1:] - accident_probs[:-1]  # (T-1,)
                    temporal_loss[b] = torch.relu(-temporal_diffs).mean()
                
                # ---------------------------
                # Attention loss (unchanged for positives)
                if 'annotations' in batch:
                    annotations = batch['annotations'][b]
                    if annotations != []:
                        frame_attn_losses = []
                        for t in range(T):
                            ann_t = t + 1  # assuming annotations use 1-indexed frame numbers
                            if ann_t in annotations:
                                batch_boxes, valid_detections = detections[t]
                                pred_boxes = batch_boxes[b, :valid_detections[b], :4]
                                curr_attn = obj_attns[t][b]
                                
                                if valid_detections[b] == 0:
                                    continue
                                    
                                gt_boxes = []
                                for anno in annotations[ann_t]:
                                    if anno['is_related'] == 1:
                                        gt_boxes.append(anno['bbox'])
                                
                                if len(gt_boxes) > 0:
                                    gt_boxes = torch.tensor(gt_boxes, device=pred_boxes.device, dtype=pred_boxes.dtype)
                                    frame_attn_loss = accident_attention_loss(
                                        attn_weights=curr_attn,
                                        pred_boxes_xyxy=pred_boxes,
                                        gt_boxes_xyxy=gt_boxes,
                                        eps=self.iou_thresh
                                    )
                                    if frame_attn_loss.item() != 0.0:
                                        frame_attn_losses.append(frame_attn_loss)
                        if frame_attn_losses:
                            attn_loss[b] = torch.stack(frame_attn_losses).mean()
                            
            # For negative samples (non-accident videos)
            else:
                probs = torch.softmax(pred_batch[:, b, :], dim=-1)
                pred_batch[:, b, :] = probs
                r = probs[:, 1]
                # AdaLEA loss for negatives: -log(1 - r)
                neg_loss = -torch.log(1 - r + eps)
                ce_loss[b] = neg_loss.mean()
                
                # ---------------------------
                # Temporal loss for negatives (unchanged)
                accident_probs = probs[:, 1]
                if T > 2:
                    first_order_diffs = accident_probs[1:] - accident_probs[:-1]
                    second_order_diffs = first_order_diffs[1:] - first_order_diffs[:-1]
                    temporal_loss[b] = torch.mean(torch.abs(second_order_diffs))
        
        # Average the losses across the batch
        ce_loss = ce_loss.mean()
        temporal_loss = temporal_loss.mean()
        attn_loss = attn_loss.mean()
        
        total_loss = ce_loss + self.lambda_temporal * temporal_loss + self.lambda_attn * attn_loss
        
        return {
            'ce_loss': ce_loss,
            'temporal_loss': temporal_loss,
            'attn_loss': attn_loss,
            'total_loss': total_loss
        }

def accident_attention_loss(
    attn_weights: torch.Tensor,
    pred_boxes_xyxy: torch.Tensor,
    gt_boxes_xyxy: torch.Tensor,
    eps: float = 0.3
) -> torch.Tensor:
    """
    한 프레임 기준 accident-related objects에 대한 attention ranking loss
    """
    device = attn_weights.device
    
    # Squeeze attn_weights if it's 2D (1, N) -> (N,)
    if attn_weights.dim() == 2:
        attn_weights = attn_weights.squeeze(0)
    
    # 입력 텐서들의 device 확인 및 동기화
    pred_boxes_xyxy = pred_boxes_xyxy.to(device)
    gt_boxes_xyxy = gt_boxes_xyxy.to(device)
    
    # Shape 및 device 디버깅
    # print(f"[Debug] Device info:")
    # print(f"- attn_weights: {attn_weights.device}, shape: {attn_weights.shape}")
    # print(f"- pred_boxes: {pred_boxes_xyxy.device}, shape: {pred_boxes_xyxy.shape}")
    # print(f"- gt_boxes: {gt_boxes_xyxy.device}, shape: {gt_boxes_xyxy.shape}")

    # GT box가 없을 경우 처리
    if gt_boxes_xyxy.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # 1. Calculate IoU matrix
    try:
        ious = box_iou(pred_boxes_xyxy, gt_boxes_xyxy)  # (N, M)
    except Exception as e:
        print(f"[Error] IoU calculation failed: {e}")
        print(f"pred_boxes_xyxy: {pred_boxes_xyxy}")
        print(f"gt_boxes_xyxy: {gt_boxes_xyxy}")
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    # 2. Find best matching predicted boxes for each GT box
    matched_pred_indices = []  # Store indices of matched predicted boxes
    
    # Sort GT boxes by their maximum IoU scores (내림차순)
    max_ious_per_gt, _ = ious.max(dim=0)  # (M,)
    gt_priority = torch.argsort(max_ious_per_gt, descending=True)
    
    # Track which predicted boxes have been matched
    used_pred_boxes = torch.zeros(len(pred_boxes_xyxy), dtype=torch.bool, device=device)
    
    for gt_idx in gt_priority:
        # Get IoUs for current GT box
        curr_ious = ious[:, gt_idx]
        
        # Mask out already matched predicted boxes
        curr_ious[used_pred_boxes] = 0
        
        # Find best available match
        best_iou, best_pred_idx = curr_ious.max(dim=0)
        
        # If IoU is above threshold, record the match
        if best_iou > eps:
            matched_pred_indices.append(best_pred_idx.item())
            used_pred_boxes[best_pred_idx] = True
    
    # If no matches found above threshold
    if not matched_pred_indices:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    try:
        # 3. Calculate ranking loss
        matched_pred_indices = torch.tensor(matched_pred_indices, device=device)
        unmatched_pred_indices = torch.tensor(
            [i for i in range(len(pred_boxes_xyxy)) if i not in matched_pred_indices], 
            device=device
        )
        
        # Get attention scores for matched and unmatched predictions
        matched_scores = attn_weights[matched_pred_indices]      # (K,) where K is number of matches
        unmatched_scores = attn_weights[unmatched_pred_indices]  # (N-K,)
        
        # Compute pairwise ranking loss
        loss = torch.tensor(0.0, device=device)
        if len(unmatched_scores) > 0:  # Only if there are unmatched objects
            for matched_score in matched_scores:
                # Calculate margin ranking loss: max(0, margin - (matched_score - unmatched_score))
                margin = 0.1  # Can be adjusted
                ranking_diffs = margin - (matched_score - unmatched_scores)
                ranking_loss = torch.relu(ranking_diffs).mean()
                loss = loss + ranking_loss
            
            # Normalize by number of matched objects
            loss = loss / len(matched_scores)
        
        return loss
        
    except Exception as e:
        print(f"[Error] Ranking loss calculation failed: {e}")
        print(f"matched_pred_indices: {matched_pred_indices}")
        print(f"unmatched_pred_indices: {unmatched_pred_indices}")
        print(f"attn_weights shape: {attn_weights.shape}")
        return torch.tensor(0.0, device=device, requires_grad=True)


if __name__ == '__main__':
    import time
    import os
    from torchvision.io import read_image
    from torchvision.transforms import Resize
    from thop import profile
    import torch.cuda.profiler as profiler
    
    def ensure_cuda_model(model):
        """Ensures all submodules and their parameters are on CUDA"""
        for module in model.modules():
            for param in module.parameters(recurse=False):
                if param.device.type != 'cuda':
                    param.data = param.data.cuda()
            for buf in module.buffers(recurse=False):
                if buf.device.type != 'cuda':
                    buf.data = buf.data.cuda()
        return model

    # videos = ['000001', '000002', '000003', '000004']
    videos = ['000001']
    base_folder = "./taa/data/DAD/frames/training/positive"

    # Initialize list to store batches from each video
    all_video_frames = []

    for video_id in videos:
        # Create path for each video folder
        folder_path = os.path.join(base_folder, video_id)
        
        # Get all image files and sort them
        images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()  # Ensure frames are in order
        
        # Take first 100 frames
        images = images[:100]
        
        # Load and process images
        frames = [os.path.join(folder_path, img) for img in images]
        frames = [read_image(img) for img in frames]
        frames = [Resize((640, 640))(img) for img in frames]
        frames = [img / 255.0 for img in frames]
        
        # Stack frames for this video
        video_frames = torch.stack(frames)  # Shape: (100, 3, 640, 640)
        all_video_frames.append(video_frames)

    # Stack all videos together
    images = torch.stack(all_video_frames).to('cuda')  # Shape: (4, 100, 3, 640, 640)
    # images = images[:1, :, :, :, :]
    
    # Model initialization
    model = YOLOv10TAADetectionModel().to('cuda')
    model.detector.fuse()
    model = ensure_cuda_model(model)  # 모든 서브모듈을 CUDA로 이동
    model.eval()
    
    # 1. Model Parameters Analysis
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print("\n=== Model Parameters Analysis ===")
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M, {trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable parameters: {non_trainable_params:,} ({non_trainable_params/1e6:.2f}M, {non_trainable_params/total_params*100:.2f}%)")
    
    # 3. Speed Analysis (FPS)
    print("\n=== Speed Analysis ===")
    
    # Warmup
    print("Warming up...")
    with torch.no_grad():
        for _ in range(10):
            _ = model.forward_single_batch(images)
    
    # FPS measurement
    num_frames = images.shape[1]
    num_runs = 50
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.forward_single_batch(images)
    torch.cuda.synchronize()
    end = time.time()
    
    fps = (num_runs * num_frames) / (end - start)
    
    print(f"Processing FPS: {fps:.2f}")
    
    # 4. Memory Usage Analysis
    print("\n=== Memory Usage Analysis ===")
    print(f"Current GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f}GB")
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")
    print(f"Current GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f}GB")
    print(f"Maximum GPU memory reserved: {torch.cuda.max_memory_reserved()/1e9:.2f}GB")
    
    # 5. Model Size
    print("\n=== Model Size Analysis ===")
    model_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    total_size = model_size + buffer_size
    
    print(f"Model Parameters Size: {model_size/1e6:.2f}MB")
    print(f"Model Buffers Size: {buffer_size/1e6:.2f}MB")
    print(f"Total Model Size: {total_size/1e6:.2f}MB")