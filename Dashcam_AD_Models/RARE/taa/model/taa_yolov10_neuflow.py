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
from taa.model.NeuFlow.neuflow import NeuFlow
from taa.model.NeuFlow.backbone_v7 import ConvBlock, fuse_conv_and_bn
from taa.model.NeuFlow import flow_viz


YOLOv10_feature_mapping = {
    # backbone_feat, p3_feat, p4_feat, p5_feat, ROI Aligned feat
    'yolov10n': [[256, 7, 7], [64, 80, 80], [128, 40, 40], [256, 20, 20], [1, 8, 8], [2, 8, 8], [4, 8, 8]],
    'yolov10s': [[512, 20, 20], [128, 80, 80], [256, 40, 40], [512, 20, 20]],
    'yolov10m': [[576, 20, 20], [192, 80, 80], [384, 40, 40], [576, 20, 20]],
    'yolov10B': [[512, 20, 20], [256, 80, 80], [512, 40, 40], [512, 20, 20]],
    'yolov10L': [[512, 20, 20], [256, 80, 80], [512, 40, 40], [512, 20, 20]],
    'yolov10x': [[640, 20, 20], [320, 80, 80], [640, 40, 40], [640, 20, 20], [5, 8, 8], [10, 8, 8], [10, 8, 8]],
}

class YOLOv10TAANeuFlowDetectionModel(nn.Module):
    def __init__(self, yolo_id='yolov10x', yolo_ft='None', conf_thresh=0.15):
        super().__init__()

        # 1. Base detector initialization
        if yolo_ft == 'None':
            self.detector = YOLOv10.from_pretrained(f'jameslahm/{yolo_id}')
            print(f"Loaded YOLOv10 from pretrained {yolo_id}")
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

        
        # Initialize NeuFlow model
        self.flow_model = NeuFlow().to(next(self.detector.parameters()).device)
        
        # Load NeuFlow weights
        flow_checkpoint = torch.load('taa/model/weights/neuflow_mixed.pth', map_location='cuda')
        self.flow_model.load_state_dict(flow_checkpoint['model'], strict=True)
        
        # First set eval mode
        self.flow_model.eval()
        
        # Fuse Conv and BN layers and handle non-leaf tensors properly
        for m in self.flow_model.modules():
            if type(m) is ConvBlock:
                # Fuse and create new parameters that are leaf tensors
                fused_conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
                fused_conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
                
                # Create new Conv2d modules with the fused parameters
                m.conv1 = nn.Conv2d(
                    fused_conv1.in_channels,
                    fused_conv1.out_channels,
                    fused_conv1.kernel_size,
                    fused_conv1.stride,
                    fused_conv1.padding,
                    bias=True
                )
                m.conv2 = nn.Conv2d(
                    fused_conv2.in_channels,
                    fused_conv2.out_channels,
                    fused_conv2.kernel_size,
                    fused_conv2.stride,
                    fused_conv2.padding,
                    bias=True
                )
                
                # Copy the fused parameters
                m.conv1.weight.data = fused_conv1.weight.data.detach()
                m.conv1.bias.data = fused_conv1.bias.data.detach()
                m.conv2.weight.data = fused_conv2.weight.data.detach()
                m.conv2.bias.data = fused_conv2.bias.data.detach()
                
                # Remove old norm layers
                delattr(m, "norm1")
                delattr(m, "norm2")
                m.forward = m.forward_fuse
        
        # Now freeze all parameters (they should all be leaf tensors now)
        for param in self.flow_model.parameters():
            param.requires_grad = False
        
        # Initialize NeuFlow
        self.flow_model.init_bhwd(1, 640, 640, 'cuda', False)

        self.mean_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.scene_temporal = nn.GRU(input_size=self.feature_mapping[0][0], hidden_size=self.feature_mapping[0][0], num_layers=2, dropout=0.1, batch_first=True)
        # self.no_obj_head = nn.Sequential(nn.Linear(self.feature_mapping[0][0], self.feature_mapping[0][0]//4), nn.SiLU(inplace=True), nn.Linear(self.feature_mapping[0][0]//4, 2))

        self.roi_align_backbone = RoIAlign(output_size=8, spatial_scale=20/640, sampling_ratio=-1)
        self.roi_align_p3 = RoIAlign(output_size=8, spatial_scale=80/640, sampling_ratio=-1)
        self.roi_align_p4 = RoIAlign(output_size=8, spatial_scale=40/640, sampling_ratio=-1)
        self.roi_align_p5 = RoIAlign(output_size=8, spatial_scale=20/640, sampling_ratio=-1)
        self.roi_align_flow = RoIAlign(output_size=8, spatial_scale=80/640, sampling_ratio=-1)

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
        )

        self.fuse_corr = nn.Sequential(
            CBAM(c1=81+self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, kernel_size=3),
            CBAM(c1=81+self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4, kernel_size=3),
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
            nn.Linear(in_features=512+81+self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4,
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
        
        self.AttFusion = AttentionFusion(scene_dim=self.feature_mapping[0][0], obj_dim=(self.feature_mapping[0][0]//4+self.feature_mapping[1][0]//4+self.feature_mapping[2][0]//4+self.feature_mapping[3][0]//4)//2, ff_dim=128, num_heads=8, layer_norm=True)
        
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
        attns_in_each_obj = []
        obj_attns = []

        # Register hooks

        scene_hidden = None
        # fused_hidden = None
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

            if t > 0:
                with torch.no_grad():  # Ensure no gradients are computed for flow model
                    _, corrs = self.flow_model(x[:, t-1], x[:, t])
            else:
                with torch.no_grad():  # Ensure no gradients are computed for flow model
                    _, corrs = self.flow_model(x[:, t], x[:, t])
            
            corrs = corrs.detach()  # Detach correlations from computation graph

            # Extract features using hooks
            backbone_feat, p3_feat, p4_feat, p5_feat = self._features[x.device]['model.8'].clone(), self._features[x.device]['model.16'].clone(), self._features[x.device]['model.19'].clone(), self._features[x.device]['model.22'].clone()
            
            # global avg pooling on backbone feature
            scene_feat = self.mean_pool(backbone_feat)
            scene_feat = scene_feat.squeeze().unsqueeze(0).unsqueeze(0) # n: (1, 1, 256), x: (1, 1, 640)
            scene_feat, scene_hidden = self.scene_temporal(scene_feat, scene_hidden) # n: (1, 1, 256), x: (1, 1, 640)
            scene_feat = scene_feat.squeeze(0) # n: (1, 256), x: (1, 640)
            # scene_risk = self.no_obj_head(scene_feat)

            num_obj = len(clss)
            if num_obj == 0:
                xyxy = torch.tensor([[0, 0, 320, 320], [320, 320, 640, 640], [0, 320, 320, 640], [320, 0, 640, 320]], dtype=torch.float32).to(x.device)
                xywhn = torch.tensor([[160, 160, 160, 160], [480, 480, 160, 160], [160, 480, 160, 160], [480, 160, 160, 160]], dtype=torch.float32).to(x.device)
                filtered_detections.append((torch.tensor([]), torch.tensor([]), xyxy, xywhn))
            else:
                filtered_detections.append((clss, confs, xyxy))

            # xyxy: (N, 4) -> 0: x1, 1: y1, 2: x2, 3: y2 -> xyxy: (N, 5) 0: 0, 1: x1, 2: y1, 3: x2, 4: y2
            xyxy = torch.cat([torch.zeros_like(xyxy[:, :1]), xyxy], dim=1)
            
            # 각 객체에 대한 특징 추출
            obj_backbone_feats = self.roi_align_backbone(input=backbone_feat, rois=xyxy) # n:(N, 256, 8, 8), x:(N, 640, 8, 8)
            obj_p3_feats = self.roi_align_p3(input=p3_feat, rois=xyxy) # n:(N, 64, 8, 8), x:(N, 320, 8, 8)
            obj_p4_feats = self.roi_align_p4(input=p4_feat, rois=xyxy) # n:(N, 128, 8, 8), x:(N, 640, 8, 8)
            obj_p5_feats = self.roi_align_p5(input=p5_feat, rois=xyxy) # n:(N, 256, 8, 8), x:(N, 640, 8, 8)
            obj_corrs_feats = self.roi_align_flow(input=corrs, rois=xyxy) # x:(N,81, 8, 8)

            obj_backbone_feats = self.backbone_channel_reduction(obj_backbone_feats) # n:(N, 64, 8, 8), x:(N, 160, 8, 8)
            obj_p3_feats = self.p3_channel_reduction(obj_p3_feats) # n:(N, 16, 8, 8), x:(N, 80, 8, 8)
            obj_p4_feats = self.p4_channel_reduction(obj_p4_feats) # n:(N, 32, 8, 8), x:(N, 160, 8, 8)
            obj_p5_feats = self.p5_channel_reduction(obj_p5_feats) # n:(N, 64, 8, 8), x:(N, 160, 8, 8)

            obj_feats = torch.cat([obj_backbone_feats, obj_p3_feats, obj_p4_feats, obj_p5_feats], dim=1) # n:(N, 448, 8, 8), x:(N, 1280, 8, 8)
            obj_feats = self.fuse_backbone_p3_p4_p5(obj_feats) # n:(N, 448, 8, 8), x:(N, 1280, 8, 8)
            
            # obj_feats = torch.cat([obj_p3_feats, obj_p4_feats, obj_p5_feats], dim=1) # n:(N, 112, 8, 8), x:(N, 400, 8, 8)
            # obj_feats = self.fuse_p3_p4_p5(obj_feats) # n:(N, 112, 8, 8), x:(N, 400, 8, 8)
            # obj_feats, attn_in_each_obj = self.MHSA(obj_feats)
            # attns_in_each_obj.append(attn_in_each_obj)

            obj_feats = torch.cat([obj_corrs_feats, obj_feats], dim=1)
            obj_feats = self.fuse_corr(obj_feats)

            # global avg pooling
            obj_feats = self.mean_pool(obj_feats).squeeze(2).squeeze(2) # n:(N, 112), x:(N, 400)
            obj_coord = self.coord_embed(xywhn) # n:(N, 512), x:(N, 512)
            obj_feats = torch.cat([obj_feats, obj_coord], dim=-1) # n:(N, 640), x:(N, 1024)
            obj_feats = self.fuse_coord_feat(obj_feats) # n:(N, 640), x:(N, 1024)
            
            
            
            fused_feat, obj_attn = self.AttFusion(scene_feat, obj_feats) # (1, 64), (1, N)
            obj_attns.append(obj_attn)
            # fused_feat, fused_hidden = self.fuse_temporal(fused_feat, fused_hidden)
            obj_risk = self.risk_head(fused_feat) # (1, 2)
            risk_scores.append(obj_risk)

            
            
        '''
        Return:
            risk_score: List[torch.Tensor] : length of T, each shape: (1, 2)
            detections: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] : length of T, each Tuple: (cls: N, conf: N, xyxy: N, 4)
        '''

        return {
            'risk_score': risk_scores,
            'detections': filtered_detections,
            'attns_in_each_obj': attns_in_each_obj,
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

        # Set train/eval mode for modules
        for name, module in self.named_children():
            if name == 'detector' or name == 'flow_model':
                for param in module.parameters():
                    param.requires_grad = False
            else:
                module.train(mode)
                for param in module.parameters():
                    param.requires_grad = True

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
            
            print(penalty)
            
            ce_loss = torch.mean(penalty * ce_losses)
            
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
    import time

    # read real images from folder "./taa/data/DAD/frames/training/positive/000001" and "./taa/data/DAD/frames/training/positive/000002"
    import os
    from torchvision.io import read_image
    from torchvision.transforms import Resize
    
    # read images from folder
    folder_path = "./taa/data/DAD/frames/training/positive/000001"
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    images = [os.path.join(folder_path, img) for img in images]
    images = [read_image(img) for img in images]
    images = [Resize((640, 640))(img) for img in images]
    images = [img / 255.0 for img in images]
    images = torch.stack(images).unsqueeze(0).to('cuda')
    
    # read images from folder
    # folder_path = "./taa/data/DAD/frames/training/positive/000002"
    # images2 = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    # images2 = [os.path.join(folder_path, img) for img in images2]
    # images2 = [read_image(img) for img in images2]
    # images2 = [Resize((640, 640))(img) for img in images2]
    # images2 = [img / 255.0 for img in images2]
    # images2 = torch.stack(images2).unsqueeze(0).to('cuda')
    #
    # images = torch.cat([images, images2], dim=0)

    # simple model forward test
    model = YOLOv10TAANeuFlowDetectionModel().to('cuda')
    
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
            _ = model(images)
    
    # FPS measurement
    num_frames = images.shape[1]
    num_runs = 1
    
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(images)
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