import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, List

class SimpleBaselineModel(nn.Module):
    """
    Simple baseline model for traffic accident anticipation.
    Uses a CNN backbone (ResNet50 or MobileNetV2) followed by GRU and Linear layers.
    """
    def __init__(self, backbone_type: str = 'resnet50', pretrained: bool = True):
        super().__init__()
        self.backbone_type = backbone_type

        # Initialize backbone
        if backbone_type == 'resnet50':
            backbone = models.resnet50(weights='IMAGENET1K_V2' if pretrained else None)
            self.feature_dim = 2048  # ResNet50's final feature dimension
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC layer
        elif backbone_type == 'mobilenet_v2':
            backbone = models.mobilenet_v2(weights='IMAGENET1K_V2' if pretrained else None)
            self.feature_dim = 1280  # MobileNetV2's final feature dimension
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])  # Remove final FC layer
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")

        # GRU for temporal modeling
        self.hidden_dim = 256
        self.gru = nn.GRU(
            input_size=self.feature_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True
        )

        # Risk prediction head
        self.risk_head = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> Dict[str, List[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor of shape (B, T, C, H, W)
                B: batch size
                T: temporal length
                C: channels
                H, W: height, width
        
        Returns:
            Dictionary containing:
                - risk_score: List of tensors, each of shape (B, 2) for each timestep
        """
        B, T = x.shape[:2]
        risk_scores = []

        # Process each frame
        features = []
        for t in range(T):
            # Extract features using backbone
            feat = self.backbone(x[:, t])  # (B, feature_dim, 1, 1)
            feat = feat.squeeze(-1).squeeze(-1)  # (B, feature_dim)
            features.append(feat)

        # Stack features and process through GRU
        features = torch.stack(features, dim=1)  # (B, T, feature_dim)
        temporal_features, _ = self.gru(features)  # (B, T, hidden_dim)

        # Generate risk scores for each timestep
        for t in range(T):
            risk_score = self.risk_head(temporal_features[:, t])  # (B, 2)
            risk_scores.append(risk_score)

        return {
            'risk_score': risk_scores
        }

    def train(self, mode: bool = True):
        """
        Sets the module in training mode.
        Freezes backbone parameters and keeps other modules trainable.
        """
        super().train(mode)

        # Backbone은 항상 eval 모드로 유지하고 파라미터를 freeze
        self.backbone.eval()
        for param in self.backbone.parameters():
            param.requires_grad = False

        # GRU와 risk_head는 학습 가능하도록 설정
        if mode:
            self.gru.train()
            self.risk_head.train()
            for module in [self.gru, self.risk_head]:
                for param in module.parameters():
                    param.requires_grad = True

        return self

    def eval(self):
        """
        Sets the module in evaluation mode.
        """
        return super().eval()


class TAATemporal(nn.Module):
    """TAA를 위한 수정된 Loss 함수
    
    1. Cross Entropy: 각 타임스텝의 예측에 대한 손실 (시간 민감성 반영)
    2. Temporal Consistency: 연속된 프레임 간의 예측 일관성
    """
    def __init__(self, lambda_temporal: float = 0.1, fps: float = 10.0, toa: int = 90):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction='none')
        self.lambda_temporal = lambda_temporal
        self.fps = fps
        self.toa = toa

    def forward(self, pred, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred (List[torch.Tensor]): length of T, each shape: (B, 2)
            target (torch.Tensor): 정답 (B,), bool

        Returns:
            Dict[str, torch.Tensor]: {
                'ce_loss': Cross Entropy Loss,
                'temporal_loss': Temporal Consistency Loss,
                'total_loss': 전체 Loss
            }
        """
        T = len(pred)  # 타임스텝 길이
        B = pred[0].shape[0]  # 배치 크기

        # 예측 결합: (B*T, 2) 형태로 변환
        pred_batch = torch.cat(pred, dim=0)  # (B*T, 2)
        pred_batch = pred_batch.view(B, T, 2)

        # 1. Cross Entropy Loss
        pred_flat = pred_batch.view(-1, 2)  # (B*T, 2)
        target_expanded = target.unsqueeze(1).expand(-1, T).flatten().long()  # (B*T,)

        # Positive/Negative 분리
        target_positive = (target_expanded == 1).float()  # Positive class mask
        target_negative = (target_expanded == 0).float()  # Negative class mask

        # Time-to-Accident Penalty 계산
        time_steps = torch.arange(T, device=pred[0].device).unsqueeze(0).repeat(B, 1)  # (B, T)
        
        # 선형 감소로 변경하고, 최소 가중치 설정
        min_weight = 0.1  # 최소 가중치
        penalty_positive = torch.clamp(
            1.0 - (self.toa - time_steps - 1) / (self.toa * 2),  # 더 완만한 감소
            min=min_weight
        )
        # ToA 이후 프레임은 최소 가중치 적용
        penalty_positive[:, self.toa:] = min_weight
        
        # Normalize positive weights to sum to T (프레임 수)
        normalizer = (T / penalty_positive.sum()) 
        penalty_positive = penalty_positive * normalizer
        penalty_positive = penalty_positive.view(-1)

        # Negative examples에는 균등한 가중치 적용
        penalty_negative = torch.ones_like(penalty_positive).view(-1)  # 모든 프레임에 동일한 가중치

        # Cross Entropy Loss 계산
        ce_loss = self.ce(pred_flat, target_expanded)
        weighted_ce_loss = (
            (ce_loss * penalty_positive * target_positive).sum() +
            (ce_loss * penalty_negative * target_negative).sum()
        ) / T

        # print("Positive or Negative:", target_positive[0], target_negative[0])
        # print("Weighted CE Loss:", weighted_ce_loss)

        # 2. Temporal Consistency Loss
        # Softmax를 적용하여 anomaly probability 추출
        anomaly_probs = nn.functional.softmax(pred_flat, dim=-1)[:, 1].view(B, T)  # (B, T)
        temporal_loss = torch.abs(anomaly_probs[:, 1:] - anomaly_probs[:, :-1]).mean()
        
        # Total Loss
        total_loss = weighted_ce_loss + self.lambda_temporal * temporal_loss

        return {
            'ce_loss': weighted_ce_loss,
            'temporal_loss': temporal_loss,
            'total_loss': total_loss
        }

if __name__ == '__main__':
    # Simple test
    model = SimpleBaselineModel(backbone_type='resnet50')
    x = torch.randn(2, 100, 3, 224, 224)  # (B, T, C, H, W)
    output = model(x)
    print("Output shape:", len(output['risk_score']), output['risk_score'][0].shape) 