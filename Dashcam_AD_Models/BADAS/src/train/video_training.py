"""
Enhanced Video Classification Training System with Early Stopping
===============================================================

Features:
1. Support for V-JEPA 2 and VideoMAE models
2. Frame-level feature extraction for VideoMAE
3. Configurable classification heads (Linear, MLP, Attention)
4. Temporal processing options
5. Early stopping mechanism
6. All original training capabilities preserved
"""

import os
import re
import cv2
import time
import random
import psutil
import logging
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoConfig, AutoModelForVideoClassification
from albumentations.pytorch import ToTensorV2
import csv
from typing import Optional, Dict, Any, List

# Try to import processors
try:
    from transformers import VideoMAEImageProcessor
    HAS_VIDEOMAE_PROCESSOR = True
except ImportError:
    HAS_VIDEOMAE_PROCESSOR = False

try:
    from transformers import AutoImageProcessor
    HAS_AUTO_PROCESSOR = True
except ImportError:
    HAS_AUTO_PROCESSOR = False

try:
    from transformers import AutoVideoProcessor
    HAS_AUTO_VIDEO_PROCESSOR = True
except ImportError:
    HAS_AUTO_VIDEO_PROCESSOR = False

# =============================================================================
# EARLY STOPPING CLASS - FIXED VERSION
# =============================================================================

class EarlyStopping:
    """Early stopping utility to stop training when validation doesn't improve"""
    
    def __init__(self, patience: int = 2, min_delta: float = 0.001, mode: str = 'max', 
                 restore_best_weights: bool = True, verbose: bool = True):
        """
        Args:
            patience: Number of epochs to wait after last improvement
            min_delta: Minimum change to qualify as an improvement
            mode: 'max' for metrics where higher is better (accuracy), 'min' for lower is better (loss)
            restore_best_weights: Whether to restore model to best weights when stopping
            verbose: Whether to print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
        
        if mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
        else:
            self.is_better = lambda current, best: current < best - min_delta
    
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """
        Call early stopping check
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if should stop training, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            self._save_checkpoint(model)
            if self.verbose:
                print(f"🎯 Early stopping: Initial score {score:.4f}")
            
        elif self.is_better(score, self.best_score):
            if self.verbose:
                improvement = score - self.best_score if self.mode == 'max' else self.best_score - score
                print(f"✅ Early stopping: Score improved from {self.best_score:.4f} to {score:.4f} "
                      f"(+{improvement:.4f}). Reset counter.")
            
            self.best_score = score
            self.counter = 0
            self._save_checkpoint(model)
            
        else:
            self.counter += 1
            if self.verbose:
                print(f"⏳ Early stopping: No improvement for {self.counter}/{self.patience} epochs "
                      f"(current: {score:.4f}, best: {self.best_score:.4f})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"🛑 Early stopping triggered! No improvement for {self.patience} epochs.")
                
                # Restore weights IMMEDIATELY when stopping is triggered
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print(f"🔄 Restored model to best weights (score: {self.best_score:.4f})")
        
        return self.early_stop
    
    def _save_checkpoint(self, model: torch.nn.Module):
        """Save model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

# =============================================================================
# ENHANCED CLASSIFICATION HEADS
# =============================================================================

class LinearHead(nn.Module):
    """Simple linear classification head"""
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dropout(x)
        return self.classifier(x)

class MLPHead(nn.Module):
    """Multi-layer perceptron classification head"""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout)
            ])
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)

class AttentionHead(nn.Module):
    """Enhanced attention-based classification head"""
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 512,
                 num_heads: int = 8, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Multi-head attention layers
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            ) for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.global_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
        # Initialize weights
        nn.init.normal_(self.global_query, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, input_dim) or (batch_size, input_dim)
        
        if len(x.shape) == 2:
            # Single vector input - expand to sequence
            x = x.unsqueeze(1)  # (batch_size, 1, input_dim)
            
        batch_size = x.size(0)
        
        # Project to hidden dimension
        x = self.input_projection(x)
        
        # Apply attention layers
        for layer in self.attention_layers:
            x = layer(x)
            
        # Global attention pooling
        query = self.global_query.expand(batch_size, -1, -1)
        attended, _ = self.global_attention(query=query, key=x, value=x)
        attended = self.norm(attended)
        
        # Classification
        attended = attended.squeeze(1)
        return self.classifier(attended)

# =============================================================================
# TEMPORAL PROCESSING MODULES
# =============================================================================

class TemporalProcessor(nn.Module):
    """Base class for temporal processing"""
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

class MeanPooling(TemporalProcessor):
    """Simple mean pooling across time dimension"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

class MaxPooling(TemporalProcessor):
    """Max pooling across time dimension"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.max(dim=1)[0]

class LSTMProcessor(TemporalProcessor):
    """LSTM-based temporal processing"""
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_layers: int = 2):
        super().__init__(input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.1 if num_layers > 1 else 0)
        self.output_dim = hidden_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, (h_n, c_n) = self.lstm(x)
        return h_n[-1]  # Take last hidden state

class AttentionProcessor(TemporalProcessor):
    """Attention-based temporal processing"""
    def __init__(self, input_dim: int, num_heads: int = 8):

        #print(f"🔍 DEBUG: AttentionProcessor.__init__: input_dim={input_dim}, num_heads={num_heads}")
        #print(f"🔍 DEBUG: input_dim % num_heads = {input_dim % num_heads}")
        super().__init__(input_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        attn_out = self.norm(attn_out)
        return attn_out.mean(dim=1)

# =============================================================================
# ENHANCED VIDEO CLASSIFIER
# =============================================================================

class EnhancedVideoClassifier(nn.Module):
    """
    Enhanced Video Classifier with proper temporal processing flow
    
    Flow:
    1. Extract features from video (sequence or single vector)
    2. If predictor enabled: predict future and combine with present
    3. Apply temporal processing to reduce sequence to single vector
    4. Apply classification head
    """
    
    def __init__(self, config: argparse.Namespace, model_info: dict):
        super().__init__()
        self.config = config
        self.model_info = model_info
        
        # Core configuration
        self.use_frame_level = getattr(config, 'use_frame_level', False)
        self.use_future_prediction = getattr(config, 'use_future_prediction', False)
        self.use_classification_model = getattr(config, 'use_classification_model', False)
        
        # Initialize components
        self.backbone = None
        self.predictor = None
        self.temporal_processor = None
        self.classifier = None
        
        # State tracking
        self.has_predictor = False
        self.use_custom_head = True
        self.feature_dim = None
        
        # Build model
        self._build_model()
        
    def _build_model(self):
        """Build the complete model pipeline"""
        # print(f"🏗️ Building Enhanced Video Classifier")
        # print(f"   Model: {self.config.model_name}")
        # print(f"   Type: {self._get_model_type_name()}")
        
        # Step 1: Load backbone model
        self._load_backbone()
        
        # Step 2: Setup predictor if needed
        self._setup_predictor()
        
        # Step 3: Determine feature dimensions
        self.feature_dim = self._calculate_feature_dimension()
        
        # Step 4: Setup temporal processing (always for sequences)
        self._setup_temporal_processing()
        
        # Step 5: Setup classification head
        self._setup_classification_head()
        
        # Step 6: Report final architecture
        #self._report_architecture()
    
    def _get_model_type_name(self) -> str:
        """Get human-readable model type name"""
        if self.model_info['is_videomae']:
            return "VideoMAE"
        elif self.model_info['is_vjepa2']:
            return "V-JEPA 2"
        else:
            return "Generic Video Model"
    
    def _load_backbone(self):
        """Load the backbone model"""
        #print(f"📦 Loading backbone...")
        
        if self.use_classification_model:
            # Use built-in classification model
            self.backbone = AutoModelForVideoClassification.from_pretrained(
                self.config.model_name,
                num_labels=self.config.num_classes,
                trust_remote_code=True
            )
            self.use_custom_head = False
            #print("   ✅ Loaded with built-in classification head")
            
        else:
            # Use encoder only with custom head
            model_config = AutoConfig.from_pretrained(self.config.model_name, trust_remote_code=True)
            if hasattr(model_config, "drop_path_rate"):
                model_config.drop_path_rate = getattr(self.config, 'drop_path_rate', 0.1)
                
            self.backbone = AutoModel.from_pretrained(
                self.config.model_name,
                config=model_config,
                trust_remote_code=True
            )
            self.use_custom_head = True
            print("Loaded encoder with custom head")
    
    def _setup_predictor(self):
        """Setup predictor for V-JEPA 2 models"""
        if not (self.use_future_prediction and self.use_custom_head and self.model_info['is_vjepa2']):
            if self.use_future_prediction and not self.model_info['is_vjepa2']:
                print("⚠️  Future prediction only available for V-JEPA 2 models")
            return
            
        #print("Setting up V-JEPA 2 predictor...")
        
        try:
            if hasattr(self.backbone, 'predictor'):
                self.predictor = self.backbone.predictor
                self.has_predictor = True
                print(f"Found predictor: {type(self.predictor).__name__}")
            else:
                print("No predictor found in model")
                self.use_future_prediction = False
                
        except Exception as e:
            print(f"Failed to access predictor: {e}")
            self.use_future_prediction = False
    
    def _calculate_feature_dimension(self) -> int:
        """Calculate backbone feature dimension through probing"""
        if self.model_info['is_vjepa2']:
            dim = self._get_vjepa2_dimension()
        else:
            dim = self._probe_backbone_dimension()
        
        #print(f"🔍 DEBUG: Calculated feature_dim = {dim}")
        return dim
    
    def _get_vjepa2_dimension(self) -> int:
        """Get V-JEPA 2 model dimension based on model name"""
        model_name = self.config.model_name.lower()
        
        if 'vitl' in model_name:
            dim = 1024
        elif 'vith' in model_name:
            dim = 1280
        elif 'vitg' in model_name:
            dim = 1408
        elif 'vitb' in model_name:
            dim = 768
        else:
            dim = 1408  # Default to giant
        
        #print(f"🔍 DEBUG: V-JEPA2 model_name='{self.config.model_name}' → dim={dim}")
        return dim
    
    
    def _probe_backbone_dimension(self) -> int:
        """Probe backbone to determine output dimension"""
        self.backbone.eval()
        
        try:
            with torch.no_grad():
                # Create dummy input
                if self.model_info['is_videomae']:
                    dummy_input = torch.randn(1, self.config.frame_count, 3, 
                                            self.config.img_size, self.config.img_size)
                    dummy_input = dummy_input.permute(0, 2, 1, 3, 4)  # VideoMAE format
                    output = self.backbone(pixel_values=dummy_input)
                else:
                    dummy_input = torch.randn(1, self.config.frame_count, 3, 
                                            self.config.img_size, self.config.img_size)
                    output = self.backbone(dummy_input)
                
                # Extract dimension
                if hasattr(output, 'last_hidden_state'):
                    return output.last_hidden_state.shape[-1]
                else:
                    return output.shape[-1] if len(output.shape) > 1 else 768
        
        finally:
            self.backbone.train()
    
    def _setup_temporal_processing(self):
        """Setup temporal processing - always available for custom heads"""
        if not self.use_custom_head:
            print("   🚫 Temporal processing not needed for built-in classification")
            return
        
        #print(f"🔍 DEBUG: Before temporal setup - feature_dim = {self.feature_dim}")
        
        temporal_method = getattr(self.config, 'temporal_method', 'mean')
        
        if temporal_method == 'mean':
            self.temporal_processor = MeanPooling(self.feature_dim)
        elif temporal_method == 'max':
            self.temporal_processor = MaxPooling(self.feature_dim)
        elif temporal_method == 'lstm':
            hidden_dim = getattr(self.config, 'temporal_hidden_dim', 512)
            self.temporal_processor = LSTMProcessor(self.feature_dim, hidden_dim)
            # LSTM changes output dimension
            self.feature_dim = hidden_dim
            #print(f"🔍 DEBUG: LSTM changed feature_dim to {self.feature_dim}")
        elif temporal_method == 'attention':
            num_heads = getattr(self.config, 'temporal_num_heads', 8)
            #print(f"🔍 DEBUG: Creating AttentionProcessor with input_dim={self.feature_dim}, num_heads={num_heads}")
            #print(f"🔍 DEBUG: Divisibility check: {self.feature_dim} % {num_heads} = {self.feature_dim % num_heads}")
            self.temporal_processor = AttentionProcessor(self.feature_dim, num_heads)
        else:
            # Default fallback
            self.temporal_processor = MeanPooling(self.feature_dim)
            
        #print(f"   🕒 Temporal processing: {temporal_method}")
    
    def _setup_classification_head(self):
        """Setup classification head"""
        if not self.use_custom_head:
            return
            
        head_type = getattr(self.config, 'head_type', 'linear')
        
        if head_type == 'linear':
            self.classifier = LinearHead(
                self.feature_dim, 
                self.config.num_classes,
                getattr(self.config, 'head_dropout', 0.1)
            )
        elif head_type == 'mlp':
            self.classifier = MLPHead(
                self.feature_dim, 
                self.config.num_classes,
                getattr(self.config, 'head_hidden_dim', 512),
                getattr(self.config, 'head_num_layers', 2),
                getattr(self.config, 'head_dropout', 0.1)
            )
        elif head_type == 'attention':
            self.classifier = AttentionHead(
                self.feature_dim, 
                self.config.num_classes,
                getattr(self.config, 'head_hidden_dim', 512),
                getattr(self.config, 'head_num_heads', 8),
                getattr(self.config, 'head_num_layers', 2),
                getattr(self.config, 'head_dropout', 0.1)
            )
        else:
            # Default fallback
            self.classifier = LinearHead(self.feature_dim, self.config.num_classes)
            
        #print(f"   🎯 Classification head: {head_type}")
    
    def _report_architecture(self):
        """Report final model architecture"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n🤖 Enhanced Video Classifier Summary:")
        print(f"   📱 Model: {self.config.model_name}")
        print(f"   🎭 Type: {self._get_model_type_name()}")
        print(f"   🎯 Classes: {self.config.num_classes}")
        print(f"   📐 Feature dim: {self.feature_dim}")
        print(f"   🎬 Frame-level: {self.use_frame_level}")
        print(f"   🔮 Future prediction: {self.use_future_prediction}")
        if self.use_future_prediction:
            print(f"   🔗 Feature combination: {getattr(self.config, 'predictor_combination_method', 'concat')}")
        if self.temporal_processor:
            print(f"   🕒 Temporal processing: {getattr(self.config, 'temporal_method', 'mean')}")
        print(f"   ⚙️ Total params: {total_params:,}")
        print(f"   🏋️ Trainable params: {trainable_params:,}")
        print(f"   🔒 Frozen params: {total_params - trainable_params:,}")
    
    def _extract_video_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from video input
        Returns either sequence of vectors or single vector
        """
        if self.model_info['is_vjepa2']:
            # V-JEPA 2 processing
            if hasattr(self.backbone, 'get_vision_features'):
                features = self.backbone.get_vision_features(pixel_values_videos=x)
            else:
                output = self.backbone(pixel_values_videos=x)
                features = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
                
        elif self.model_info['is_videomae']:
            # VideoMAE processing
            x = x.permute(0, 2, 1, 3, 4)  # VideoMAE expects (B, C, T, H, W)
            
            if self.use_frame_level:
                # Frame-level feature extraction
                features = self._extract_frame_features(x)
            else:
                # Standard VideoMAE processing
                output = self.backbone(pixel_values=x)
                features = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
        else:
            # Generic model processing
            output = self.backbone(x)
            features = output.last_hidden_state if hasattr(output, 'last_hidden_state') else output
            
        return features
    
    def _extract_frame_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract frame-level features for VideoMAE"""
        features = {}
        
        def hook_fn(module, input, output):
            features['output'] = output
        
        # Hook on the last transformer block
        hook = self.backbone.model.blocks[-1].register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = self.backbone(pixel_values=x)
                
                if 'output' not in features:
                    raise RuntimeError("Failed to capture frame-level features")
                
                # Reshape to frame-level features
                all_tokens = features['output']  # (B, seq_len, hidden_dim)
                B, seq_len, D = all_tokens.shape
                
                patches_per_frame = seq_len // self.config.frame_count
                frame_features = all_tokens.view(B, self.config.frame_count, patches_per_frame, D)
                return frame_features.mean(dim=2)  # (B, frames, D) - average patches per frame
                
        finally:
            hook.remove()
            
    def _predict_future_features(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Predict future features using V-JEPA 2 predictor
        
        GRADIENT-SAFE & OPTIMIZED SOLUTION
        
        This version avoids all in-place operations to ensure gradient computation
        works correctly during backpropagation. Each tensor operation creates new
        tensors instead of modifying existing ones.
        """
        B, seq_len, D = encoder_features.shape
        
        # Calculate future tokens needed
        future_seconds = getattr(self.config, 'future_prediction_seconds', 1.0)
        original_fps = getattr(self.config, 'original_fps', 4)
        present_seconds = self.config.frame_count / original_fps
        future_tokens_needed = int(seq_len * (future_seconds / present_seconds))
        
        # Ensure we predict at least 1 token
        future_tokens_needed = max(future_tokens_needed, 1)
        
        device = encoder_features.device
        dtype = encoder_features.dtype
        expanded_seq_len = seq_len + future_tokens_needed
        
        # Pre-create reusable masks (these don't need gradients)
        context_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        target_indices = torch.arange(seq_len, expanded_seq_len, device=device).unsqueeze(0)
        
        # Collect results without pre-allocation to avoid gradient issues
        results = []
        
        # Process each sample individually
        for i in range(B):
            # Extract single sample - creates new tensor, gradient-safe
            single_sample = encoder_features[i:i+1]  # (1, seq_len, D)
            
            # Create expanded features - NEW tensor each time, gradient-safe
            zeros_padding = torch.zeros(1, future_tokens_needed, D, device=device, dtype=dtype)
            expanded_features = torch.cat([single_sample, zeros_padding], dim=1)  # (1, expanded_seq_len, D)
            
            # Predictor call
            predicted_output = self.predictor(
                expanded_features,
                context_mask=[context_indices],
                target_mask=[target_indices]
            )
            
            # Extract result - creates new tensor, gradient-safe
            future_prediction = predicted_output.last_hidden_state  # (1, future_tokens_needed, D)
            results.append(future_prediction)
        
        # Combine results - creates new tensor, gradient-safe
        combined_results = torch.cat(results, dim=0)  # (B, future_tokens_needed, D)
        
        return combined_results
    
    def _combine_present_and_future(self, present_features: torch.Tensor, 
                                   future_features: torch.Tensor) -> torch.Tensor:
        """
        Combine present and future features based on configuration
        """
        combination_method = getattr(self.config, 'predictor_combination_method', 'concat')
        
        if combination_method == 'concat':
            # Simple concatenation along sequence dimension
            combined = torch.cat([present_features, future_features], dim=1)
            return combined
            
        elif combination_method == 'weighted_sum':
            # Weighted sum requires same sequence length
            min_len = min(present_features.size(1), future_features.size(1))
            present_truncated = present_features[:, :min_len, :]
            future_truncated = future_features[:, :min_len, :]
            
            if not hasattr(self, 'temporal_fusion_weights'):
                self.temporal_fusion_weights = nn.Parameter(torch.tensor([0.7, 0.3]))
            
            weights = torch.softmax(self.temporal_fusion_weights, dim=0)
            combined = weights[0] * present_truncated + weights[1] * future_truncated
            return combined
            
        elif combination_method == 'attention_fusion':
            # Attention-based fusion
            if not hasattr(self, 'temporal_fusion_attention'):
                self.temporal_fusion_attention = nn.MultiheadAttention(
                    embed_dim=present_features.size(-1),
                    num_heads=8,
                    batch_first=True
                )
            
            # Stack present and future as two sequences
            B, seq_len_p, D = present_features.shape
            seq_len_f = future_features.size(1)
            
            # Create combined sequence with markers
            combined_seq = torch.cat([present_features, future_features], dim=1)  # (B, seq_len_p + seq_len_f, D)
            
            # Apply self-attention
            fused, _ = self.temporal_fusion_attention(combined_seq, combined_seq, combined_seq)
            return fused
            
        else:
            # Default to concatenation
            return torch.cat([present_features, future_features], dim=1)
    
    def _apply_temporal_processing(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal processing to convert sequence to single vector
        Only applied when features have sequence dimension
        """
        if len(features.shape) != 3:
            # Already a single vector per sample, no temporal processing needed
            return features
            
        if self.temporal_processor is None:
            # Fallback to mean pooling
            return features.mean(dim=1)
            
        return self.temporal_processor(features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with proper temporal processing flow
        
        Flow:
        1. Extract features (sequence or single vector)
        2. If predictor: predict future and combine with present
        3. Apply temporal processing to get single vector
        4. Apply classification head
        """
        x = x.float().to(next(self.parameters()).device)
        
        # Built-in classification model - direct output
        if not self.use_custom_head:
            if self.model_info['is_vjepa2']:
                return self.backbone(pixel_values_videos=x).logits
            else:
                if self.model_info['is_videomae']:
                    x = x.permute(0, 2, 1, 3, 4)
                return self.backbone(pixel_values=x).logits
        
        # Custom head pipeline
        # Step 1: Extract video features
        features = self._extract_video_features(x)
        
        # Step 2: Apply predictor if enabled
        if self.has_predictor and self.use_future_prediction:
            future_features = self._predict_future_features(features)
            features = self._combine_present_and_future(features, future_features)
        
        # Step 3: Apply temporal processing (sequence → single vector)
        features = self._apply_temporal_processing(features)
        
        # Step 4: Classification
        return self.classifier(features)

# =============================================================================
# CONFIGURATION AND UTILS
# =============================================================================

def get_config():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Enhanced Video Classification Training with Early Stopping")
    
    # === BASIC CONFIGURATION ===
    parser.add_argument('--model_name', type=str, required=True,
                       help='Model name (HuggingFace or local path)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Root directory containing train/val folders')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for results')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes for classification')
    
    # === TRAINING PARAMETERS ===
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--accumulation_steps', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=6)
    
    # === EARLY STOPPING ===
    parser.add_argument('--early_stopping_patience', type=int, default=2,
                       help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.001,
                       help='Minimum change to qualify as improvement')
    parser.add_argument('--early_stopping_metric', type=str, default='accuracy',
                       choices=['accuracy', 'loss'],
                       help='Metric to monitor for early stopping')
    parser.add_argument('--disable_early_stopping', action='store_true',
                       help='Disable early stopping mechanism')
    
    # === MODEL ARCHITECTURE ===
    parser.add_argument('--use_classification_model', action='store_true',
                       help='Use built-in classification model instead of custom head')
    parser.add_argument('--use_custom_head', action='store_true',
                       help='Force use of custom classification head (overrides use_classification_model)')
    parser.add_argument('--use_frame_level', action='store_true',
                       help='Extract frame-level features (VideoMAE only)')
    
    # === FUTURE PREDICTION (V-JEPA 2 ONLY) ===
    parser.add_argument('--use_future_prediction', action='store_true',
                       help='Enable future prediction using V-JEPA 2 predictor')
    parser.add_argument('--predictor_combination_method', type=str, default='concat',
                       choices=['concat', 'weighted_sum', 'attention_fusion'],
                       help='Method to combine encoder and predictor features')
    parser.add_argument('--future_prediction_seconds', type=float, default=1.0,
                       help='How many seconds into the future to predict')
    parser.add_argument('--original_fps', type=int, default=4,
                       help='Original FPS of the video data (for time calculations)')

    # === TEMPORAL PROCESSING ===
    parser.add_argument('--temporal_method', type=str, default='mean',
                       choices=['mean', 'max', 'lstm', 'attention'],
                       help='Temporal processing method')
    parser.add_argument('--temporal_hidden_dim', type=int, default=512,
                       help='Hidden dimension for LSTM temporal processing')
    parser.add_argument('--temporal_num_heads', type=int, default=8,
                       help='Number of heads for attention temporal processing')
   
    # === CLASSIFICATION HEAD ===
    parser.add_argument('--head_type', type=str, default='linear',
                       choices=['linear', 'mlp', 'attention'],
                       help='Type of classification head')
    parser.add_argument('--head_hidden_dim', type=int, default=512,
                       help='Hidden dimension for MLP/Attention heads')
    parser.add_argument('--head_num_layers', type=int, default=2,
                       help='Number of layers for MLP/Attention heads')
    parser.add_argument('--head_num_heads', type=int, default=8,
                       help='Number of heads for Attention head')
    parser.add_argument('--head_dropout', type=float, default=0.1,
                       help='Dropout rate for classification head')
    
    # === VIDEO PARAMETERS ===
    parser.add_argument('--frame_count', type=int, default=16,
                       help='Number of frames per video clip')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size for VideoMAE')
    parser.add_argument('--vjepa2_crop_size', type=int, default=256,
                       help='Input crop size for V-JEPA 2')
    
    # === TRAINING ENHANCEMENTS ===
    parser.add_argument('--use_temperature_scaling', action='store_true', default=True,
                       help='Apply temperature scaling to logits')
    parser.add_argument('--temperature', type=float, default=2.0,
                       help='Temperature for scaling')
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                       help='Drop path rate for model')
    parser.add_argument('--scheduler_step_size', type=int, default=600,
                       help='Steps between scheduler updates')
    parser.add_argument('--scheduler_T_max', type=int, default=2,
                       help='Maximum iterations for cosine scheduler')
    parser.add_argument('--scheduler_eta_min', type=float, default=1e-6,
                       help='Minimum learning rate for cosine scheduler')
    
    return parser.parse_args()

def detect_model_type(model_name: str) -> dict:
    """Detect model type and configuration"""
    model_info = {
        'is_vjepa2': 'vjepa2' in model_name.lower(),
        'is_vjepa': 'vjepa' in model_name.lower(),
        'is_videomae': 'videomae' in model_name.lower(),
        'frame_count': None,
        'crop_size': None,
        'processor_type': None
    }
    
    # Extract frame count
    frame_match = re.search(r'fpc(\d+)', model_name)
    if frame_match:
        model_info['frame_count'] = int(frame_match.group(1))
    
    # Extract crop size
    crop_match = re.search(r'-(\d+)(?:$|[^0-9])', model_name)
    if crop_match:
        model_info['crop_size'] = int(crop_match.group(1))
    
    # Determine processor type
    if model_info['is_vjepa2'] and HAS_AUTO_VIDEO_PROCESSOR:
        model_info['processor_type'] = 'auto_video'
    elif model_info['is_vjepa'] and HAS_AUTO_PROCESSOR:
        model_info['processor_type'] = 'auto_image'
    elif model_info['is_videomae'] and HAS_VIDEOMAE_PROCESSOR:
        model_info['processor_type'] = 'videomae'
    else:
        model_info['processor_type'] = 'manual'
    
    return model_info

def set_seed(seed: int):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logger(log_dir: str) -> logging.Logger:
    """Setup logging"""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def get_memory_usage() -> str:
    """Get memory usage"""
    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 ** 2)
    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        return f"RAM: {ram_mb:.1f} MB | GPU: {gpu_mb:.1f} MB"
    return f"RAM: {ram_mb:.1f} MB"

def apply_temperature_scaling(logits, config):
    """Apply temperature scaling"""
    if config.use_temperature_scaling:
        return logits / config.temperature
    return logits

def save_results_to_csv(config, best_accuracy, output_dir, stopped_early=False, final_epoch=None):
    """Save results to CSV with proper formatting"""
    csv_file = os.path.join(output_dir, "master_results_summary.csv")
    
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)  # Added proper quoting
        
        if not file_exists:
            headers = [
                'timestamp', 'model_name', 'data_root', 'epochs', 'final_epoch', 
                'early_stopped', 'batch_size', 'learning_rate', 'weight_decay', 
                'frame_count', 'img_size', 'use_frame_level', 'temporal_method', 
                'head_type', 'best_val_accuracy', 'early_stopping_patience', 'seed'
            ]
            writer.writerow(headers)
        
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Ensure all values are properly formatted and no commas in values
        row = [
            timestamp, 
            str(config.model_name).replace(',', ';'),  # Replace commas to avoid CSV issues
            str(config.data_root).replace(',', ';'),
            config.epochs,
            final_epoch if final_epoch is not None else config.epochs,
            stopped_early,
            config.batch_size, 
            config.learning_rate, 
            config.weight_decay,
            config.frame_count, 
            config.img_size, 
            config.use_frame_level,
            str(getattr(config, 'temporal_method', 'N/A')).replace(',', ';'),
            str(getattr(config, 'head_type', 'N/A')).replace(',', ';'),
            f"{best_accuracy:.2f}",  # Remove % sign to avoid parsing issues
            getattr(config, 'early_stopping_patience', 'N/A'),
            config.seed
        ]
        writer.writerow(row)

# =============================================================================
# DATASET CLASS
# =============================================================================

class VideoDataset(Dataset):
    """Enhanced video dataset"""
    def __init__(self, root_dir: str, config: argparse.Namespace, processor=None, transform=None):
        self.config = config
        self.processor = processor
        self.transform = transform
        self.frame_count = config.frame_count
        self.model_info = detect_model_type(config.model_name)

        if not os.path.isdir(root_dir):
             raise FileNotFoundError(f"Dataset directory not found: {root_dir}")
        
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for video_file in os.listdir(class_dir):
                if video_file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    self.samples.append((os.path.join(class_dir, video_file), self.class_to_idx[class_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        video_path, label = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._get_frame_indices(total_frames)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif frames:
                frames.append(frames[-1].copy())
        cap.release()
        
        while len(frames) < self.frame_count:
            frames.append(frames[-1].copy())
        
        # Process frames
        if self.processor:
            try:
                if self.model_info['is_vjepa2']:
                    frames_array = np.array(frames)
                    inputs = self.processor(frames_array, return_tensors="pt")
                    if 'pixel_values_videos' in inputs:
                        video_tensor = inputs['pixel_values_videos'].squeeze(0)
                    elif 'pixel_values' in inputs:
                        video_tensor = inputs['pixel_values'].squeeze(0)
                    else:
                        video_tensor = list(inputs.values())[0].squeeze(0)
                else:
                    inputs = self.processor(images=frames, return_tensors="pt")
                    video_tensor = inputs['pixel_values'].squeeze(0)
            except Exception as e:
                print(f"Warning: Processor failed ({e}), using manual transform")
                video_tensor = self._manual_transform(frames, idx)
        else:
            video_tensor = self._manual_transform(frames, idx)

        return video_tensor, torch.tensor(label, dtype=torch.long)

    def _manual_transform(self, frames, idx):
        """Manual transformation fallback"""
        if self.transform:
            rseed = self.config.seed + idx
            random.seed(rseed)
            np.random.seed(rseed)
            transformed_frames = [self.transform(image=f)["image"] for f in frames]
            return torch.stack(transformed_frames)
        else:
            frames_array = np.array(frames).transpose(0, 3, 1, 2)
            return torch.from_numpy(frames_array).float() / 255.0

    def _get_frame_indices(self, total_frames: int) -> np.ndarray:
        """Get frame indices for sampling"""
        if total_frames >= self.frame_count:
            step = total_frames / self.frame_count
            return np.array([int(i * step) for i in range(self.frame_count)])
        else:
            return np.array(list(range(total_frames)) + [total_frames - 1] * (self.frame_count - total_frames))

# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def validate(model, dataloader, criterion, config, device):
    """Validation loop"""
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for videos, labels in dataloader:
            labels = labels.to(device)
            with autocast():
                outputs = model(videos)
                loss = criterion(apply_temperature_scaling(outputs, config), labels)
            
            total_loss += loss.item() * videos.size(0)
            preds = outputs.argmax(dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = (correct_predictions / total_samples) * 100
    return avg_loss, accuracy

def train(config: argparse.Namespace, logger: logging.Logger, device: torch.device):
    """Main training loop with early stopping"""
    logger.info("="*60)
    logger.info(f"Enhanced Video Classification Training with Early Stopping")
    logger.info("="*60)

    # Model Detection and Configuration
    model_info = detect_model_type(config.model_name)
    logger.info(f"Model type detected: {model_info}")
    
    # Override parameters from model name
    if model_info['frame_count']:
        config.frame_count = model_info['frame_count']
        logger.info(f"Frame count set to {config.frame_count} from model name")
    
    if model_info['crop_size'] and model_info['is_vjepa2']:
        config.vjepa2_crop_size = model_info['crop_size']
        logger.info(f"V-JEPA 2 crop size set to {config.vjepa2_crop_size} from model name")

    # Early Stopping Setup
    early_stopping = None
    if not getattr(config, 'disable_early_stopping', False):
        metric_mode = 'max' if config.early_stopping_metric == 'accuracy' else 'min'
        early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
            min_delta=config.early_stopping_min_delta,
            mode=metric_mode,
            restore_best_weights=True,
            verbose=True
        )
        logger.info(f"🛡️ Early stopping enabled:")
        logger.info(f"   Patience: {config.early_stopping_patience} epochs")
        logger.info(f"   Metric: {config.early_stopping_metric}")
        logger.info(f"   Min delta: {config.early_stopping_min_delta}")
    else:
        logger.info("⚠️ Early stopping disabled")

    # Processor Setup
    processor = None
    if model_info['processor_type'] == 'auto_video':
        try:
            processor = AutoVideoProcessor.from_pretrained(config.model_name)
            logger.info("Using AutoVideoProcessor for V-JEPA 2")
        except Exception as e:
            logger.warning(f"Failed to load AutoVideoProcessor ({e})")
    elif model_info['processor_type'] == 'videomae':
        try:
            processor = VideoMAEImageProcessor.from_pretrained(config.model_name)
            logger.info("Using VideoMAEImageProcessor")
        except Exception as e:
            logger.warning(f"Failed to load VideoMAEImageProcessor ({e})")
    elif model_info['processor_type'] == 'auto_image':
        try:
            processor = AutoImageProcessor.from_pretrained(config.model_name)
            logger.info("Using AutoImageProcessor")
        except Exception as e:
            logger.warning(f"Failed to load AutoImageProcessor ({e})")

    # Define transforms
    img_size = config.vjepa2_crop_size if model_info['is_vjepa2'] else config.img_size
    
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.2, 0.2, p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    # Dataset Setup
    train_dataset = VideoDataset(os.path.join(config.data_root, 'train'), config, processor, train_transform)
    val_dataset = VideoDataset(os.path.join(config.data_root, 'val'), config, processor, val_transform)

    def worker_init_fn(worker_id):
        worker_seed = config.seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, 
                             num_workers=config.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, 
                           num_workers=config.num_workers, pin_memory=True, worker_init_fn=worker_init_fn)

    # Model Setup
    model = EnhancedVideoClassifier(config, model_info).to(device)
    
    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=config.scheduler_T_max, eta_min=config.scheduler_eta_min)
    
    # Training Loop
    best_accuracy = 0.0
    global_batch = 0
    model_save_path = os.path.join(config.output_dir, "best_model.pth")
    stopped_early = False
    final_epoch = None
    
    logger.info(f"Train/Val samples: {len(train_dataset)}/{len(val_dataset)}")
    logger.info(f"Configuration:")
    logger.info(f"  - Use frame-level: {config.use_frame_level}")
    logger.info(f"  - Temporal method: {getattr(config, 'temporal_method', 'N/A')}")
    logger.info(f"  - Head type: {getattr(config, 'head_type', 'N/A')}")
    logger.info(f"  - Classification model: {config.use_classification_model}")

    for epoch in range(config.epochs):
        model.train()
        train_loss = 0.0
        epoch_start_time = time.time()
        
        optimizer.zero_grad()
        logger.info(f"Epoch {epoch+1:02d}/{config.epochs}")
        
        for batch_idx, (videos, labels) in enumerate(train_loader):
            batch_start_time = time.time()
            labels = labels.to(device)
            
            with autocast():
                outputs = model(videos)
                loss = criterion(apply_temperature_scaling(outputs, config), labels) / config.accumulation_steps
            
            scaler.scale(loss).backward()
            train_loss += loss.item() * config.accumulation_steps * videos.size(0)
            
            if (batch_idx + 1) % config.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            global_batch += 1
            if global_batch % config.scheduler_step_size == 0:
                scheduler.step()
                logger.info(f"LR stepped @ batch {global_batch}: {scheduler.get_last_lr()[0]:.2e}")

            if batch_idx % 10 == 0:
                batch_elapsed = time.time() - batch_start_time
                curr_lr = optimizer.param_groups[0]['lr']
                avg_loss = train_loss / ((batch_idx + 1) * videos.size(0))
                logger.info(f"B{batch_idx:04d}/{len(train_loader)} | Loss {loss.item() * config.accumulation_steps:.4f} "
                           f"Avg {avg_loss:.4f} | LR {curr_lr:.2e} | {batch_elapsed:.2f}s | {get_memory_usage()}")

        # Validation
        val_loss, val_accuracy = validate(model, val_loader, criterion, config, device)
        epoch_duration = time.time() - epoch_start_time
        avg_train_loss = train_loss / len(train_dataset)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'val_acc': val_accuracy,
                'config': vars(config),
                'model_info': model_info,
            }, model_save_path)
            logger.info(f"Saved new best model (acc={val_accuracy:.2f}%)")
        
        # FIXED: Early stopping check - moved BEFORE the epoch summary log
        if early_stopping is not None:
            metric_value = val_accuracy if config.early_stopping_metric == 'accuracy' else val_loss
            should_stop = early_stopping(metric_value, model)
            
            if should_stop:
                stopped_early = True
                final_epoch = epoch + 1
                # Log epoch summary AFTER early stopping messages
                logger.info(f"Epoch {epoch+1} done in {epoch_duration:.1f}s | TrainLoss {avg_train_loss:.4f} | "
                           f"ValLoss {val_loss:.4f} | ValAcc {val_accuracy:.2f}% | Best {best_accuracy:.2f}% | "
                           f"LR {scheduler.get_last_lr()[0]:.2e}")
                logger.info(f"🛑 Training stopped early at epoch {epoch+1}")
                logger.info(f"📊 Best {config.early_stopping_metric}: {early_stopping.best_score:.4f}")
                break
        
        # Log epoch summary only if not stopping early
        logger.info(f"Epoch {epoch+1} done in {epoch_duration:.1f}s | TrainLoss {avg_train_loss:.4f} | "
                   f"ValLoss {val_loss:.4f} | ValAcc {val_accuracy:.2f}% | Best {best_accuracy:.2f}% | "
                   f"LR {scheduler.get_last_lr()[0]:.2e}")
        
        logger.info("-" * 60)

    # Final results
    if not stopped_early:
        final_epoch = config.epochs
        
    logger.info("Training complete!")
    logger.info(f"📊 Final Results:")
    logger.info(f"   Best Val Accuracy: {best_accuracy:.2f}%")
    logger.info(f"   Total Epochs: {final_epoch}/{config.epochs}")
    logger.info(f"   Early Stopped: {stopped_early}")
    if early_stopping is not None:
        logger.info(f"   Final Best {config.early_stopping_metric}: {early_stopping.best_score:.4f}")
    
    save_results_to_csv(config, best_accuracy, config.output_dir, stopped_early, final_epoch)
    return best_accuracy

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    config = get_config()
    set_seed(config.seed)
    logger = setup_logger(config.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    logger.info(f"Using device: {device}")
    logger.info(f"Available processors: VideoMAE={HAS_VIDEOMAE_PROCESSOR}, "
                f"AutoImage={HAS_AUTO_PROCESSOR}, AutoVideo={HAS_AUTO_VIDEO_PROCESSOR}")
    
    logger.info("Configuration:")
    for key, value in vars(config).items():
        logger.info(f"  - {key}: {value}")
    
    # Run training
    best_acc = train(config, logger, device)
    logger.info(f"Final result: {best_acc:.2f}%")