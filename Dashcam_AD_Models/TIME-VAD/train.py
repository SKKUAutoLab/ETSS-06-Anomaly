import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import L1Loss, MSELoss


def sparsity_loss(scores, batch_size, lambda_sparse):
    """
    Compute sparsity loss to encourage sparse predictions.
    
    Args:
        scores: Prediction scores
        batch_size: Batch size
        lambda_sparse: Sparsity loss weight
        
    Returns:
        Sparsity loss value
    """
    loss = torch.mean(torch.norm(scores, dim=0))
    return lambda_sparse * loss


def smoothness_loss(scores, lambda_smooth):
    """
    Compute temporal smoothness loss.
    
    Args:
        scores: Prediction scores
        lambda_smooth: Smoothness loss weight
        
    Returns:
        Smoothness loss value
    """
    scores_shifted = torch.zeros_like(scores)
    scores_shifted[:-1] = scores[1:]
    scores_shifted[-1] = scores[-1]
    
    loss = torch.sum((scores_shifted - scores) ** 2)
    return lambda_smooth * loss


def l1_penalty(var):
    """Compute L1 penalty for regularization."""
    return torch.mean(torch.norm(var, dim=0))


def contrastive_loss(dot_product, labels, temperature):
    """
    Compute contrastive loss for feature learning.
    
    Args:
        dot_product: Dot product between features and anchors
        labels: Binary labels
        temperature: Temperature parameter for softmax
        
    Returns:
        Contrastive loss value
    """
    labels = labels.to(dot_product.device)
    dot_product = dot_product.mean(1)  # [batch_size, 1]
    exp_dot_product = torch.exp(dot_product / temperature)
    
    mask_denominator = 1 - labels
    denominator = exp_dot_product * mask_denominator
    denominator_sum = denominator.sum(0)
    
    numerator = exp_dot_product / denominator_sum
    log_prob = -torch.log(numerator)
    masked_log_prob = log_prob * labels
    
    loss = torch.sum(masked_log_prob) / torch.sum(labels)
    return loss


class CustomContrastiveLoss(nn.Module):
    """Custom contrastive loss using positive and negative anchors."""
    
    def __init__(self, positive_anchor, negative_anchor, temperature=1.0):
        super(CustomContrastiveLoss, self).__init__()
        self.temperature = temperature
        
        # Normalize anchor vectors
        self.positive_anchor = nn.functional.normalize(positive_anchor, dim=1, p=2)
        self.negative_anchor = nn.functional.normalize(negative_anchor, dim=1, p=2)

    def forward(self, features, labels):
        """
        Compute contrastive loss.
        
        Args:
            features: Input features
            labels: Binary labels
            
        Returns:
            Total contrastive loss
        """
        # Normalize features
        features = nn.functional.normalize(features, dim=1, p=2)
        
        # Compute similarities with anchors
        positive_similarity = torch.matmul(features, self.positive_anchor.t())
        positive_loss = contrastive_loss(positive_similarity, labels, self.temperature)
        
        negative_similarity = torch.matmul(features, self.negative_anchor.t())
        negative_labels = 1 - labels
        negative_loss = contrastive_loss(negative_similarity, negative_labels, self.temperature)
        
        total_loss = positive_loss + negative_loss
        return total_loss


class SigmoidMAELoss(torch.nn.Module):
    """Sigmoid Mean Absolute Error Loss."""
    
    def __init__(self):
        super(SigmoidMAELoss, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.mse_loss = MSELoss()

    def forward(self, pred, target):
        return self.mse_loss(pred, target)


class SigmoidCrossEntropyLoss(torch.nn.Module):
    """Sigmoid Cross Entropy Loss implementation."""
    
    def __init__(self):
        super(SigmoidCrossEntropyLoss, self).__init__()

    def forward(self, x, target):
        tmp = 1 + torch.exp(-torch.abs(x))
        return torch.abs(torch.mean(-x * target + torch.clamp(x, min=0) + torch.log(tmp)))


class TimeVADLoss(torch.nn.Module):
    """
    Main loss function for TIME-VAD combining multiple loss components.
    """
    
    def __init__(self, alpha, margin, positive_anchor, negative_anchor, temperature):
        super(TimeVADLoss, self).__init__()
        self.alpha = alpha
        self.margin = margin
        self.sigmoid = torch.nn.Sigmoid()
        self.mae_criterion = SigmoidMAELoss()
        self.bce_criterion = torch.nn.BCELoss()
        self.contrastive_loss = CustomContrastiveLoss(positive_anchor, negative_anchor, temperature)

    def forward(self, score_normal, score_abnormal, normal_label, abnormal_label, 
                normal_features, abnormal_features):
        """
        Compute the total TIME-VAD loss.
        
        Args:
            score_normal: Normal video scores
            score_abnormal: Abnormal video scores
            normal_label: Normal video labels
            abnormal_label: Abnormal video labels
            normal_features: Normal video features
            abnormal_features: Abnormal video features
            
        Returns:
            Total loss value
        """
        # Combine labels and scores
        labels = torch.cat((normal_label, abnormal_label), 0)
        scores = torch.cat((score_normal, score_abnormal), 0)
        scores = scores.squeeze()
        labels = labels.cuda()
        
        # Binary cross-entropy loss
        loss_bce = self.bce_criterion(scores, labels)
        
        # Contrastive loss
        combined_features = torch.cat((normal_features, abnormal_features), 0).view(-1, 768)
        num_features = int(combined_features.size(0) / 2)
        contrastive_labels = torch.cat([torch.zeros(num_features), torch.ones(num_features)], dim=0)
        loss_contrastive = self.contrastive_loss(combined_features, contrastive_labels)
        
        # Feature magnitude losses
        loss_abnormal = torch.norm(torch.mean(abnormal_features, dim=1), p=2, dim=1)
        loss_normal = torch.abs(self.margin - torch.norm(torch.mean(normal_features, dim=1), p=2, dim=1))
        
        # magnitude loss
        loss_magnitude = torch.mean((loss_abnormal + loss_normal) ** 2)
        
        # Total loss
        total_loss = loss_bce + self.alpha * loss_magnitude + loss_contrastive
        
        return total_loss


def load_anchor_features(positive_path, negative_path, device):
    """
    Load positive and negative anchor features for contrastive learning.
    
    Args:
        positive_path: Path to positive anchor features
        negative_path: Path to negative anchor features
        device: Device to load features on
        
    Returns:
        tuple: (positive_anchor, negative_anchor)
    """    
    positive_anchor = torch.from_numpy(np.load(positive_path)).to(device)
    negative_anchor = torch.from_numpy(np.load(negative_path)).to(device)
    print(f"Loaded text anchor features: positive {positive_anchor.shape}, negative {negative_anchor.shape}")
    return positive_anchor, negative_anchor


def save_features_for_analysis(normal_features, abnormal_features, epoch, save_dir='./analysis'):
    """
    Save features for analysis (optional debugging function).
    
    Args:
        normal_features: Normal video features
        abnormal_features: Abnormal video features
        epoch: Current epoch
        save_dir: Directory to save features
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Process features for saving
    normal_data = normal_features.view(32, 10, 4, 768).mean(1).view(128, 768)
    abnormal_data = abnormal_features.view(32, 10, 4, 768).mean(1).view(128, 768)
    
    # Save to files
    normal_path = os.path.join(save_dir, f'normal_features_epoch_{epoch}.npy')
    abnormal_path = os.path.join(save_dir, f'abnormal_features_epoch_{epoch}.npy')
    
    np.save(normal_path, normal_data.cpu().detach().numpy())
    np.save(abnormal_path, abnormal_data.cpu().detach().numpy())


def train(normal_loader_iter, abnormal_loader_iter, model, batch_size, optimizer, viz, device, args=None):
    """
    Perform one training step.
    
    Args:
        normal_loader_iter: Iterator for normal video loader
        abnormal_loader_iter: Iterator for abnormal video loader
        model: TIME-VAD model
        batch_size: Batch size
        optimizer: Optimizer
        viz: Visualizer for plotting
        device: Training device
        args: Command line arguments (optional, for configuration)
    """
    model.train()
    
    with torch.set_grad_enabled(True):
        # Get batch data
        normal_input, normal_label = next(normal_loader_iter)
        abnormal_input, abnormal_label = next(abnormal_loader_iter)
        
        # Combine inputs
        combined_input = torch.cat((normal_input, abnormal_input), 0).to(device)
        
        # Forward pass
        (score_abnormal, score_normal, selected_abnormal_features, 
         selected_normal_features, all_scores, feature_magnitudes) = model(combined_input)
        
        # Process scores for regularization
        all_scores = all_scores.view(batch_size * 50 * 2, -1).squeeze()
        abnormal_scores = all_scores[batch_size * 50:]
        
        # Process labels
        normal_label = normal_label[0:batch_size]
        abnormal_label = abnormal_label[0:batch_size]
        
        # Load anchor features (if args provided)
        if args is not None:
            positive_anchor, negative_anchor = load_anchor_features(
                args.positive_anchor_path, args.negative_anchor_path, device
            )
            temperature = args.temperature
            alpha = args.alpha
            margin = args.margin
            sparsity_lambda = args.sparsity_lambda
            smooth_lambda = args.smooth_lambda
        else:
            # Use default values if args not provided (for backward compatibility)
            positive_anchor = torch.randn(100, 768).to(device)
            negative_anchor = torch.randn(100, 768).to(device)
            temperature = 0.1
            alpha = 0.0001
            margin = 110.0
            sparsity_lambda = 8e-3
            smooth_lambda = 8e-4
        
        # Initialize loss function
        loss_criterion = TimeVADLoss(alpha, margin, positive_anchor, negative_anchor, temperature)
        
        # Compute main loss
        main_loss = loss_criterion(
            score_normal, score_abnormal, normal_label, abnormal_label,
            selected_normal_features, selected_abnormal_features
        )
        
        # Compute regularization losses
        sparsity_loss_val = sparsity_loss(abnormal_scores, batch_size, sparsity_lambda)
        smoothness_loss_val = smoothness_loss(abnormal_scores, smooth_lambda)
        
        # Total loss
        total_loss = main_loss + sparsity_loss_val + smoothness_loss_val
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Visualization
        if viz is not None:
            viz.plot_lines('total_loss', total_loss.item())
            viz.plot_lines('main_loss', main_loss.item())
            viz.plot_lines('sparsity_loss', sparsity_loss_val.item())
            viz.plot_lines('smoothness_loss', smoothness_loss_val.item())


def setup_loss_function(args, device):
    """ 
    Args:
        args: Command line arguments
        device: Training device
        
    Returns:
        TimeVADLoss: Configured loss function
    """
    # Load anchor features once
    positive_anchor, negative_anchor = load_anchor_features(
        args.positive_anchor_path, args.negative_anchor_path, device
    )
    
    # Create loss function
    loss_criterion = TimeVADLoss(
        alpha=args.alpha, 
        margin=args.margin, 
        positive_anchor=positive_anchor, 
        negative_anchor=negative_anchor, 
        temperature=args.temperature
    )
    
    return loss_criterion


def train_with_loss_function(normal_loader_iter, abnormal_loader_iter, model, batch_size, 
                           optimizer, viz, device, loss_criterion, args):
    """
    Training function that uses pre-configured loss function.
    This avoids loading anchor features every epoch.
    
    Args:
        normal_loader_iter: Iterator for normal video loader
        abnormal_loader_iter: Iterator for abnormal video loader
        model: TIME-VAD model
        batch_size: Batch size
        optimizer: Optimizer
        viz: Visualizer for plotting
        device: Training device
        loss_criterion: Pre-configured loss function
        args: Command line arguments
    """
    model.train()
    
    with torch.set_grad_enabled(True):
        # Get batch data
        normal_input, normal_label = next(normal_loader_iter)
        abnormal_input, abnormal_label = next(abnormal_loader_iter)
        
        # Combine inputs
        combined_input = torch.cat((normal_input, abnormal_input), 0).to(device)
        
        # Forward pass
        (score_abnormal, score_normal, selected_abnormal_features, 
         selected_normal_features, all_scores, feature_magnitudes) = model(combined_input)
        
        # Process scores for regularization
        all_scores = all_scores.view(batch_size * 50 * 2, -1).squeeze()
        abnormal_scores = all_scores[batch_size * 50:]
        
        # Process labels
        normal_label = normal_label[0:batch_size]
        abnormal_label = abnormal_label[0:batch_size]
        
        # Compute main loss using pre-configured loss function
        main_loss = loss_criterion(
            score_normal, score_abnormal, normal_label, abnormal_label,
            selected_normal_features, selected_abnormal_features
        )
        
        # Compute regularization losses
        sparsity_loss_val = sparsity_loss(abnormal_scores, batch_size, args.sparsity_lambda)
        smoothness_loss_val = smoothness_loss(abnormal_scores, args.smooth_lambda)
        
        # Total loss
        total_loss = main_loss + sparsity_loss_val + smoothness_loss_val
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Visualization
        if viz is not None:
            viz.plot_lines('total_loss', total_loss.item())
            viz.plot_lines('main_loss', main_loss.item())
            viz.plot_lines('sparsity_loss', sparsity_loss_val.item())
            viz.plot_lines('smoothness_loss', smoothness_loss_val.item())


def train_with_args(normal_loader_iter, abnormal_loader_iter, model, batch_size, optimizer, viz, device, args):
    """
    Training function that uses args for configuration.
    This is the recommended interface for the new TIME-VAD code.
    
    DEPRECATED: Use train_with_loss_function instead to avoid loading anchors every epoch.
    """
    return train(normal_loader_iter, abnormal_loader_iter, model, batch_size, optimizer, viz, device, args)


if __name__ == '__main__':
    # Test training components
    print("Testing TIME-VAD training components...")
    
    # Test loss functions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 4
    feature_dim = 768
    
    normal_features = torch.randn(batch_size, 10, 4, feature_dim).to(device)
    abnormal_features = torch.randn(batch_size, 10, 4, feature_dim).to(device)
    normal_scores = torch.rand(batch_size, 1).to(device)
    abnormal_scores = torch.rand(batch_size, 1).to(device)
    normal_labels = torch.zeros(batch_size).to(device)
    abnormal_labels = torch.ones(batch_size).to(device)
    
    # Test anchor features
    positive_anchor = torch.randn(100, feature_dim).to(device)
    negative_anchor = torch.randn(100, feature_dim).to(device)
    
    # Test loss function
    loss_fn = TimeVADLoss(0.0001, 110.0, positive_anchor, negative_anchor, 0.1)
    
    loss = loss_fn(
        normal_scores, abnormal_scores, normal_labels, abnormal_labels,
        normal_features.view(-1, feature_dim), abnormal_features.view(-1, feature_dim)
    )
    
    print(f"Test loss computation successful: {loss.item():.4f}")
    
    # Test regularization losses
    dummy_scores = torch.rand(100).to(device)
    sparse_loss = sparsity_loss(dummy_scores, batch_size, 8e-3)
    smooth_loss = smoothness_loss(dummy_scores, 8e-4)
    
    print(f"Sparsity loss: {sparse_loss.item():.6f}")
    print(f"Smoothness loss: {smooth_loss.item():.6f}")
    
    print("All training components working correctly!")