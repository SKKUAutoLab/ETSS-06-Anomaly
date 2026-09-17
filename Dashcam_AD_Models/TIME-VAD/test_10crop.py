import os
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

from utils import frame_process_label, process_feat
from eval_tools import evaluation_P_R80, evaluation_NP_R80
from option import get_dataset_paths


def load_ground_truth(args):
    """
    Load ground truth labels for the specified dataset.
    
    Args:
        args: Command line arguments
        
    Returns:
        numpy.ndarray: Ground truth labels
    """
    dataset_paths = get_dataset_paths(args)
    if dataset_paths is None:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    gt_path = dataset_paths['ground_truth']
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    gt = np.load(gt_path)
    print(f"Loaded ground truth from {gt_path}: {gt.shape}")
    return gt


def process_test_features(input_tensor, num_segments=50):
    """
    Process test features by dividing into segments.
    
    Args:
        input_tensor: Input tensor from dataloader
        num_segments: Number of segments to divide each video
        
    Returns:
        torch.Tensor: Processed features
    """
    input_numpy = input_tensor.squeeze(0).cpu().numpy()
    
    # Transpose to [10, T, F] format (10-crop)
    features = input_numpy.transpose(1, 0, 2)
    
    # Process each crop
    processed_features = []
    for feature in features:
        segmented_feature = process_feat(feature, num_segments)
        processed_features.append(segmented_feature)
    
    processed_features = np.array(processed_features, dtype=np.float32)
    processed_tensor = torch.from_numpy(processed_features).unsqueeze(0)
    
    return processed_tensor


def test(dataloader, model, args, viz, device):
    """
    Test the TIME-VAD model on the test dataset.
    
    Args:
        dataloader: Test data loader
        model: Trained TIME-VAD model
        args: Command line arguments
        viz: Visualizer for plotting
        device: Device for computation
        
    Returns:
        tuple: (AUC, AP_video, AP, mTTA, TTA_R80, P_R80)
    """
    print(f"Starting evaluation on {args.dataset.upper()} dataset...")
    
    # Load ground truth
    try:
        ground_truth = load_ground_truth(args)
    except (ValueError, FileNotFoundError) as e:
        print(f"Error loading ground truth: {e}")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    model.eval()
    
    with torch.no_grad():
        predictions = torch.zeros(0, device=device)
        processed_labels = []
        video_labels = []
        accident_times = []
        
        frame_idx = 0
        
        print("Processing test videos...")
        for i, input_data in enumerate(dataloader):
            # Get frame-level labels for current video
            video_length = input_data.shape[1]
            video_gt = ground_truth[frame_idx:frame_idx + video_length]
            
            # Video-level label (1 if any frame is abnormal, 0 otherwise)
            video_label = 1 if np.any(video_gt == 1) else 0
            video_labels.append(video_label)
            
            # Time of accident (first abnormal frame or video length)
            if video_label == 1:
                accident_time = np.where(video_gt == 1)[0][0]
            else:
                accident_time = len(video_gt)
            # Predictions are per-segment, not per-frame, so express the accident
            # time in segment units. Both this function and evaluation_P_R80
            # index the prediction array with it.
            accident_time = accident_time * args.num_segments / video_length
            accident_times.append(accident_time)
            
            frame_idx += video_length
            
            # Process frame-level labels for evaluation
            frame_labels = frame_process_label(video_gt, length=args.num_segments)
            
            # Prepare input: [1, T, 10, F] -> [1, 10, num_segments, F]
            input_data = process_test_features(input_data, args.num_segments)
            input_data = input_data.to(device)
            
            # Forward pass
            (score_abnormal, score_normal, feat_select_abn, 
             feat_select_normal, logits, feat_magnitudes) = model(inputs=input_data)
            
            # Process predictions
            logits = torch.squeeze(logits, 1)
            logits = torch.mean(logits, 0)  # Average across crops
            
            predictions = torch.cat((predictions, logits))
            processed_labels.extend(frame_labels)
        
        print(f"Processed {len(dataloader)} test videos")
        
        # Convert to numpy
        predictions_np = predictions.cpu().detach().numpy()
        video_labels_np = np.array(video_labels)
        accident_times_np = np.array(accident_times)
        
        # Reshape predictions for evaluation
        num_videos = len(predictions_np) // args.num_segments
        predictions_reshaped = predictions_np.reshape(num_videos, args.num_segments)
        
        print(f"Evaluation shapes: predictions {predictions_reshaped.shape}, "
              f"video labels {video_labels_np.shape}, accidents {accident_times_np.shape}")
        
        # Compute video-level metrics
        video_scores = []
        for i, (pred, acc_time) in enumerate(zip(predictions_reshaped, accident_times_np)):
            if video_labels_np[i] > 0:
                # For abnormal videos, use max score up to accident time
                video_score = np.max(pred[:int(acc_time)])
            else:
                # For normal videos, use max score of entire video
                video_score = np.max(pred)
            video_scores.append(video_score)
        
        video_scores_np = np.array(video_scores)
        
        # Compute Average Precision for video-level detection
        AP_video = average_precision_score(video_labels_np, video_scores_np)
        print(f"Video-level AP: {AP_video:.4f}")
        
        # Compute frame-level AUC
        fpr, tpr, _ = roc_curve(processed_labels, predictions_np)
        frame_auc = auc(fpr, tpr)
        print(f"Frame-level AUC: {frame_auc:.4f}")
        
        # Save ROC data
        if hasattr(args, 'output_dir'):
            np.save(os.path.join(args.output_dir, 'fpr.npy'), fpr)
            np.save(os.path.join(args.output_dir, 'tpr.npy'), tpr)
        
        # Compute Precision-Recall AUC
        precision, recall, _ = precision_recall_curve(processed_labels, predictions_np)
        pr_auc = auc(recall, precision)
        
        # Save PR data
        if hasattr(args, 'output_dir'):
            np.save(os.path.join(args.output_dir, 'precision.npy'), precision)
            np.save(os.path.join(args.output_dir, 'recall.npy'), recall)
        
        # Compute accident prediction metrics
        try:
            AP, mTTA, TTA_R80, P_R80 = evaluation_P_R80(
                predictions_reshaped, video_labels_np, accident_times_np, fps=args.fps
            )
            print(f"Accident prediction - AP: {AP:.4f}, mTTA: {mTTA:.4f}, TTA_R80: {TTA_R80:.4f}, P_R80: {P_R80:.4f}")
        except Exception as e:
            print(f"Error computing accident prediction metrics: {e}")
            AP, mTTA, TTA_R80, P_R80 = 0.0, 0.0, 0.0, 0.0
        
        # Visualization
        if viz is not None:
            viz.plot_lines('test_auc', frame_auc)
            viz.plot_lines('test_pr_auc', pr_auc)
            viz.plot_lines('test_ap_video', AP_video)
            viz.plot_lines('test_mtta', mTTA)
        
        print(f"Evaluation completed successfully!")
        return frame_auc, AP_video, AP, mTTA, TTA_R80, P_R80


def evaluate_model(model_path, args, device=None):
    """
    Evaluate a saved model on the test dataset.
    
    Args:
        model_path: Path to saved model checkpoint
        args: Command line arguments
        device: Device for computation (optional)
        
    Returns:
        dict: Evaluation results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    from model import Model
    from dataset import Dataset
    from torch.utils.data import DataLoader
    from utils import Visualizer
    
    model = Model(
        n_features=args.feature_size,
        batch_size=1,  # Use batch size 1 for testing
        temporal_model=args.temporal_model
    )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model weights
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print(f"Loaded model from {model_path}")
    
    # Create test dataloader
    test_dataset = Dataset(args, test_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )
    
    # Create visualizer
    viz = Visualizer(env=f'TIME-VAD-{args.dataset}-eval', use_incoming_socket=False)
    
    # Evaluate
    auc, AP_video, AP, mTTA, TTA_R80, P_R80 = test(test_loader, model, args, viz, device)
    
    results = {
        'AUC': auc,
        'AP_video': AP_video,
        'AP': AP,
        'mTTA': mTTA,
        'TTA_R80': TTA_R80,
        'P_R80': P_R80
    }
    
    return results


def print_evaluation_summary(results, dataset_name):
    """
    Print a formatted summary of evaluation results.
    
    Args:
        results: Dictionary of evaluation results
        dataset_name: Name of the dataset
    """
    print(f"\n{'='*50}")
    print(f"TIME-VAD Evaluation Results on {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"Frame-level AUC:      {results['AUC']:.4f}")
    print(f"Video-level AP:       {results['AP_video']:.4f}")
    print(f"Accident AP:          {results['AP']:.4f}")
    print(f"Mean TTA:             {results['mTTA']:.4f}")
    print(f"TTA @ Recall 80%:     {results['TTA_R80']:.4f}")
    print(f"Precision @ Recall 80%: {results['P_R80']:.4f}")
    print(f"{'='*50}")


if __name__ == '__main__':
    # Test evaluation functionality
    import argparse
    from option import parser
    
    # Parse arguments
    test_args = parser.parse_args(['--dataset', 'ccd', '--batch-size', '1'])
    
    print("Testing TIME-VAD evaluation components...")
    
    # Test ground truth loading
    try:
        gt = load_ground_truth(test_args)
        print(f"Ground truth shape: {gt.shape}")
    except Exception as e:
        print(f"Could not test ground truth loading: {e}")
    
    # Test feature processing
    dummy_input = torch.randn(1, 100, 10, 768)  # [batch, time, crops, features]
    processed = process_test_features(dummy_input, num_segments=50)
    print(f"Feature processing test: {dummy_input.shape} -> {processed.shape}")
    
    print("Evaluation components working correctly!")