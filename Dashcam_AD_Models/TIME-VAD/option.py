import argparse
import os

parser = argparse.ArgumentParser(description='TIME-VAD: Temporal Video Anomaly Detection')

# Dataset and Data Loading
parser.add_argument('--dataset', default='ccd', choices=['ccd', 'dad', 'dota'],
                    help='Dataset to train on (default: ccd)')
parser.add_argument('--data-root', type=str, default='./data',
                    help='Root directory for datasets (default: ./data)')
parser.add_argument('--workers', type=int, default=4,
                    help='Number of workers in dataloader (default: 4)')

# Model Architecture
parser.add_argument('--feature-size', type=int, default=768,
                    help='Size of input features (default: 768)')
parser.add_argument('--temporal-model', default='dtca', 
                    choices=['dtca', 'tcn', 'transformer', 'lstm', 'convlstm'],
                    help='Temporal modeling approach (default: dtca)')
parser.add_argument('--num-classes', type=int, default=1,
                    help='Number of classes (default: 1)')

# Training Parameters
parser.add_argument('--batch-size', type=int, default=16,
                    help='Batch size for training (default: 32)')
parser.add_argument('--max-epoch', type=int, default=15000,
                    help='Maximum number of training epochs (default: 15000)')
parser.add_argument('--lr', type=str, default='[0.0005]*15000',
                    help='Learning rate schedule as string (default: [0.0005]*15000)')
parser.add_argument('--weight-decay', type=float, default=0.005,
                    help='Weight decay for optimizer (default: 0.005)')

# Evaluation and Checkpointing
parser.add_argument('--eval-freq', type=int, default=5,
                    help='Evaluation frequency in epochs (default: 1)')
parser.add_argument('--eval-start', type=int, default=10,
                    help='Epoch to start evaluation (default: 25)')
parser.add_argument('--eval-initial', action='store_true', default=True,
                    help='Perform initial evaluation before training')
parser.add_argument('--save-freq', type=int, default=1000,
                    help='Model checkpoint saving frequency in epochs (default: 1000)')

# Model and Output Paths
parser.add_argument('--model-name', default='time_vad',
                    help='Name prefix for saved models (default: time_vad)')
parser.add_argument('--checkpoint-dir', default='./checkpoints',
                    help='Directory to save model checkpoints (default: ./checkpoints)')
parser.add_argument('--output-dir', default='./results',
                    help='Directory to save results (default: ./results)')
parser.add_argument('--pretrained-ckpt', default="./checkpoints/b8.pkl",
                    help='Path to pretrained checkpoint for initialization')

# Hardware Configuration
parser.add_argument('--gpu-id', type=int, default=1,
                    help='GPU device ID to use (-1 for CPU) (default: 0)')

# Feature Extraction (for compatibility)
parser.add_argument('--feat-extractor', default='i3d', choices=['i3d', 'c3d'],
                    help='Feature extractor used (default: i3d)')
parser.add_argument('--modality', default='RGB', choices=['RGB', 'AUDIO', 'MIX'],
                    help='Input modality (default: RGB)')

# Visualization
parser.add_argument('--plot-freq', type=int, default=10,
                    help='Frequency of plotting during training (default: 10)')

# Loss Function Parameters
parser.add_argument('--alpha', type=float, default=0.0001,
                    help='Weight for TIME-VAD loss component (default: 0.0001)')
parser.add_argument('--margin', type=float, default=110.0,
                    help='Margin for feature magnitude loss (default: 110.0)')
parser.add_argument('--temperature', type=float, default=0.1,
                    help='Temperature for contrastive loss (default: 0.1)')
parser.add_argument('--sparsity-lambda', type=float, default=8e-3,
                    help='Weight for sparsity loss (default: 8e-3)')
parser.add_argument('--smooth-lambda', type=float, default=8e-4,
                    help='Weight for smoothness loss (default: 8e-4)')

# CLIP Features for Contrastive Learning
parser.add_argument('--positive-anchor-path', type=str, 
                    default='./data/clip_features2.npy',
                    help='Path to positive anchor features (default: ./data/clip_features2.npy)')
parser.add_argument('--negative-anchor-path', type=str,
                    default='./data/clip_features1.npy',
                    help='Path to negative anchor features (default: ./data/clip_features1.npy)')

# Video Processing Parameters
parser.add_argument('--num-segments', type=int, default=50,
                    help='Number of segments to divide each video (default: 50)')
parser.add_argument('--fps', type=float, default=10.0,
                    help='Frames per second for evaluation metrics (default: 10.0)')

def get_dataset_paths(args):
    """
    Get dataset-specific file paths based on the dataset and data root directory.
    """
    dataset_paths = {
        'ccd': {
            'train_list': os.path.join(args.data_root, 'CCD', 'train_ccd.list'),
            'test_list': os.path.join(args.data_root, 'CCD', 'test_ccd.list'),
            'ground_truth': os.path.join(args.data_root, 'CCD', 'ground_truth.npy'),
            'normal_train_start': 1200,  # Normal videos start after this index
        },
        'dad': {
            'train_list': os.path.join(args.data_root, 'DAD', 'train.list'),
            'test_list': os.path.join(args.data_root, 'DAD', 'test.list'),
            'ground_truth': os.path.join(args.data_root, 'DAD', 'ground_truth.npy'),
            'normal_train_end': 829,  # Normal (negative) videos end at this index
        },
        'dota': {
            'train_list': os.path.join(args.data_root, 'DOTA', 'train_dota.list'),
            'test_list': os.path.join(args.data_root, 'DOTA', 'DOTA_val.list'),
            'ground_truth': os.path.join(args.data_root, 'DOTA', 'test_labels.npy'),
            'normal_train_end': 4130,  # Normal videos end at this index
        }
    }
    
    return dataset_paths.get(args.dataset.lower(), None)

def validate_args(args):
    """
    Validate command line arguments and set up directories.
    """
    # Create output directories if they don't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.data_root, exist_ok=True)
    
    # Validate dataset
    if args.dataset.lower() not in ['ccd', 'dad', 'dota']:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    # Validate temporal model
    valid_temporal_models = ['dtca', 'tcn', 'transformer', 'lstm', 'convlstm']
    if args.temporal_model not in valid_temporal_models:
        raise ValueError(f"Unsupported temporal model: {args.temporal_model}")
    
    # Validate paths
    dataset_paths = get_dataset_paths(args)
    if dataset_paths is None:
        raise ValueError(f"No configuration found for dataset: {args.dataset}")
    
    # Check if CLIP features exist
    if not os.path.exists(args.positive_anchor_path):
        print(f"Warning: Positive anchor features not found at {args.positive_anchor_path}")
    if not os.path.exists(args.negative_anchor_path):
        print(f"Warning: Negative anchor features not found at {args.negative_anchor_path}")
    
    return args

if __name__ == '__main__':
    # Test argument parsing
    args = parser.parse_args()
    args = validate_args(args)
    
    print("TIME-VAD Configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Temporal Model: {args.temporal_model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Max Epochs: {args.max_epoch}")
    print(f"GPU ID: {args.gpu_id}")
    print(f"Data Root: {args.data_root}")
    print(f"Checkpoint Dir: {args.checkpoint_dir}")
    print(f"Output Dir: {args.output_dir}")