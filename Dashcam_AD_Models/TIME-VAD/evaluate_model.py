#!/usr/bin/env python3
"""
Standalone evaluation script for TIME-VAD models.
Usage: python evaluate_model.py --dataset ccd --model-path checkpoints/time_vad_best.pkl
"""

import os
import json
import torch
import argparse
from torch.utils.data import DataLoader

from model import Model
from dataset import Dataset
from test_10crop import test, load_ground_truth
from utils import Visualizer
from option import get_dataset_paths


def parse_eval_args():
    """Parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description='TIME-VAD Model Evaluation')
    
    # Required arguments
    parser.add_argument('--dataset', default='dota', choices=['ccd', 'dad', 'dota'],
                        help='Dataset to evaluate on')
    parser.add_argument('--model-path', required=True, 
                        help='Path to model checkpoint')
                        
    
    # Optional arguments
    parser.add_argument('--data-root', default='./data',
                        help='Root directory for datasets')
    parser.add_argument('--output-dir', default='./results',
                        help='Directory to save results')
    parser.add_argument('--feature-size', type=int, default=768,
                        help='Feature dimension')
    parser.add_argument('--temporal-model', default='dtca',
                        choices=['dtca', 'tcn', 'transformer', 'lstm', 'convlstm'],
                        help='Temporal model type')
    parser.add_argument('--num-segments', type=int, default=50,
                        help='Number of temporal segments')
    parser.add_argument('--fps', type=float, default=10.0,
                        help='Frames per second for metrics')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of dataloader workers')
    parser.add_argument('--gpu-id', type=int, default=0,
                        help='GPU device ID (-1 for CPU)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size for testing')
    parser.add_argument('--modality', default='RGB', choices=['RGB', 'AUDIO', 'MIX'],
                    help='Input modality (default: RGB)')
    
    return parser.parse_args()


def setup_device(gpu_id):
    """Setup computation device."""
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device


def validate_files(args):
    """Validate required files exist."""
    # Check model file
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    
    # Check dataset files
    dataset_paths = get_dataset_paths(args)
    if not dataset_paths:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    required_files = ['test_list', 'ground_truth']
    for file_key in required_files:
        file_path = dataset_paths[file_key]
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    print("✅ All required files found")


def load_model(args, device):
    """Load model from checkpoint."""
    print(f"Loading model from: {args.model_path}")
    
    # Create model
    model = Model(
        n_features=args.feature_size,
        batch_size=args.batch_size,
        temporal_model=args.temporal_model
    )
    
    # Load weights
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    
    print(f"✅ Model loaded successfully ({args.temporal_model})")
    return model


def create_test_loader(args):
    """Create test data loader."""
    test_dataset = Dataset(args, test_mode=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )
    
    print(f"✅ Test loader created: {len(test_dataset)} samples")
    return test_loader


def run_evaluation(args):
    """Run complete evaluation pipeline."""
    print("=" * 70)
    print(f"TIME-VAD Evaluation: {args.dataset.upper()} Dataset")
    print("=" * 70)
    
    # Setup
    device = setup_device(args.gpu_id)
    validate_files(args)
    
    # Load model and data
    model = load_model(args, device)
    test_loader = create_test_loader(args)
    
    # Create visualizer
    viz = Visualizer(env=f'TIME-VAD-{args.dataset}-eval', use_incoming_socket=False)
    
    # Run evaluation
    print("\n🔄 Running evaluation...")
    auc, AP_video, AP, mTTA, TTA_R80, P_R80 = test(
        test_loader, model, args, viz, device
    )
    
    # Collect results
    results = {
        'dataset': args.dataset,
        'model_path': args.model_path,
        'temporal_model': args.temporal_model,
        'num_segments': args.num_segments,
        'AUC': float(auc),
        'AP_video': float(AP_video),
        'AP': float(AP),
        'mTTA': float(mTTA),
        'TTA_R80': float(TTA_R80),
        'P_R80': float(P_R80)
    }
    
    return results


def print_results(results):
    """Print formatted results."""
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    print(f"Dataset:              {results['dataset'].upper()}")
    print(f"Model:                {results['model_path']}")
    print(f"Temporal Model:       {results['temporal_model']}")
    print(f"Segments:             {results['num_segments']}")
    print("-" * 70)
    print(f"Frame-level AUC:      {results['AUC']:.4f}")
    print(f"Video-level AP:       {results['AP_video']:.4f}")
    print(f"Accident AP:          {results['AP']:.4f}")
    print(f"Mean TTA:             {results['mTTA']:.4f} seconds")
    print(f"TTA @ Recall 80%:     {results['TTA_R80']:.4f} seconds")
    print(f"Precision @ Recall 80%: {results['P_R80']:.4f}")
    print("=" * 70)


def save_results(results, output_dir):
    """Save results to file."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_file = os.path.join(output_dir, f"{results['dataset']}_evaluation_results.json")
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save readable text
    txt_file = os.path.join(output_dir, f"{results['dataset']}_evaluation_results.txt")
    with open(txt_file, 'w') as f:
        f.write(f"TIME-VAD Evaluation Results\n")
        f.write(f"Dataset: {results['dataset'].upper()}\n")
        f.write(f"Model: {results['model_path']}\n")
        f.write(f"Temporal Model: {results['temporal_model']}\n")
        f.write(f"Segments: {results['num_segments']}\n")
        f.write(f"Frame-level AUC: {results['AUC']:.6f}\n")
        f.write(f"Video-level AP: {results['AP_video']:.6f}\n")
        f.write(f"Accident AP: {results['AP']:.6f}\n")
        f.write(f"Mean TTA: {results['mTTA']:.6f}\n")
        f.write(f"TTA @ Recall 80%: {results['TTA_R80']:.6f}\n")
        f.write(f"Precision @ Recall 80%: {results['P_R80']:.6f}\n")
    
    print(f"📁 Results saved to: {json_file}")
    print(f"📁 Results saved to: {txt_file}")


def main():
    """Main evaluation function."""
    try:
        args = parse_eval_args()
        
        # Run evaluation
        results = run_evaluation(args)
        
        # Display results
        print_results(results)
        
        # Save results
        save_results(results, args.output_dir)
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())