#!/usr/bin/env python3

import os
import json
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score

from model import Model
from dataset import Dataset
from utils import frame_process_label, process_feat, Visualizer
from option import get_dataset_paths

def load_compatible_checkpoint(model, checkpoint_path, temporal_model='convlstm'):
    """Load checkpoint - try exact first, then compatible."""
    try:
        # Try exact loading first
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)
        print(f"✅ Loaded checkpoint with exact weights (strict=True)")
        return True
    except Exception as e:
        print(f"⚠️ Exact loading failed: {e}")
        try:
            # Fallback to compatible loading
            model.load_state_dict(state_dict, strict=False)
            print(f"✅ Loaded checkpoint with compatible weights (strict=False)")
            return True
        except Exception as e2:
            print(f"❌ Error loading checkpoint: {e2}")
            return False

def parse_dota_args():
    """Parse command line arguments for DOTA evaluation."""
    parser = argparse.ArgumentParser(description='TIME-VAD DOTA Dataset Evaluation')
    
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--data-root', default='./data', help='Root directory for datasets')
    parser.add_argument('--output-dir', default='./dota_results', help='Directory to save results')
    parser.add_argument('--feature-size', type=int, default=768, help='Feature dimension')
    parser.add_argument('--temporal-model', default='dtca',  # Changed back to dtca to match original
                        choices=['dtca', 'tcn', 'transformer', 'lstm', 'convlstm'],
                        help='Temporal model type')
    parser.add_argument('--num-segments', type=int, default=50, help='Number of temporal segments')
    parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID (-1 for CPU)')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for testing')
    
    # Add ground truth path option
    parser.add_argument('--gt-path', type=str, default="/home/sumit/P2RTFM/code/data/DOTA/test_labels.npy", 
                        help='Path to ground truth file')
    
    args = parser.parse_args()
    args.dataset = 'dota'
    args.modality = 'RGB'
    
    print(f"🔧 Configuration:")
    print(f"   Model: {args.model_path}")
    print(f"   Temporal Model: {args.temporal_model}")
    print(f"   Ground Truth: {args.gt_path}")
    print(f"   GPU ID: {args.gpu_id}")
    
    return args

def setup_device(gpu_id):
    """Setup computation device."""
    if torch.cuda.is_available() and gpu_id >= 0:
        device = torch.device(f'cuda:{gpu_id}')
        print(f"Using GPU: {gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    return device

def test(dataloader, model, args, viz, device,  output_dir=None):
    """
    EXACT REPLICA of your original test function with minimal modifications.
    """
    with torch.no_grad():
        model.eval()
        pred = torch.zeros(0)  # Start with CPU tensor for concatenation
        new_gt = []
        gt = np.load(args.gt_path)  # Use the provided path
            
        k = 0  # Exact same variable name as original
        vid_lab = []
        acc = []

        print(f"✅ Loaded ground truth: {gt.shape}")
       
        
        for i, input in enumerate(dataloader):  
            l = k + np.shape(input)[1]
            label = gt[k:l]
            
            k += np.shape(input)[1]
            num_frame = np.shape(input)[1]

            if len(label) < 50:
                label = frame_process_label(label, length=50)

            vid_lab.append(1 if 1 in label else 0)
            if 1 in label:
                if np.where(label == 1)[0][0] == 0:
                    acc.append(1)
                else:
                    acc.append(np.where(label == 1)[0][0])
            else:
                acc.append(len(label))
            
            if num_frame < 50:

                input = input.squeeze(0).cpu().numpy()
                
                features = input.transpose(1, 0, 2)  # [10, B, T, F] - exact comment from original
                
                divided_features = []
                for feature in features:
                    feature = process_feat(feature, 50)  # divide a video into 50 segments 
                    divided_features.append(feature)
                divided_features = np.array(divided_features, dtype=np.float32)
                divided_features = torch.from_numpy(divided_features).unsqueeze(0)

                input = divided_features.to(device)
                score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits, feat_magnitudes = model(inputs=input)
                logits = torch.squeeze(logits, 1)
                logits = torch.mean(logits, 0)
                sig = logits.cpu()  # Move to CPU for concatenation
                pred = torch.cat((pred, sig))
                
            else:
                k = 0  
                for start in range(0, num_frame - 50 + 1, 1):
                    end = start + 50
                    subset_input = input[:, start:end, :, :]
                    subset_input = subset_input.permute(0, 2, 1, 3).to(device)  # Need device for model compatibility
                    
                    score_abnormal, score_normal, feat_select_abn, feat_select_normal, logits, feat_magnitudes = model(inputs=subset_input)
                    logits = torch.squeeze(logits, 1)
                    logits = torch.mean(logits, 0)
                    
                    if k == 0: 
                        sig = logits.cpu()  # Move to CPU for concatenation
                        pred = torch.cat((pred, sig))
                    else:
                        sig = logits[-1,:].unsqueeze(0).cpu()  # Move to CPU for concatenation
                        pred = torch.cat((pred, sig))
                    k = 1
            
            new_gt.extend(label)
            
            # Progress indicator
            if (i + 1) % 200 == 0:
                print(f"   Processed {i+1}/{len(dataloader)} videos")

        pred = list(pred.cpu().detach().numpy())

        fpr, tpr, threshold = roc_curve(new_gt, pred)
        # CHANGE: Save to output directory if provided
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            np.save(os.path.join(output_dir, 'fpr.npy'), fpr)
            np.save(os.path.join(output_dir, 'tpr.npy'), tpr)
        else:
            # Fallback to current directory (original behavior)
            np.save('fpr.npy', fpr)
            np.save('tpr.npy', tpr)
        
        rec_auc = auc(fpr, tpr)
        print('auc : ' + str(rec_auc))
        
        precision, recall, th = precision_recall_curve(new_gt, pred)
        pr_auc = auc(recall, precision)
        
        # CHANGE: Save to output directory if provided
        if output_dir is not None:
            np.save(os.path.join(output_dir, 'precision.npy'), precision)
            np.save(os.path.join(output_dir, 'recall.npy'), recall)
        else:
            # Fallback to current directory (original behavior)
            np.save('precision.npy', precision)
            np.save('recall.npy', recall)
        
        if viz is not None:
            viz.plot_lines('pr_auc', pr_auc)
            viz.plot_lines('auc', rec_auc)
        
        # Return format matching your original (6 returns)
        return rec_auc, rec_auc, rec_auc, rec_auc, rec_auc, rec_auc

def evaluate_dota_dataset_original(args):
    """
    Main evaluation function 
    """
    print("=" * 70)
    print("TIME-VAD DOTA Evaluation")
    print("=" * 70)
    
    # Setup
    device = setup_device(args.gpu_id)
    
    # Validate files
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not os.path.exists(args.gt_path):
        raise FileNotFoundError(f"Ground truth not found: {args.gt_path}")
    
    # Load model 
    print(f"Loading model from: {args.model_path}")
    model = Model(
        n_features=args.feature_size,
        batch_size=args.batch_size,
        temporal_model=args.temporal_model
    )
    
    success = load_compatible_checkpoint(model, args.model_path, args.temporal_model)
    if not success:
        raise RuntimeError(f"Failed to load model from {args.model_path}")
    
    model = model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully ({args.temporal_model})")
    
    test_dataset = Dataset(args, test_mode=True)
    dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )
    print(f"✅ Test loader created: {len(test_dataset)} samples")
    eval_output_dir = os.path.join(args.output_dir, 'evaluation_curves')
    # Create visualizer (matching original signature)
    viz = Visualizer(env='TIME-VAD-DOTA-eval', use_incoming_socket=False)
    
    # Run the EXACT original test function
    auc_result, _, _, _, _, _ = test(dataloader, model, args, viz, device, eval_output_dir)
    
    print(f"\n🎯 FINAL ROC AUC: {auc_result:.4f}")
    
    # Collect results
    results = {
        'dataset': 'DOTA',
        'model_path': args.model_path,
        'temporal_model': args.temporal_model,
        'frame_auc': float(auc_result),
        'evaluation_method': 'exact_original_logic'
    }
    
    return results

def main():
    """Main function."""
    try:
        args = parse_dota_args()
        
        results = evaluate_dota_dataset_original(args)
        
        # Print results
        print("=" * 70)
        print(f"Model:           {results['model_path']}")
        print(f"Temporal Model:  {results['temporal_model']}")
        print(f"ROC AUC:         {results['frame_auc']:.4f}")
        print(f"Method:          {results['evaluation_method']}")
        print("=" * 70)
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, 'dota_results_exact_original.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"📁 Results saved to: {results_file}")
        
        print("✅ Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())