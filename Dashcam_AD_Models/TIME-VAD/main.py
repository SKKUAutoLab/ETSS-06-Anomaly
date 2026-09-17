import os
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Model
from dataset import create_data_loaders, validate_dataset_files
from train import train_with_loss_function, setup_loss_function
from test_10crop import test
from utils import save_best_record, Visualizer
from config import Config
from option import validate_args, get_dataset_paths  # Import get_dataset_paths
import option
from evaluate_dota import evaluate_dota_dataset_original


def setup_model_and_optimizer(args, device):
    """Setup model, optimizer, and loss function"""
    model = Model(
        n_features=args.feature_size, 
        batch_size=args.batch_size,
        temporal_model=args.temporal_model
    )
    
    model = model.to(device)
    
    # Load pretrained model if specified
    if args.pretrained_ckpt and os.path.exists(args.pretrained_ckpt):
        print(f"Loading pretrained model from {args.pretrained_ckpt}")
        try:
            state_dict = torch.load(args.pretrained_ckpt, map_location=device)
            model.load_state_dict(state_dict)
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"⚠️  Could not load pretrained model: {e}")
            print("Continuing with random initialization")
    
    # Setup optimizer
    config = Config(args)
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.lr[0], 
        weight_decay=args.weight_decay
    )
    
    # Setup loss function (load anchor features once)
    print("Setting up loss function with anchor features...")
    loss_criterion = setup_loss_function(args, device)
    
    return model, optimizer, config, loss_criterion


def setup_device(args):
    """Setup training device"""
    if torch.cuda.is_available() and args.gpu_id >= 0:
        device = torch.device(f'cuda:{args.gpu_id}')
        torch.cuda.set_device(args.gpu_id)
        print(f"Using GPU: {args.gpu_id}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def create_checkpoint_dir(args):
    """Create checkpoint directory"""
    ckpt_dir = args.checkpoint_dir
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f"Created checkpoint directory: {ckpt_dir}")
    return ckpt_dir


def run_dota_evaluation_direct(model, args, device, evaluation_name="temp_eval"):
    """
    Run DOTA-specific evaluation using the already-loaded model directly.
    This avoids redundant model loading and saving.
    """
    
    try:
        # Get DOTA ground truth path using option.py function
        dataset_paths = get_dataset_paths(args)
        gt_path = dataset_paths['ground_truth']

        eval_args = type('Args', (), {
            'data_root': args.data_root,
            'dataset': 'dota',
            'feature_size': args.feature_size,
            'temporal_model': args.temporal_model,
            'num_segments': args.num_segments,
            'workers': args.workers,
            'gpu_id': args.gpu_id,
            'batch_size': 1,
            'output_dir': os.path.join(args.output_dir, evaluation_name),
            'modality': 'RGB',
            'gt_path': gt_path
        })()
        
        # Import the test function directly
        from evaluate_dota import test
        from dataset import Dataset
        from torch.utils.data import DataLoader
        from utils import Visualizer
        
        print("="*70)
        print("TIME-VAD DOTA Evaluation")
        print("="*70)
        
        # Create test dataset and dataloader
        test_dataset = Dataset(eval_args, test_mode=True)
        dataloader = DataLoader(
            test_dataset,
            batch_size=eval_args.batch_size,
            shuffle=False,
            num_workers=eval_args.workers,
            pin_memory=False
        )
        print(f"✅ Test loader created: {len(test_dataset)} samples")
        
        # Create visualizer
        viz = Visualizer(env='TIME-VAD-DOTA-eval', use_incoming_socket=False)
        
        # Set model to eval mode
        model.eval()
        
        # Run evaluation using the already-loaded model
        eval_output_dir = os.path.join(eval_args.output_dir, 'evaluation_curves')
        auc_result, _, _, _, _, _ = test(dataloader, model, eval_args, viz, device, eval_output_dir)
        
        print(f"\n🎯 DOTA ROC AUC: {auc_result:.4f}")
        
        # Collect results
        results = {
            'dataset': 'DOTA',
            'temporal_model': args.temporal_model,
            'frame_auc': float(auc_result),
            'evaluation_method': 'direct_model_reuse'
        }
        
        return results['frame_auc']
        
    except Exception as e:
        print(f"⚠️  DOTA evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_final_dota_evaluation(args, ckpt_dir):
    """Run final DOTA evaluation after training."""
    if args.dataset == 'dota':
        print("\n" + "="*60)
        print("🎯 RUNNING FINAL DOTA EVALUATION")
        print("="*60)
        
        # Get DOTA ground truth path using option.py function
        dataset_paths = get_dataset_paths(args)
        
        gt_path = dataset_paths['ground_truth']
        
        # Create evaluation args for the working evaluate_dota.py
        eval_args = type('Args', (), {
            'model_path': os.path.join(ckpt_dir, f'{args.model_name}_best.pkl'),
            'data_root': args.data_root,
            'dataset': 'dota',
            'feature_size': args.feature_size,
            'temporal_model': args.temporal_model,
            'num_segments': args.num_segments,
            'workers': args.workers,
            'gpu_id': args.gpu_id,
            'batch_size': 1,
            'output_dir': os.path.join(args.output_dir, 'final_dota_evaluation'),
            'modality': 'RGB',
            'gt_path': gt_path
        })()
        
        try:
            # Use the working function
            results = evaluate_dota_dataset_original(eval_args)
            
            # Print results
            print("\n" + "=" * 70)
            print("📊 FINAL DOTA EVALUATION RESULTS")
            print("=" * 70)
            print(f"Model:           {results['model_path']}")
            print(f"Temporal Model:  {results['temporal_model']}")
            print(f"ROC AUC:         {results['frame_auc']:.4f}")
            print(f"Method:          {results['evaluation_method']}")
            print("=" * 70)
            
            # Save results
            results_file = os.path.join(eval_args.output_dir, 'final_dota_results.json')
            os.makedirs(eval_args.output_dir, exist_ok=True)
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"📁 Final results saved to: {results_file}")
            
        except Exception as e:
            print(f"Final DOTA evaluation failed: {e}")
            import traceback
            traceback.print_exc()


def main():
    # Parse and validate arguments
    args = option.parser.parse_args()
    args = validate_args(args)
    
    # Validate dataset files exist
    if not validate_dataset_files(args):
        print("Error: Missing required dataset files. Please check data paths.")
        return
    
    # Setup device
    device = setup_device(args)
    
    # Create checkpoint directory
    ckpt_dir = create_checkpoint_dir(args)
    
    # Setup visualizer
    viz = Visualizer(env=f'TIME-VAD-{args.dataset}-{args.temporal_model}', use_incoming_socket=False)
    
    # Setup data loaders
    print("Setting up data loaders...")
    train_normal_loader, train_abnormal_loader, test_loader = create_data_loaders(args)
    print(f"Training normal samples: {len(train_normal_loader.dataset)}")
    print(f"Training abnormal samples: {len(train_abnormal_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Setup model and optimizer
    print("Setting up model and optimizer...")
    model, optimizer, config, loss_criterion = setup_model_and_optimizer(args, device)
    print(f"Using temporal model: {args.temporal_model}")
    print("Anchor features loaded successfully!")
        
    # Initialize tracking variables
    test_info = {
        "epoch": [], 
        "test_AUC": [], 
        "AP_video": [], 
        "AP": [], 
        "mTTA": [], 
        "TTA_R80": [], 
        "P_R80": []
    }
    best_AUC = -1
    
    # Initial evaluation
    if args.eval_initial:
        print("Performing initial evaluation...")
        
        if args.dataset == 'dota':
            # Use the direct evaluation function instead of the redundant one
            initial_auc = run_dota_evaluation_direct(model, args, device, "initial_evaluation")
            if initial_auc is not None:
                print(f"Initial DOTA ROC-AUC: {initial_auc:.4f}")
            else:
                print("⚠️  Initial DOTA evaluation failed")
        
        else:
            auc, AP_video, AP, mTTA, TTA_R80, P_R80 = test(test_loader, model, args, viz, device)
            print(f"Initial - AUC: {auc:.4f}, AP: {AP:.4f}, mTTA: {mTTA:.4f}")
    
    # Training loop
    print(f"Starting training for {args.max_epoch} epochs...")
    print("Note: Anchor features loaded once for efficient training (no repeated loading per epoch)")
    
    for epoch in tqdm(range(1, args.max_epoch + 1), desc="Training", dynamic_ncols=True):
        
        # Update learning rate if needed
        if epoch > 1 and config.lr[epoch - 1] != config.lr[epoch - 2]:
            for param_group in optimizer.param_groups:
                param_group["lr"] = config.lr[epoch - 1]
            print(f"Updated learning rate to: {config.lr[epoch - 1]}")
        
        # Reset data loader iterators
        if (epoch - 1) % len(train_normal_loader) == 0:
            normal_iter = iter(train_normal_loader)
        
        if (epoch - 1) % len(train_abnormal_loader) == 0:
            abnormal_iter = iter(train_abnormal_loader)
        
        # Training step
        train_with_loss_function(normal_iter, abnormal_iter, model, args.batch_size, 
                               optimizer, viz, device, loss_criterion, args)
        
        # Evaluation
        if epoch % args.eval_freq == 0 and epoch > args.eval_start:
            print(f"\nEvaluating at epoch {epoch}...")
            
            # Use DOTA-specific evaluation if dataset is DOTA and available
            if args.dataset == 'dota':
                # Use direct evaluation instead of redundant loading
                dota_auc = run_dota_evaluation_direct(model, args, device, f"epoch_{epoch}_evaluation")
                if dota_auc is not None:
                    auc = dota_auc
                    # Set dummy values for other metrics for DOTA
                    AP_video = AP = mTTA = TTA_R80 = P_R80 = 0.0
                    print(f"Epoch {epoch} - DOTA ROC-AUC: {auc:.4f}")
                else:
                    print(f"⚠️  DOTA evaluation failed, skipping evaluation at epoch {epoch}")
                    continue
            else:
                # Standard evaluation for non-DOTA datasets
                auc, AP_video, AP, mTTA, TTA_R80, P_R80 = test(test_loader, model, args, viz, device)
                print(f"Epoch {epoch} - AUC: {auc:.4f}, AP: {AP:.4f}, mTTA: {mTTA:.4f}")
            
            # Store results
            test_info["epoch"].append(epoch)
            test_info["test_AUC"].append(auc)
            test_info["AP_video"].append(AP_video)
            test_info["AP"].append(AP)
            test_info["mTTA"].append(mTTA)
            test_info["TTA_R80"].append(TTA_R80)
            test_info["P_R80"].append(P_R80)
            
            # Save best model based on ROC-AUC
            if auc > best_AUC:
                best_AUC = auc
                best_model_path = os.path.join(ckpt_dir, f'{args.model_name}_best.pkl')
                torch.save(model.state_dict(), best_model_path)
                
                # Save best results
                best_results_path = os.path.join(args.output_dir, f'best_results_epoch_{epoch}.txt')
                save_best_record(test_info, best_results_path)
                
                print(f"🏆 New best ROC-AUC: {best_AUC:.4f} - Model saved to {best_model_path}")
        
        # Save periodic checkpoint
        if epoch % args.save_freq == 0:
            checkpoint_path = os.path.join(ckpt_dir, f'{args.model_name}_epoch_{epoch}.pkl')
            torch.save(model.state_dict(), checkpoint_path)
    
    # Save final model
    final_model_path = os.path.join(ckpt_dir, f'{args.model_name}_final.pkl')
    torch.save(model.state_dict(), final_model_path)
    print(f"Training completed. Final model saved to {final_model_path}")
    
    # Final evaluation using the working DOTA script
    run_final_dota_evaluation(args, ckpt_dir)
    
    # Print final results
    if test_info["test_AUC"]:
        print(f"\n🎯 FINAL TRAINING RESULTS:")
        print(f"Best ROC-AUC: {best_AUC:.4f}")
        print(f"Final ROC-AUC: {test_info['test_AUC'][-1]:.4f}")
        if args.dataset != 'dota':
            print(f"Final AP: {test_info['AP'][-1]:.4f}")
            print(f"Final mTTA: {test_info['mTTA'][-1]:.4f}")


if __name__ == '__main__':
    main()