import os
import torch
import torch.utils.data as data
import numpy as np
from utils import process_feat
from torch.utils.data import DataLoader
from option import get_dataset_paths


class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        """
        TIME-VAD Dataset class for loading video anomaly detection data.
        
        Args:
            args: Command line arguments
            is_normal: Whether to load normal (True) or abnormal (False) videos
            transform: Optional data transformation
            test_mode: Whether in test mode (True) or train mode (False)
        """
        self.args = args
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset.lower()
        self.transform = transform
        self.test_mode = test_mode
        self.num_segments = args.num_segments
        
        # Get dataset-specific paths
        self.dataset_paths = get_dataset_paths(args)
        if self.dataset_paths is None:
            raise ValueError(f"Dataset '{self.dataset}' is not supported")
        
        # Set list file path based on mode
        if test_mode:
            self.list_file = self.dataset_paths['test_list']
        else:
            self.list_file = self.dataset_paths['train_list']
        
        # Verify list file exists
        if not os.path.exists(self.list_file):
            raise FileNotFoundError(f"Dataset list file not found: {self.list_file}")
        
        # Parse the list file
        self._parse_list()
        
        self.num_frame = 0
        self.labels = None
        
        # print(f"Loaded {len(self.list)} {'test' if test_mode else 'train'} "
        #       f"{'normal' if is_normal else 'abnormal'} samples for {self.dataset.upper()}")

    def _parse_list(self):
        """Parse the dataset list file and filter based on normal/abnormal split."""
        with open(self.list_file, 'r') as f:
            self.list = [line.strip() for line in f.readlines()]
        
        if not self.test_mode:
            self._split_normal_abnormal()

    def _split_normal_abnormal(self):
        """Split the training data into normal and abnormal based on dataset configuration."""
        if self.dataset == 'dad':
            # train.list is [829 negative (no accident), 455 positive (accident)]
            normal_end_idx = self.dataset_paths['normal_train_end']
            if self.is_normal:
                self.list = self.list[:normal_end_idx]
                print(f'Loaded {len(self.list)} normal training samples for DAD')
            else:
                self.list = self.list[normal_end_idx:]
                print(f'Loaded {len(self.list)} abnormal training samples for DAD')
                
        elif self.dataset == 'ccd':
            normal_start_idx = self.dataset_paths['normal_train_start']
            if self.is_normal:
                self.list = self.list[normal_start_idx:]
                print(f'Loaded {len(self.list)} normal training samples for CCD')
            else:
                self.list = self.list[:normal_start_idx]
                print(f'Loaded {len(self.list)} abnormal training samples for CCD')
                
        elif self.dataset == 'dota':
            normal_end_idx = self.dataset_paths['normal_train_end']
            if self.is_normal:
                self.list = self.list[:normal_end_idx]
                print(f'Loaded {len(self.list)} normal training samples for DOTA')
            else:
                self.list = self.list[normal_end_idx:]
                print(f'Loaded {len(self.list)} abnormal training samples for DOTA')

    def __getitem__(self, index):
        """
        Get a single data sample.
        
        Args:
            index: Sample index
            
        Returns:
            features: Processed video features
            label: Video-level label (for training) or None (for testing)
        """
        try:
            # Load features
            feature_path = self.list[index].strip()
            if not os.path.exists(feature_path):
                raise FileNotFoundError(f"Feature file not found: {feature_path}")
                
            features = np.load(feature_path, allow_pickle=True)
            features = np.array(features, dtype=np.float32)
            
            # Apply transform if specified
            if self.transform is not None:
                features = self.transform(features)
            
            if self.test_mode:
                # For testing, return features as-is (will be processed in test script)
                return features
            else:
                # For training, process features
                features = self._process_training_features(features)
                label = self.get_label()
                return features, label
                
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            # Return a dummy sample to avoid crashing
            if self.test_mode:
                return np.zeros((10, self.num_segments, self.args.feature_size), dtype=np.float32)
            else:
                dummy_features = np.zeros((10, self.num_segments, self.args.feature_size), dtype=np.float32)
                return dummy_features, self.get_label()

    def _process_training_features(self, features):
        """
        Process features for training mode.
        
        Args:
            features: Raw features from file
            
        Returns:
            processed_features: Features divided into segments
        """
        # Transpose to [10, T, F] format (10-crop)
        features = features.transpose(1, 0, 2)
        
        # Process each crop
        processed_features = []
        for feature in features:
            # Divide video into segments (default: 50)
            segmented_feature = process_feat(feature, self.num_segments)
            processed_features.append(segmented_feature)
        
        processed_features = np.array(processed_features, dtype=np.float32)
        return processed_features

    def get_label(self):
        """
        Get the video-level label.
        
        Returns:
            label: 0.0 for normal, 1.0 for abnormal
        """
        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)
        return label

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.list)

    def get_num_frames(self):
        """Return the number of frames (for compatibility)."""
        return self.num_frame

    def get_dataset_info(self):
        """
        Get information about the dataset.
        
        Returns:
            dict: Dataset information
        """
        return {
            'dataset': self.dataset,
            'mode': 'test' if self.test_mode else 'train',
            'type': 'normal' if self.is_normal else 'abnormal',
            'num_samples': len(self.list),
            'num_segments': self.num_segments,
            'feature_size': self.args.feature_size,
            'list_file': self.list_file
        }


def create_data_loaders(args):
    """
    Create train and test data loaders for TIME-VAD.
    
    Args:
        args: Command line arguments
        
    Returns:
        tuple: (train_normal_loader, train_abnormal_loader, test_loader)
    """
    # Training loaders
    train_normal_dataset = Dataset(args, test_mode=False, is_normal=True)
    train_abnormal_dataset = Dataset(args, test_mode=False, is_normal=False)
    test_dataset = Dataset(args, test_mode=True)
    
    train_normal_loader = DataLoader(
        train_normal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True
    )
    
    train_abnormal_loader = DataLoader(
        train_abnormal_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False
    )
    
    return train_normal_loader, train_abnormal_loader, test_loader


def validate_dataset_files(args):
    """
    Validate that all required dataset files exist.
    
    Args:
        args: Command line arguments
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    dataset_paths = get_dataset_paths(args)
    if dataset_paths is None:
        print(f"Error: Unknown dataset '{args.dataset}'")
        return False
    
    required_files = ['train_list', 'test_list', 'ground_truth']
    missing_files = []
    
    for file_key in required_files:
        file_path = dataset_paths[file_key]
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Missing required dataset files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    print(f"All required files found for dataset '{args.dataset.upper()}'")
    return True


if __name__ == '__main__':
    # Test dataset loading
    import argparse
    from option import parser
    
    args = parser.parse_args(['--dataset', 'ccd'])
    
    if validate_dataset_files(args):
        try:
            train_normal_loader, train_abnormal_loader, test_loader = create_data_loaders(args)
            
            print("\nDataset Statistics:")
            print(f"Normal training samples: {len(train_normal_loader.dataset)}")
            print(f"Abnormal training samples: {len(train_abnormal_loader.dataset)}")
            print(f"Test samples: {len(test_loader.dataset)}")
            
            # Test loading a sample
            sample_normal = train_normal_loader.dataset[0]
            print(f"\nSample shapes:")
            print(f"Normal training sample: {sample_normal[0].shape}")
            print(f"Normal training label: {sample_normal[1]}")
            
        except Exception as e:
            print(f"Error testing dataset: {e}")
    else:
        print("Cannot test dataset due to missing files")