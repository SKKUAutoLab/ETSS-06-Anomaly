#!/usr/bin/env python
# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from ultralytics.data.utils import HELP_URL, LOGGER, get_hash
from ultralytics.utils import NUM_THREADS, ROOT, colorstr, ops
from ultralytics.utils.checks import check_requirements

# Global Constants
HDF5_COMPRESSION = "lzf"  # Compression algorithm for HDF5 files


def create_hdf5_dataset(path="../datasets/coco8/", split='train'):
    """
    Creates an HDF5 dataset for rotated object detection.
    
    Args:
        path (str): Path to dataset root directory
        split (str): Dataset split ('train', 'val', or 'test')
    """
    check_requirements('h5py')
    
    # Initialize paths
    path = Path(path)
    labels_dir = path / 'labels' / split
    images_dir = path / 'images' / split
    hdf5_path = path / f'{split}.hdf5'
    
    if hdf5_path.exists():
        LOGGER.info(f'{colorstr("HDF5 dataset already exists: ")} {hdf5_path}')
        return hdf5_path
    
    # Get image and label files
    img_files = sorted(images_dir.rglob('*.*'))
    img_files = [x for x in img_files if x.suffix[1:].lower() in ('bmp', 'jpg', 'jpeg', 'png')]
    label_files = [labels_dir / f'{x.stem}.txt' for x in img_files]
    
    # Verify dataset
    num_imgs = len(img_files)
    if num_imgs == 0:
        LOGGER.warning(f'No images found in {images_dir}. See {HELP_URL}')
        return None
    
    # Create HDF5 file
    LOGGER.info(f'{colorstr("Creating HDF5 dataset: ")} {hdf5_path}')
    with h5py.File(hdf5_path, 'w') as h5_file:
        # Create datasets
        dt = h5py.special_dtype(vlen=np.dtype('float32'))
        images = h5_file.create_dataset('images', shape=(num_imgs,), dtype=dt)
        labels = h5_file.create_dataset('labels', shape=(num_imgs,), dtype=dt)
        shapes = h5_file.create_dataset('shapes', shape=(num_imgs, 2), dtype='int32')
        
        # Store hash for dataset verification
        hash_str = get_hash([str(x) for x in img_files] + [str(x) for x in label_files])
        h5_file.attrs['hash'] = hash_str
        
        # Process images and labels
        for i, (img_file, label_file) in enumerate(tqdm(zip(img_files, label_files), total=num_imgs)):
            try:
                # Read and process image
                img = cv2.imread(str(img_file))
                if img is None:
                    raise FileNotFoundError(f'Image not found: {img_file}')
                h, w = img.shape[:2]
                shapes[i] = (h, w)
                
                # Store compressed image
                img_bytes = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                images[i] = np.frombuffer(img_bytes, dtype='uint8')
                
                # Read and process labels
                if label_file.exists():
                    with open(label_file) as f:
                        lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32)
                        if len(lb):
                            # Store rotated bounding box coordinates (cls, cx, cy, w, h, angle)
                            labels[i] = lb.ravel()
                        else:
                            labels[i] = np.zeros(0, dtype=np.float32)
                else:
                    labels[i] = np.zeros(0, dtype=np.float32)
                    
            except Exception as e:
                LOGGER.warning(f'Error processing {img_file}: {e}')
                
        # Store additional metadata
        h5_file.attrs['num_imgs'] = num_imgs
        h5_file.attrs['split'] = split
        
    LOGGER.info(f'HDF5 dataset saved to {hdf5_path}')
    return hdf5_path


def verify_hdf5(path):
    """
    Verify the integrity of an HDF5 dataset.
    
    Args:
        path (str): Path to HDF5 file
    """
    try:
        with h5py.File(path, 'r') as h5_file:
            stored_hash = h5_file.attrs.get('hash', None)
            num_imgs = h5_file.attrs.get('num_imgs', 0)
            
            if stored_hash is None:
                LOGGER.warning(f'No hash found in {path}')
                return False
                
            LOGGER.info(f'Verifying {num_imgs} images in {path}')
            for i in tqdm(range(num_imgs)):
                if h5_file['images'][i] is None or h5_file['labels'][i] is None:
                    LOGGER.warning(f'Corrupted data found at index {i}')
                    return False
            
            LOGGER.info(f'{colorstr("Verification complete: ")} {path} ✅')
            return True
            
    except Exception as e:
        LOGGER.warning(f'Error verifying {path}: {e}')
        return False


if __name__ == '__main__':
    # Example usage
    path = 'taa/data/ROL'  # path to dataset
    for split in ['train', 'val']:
        hdf5_path = create_hdf5_dataset(path=path, split=split)
        if hdf5_path:
            verify_hdf5(hdf5_path)