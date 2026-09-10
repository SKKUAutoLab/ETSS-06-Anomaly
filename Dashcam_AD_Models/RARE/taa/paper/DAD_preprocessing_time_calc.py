'''
This script is for calculating the preprocessing time of DAD dataset.

1. Object detection by faster rcnn
2. Feature extraction by vgg16 of frame and 19 detection proposals
'''

import time
import torch
import torchvision.models as models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import numpy as np
import torch.nn as nn
import os
import cv2
from torchvision.transforms import Resize

def setup_models(device):
    # Initialize models
    detector = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    detector.eval()
    
    # VGG16 setup - remove classifier to get features
    feature_extractor = models.vgg16(pretrained=True).to(device)
    feature_extractor.classifier = nn.Sequential(*list(feature_extractor.classifier.children())[:-1])
    feature_extractor.eval()
    
    return detector, feature_extractor

def measure_memory_and_time(image, original_image, detector, feature_extractor, device, num_runs=10):
    results = {
        'detection': {'time': [], 'memory': []},
        'full_frame_feature': {'time': [], 'memory': []},
        'proposals_batch': {'time': [], 'memory': []},
        'proposals_sequential': {'time': [], 'memory': []}
    }
    
    image = image.to(device)
    original_image = original_image.to(device)
    for _ in range(num_runs):
        # Detection memory and time
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.time()
        with torch.no_grad():
            detections = detector([original_image])
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        detection_time = time.time() - start
        detection_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        results['detection']['time'].append(detection_time)
        results['detection']['memory'].append(detection_memory)

        # Full frame feature extraction
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.time()
        with torch.no_grad():
            frame_features = feature_extractor(image.unsqueeze(0))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        frame_time = time.time() - start
        frame_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        results['full_frame_feature']['time'].append(frame_time)
        results['full_frame_feature']['memory'].append(frame_memory)

        # Batch processing of proposals
        proposals_batch = torch.stack([image] * 20)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.time()
        with torch.no_grad():
            proposal_features_batch = feature_extractor(proposals_batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        batch_time = time.time() - start
        batch_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        results['proposals_batch']['time'].append(batch_time)
        results['proposals_batch']['memory'].append(batch_memory)

        # Sequential processing of proposals
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        start = time.time()
        with torch.no_grad():
            for _ in range(20):
                proposal_features_seq = feature_extractor(image.unsqueeze(0))
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        
        seq_time = time.time() - start
        seq_memory = torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
        
        results['proposals_sequential']['time'].append(seq_time)
        results['proposals_sequential']['memory'].append(seq_memory)

    # Calculate means
    for component in results:
        results[component]['time'] = np.mean(results[component]['time'])
        results[component]['memory'] = np.mean(results[component]['memory'])
    
    return results

def load_real_images(num_images=100):
    folder_path = "./taa/data/DAD/frames/training/positive/000001"
    all_images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # If we have less than num_images, we'll cycle through the available images
    selected_images = []
    while len(selected_images) < num_images:
        selected_images.extend(all_images[:num_images - len(selected_images)])
    
    selected_images = selected_images[:num_images]
    
    # Load and preprocess images
    original_images = []
    processed_images = []
    for img_name in selected_images:
        img_path = os.path.join(folder_path, img_name)
        # Read image using OpenCV
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        # Store original image after normalizing
        original_img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        original_images.append(original_img/255.0)
        
        # Process image for VGG16
        img = cv2.resize(img, (224, 224))  # Resize for VGG16
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()  # Convert to torch tensor
        img = img / 255.0  # Normalize
        processed_images.append(img)
    
    return original_images, processed_images

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def analyze_efficiency():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    detector, feature_extractor = setup_models(device)
    
    # Calculate and print model parameters
    detector_params = count_parameters(detector)
    feature_extractor_params = count_parameters(feature_extractor)
    
    print("\nModel Parameters:")
    print("-" * 60)
    print(f"Faster R-CNN Parameters: {detector_params:,}")
    print(f"VGG16 Feature Extractor Parameters: {feature_extractor_params:,}")
    print(f"Total Parameters: {detector_params + feature_extractor_params:,}")
    print("-" * 60)
    
    # Load real images
    print("\nLoading images...")
    original_images, images = load_real_images(num_images=100)
    print(f"Loaded {len(images)} images")
    
    # Store results for each run
    all_runs_results = []
    
    print("\nRunning measurements...")
    for run in range(10):
        print(f"\nRun {run + 1}/10")
        run_results = []
        
        for i, (original_image, image) in enumerate(zip(original_images, images), 1):
            if i % 10 == 0:
                print(f"Processing image {i}/100")
            
            results = measure_memory_and_time(image, original_image, detector, feature_extractor, device, num_runs=1)
            run_results.append(results)
        
        # Average results for this run
        run_avg_results = {
            'detection': {'time': 0, 'memory': 0},
            'full_frame_feature': {'time': 0, 'memory': 0},
            'proposals_batch': {'time': 0, 'memory': 0},
            'proposals_sequential': {'time': 0, 'memory': 0}
        }
        
        for component in run_avg_results:
            run_avg_results[component]['time'] = np.mean([r[component]['time'] for r in run_results])
            run_avg_results[component]['memory'] = np.mean([r[component]['memory'] for r in run_results])
        
        all_runs_results.append(run_avg_results)
    
    # Calculate final averages and standard deviations across all runs
    final_results = {
        'detection': {'time': 0, 'memory': 0},
        'full_frame_feature': {'time': 0, 'memory': 0},
        'proposals_batch': {'time': 0, 'memory': 0},
        'proposals_sequential': {'time': 0, 'memory': 0}
    }
    
    std_results = {
        'detection': {'time': 0, 'memory': 0},
        'full_frame_feature': {'time': 0, 'memory': 0},
        'proposals_batch': {'time': 0, 'memory': 0},
        'proposals_sequential': {'time': 0, 'memory': 0}
    }
    
    for component in final_results:
        final_results[component]['time'] = np.mean([run[component]['time'] for run in all_runs_results])
        final_results[component]['memory'] = np.mean([run[component]['memory'] for run in all_runs_results])
        std_results[component]['time'] = np.std([run[component]['time'] for run in all_runs_results])
        std_results[component]['memory'] = np.std([run[component]['memory'] for run in all_runs_results])
    
    print(f"\nFinal Efficiency Analysis (Device: {device})")
    print(f"Average over 10 runs of 100 images each")
    print("-" * 60)
    
    print("\n1. Object Detection (Faster R-CNN):")
    print(f"   Time: {final_results['detection']['time']:.3f} ± {std_results['detection']['time']:.3f} seconds")
    print(f"   Memory: {final_results['detection']['memory']:.2f} ± {std_results['detection']['memory']:.2f} MB")
    
    print("\n2. Full Frame Feature Extraction (VGG16):")
    print(f"   Time: {final_results['full_frame_feature']['time']:.3f} ± {std_results['full_frame_feature']['time']:.3f} seconds")
    print(f"   Memory: {final_results['full_frame_feature']['memory']:.2f} ± {std_results['full_frame_feature']['memory']:.2f} MB")
    
    print("\n3. Proposal Regions Feature Extraction (19 regions):")
    print("   Batch Processing:")
    print(f"   Time: {final_results['proposals_batch']['time']:.3f} ± {std_results['proposals_batch']['time']:.3f} seconds")
    print(f"   Memory: {final_results['proposals_batch']['memory']:.2f} ± {std_results['proposals_batch']['memory']:.2f} MB")
    print("   Sequential Processing:")
    print(f"   Time: {final_results['proposals_sequential']['time']:.3f} ± {std_results['proposals_sequential']['time']:.3f} seconds")
    print(f"   Memory: {final_results['proposals_sequential']['memory']:.2f} ± {std_results['proposals_sequential']['memory']:.2f} MB")
    
    print("\nComparison - Batch vs Sequential Processing:")
    time_speedup = final_results['proposals_sequential']['time'] / final_results['proposals_batch']['time']
    memory_increase = final_results['proposals_batch']['memory'] / final_results['proposals_sequential']['memory']
    print(f"Time Speedup with Batch Processing: {time_speedup:.2f}x")
    print(f"Memory Usage Increase with Batch Processing: {memory_increase:.2f}x")
    
    # Print individual run averages
    print("\nIndividual Run Averages:")
    for run_idx, run_results in enumerate(all_runs_results, 1):
        print(f"\nRun {run_idx}:")
        print(f"Detection Time: {run_results['detection']['time']:.3f} s")
        print(f"Full Frame Time: {run_results['full_frame_feature']['time']:.3f} s")
        print(f"Batch Proposals Time: {run_results['proposals_batch']['time']:.3f} s")
        print(f"Sequential Proposals Time: {run_results['proposals_sequential']['time']:.3f} s")

if __name__ == "__main__":
    analyze_efficiency()

