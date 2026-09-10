import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('../')
from taa.model.taa_yolov10 import YOLOv10TAADetectionModel
from taa.model.taa_yolov10_neuflow import YOLOv10TAANeuFlowDetectionModel
from taa.util.visualization import visualize_patch_attention
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio

def load_model(weights_path, device):
    """Load the trained model with weights."""
    model = YOLOv10TAADetectionModel()
    model.detector.fuse()
    model = model.train()
    model = model.eval()
    # model = YOLOv10TAANeuFlowDetectionModel()
    # Load the full state dictionary from the checkpoint
    full_state_dict = torch.load(weights_path, map_location=device)['model_state_dict']
    
    # Filter out keys related to detector and flow_model
    filtered_state_dict = {k: v for k, v in full_state_dict.items() if not (k.startswith('detector') or k.startswith('flow_model'))}
    
    # Load the filtered state dictionary into the model
    model.load_state_dict(filtered_state_dict, strict=False)
    
    # Move the model to the specified device
    model.to(device)
    model.eval()
    return model

def process_video(video_path, model, device, output_dir, attention_threshold=0.):
    """Process video frames and visualize detection results with attention scores."""
    # Extract video_id from video_path
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_dir = os.path.join(output_dir, video_id)
    frames_dir = os.path.join(video_dir, 'frames')
    os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    all_predictions = []
    frame_indices = []
    current_frame_idx = 0

    frames = []
    frames_org = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame
        frame, resized_frame = preprocess_frame(frame, device)
        frames_org.append(resized_frame)
        frames.append(frame)

        if len(frames) == 100:  # Process in batches of 100 frames
            input_frames = torch.stack(frames).unsqueeze(0)
            frames.clear()

            # Get model predictions
            with torch.no_grad():
                predictions = model(input_frames)

            # Get risk scores and store them
            risk_scores = torch.softmax(torch.stack(predictions['risk_score'], dim=0)[:, 0, :], dim=1)[:, 1]
            all_predictions.extend(risk_scores.cpu().numpy())
            frame_indices.extend(range(current_frame_idx, current_frame_idx + len(risk_scores)))
            current_frame_idx += len(risk_scores)

            # Extract attention scores and bounding boxes
            attention_scores = predictions['obj_attns']
            bboxes = predictions['detections']

            # Visualize attention and detections
            visualize_frames(frames_org, bboxes, attention_scores, frames_dir, attention_threshold)
            frames_org.clear()

    cap.release()

    # Save predictions to CSV
    # Create DataFrame and save to CSV
    df = pd.DataFrame({
        'frame_idx': frame_indices,
        'risk_score': all_predictions
    })
    csv_path = os.path.join(video_dir, f'{video_id}_predictions.csv')
    df.to_csv(csv_path, index=False)

    # Create prediction plot using Plotly
    fig = go.Figure()
    
    # Add risk score line
    fig.add_trace(
        go.Scatter(
            x=df['frame_idx'],
            y=df['risk_score'],
            mode='lines',
            name='Risk Score',
            line=dict(color='black', width=2)
        )
    )
    
    # Add threshold line
    fig.add_trace(
        go.Scatter(
            x=[df['frame_idx'].min(), df['frame_idx'].max()],
            y=[0.5, 0.5],
            mode='lines',
            name='Threshold',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Risk Score Predictions - {video_id}',
            x=0.5,
            font=dict(size=24)
        ),
        xaxis_title=dict(text='Frame Index', font=dict(size=18)),
        yaxis_title=dict(text='Risk Score', font=dict(size=18)),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            tickfont=dict(size=14)
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[0, 1],
            tickfont=dict(size=14)
        ),
        showlegend=True,
        legend=dict(
            font=dict(size=16),
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        plot_bgcolor='white'
    )
    
    # Save high-quality plot
    plot_path = os.path.join(video_dir, f'{video_id}_predictions.png')
    pio.write_image(fig, plot_path, format="png", width=1200, height=800, scale=3)
    
    # Also save interactive HTML version
    html_path = os.path.join(video_dir, f'{video_id}_predictions.html')
    fig.write_html(html_path)

def preprocess_frame(frame, device):
    """Preprocess the frame for model input."""
    # Resize to 640x640, convert frame to tensor and normalize
    resized_frame = cv2.resize(frame, (640, 640))
    frame_tensor = torch.from_numpy(resized_frame).permute(2, 0, 1).float() / 255.0
    return frame_tensor.to(device), resized_frame

def visualize_frames(frames, bboxes, attention_scores, output_dir, attention_threshold):
    """Visualize detections and attention scores on the frames."""
    print("\nDebugging Visualization:")
    print(f"Number of frames: {len(frames)}")
    print(f"Number of bbox predictions: {len(bboxes)}")
    print(f"Number of attention scores: {len(attention_scores)}")
    
    for idx, (frame, bboxes_frame, scores_frame) in enumerate(zip(frames, bboxes, attention_scores)):
        # print(f"scores_frame shape: {type(scores_frame)}, length: {len(scores_frame)}")
        
        # bboxes_frame is a tuple where first element contains boxes and second element contains valid count
        boxes = bboxes_frame[0][0]  # Get the actual boxes (remove batch dimension)
        valid_count = bboxes_frame[1].item()  # Get number of valid boxes as integer
        
        # Only take valid boxes and scores
        valid_boxes = boxes[:valid_count]
        valid_scores = scores_frame[0][:valid_count]
        
        # Convert tensors to numpy if needed
        if isinstance(valid_boxes, torch.Tensor):
            valid_boxes = valid_boxes.cpu().numpy()
        if isinstance(valid_scores, torch.Tensor):
            valid_scores = valid_scores.cpu().numpy()
        
        # Filter boxes based on attention threshold
        filtered_boxes = []
        for bbox, score in zip(valid_boxes, valid_scores):
            if score > attention_threshold:
                # Only take the first 4 values (x1, y1, x2, y2) from bbox
                filtered_boxes.append((bbox[:4], score))
        
        # Sort boxes by score in descending order
        filtered_boxes.sort(key=lambda x: x[1], reverse=True)
        
        # Create a copy of the frame for visualization
        frame_vis = frame.copy()
        
        # Draw boxes with different colors based on score ranking
        for i, (bbox, score) in enumerate(filtered_boxes):
            x1, y1, x2, y2 = map(int, bbox)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            if i == 0:  # Highest score - red
                color = (0, 0, 255)
                thickness = 4
            elif i == 1:  # Second highest - orange
                color = (0, 165, 255)
                thickness = 4
            else:  # Rest - light green
                color = (0, 255, 0)
                thickness = 2
                
            cv2.rectangle(frame_vis, (x1, y1), (x2, y2), color, thickness)
            # cv2.putText(frame_vis, f"{score:.2f}", (x1, y1 - 10), 
            #            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save the visualized frame
        output_path = os.path.join(output_dir, f"frame_{idx:06d}.jpg")
        cv2.imwrite(output_path, frame_vis)

def main():
    video_path = 'data/DAD/videos/testing/positive/000525.mp4'
    weights_path = '_experiments/20251102_153919/checkpoints/best_model.pth'
    output_dir = '_demo'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(weights_path, device)

    # Process video and visualize
    # process_video(video_path, model, device, output_dir)
    
    video_dir = 'data/DAD/videos/testing/positive'
    # Get all MP4 files in the directory
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    # Process each video
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(video_dir, video_file)
        print(f"\nProcessing video: {video_file}")
        try:
            process_video(video_path, model, device, output_dir)
        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")
            continue
        
if __name__ == '__main__':
    main()
