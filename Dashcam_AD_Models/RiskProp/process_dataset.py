import os
import pandas as pd
import subprocess
import signal
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count

def signal_handler(sig, frame):
    print("\nProcess interrupted by user. Exiting...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

def extract_frames(video_path, output_dir, fps=30):
    """Extract frames from video at specified fps"""
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"ffmpeg -i {video_path} -vf fps={fps} -start_number 0 {output_dir}/%06d.jpg -loglevel error"
    subprocess.run(cmd, shell=True, check=True)

def process_train_video(args):
    idx, row, data_root = args
    video_id = str(int(row['id'])).zfill(5)
    video_path = f"{data_root}/train/{video_id}.mp4"
    
    if not os.path.exists(video_path):
        return None
    
    target = int(row['target'])
    
    if target == 1:
        accident_frame = int(float(row['time_of_event']) * 30)
        abnormal_start_frame = int(float(row['time_of_alert']) * 30)
    else:
        accident_frame = None
        abnormal_start_frame = None
    
    frame_dir = f"{data_root}/train_raw_frames/{video_id}"
    
    # Extract frames only if directory doesn't exist or is empty
    if not os.path.exists(frame_dir) or not os.listdir(frame_dir):
        extract_frames(video_path, frame_dir)
    
    # Count actual extracted frames
    total_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    
    if idx % 50 == 0:
        print(f"Train progress: {idx+1}")
    
    return [int(row['id']), 0, total_frames, None, abnormal_start_frame, accident_frame, target]

def process_test_video(args):
    idx, row, data_root = args
    video_id = str(int(row['id'])).zfill(5)
    video_path = f"{data_root}/test/{video_id}.mp4"
    
    if not os.path.exists(video_path):
        return None
    
    frame_dir = f"{data_root}/test_raw_frames/{video_id}"
    
    # Extract frames only if directory doesn't exist or is empty
    if not os.path.exists(frame_dir) or not os.listdir(frame_dir):
        extract_frames(video_path, frame_dir)
    
    # Count actual extracted frames
    total_frames = len([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    
    if idx % 50 == 0:
        print(f"Test progress: {idx+1}")
    
    return [int(row['id']), 1, total_frames, None, None, None, None]

def process_nexar_dataset():
    data_root = "data/nexar-collision-prediction"
    train_df = pd.read_csv(f"{data_root}/train.csv")
    test_df = pd.read_csv(f"{data_root}/test.csv")
    
    num_workers = min(8, cpu_count())
    print(f"Using {num_workers} workers")
    
    # Process training videos
    print(f"Processing {len(train_df)} training videos...")
    train_args = [(idx, row, data_root) for idx, row in train_df.iterrows()]
    with Pool(num_workers) as pool:
        train_results = pool.map(process_train_video, train_args)
    train_annotations = [r for r in train_results if r is not None]
    
    # Process test videos
    print(f"Processing {len(test_df)} test videos...")
    test_args = [(idx, row, data_root) for idx, row in test_df.iterrows()]
    with Pool(num_workers) as pool:
        test_results = pool.map(process_test_video, test_args)
    test_annotations = [r for r in test_results if r is not None]
    
    # Save annotations.csv
    all_annotations = train_annotations + test_annotations
    ann_df = pd.DataFrame(all_annotations, columns=[
        'id', 'is_test', 'total_frames', 'type', 
        'abnormal_start_frame', 'accident_frame', 'target'
    ])
    ann_df.to_csv(f"{data_root}/annotations.csv", index=False)
    print(f"Saved annotations.csv with {len(ann_df)} entries")

if __name__ == "__main__":
    process_nexar_dataset()
