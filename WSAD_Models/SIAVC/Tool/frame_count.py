import os
import pickle

folder_path = r'F:\KeyPoint\Code\FixMatch-pytorch-master\dataset\Firesense_Test\Smoke'
total_frames = 0
pkl_files = [file for file in os.listdir(folder_path) if file.endswith('.pkl')]
for pkl_file in pkl_files:
    pkl_path = os.path.join(folder_path, pkl_file)
    with open(pkl_path, 'rb') as f:
        video_file = pickle.load(f)
    total_frames += video_file.shape[0]
print(f"Total frames in all videos: {total_frames}")