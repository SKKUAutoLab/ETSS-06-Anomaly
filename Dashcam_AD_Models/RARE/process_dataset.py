import os
import cv2

video_root = "data/DAD/videos"
frame_root = "taa/data/DAD/frames"
splits = ["training", "testing"]
categories = ["positive", "negative"]
for split in splits:
    for category in categories:
        input_folder = os.path.join(video_root, split, category)
        output_folder_base = os.path.join(frame_root, split, category)
        os.makedirs(output_folder_base, exist_ok=True)
        mp4_files = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
        for mp4_file in mp4_files:
            video_path = os.path.join(input_folder, mp4_file)
            output_folder = os.path.join(output_folder_base, mp4_file)
            os.makedirs(output_folder, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                frame_name = f"{frame_count:06d}.jpg"
                frame_path = os.path.join(output_folder, frame_name)
                cv2.imwrite(frame_path, frame)
            cap.release()
            print(f"✅ Extracted {frame_count} frames from {split}/{category}/{mp4_file}")
print("\n🎉 All videos processed successfully.")
