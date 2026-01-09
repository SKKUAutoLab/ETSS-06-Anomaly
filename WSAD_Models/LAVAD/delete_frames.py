import os
import shutil

annotation_file = "datasets/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
frames_dir = "datasets/ucf_crime/frames"
print("Reading annotation file...")
valid_videos = set()
with open(annotation_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()
            if parts:
                video_name = parts[0]
                folder_name = video_name.replace('.mp4', '')
                valid_videos.add(folder_name)
print(f"Found {len(valid_videos)} valid videos in annotation file")
print("\nScanning frames directory...")
all_folders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
print(f"Found {len(all_folders)} total subfolders in frames directory")
folders_to_delete = [f for f in all_folders if f not in valid_videos]
print(f"\nFolders to delete: {len(folders_to_delete)}")
print(f"Folders to keep: {len(all_folders) - len(folders_to_delete)}")
if folders_to_delete:
    print("\nSample folders to be deleted:")
    for folder in folders_to_delete[:5]:
        print(f"  - {folder}")
    if len(folders_to_delete) > 5:
        print(f"  ... and {len(folders_to_delete) - 5} more")
    response = input("\nDo you want to proceed with deletion? (yes/no): ")
    if response.lower() == 'yes':
        print("\nDeleting folders...")
        deleted_count = 0
        for folder in folders_to_delete:
            folder_path = os.path.join(frames_dir, folder)
            try:
                shutil.rmtree(folder_path)
                deleted_count += 1
                if deleted_count % 10 == 0:
                    print(f"  Deleted {deleted_count}/{len(folders_to_delete)} folders...")
            except Exception as e:
                print(f"  Error deleting {folder}: {e}")
        print(f"\nDeletion complete! Removed {deleted_count} folders.")
    else:
        print("\nDeletion cancelled.")
else:
    print("\nNo folders to delete. All subfolders are valid!")
print("\nFinal summary:")
remaining_folders = [f for f in os.listdir(frames_dir) if os.path.isdir(os.path.join(frames_dir, f))]
print(f"  Remaining folders: {len(remaining_folders)}")
