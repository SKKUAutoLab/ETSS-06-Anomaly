import os

temporal_annotation_file = "datasets/ucf_crime/annotations/Temporal_Anomaly_Annotation_for_Testing_Videos.txt"
test_file = "datasets/ucf_crime/annotations/test.txt"
output_file = "datasets/ucf_crime/annotations/cut_test.txt"
print("Reading temporal annotation file...")
valid_videos = set()
with open(temporal_annotation_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()
            if parts:
                video_name = parts[0]
                folder_name = video_name.replace('.mp4', '')
                valid_videos.add(folder_name)
print(f"Found {len(valid_videos)} valid videos in temporal annotation file")
print("\nReading test.txt...")
filtered_lines = []
kept_count = 0
skipped_count = 0
with open(test_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            parts = line.split()
            if parts:
                video_name = parts[0]
                if video_name in valid_videos:
                    filtered_lines.append(line)
                    kept_count += 1
                else:
                    skipped_count += 1
print(f"Lines to keep: {kept_count}")
print(f"Lines to skip: {skipped_count}")
print(f"\nWriting to {output_file}...")
with open(output_file, 'w') as f:
    for line in filtered_lines:
        f.write(line + '\n')
print(f"Done! Created {output_file} with {kept_count} lines.")
print("\nPreview of cut_test.txt (first 5 lines):")
for i, line in enumerate(filtered_lines[:5]):
    print(f"  {line}")
if len(filtered_lines) > 5:
    print(f"  ... and {len(filtered_lines) - 5} more lines")
