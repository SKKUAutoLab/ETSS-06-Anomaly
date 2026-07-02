import json
import os
from pathlib import Path

def replace_paths_in_json(file_path, old_path, new_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    modified_count = 0
    for item in data:
        if 'video' in item and old_path in item['video']:
            item['video'] = item['video'].replace(old_path, new_path)
            modified_count += 1
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"✓ Processed {file_path}")
    print(f"  Modified {modified_count} video paths")
    return modified_count

def main():
    annotation_dir = "datasets/hawk/Annotation/DoTA"
    old_path = "/remote-home/share/jiaqitang/Data/"
    new_path = ""
    json_files = ["DoTA.json"]
    total_modified = 0
    print(f"Starting path replacement...")
    print(f"Old path: {old_path}")
    print(f"New path: {new_path}")
    print(f"Directory: {annotation_dir}\n")
    for json_file in json_files:
        file_path = os.path.join(annotation_dir, json_file)
        if os.path.exists(file_path):
            count = replace_paths_in_json(file_path, old_path, new_path)
            total_modified += count
        else:
            print(f"⚠ File not found: {file_path}")
    print(f"\n{'='*50}")
    print(f"Total video paths modified: {total_modified}")
    print(f"Replacement complete!")

if __name__ == "__main__":
    main()
