import os
import shutil

def replace_paths_in_file(file_path):
    backup_path = file_path + '.backup'
    shutil.copy2(file_path, backup_path)
    print(f"Created backup: {backup_path}")
    with open(file_path, 'r') as f:
        lines = f.readlines()
    modified_lines = []
    changes_made = 0
    for line in lines:
        original_line = line
        if '/scratch/kf09/lz1278/' in line:
            modified_line = line.replace('/scratch/kf09/lz1278/', '../datasets/MSAD_feature/')
            modified_lines.append(modified_line)
            changes_made += 1
            print(f"Changed: {original_line.strip()} -> {modified_line.strip()}")
        else:
            modified_lines.append(line)
    with open(file_path, 'w') as f:
        f.writelines(modified_lines)
    print(f"Modified {changes_made} lines in {file_path}")
    return changes_made

def main():
    files_to_modify = ['RTFM/list/msad-i3d-test.list', 'RTFM/list/msad-i3d.list', 'MGFN/data/msad/msad-i3d.list', 'MGFN/data/msad/msad-i3d-test.list']
    total_changes = 0
    for file_path in files_to_modify:
        if os.path.exists(file_path):
            print(f"\nProcessing {file_path}...")
            changes = replace_paths_in_file(file_path)
            total_changes += changes
        else:
            print(f"Warning: File not found - {file_path}")
    print(f"\nTotal changes made: {total_changes}")
    print("Backup files created with .backup extension")

if __name__ == "__main__":
    main()