import csv
import shutil

# List of files to process in-place
files_to_process = [
    "list/ucf_CLIP_rgb.csv",
    "list/ucf_CLIP_rgbtest.csv",
    "list/xd_CLIP_rgb.csv",
    "list/xd_CLIP_rgbtest.csv",
]

for file_path in files_to_process:
    # Create a backup (just in case)
    backup_path = file_path + ".bak"
    shutil.copyfile(file_path, backup_path)
    
    # Temporary file for writing the fixed content
    temp_path = file_path + ".tmp"
    
    with open(file_path, 'r', newline='', encoding='utf-8') as infile, \
         open(temp_path, 'w', newline='', encoding='utf-8') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        for row in reader:
            if not row:
                writer.writerow(row)
                continue
                
            original_path = row[0].strip()
            
            # Replace both prefixes
            new_path = original_path.replace("/data/VAD/UCFCrime", "datasets") \
                                   .replace("/data/VAD/XD", "datasets")
            
            row[0] = new_path
            writer.writerow(row)
    
    # Overwrite the original file with the fixed one
    shutil.move(temp_path, file_path)
    
    print(f"Updated in-place: {file_path} (backup saved as {backup_path})")

print("All files have been directly updated!")
