import os
import shutil

raw_root = "data/raw"
dada_root = "data/DADA2000"

for sub in os.listdir(raw_root):
    raw_sub = os.path.join(raw_root, sub)
    dada_sub = os.path.join(dada_root, sub)

    if not os.path.isdir(raw_sub) or not os.path.exists(dada_sub):
        continue  # skip if folder not in both

    for subsub in os.listdir(raw_sub):
        raw_subsub = os.path.join(raw_sub, subsub)
        dada_subsub = os.path.join(dada_sub, subsub)

        if not os.path.isdir(raw_subsub) or not os.path.exists(dada_subsub):
            continue  # skip if subsubfolder not in both

        # Define image and map folders
        for folder in ["images", "maps"]:
            raw_folder = os.path.join(raw_subsub, folder)
            dada_folder = os.path.join(dada_subsub, folder)

            if os.path.exists(raw_folder) and os.path.exists(dada_folder):
                for file in os.listdir(raw_folder):
                    src_file = os.path.join(raw_folder, file)
                    dst_file = os.path.join(dada_folder, file)

                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)  # copy with metadata
                        print(f"Copied {src_file} -> {dst_file}")
