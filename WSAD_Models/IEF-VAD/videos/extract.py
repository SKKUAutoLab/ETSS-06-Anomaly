import zipfile
import os
import glob

special_zips = {"Testing_Normal_Videos.zip", "Training-Normal-Videos-Part-1.zip", "Training-Normal-Videos-Part-2.zip"}
normal_dir = "./Normal"
os.makedirs(normal_dir, exist_ok=True)
for zip_path in glob.glob("*.zip"):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        top_level = members[0].split('/')[0] if '/' in members[0] else ""
        for member in members:
            target_path = member
            if top_level and target_path.startswith(top_level + "/"):
                target_path = target_path[len(top_level) + 1:]
            if not target_path:
                continue
            if os.path.basename(zip_path) in special_zips:
                final_path = os.path.join(normal_dir, target_path)
            else:
                final_path = os.path.join("./", target_path)
            os.makedirs(os.path.dirname(final_path), exist_ok=True)
            if not member.endswith('/'):
                with zip_ref.open(member) as src, open(final_path, "wb") as dst:
                    dst.write(src.read())
print("✅ Extraction completed (special zips → Normal/, others → current dir).")
