import zipfile
import os
from pathlib import Path

zip_dir = "datasets/ucf_crime/videos"
extract_to = "datasets/ucf_crime/videos"
os.makedirs(extract_to, exist_ok=True)
for zip_path in Path(zip_dir).glob("*.zip"):
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        for member in zip_ref.namelist():
            if member.endswith("/"):
                continue
            parts = Path(member).parts
            if len(parts) > 1:
                relative_path = Path(*parts[1:])
            else:
                relative_path = Path(parts[0])
            target_path = Path(extract_to) / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_ref.open(member) as source, open(target_path, "wb") as target:
                target.write(source.read())
    print(f"✅ Extracted: {zip_path.name}")
