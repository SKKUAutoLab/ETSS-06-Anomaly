from huggingface_hub import hf_hub_download
import sys
import os
loader_path = hf_hub_download(repo_id="nexar-ai/badas-open", filename="badas_loader.py")
sys.path.insert(0, os.path.dirname(loader_path))
from badas_loader import load_badas_model

model = load_badas_model()
predictions = model.predict("dashcam_video.mp4")
for frame_idx, prob in enumerate(predictions):
    if prob > 0.8:
        print(f"⚠️ Collision risk at frame {frame_idx}: {prob:.2%}")

