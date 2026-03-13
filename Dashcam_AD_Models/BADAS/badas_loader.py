import sys
import os
from pathlib import Path
from huggingface_hub import snapshot_download

def load_badas_model(device="cuda"):
    print("Downloading BADAS model...")
    repo_path = snapshot_download(repo_id="nexar-ai/nexight", allow_patterns=["src/*", "weights/*"])
    
    src_path = Path(repo_path) / "src"
    sys.path.insert(0, str(src_path))
    
    from models.vjepa import VJEPAModel
    checkpoint_path = Path(repo_path) / "weights" / "badas_open.pth"
    
    model = VJEPAModel(
        model_name="facebook/vjepa2-vitl-fpc16-256-ssv2",
        checkpoint_path=str(checkpoint_path),
        frame_count=16,
        img_size=224,
        window_stride=1,
        target_fps=8.0,
        use_sliding_window=True,
    )
    model.load()
    return model
