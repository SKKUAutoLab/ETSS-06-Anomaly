import torch
import os
from pathlib import Path

output_dir = "pytorch_model.bin"
shards = []
shard_files = sorted([f for f in os.listdir(output_dir) if f.startswith("pytorch_model-") and f.endswith(".bin")])
state_dict = {}
for shard_file in shard_files:
    shard_path = os.path.join(output_dir, shard_file)
    shard_state = torch.load(shard_path, map_location="cpu")
    state_dict.update(shard_state)
torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
