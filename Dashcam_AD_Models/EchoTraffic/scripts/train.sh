# pre-training
torchrun --nproc_per_node=2 train.py --cfg-path train_configs/video_audio_pretrain.yaml
# SFT
torchrun --nproc_per_node=2 train.py --cfg-path train_configs/video_audio_finetune.yaml
