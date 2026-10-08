# train with a single GPU
# CUDA_VISIBLE_DEVICES=0 python runner/main.py --fp16 --config "configs/train/poma/vst/instance_detection/vst,ins,fpn(1),prompt(1),rnn,vcl=8.py" --output "outputs/vst,ins,fpn(1),prompt(1),rnn,vcl=8/" --phase train --epoch -1
# train with multiple GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2  runner/main.py --distributed --fp16  --config "configs/train/poma/vst/instance_detection/vst,ins,fpn(1),prompt(1),rnn,vcl=8.py" --output "outputs/vst,ins,fpn(1),prompt(1),rnn,vcl=8/" --phase train --epoch -1
