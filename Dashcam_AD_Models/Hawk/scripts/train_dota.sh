export NCCL_P2P_DISABLE=1
# pretrain 
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port='10000' train.py --cfg-path  configs/train_configs/stage1_pretrain.yaml
# finetune
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port='12001' train.py --cfg-path  configs/train_configs/stage2_finetune.yaml
