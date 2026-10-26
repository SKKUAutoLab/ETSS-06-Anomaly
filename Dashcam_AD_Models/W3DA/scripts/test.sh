cd logs/Attn-7b/2025_10_31_03_17/best_ckpt_model_epoch0
python zero_to_fp32.py . ../pytorch_model.bin --tag global_step500
cd ../../../..
cd logs/Attn-7b/2025_10_31_03_17
python ../../../merge_bin.py
cd ../../..
CUDA_VISIBLE_DEVICES=0 python merge_lora_weights_and_save_hf_model.py --version="ckpts/LLaVA-7B-Lightening-v1-1" --vision-tower="ckpts/clip-vit-large-patch14" --weight="logs/Attn-7b/2025_10_31_03_17/pytorch_model.bin" --save_path="ATTN-7B"
# deepspeed --master_port=24999 train_ds.py --version="ckpts/LLaVA-7B-Lightening-v1-1" --vision-tower="ckpts/clip-vit-large-patch14" --dataset_dir="dataset" --log_base_dir="logs" --val_dataset="DReyeVE||LBW||DADA" --val_batch_size=1 --eval_only --dataset="DReyeVE||LBW||DADA" --resume="ATTN-7B"
