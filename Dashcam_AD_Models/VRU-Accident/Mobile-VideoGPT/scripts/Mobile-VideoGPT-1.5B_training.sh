#!/bin/sh


export DATASET_DIR=/Mobile-VideoGPT/playground/data
export PYTHONPATH="./:$PYTHONPATH"

BASE_LLM_PATH=Qwen/Qwen2.5-1.5B-Instruct
VISION_TOWER=OpenGVLab/VideoMamba
PROJECTOR_TYPE=etp
OUTPUT_DIR_PATH=results/pretrain/etp_qwen2point5_1point5B_instrut_video_mamba


# Stage 1: Video projection pre-training

deepspeed mobilevideogpt/train/pretrain.py \
--deepspeed scripts/zero2.json \
--tune_mm_mlp_adapter True \
--model_name_or_path "$BASE_LLM_PATH" \
--version qwen2_instruct \
--dataset_use PRETRAINING \
--vision_tower "$VISION_TOWER" \
--mm_projector_type "$PROJECTOR_TYPE" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir $OUTPUT_DIR_PATH \
--num_train_epochs 2 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none \
--num_select_k_frames_in_chunk 4 \
--topk True

IMAGE_VISION_TOWER=openai/clip-vit-base-patch16
OUTPUT_DIR_PATH=results/pretrain/etp_qwen2point5_1point5B_instrut_clip_base

# Stage 2: Image projection pre-training

deepspeed mobilevideogpt/train/pretrain.py \
--deepspeed scripts/zero2.json \
--tune_image_mm_mlp_adapter True \
--model_name_or_path "$BASE_LLM_PATH" \
--version qwen2_instruct \
--dataset_use PRETRAINING \
--image_vision_tower "$IMAGE_VISION_TOWER" \
--image_mm_projector_type "$PROJECTOR_TYPE" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--bf16 True \
--output_dir $OUTPUT_DIR_PATH \
--num_train_epochs 2 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 1e-3 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none \
--num_select_k_frames_in_chunk 4 \
--topk True


PRETRAIN_VIDEO_MLP_PATH=results/pretrain/etp_qwen2point5_1point5B_instrut_video_mamba/mm_projector.bin
PRETRAIN_IMAGE_MLP_PATH=results/pretrain/etp_qwen2point5_1point5B_instrut_clip_base/mm_projector.bin
OUTPUT_DIR_PATH=results/mobilevideogpt_finetune_qwen2point5_1point5B

# Stage 3: Mobile-VideoGPT finetuning

deepspeed mobilevideogpt/train/train.py \
--lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
--deepspeed scripts/zero3.json \
--model_name_or_path "$BASE_LLM_PATH" \
--version qwen2_instruct \
--dataset_use MobileGPT \
--vision_tower "$VISION_TOWER" \
--image_vision_tower "$IMAGE_VISION_TOWER" \
--mm_projector_type "$PROJECTOR_TYPE" \
--image_mm_projector_type "$PROJECTOR_TYPE" \
--pretrain_mm_mlp_adapter "$PRETRAIN_VIDEO_MLP_PATH" \
--pretrain_image_mm_mlp_adapter "$PRETRAIN_IMAGE_MLP_PATH" \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir $OUTPUT_DIR_PATH \
--num_train_epochs 2 \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 2 \
--evaluation_strategy "no" \
--save_strategy "steps" \
--save_steps 50000 \
--save_total_limit 1 \
--learning_rate 2e-4 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type "cosine" \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 4 \
--lazy_preprocess True \
--report_to none \
--num_select_k_frames_in_chunk 4 \
--topk True
