DATASET_PATH="datasets"
NAME="exp_controlnet_russia_crash"
OUT_DIR="outputs_controlnet/${NAME}"
PROJECT_NAME='car_crash'
WANDB_ENTITY='tdc2000'
PRETRAINED_MODEL_PATH="outputs/exp_russa_crash"
mkdir -p $OUT_DIR
SCRIPT_PATH=$0
SAVE_SCRIPT_PATH="${OUT_DIR}/train_scripts.sh"
cp $SCRIPT_PATH $SAVE_SCRIPT_PATH
echo "Saved script to ${SAVE_SCRIPT_PATH}"
CUDA_LAUNCH_BLOCKING=1 accelerate launch --config_file config/multi_gpu.yaml train_video_controlnet.py --run_name $NAME --data_root $DATASET_PATH --project_name $PROJECT_NAME --pretrained_model_name_or_path $PRETRAINED_MODEL_PATH --output_dir $OUT_DIR --variant fp16 --dataset_name russia_crash --train_batch_size 1 --learning_rate 4e-5 --checkpoints_total_limit 3 --checkpointing_steps 300 --checkpointing_time 10620 --gradient_accumulation_steps 5 --validation_steps 300 --enable_gradient_checkpointing --lr_scheduler constant --report_to wandb --seed 1234 --mixed_precision fp16 --clip_length 25 --fps 6 --min_guidance_scale 1.0 --max_guidance_scale 3.0 --noise_aug_strength 0.01 --num_demo_samples 15 --num_train_epochs 10 --dataloader_num_workers 0 --resume_from_checkpoint latest --wandb_entity $WANDB_ENTITY --train_H 256 --train_W 256 --use_action_conditioning --contiguous_bbox_masking_prob 0.75 --contiguous_bbox_masking_start_ratio 0.0 --val_on_first_step
