#!/bin/sh

MODEL_PATH=$1 # The path of the pre-trained model for Mobile-VideoGPT-0.5B or Mobile-VideoGPT-1.5B
export PYTHONPATH="../:./:$PYTHONPATH" # The python path for lmms-eval and Mobile-VideoGPT
export SLURM_NTASKS=8  # The number of available GPUs

# MVBench
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
	--model mobile_videogpt \
	--model_args "pretrained=$MODEL_PATH" \
	--tasks "mvbench" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix mobile_videogpt \
	--output_path ./logs/

# MLVU
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
        --model mobile_videogpt \
        --model_args "pretrained=$MODEL_PATH" \
        --tasks "mlvu" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix mobile_videogpt \
        --output_path ./logs/

# NeXt_QA
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
	--model mobile_videogpt \
	--model_args "pretrained=$MODEL_PATH" \
	--tasks "nextqa_mc_test" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix mobile_videogpt \
	--output_path ./logs/

# EgoSchema
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
	--model mobile_videogpt \
	--model_args "pretrained=$MODEL_PATH" \
	--tasks "egoschema" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix mobile_videogpt \
	--output_path ./logs/

# ActivityNetQA
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
	--model mobile_videogpt \
	--model_args "pretrained=$MODEL_PATH" \
	--tasks "activitynetqa" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix mobile_videogpt \
	--output_path ./logs/ \

# PerceptionTest
torchrun --nproc_per_node $SLURM_NTASKS lmms_eval/__main__.py \
	--model mobile_videogpt \
	--model_args "pretrained=$MODEL_PATH" \
	--tasks "perceptiontest_val_mc" \
	--batch_size 1 \
	--log_samples \
	--log_samples_suffix mobile_videogpt \
	--output_path ./logs/ \
