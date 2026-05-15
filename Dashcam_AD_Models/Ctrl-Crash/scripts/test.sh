python run_gen_videos.py --model_path outputs_controlnet/exp_controlnet_russia_crash/checkpoint-600 --output_path outputs_video --data_root datasets --num_demo_samples 100 --max_output_vids 100 --num_gens_per_sample 1 --eval_output --dataset russia_crash
python src/eval/video_quality_metrics_fvd_pair.py --vid_root outputs_video --samples 200 --num_frames 25 --downsample
