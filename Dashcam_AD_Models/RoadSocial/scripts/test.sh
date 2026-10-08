python infer_on_roadsocial.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'Qwen2-VL-' --model_size 2 --gpu_id 0
python infer_on_roadsocial.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'llava-ov' --model_size 0.5 --gpu_id 0
python infer_on_roadsocial.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'llava-ov_ft' --model_size 'weights/LLAVA-OV-7B_RoadSocial_Finetuned' --gpu_id 0
python llmeval_roadsocial_tasks.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'Qwen2-VL-' --model_size 2
python llmeval_roadsocial_tasks.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'llava-ov' --model_size 0.5
python llmeval_roadsocial_tasks.py --qas_dir 'data/RoadSocial/sample/' --model_prefix 'llava-ov_ft' --model_size 'weights/LLAVA-OV-7B_RoadSocial_Finetuned'
