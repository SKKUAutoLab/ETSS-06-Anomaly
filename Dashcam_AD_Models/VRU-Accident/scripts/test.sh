mkdir -p results
python main.py --task VQA --mode evaluation --load_path Model_Response/VQA/Qwen2_VL_7B --save_path results/VQA/Qwen2_VL_7B_VQA_Acc.json --model Qwen2_VL_7B --dataset All
python main.py --task Dense_Captioning --dataset All --mode test --model InternVL25_4B
