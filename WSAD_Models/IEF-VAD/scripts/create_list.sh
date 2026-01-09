cd list
python ucf_generate_file.py --base_path "../datasets/UCFCrime_dataset/vitl/rgb" --train_save_path "ucf/rgb/vitl/train.csv" --test_save_path "ucf/rgb/vitl/test.csv"
python ucf_generate_file.py --base_path "../datasets/UCFCrime_dataset/vitl/event_thr_10" --train_save_path "ucf/event/vitl/thr_10/train.csv" --test_save_path "ucf/event/vitl//thr_10/test.csv"
python ucf_generate_gt.py --csv_path "ucf/rgb/vitl/test.csv" --save_path "ucf/rgb/vitl/gt.npy"
python ucf_generate_gt.py --csv_path "ucf/event/vitl/thr_10/test.csv" --save_path "ucf/event/vitl/thr_10/gt.npy"
cd ..