cd extracting
python ucf_gen_img.py --video_dir "../videos" --save_dir "../datasets/UCFCrime_dataset/vitl/rgb"
python ucf_gen_event.py --video_dir "../videos" --save_dir "../datasets/UCFCrime_dataset/vitl/event_thr_10" --clip_ckpt "../event_vitl.pt" 
cd ..