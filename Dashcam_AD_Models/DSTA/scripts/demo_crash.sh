python demo.py --video_file demo/000007.mp4 --task extract_feature
python demo.py --task inference --feature_file demo/000007_feature.npz --ckpt_file demo/final_model_ccd.pth
python demo.py --task visualize --video_file demo/000007.mp4 --result_file demo/000007_result.npz --vis_file demo/000007_vis.avi
