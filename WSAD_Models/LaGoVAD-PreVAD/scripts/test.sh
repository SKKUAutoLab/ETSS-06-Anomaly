export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=`pwd`:$PYTHONPATH
export HTTPS_PROXY=http://127.0.0.1:7890
python src/full_length_eval.py --config "ckpts/config.yaml" --ckpt "ckpts/best.ckpt" --dataset ucf --data_root data/PreVAD/
python src/offline_evals/offline_ucf_eval.py
python src/offline_evals/offline_xd_eval.py
python src/offline_evals/offline_dota_eval.py
