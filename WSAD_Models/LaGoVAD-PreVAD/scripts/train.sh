export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=`pwd`:$PYTHONPATH
export HTTPS_PROXY=http://127.0.0.1:7890
export MPLBACKEND=agg
export TOKENIZERS_PARALLELISM=false
python src/main.py fit -c src/configs/default.yaml --trainer.devices=1
