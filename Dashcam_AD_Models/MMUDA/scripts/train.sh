export NCCL_P2P_DISABLE=1
python train.py --name exp --source C --target D --train-num 1 --mix 2 --network O
