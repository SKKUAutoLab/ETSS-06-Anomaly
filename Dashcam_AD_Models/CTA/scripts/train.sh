export CUDA_VISIBLE_DEVICES=0
python train_classifier.py --consensus_type average --random_seed 17
python train_classifier.py --consensus_type linear --random_seed 3
python train_localization.py --architecture_type forward-SST
python train_localization.py --architecture_type backward-SST
python train_localization.py --architecture_type bi-SST
python train_localization.py --architecture_type SSTCN-SST --num_layers 10 --num_epochs 100
python train_localization.py --architecture_type SSTCN-Segmentation --num_layers 10
python train_localization.py --architecture_type MSTCN-Segmentation
python train_localization.py --architecture_type SSTCN-R-C3D