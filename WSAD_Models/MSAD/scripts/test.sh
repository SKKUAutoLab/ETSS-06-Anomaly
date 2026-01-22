cd RTFM
python test.py --test-rgb-list 'list/msad-i3d-test.list' --gt 'list/gt-MSAD-WS-new.npy' --testing-model '../datasets/MSAD_feature/ckpt/rtfm-msad-i3d-0.86.pkl' --dataset 'msad'
cd ..
cd MGFN
python test.py --testing-model ../datasets/MSAD_feature/ckpt/mgfn-msad-i3d-84.96.pkl  
cd ..