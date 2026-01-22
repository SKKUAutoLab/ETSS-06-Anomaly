cd RTFM
python main.py --feat-extractor 'i3d' --feature-size 2048 --rgb-list 'list/msad-i3d.list' --test-rgb-list 'list/msad-i3d-test.list' --gt default='list/gt-MSAD-WS-new.npy' --dataset 'msad'
cd ..
cd MGFN
python main.py 
cd ..