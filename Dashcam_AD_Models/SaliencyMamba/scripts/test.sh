cd utils
python restore_weight.py --inc_path ../ckpts/model_best_salmm_TrafficGaze_256x256_incomplete.tar --c_path model_best_salmm.tar
cd ..
mkdir -p ckpts/TrafficGaze
mv utils/model_best_salmm.tar ckpts/TrafficGaze/model_best_salmm.tar
python evaluate_metrics.py --network salmm --b 1 --g 0 --category TrafficGaze --root data/TrafficGaze --test_weight ./
