CUDA_VISIBLE_DEVICES=0,1 PORT=29501 tools/dist_test.sh configs/predict_anomaly_snippet.py work_dirs/predict_anomaly_snippet/epoch_2.pth 2
tools/dist_test.sh configs/predict_anomaly_snippet.py work_dirs/predict_anomaly_snippet/epoch_2.pth 2 --dump outputs/predictions.pkl
CUDA_VISIBLE_DEVICES=0,1 PORT=29501 tools/dist_test.sh configs/predict_anomaly_frame.py work_dirs/predict_anomaly_frame/epoch_2.pth 2
tools/dist_test.sh configs/predict_anomaly_frame.py work_dirs/predict_anomaly_frame/epoch_2.pth 2 --dump outputs/predictions.pkl
