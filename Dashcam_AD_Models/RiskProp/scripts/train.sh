# train the snippet-level anticipation model
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 tools/dist_train.sh configs/predict_anomaly_snippet.py 2
# train the frame-level anticipation model
CUDA_VISIBLE_DEVICES=0,1 PORT=29500 tools/dist_train.sh configs/predict_anomaly_frame.py 2
