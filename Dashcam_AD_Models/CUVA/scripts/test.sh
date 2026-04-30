export PYTHONPATH="./:$PYTHONPATH"
cd Models/Video-ChatGPT/video_chatgpt/CUVA
bash inference_CUVA.sh
bash mmEval_demo.sh
cd ../../../..
