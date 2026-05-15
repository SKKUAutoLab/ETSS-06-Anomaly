pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118 xformers
pip install pip==24.0 setuptools==75.1.0
pip install accelerate==0.30.0 av==12.3.0 click termcolor decord==0.6.0 diffusers==0.27.2 fastapi==0.115.7 hydra-core==1.3.2 huggingface-hub==0.25.0 imageio==2.36.0 imageio-ffmpeg==0.5.1 joblib==1.4.2 matplotlib numpy==1.26.4 omegaconf==2.3.0 openai==1.60.2 opencv-python six==1.16.0 packaging==24.1 pandas==2.2.2 peft==0.10.0 protobuf==5.28.3 pydantic==2.10.6 pygments==2.18.0 pyparsing==3.2.0 pyquaternion==0.9.9 h5py pyyaml gdown tensorboard tensorboardX scikit-image scikit-learn scipy==1.13.1 seaborn sentencepiece==0.2.0 shapely==1.8.5.post1 timm==1.0.15 torchmetrics==1.5.1 transformers==4.40 ultralytics==8.3.61 wandb ftfy regex yapf==0.40.1 yacs easydict lap wandb[media] lpips
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
git clone https://github.com/facebookresearch/sam2.git 
cd sam2
# pip install -e . # for pytorch <= 2.1
python setup.py build_ext --inplace # for pytorch > 2.1
cd ..
