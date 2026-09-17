pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==65.0
pip install numpy==1.26.4 scikit-learn scikit-image h5py pyyaml gdown ftfy regex yapf==0.40.1 yacs easydict omegaconf tensorboard tensorboardX wandb termcolor matplotlib opencv-python tqdm einops six packaging timm transformers==4.40
git clone https://github.com/MzeroMiko/VMamba
cd VMamba
pip install -r requirements.txt
cd kernels/selective_scan 
pip install -e .
cd ../../..
pip install https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu11torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
pip install https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu11torch2.1cxx11abiFALSE-cp39-cp39-linux_x86_64.whl
