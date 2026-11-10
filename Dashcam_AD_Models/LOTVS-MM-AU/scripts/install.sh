cd Detection/mmdetection
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install pip==24.0 setuptools==65.0 cython ninja
pip install numpy==1.22.0 scikit-learn scikit-image opencv-python h5py pyyaml gdown ftfy regex tensorboard tensorboardX wandb imageio pandas termcolor openmim huggingface_hub transformers==4.30
mim install mmengine
mim install mmcv==2.0.1
pip install -v -e .
cd ../..
