pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==75.0 six packaging
pip install -r requirements.txt
pip install numpy==1.21.5 scikit-learn scikit-image opencv-python h5py pyyaml gdown ftfy regex yapf==0.40.1 yacs easydict tqdm einops tensorboard tensorboardX wandb openmim imageio imageio-ffmpeg matplotlib pandas termcolor thop tabulate torchinfo torchsummary seaborn librosa numba xlrd
mim install mmengine==0.10.7
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html
mim install mmaction2==1.2.0
pip install -v -e .
