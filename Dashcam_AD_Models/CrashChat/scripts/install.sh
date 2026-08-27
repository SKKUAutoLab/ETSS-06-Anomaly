pip install torch==2.4.0 torchvision==0.19.0 --extra-index-url https://download.pytorch.org/whl/cu118
pip install pip==24.0 cython ninja setuptools==65.0
pip install -r requirements.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda install -c conda-forge ffmpeg -y
