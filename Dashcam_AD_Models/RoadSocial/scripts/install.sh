pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate==1.0.1
pip install qwen-vl-utils[decord]
pip install flash_attn==2.7.3 --no-build-isolation
git clone https://github.com/LLaVA-VL/LLaVA-NeXT.git
cd LLaVA-NeXT
git checkout 00b5b84ce8675f62eb7bb4587810366ab3770613
git apply ../llavaov_pyproject.toml.patch
pip install -e ".[train]"
pip install --upgrade httpx
cd ..
pip install huggingface_hub
huggingface-cli login
mkdir -p data
cd data
git clone https://huggingface.co/datasets/chiragp26/RoadSocial
cd ..
