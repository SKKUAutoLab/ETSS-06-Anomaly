cd Models
git clone --recursive https://github.com/DAMO-NLP-SG/Video-LLaMA
git clone --recursive https://github.com/X-PLUG/mPLUG-Owl
git clone --recursive https://github.com/Luodian/Otter
git clone --recursive https://github.com/RenShuhuai-Andy/TimeChat
git clone --recursive https://github.com/yxuansu/PandaGPT
cd ..
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
pip install markupsafe==2.0.1
