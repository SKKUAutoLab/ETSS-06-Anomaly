mkdir -p ckpts
cd ckpts
git-lfs clone https://huggingface.co/DAMO-NLP-SG/Video-LLaMA-2-7B-Finetuned
wget https://huggingface.co/Jiaqi-hkust/hawk/resolve/main/finetuned.pth
wget https://huggingface.co/Jiaqi-hkust/hawk/resolve/main/pretrained.pth
cd ..
