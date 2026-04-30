mkdir -p ckpts
cd ckpts
git-lfs clone https://huggingface.co/mmaaz60/LLaVA-7B-Lightening-v1-1
git-lfs clone https://huggingface.co/openai/clip-vit-large-patch14
wget https://huggingface.co/MBZUAI/Video-ChatGPT-7B/resolve/main/video_chatgpt-7B.bin
cd ..
