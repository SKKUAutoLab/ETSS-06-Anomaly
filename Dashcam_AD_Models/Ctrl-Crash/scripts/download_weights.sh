mkdir -p stabilityai
cd stabilityai
git-lfs clone https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
cd ..
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt -P checkpoints
