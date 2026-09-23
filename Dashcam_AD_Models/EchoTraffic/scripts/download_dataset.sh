mkdir -p datasets
cd datasets
git-lfs clone https://huggingface.co/datasets/harryhsing/AV-TAU
cd AV-TAU/archives
mkdir -p ../videos
for f in videos_part{1..8}.tar.gz; do
    echo "Extracting $f ..."
    tar -xzf "$f" -C ../videos
done
cd ../../..
