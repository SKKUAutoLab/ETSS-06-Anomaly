mkdir -p datasets/MM-AU
cd datasets/MM-AU
huggingface-cli download JeffreyChou/MM-AU --repo-type dataset --include "DADA-2000*" --local-dir ./
cd ../..
cd datasets/MM-AU/DADA-2000_chunks
cat DADA2000.part_* > DADA2000.tar.gz
tar -xzvf DADA2000.tar.gz
cd ../../..
