mkdir -p data
cd data
hf download JeffreyChou/MM-AU --repo-type dataset --include "MMAU_Detv1*" --local-dir ./
cd MMAU_Detv1/images/train_chunks
cat train_part_* > train.tar.gz
tar -xzvf train.tar.gz
cd ../val_chunks
cat val_part_* > val.tar.gz
tar -xzvf val.tar.gz
cd ../test_chunks
cat test_part_* > test.tar.gz
tar -xzvf test.tar.gz
cd ../../../..
mv data/MMAU_Detv1 data/MM-AU-Detect
