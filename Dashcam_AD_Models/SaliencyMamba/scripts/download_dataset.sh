mkdir -p data
cd data
git-lfs clone https://huggingface.co/datasets/springyu/TrafficGaze
cd TrafficGaze
cat TrafficGaze.part*.zip > TrafficGaze.zip 
unzip TrafficGaze.zip
unzip fixdata.zip
cd ../..
