sudo apt-get remove libcurl4-openssl-dev
sudo apt-get remove libcurl4-openssl-dev*
sudo apt-get purge libcurl4-openssl-dev
sudo apt-get --purge autoremove libcurl4-openssl-dev
sudo apt-get purge --auto-remove libcurl4-openssl-dev
sudo dpkg --purge --force-depends libcurl4-openssl-dev
sudo dpkg --purge --force-all libcurl4-openssl-dev
sudo rm -rf ~/.unsloth
sed -i '/unsloth/d' ~/.bashrc
sed -i '/\.unsloth/d' ~/.bashrc
sed -i '/unsloth/d' ~/.zshrc
sed -i '/\.unsloth/d' ~/.zshrc
sudo apt-get remove unsloth
sudo apt-get remove unsloth*
sudo apt-get purge unsloth
sudo apt-get --purge autoremove unsloth
sudo apt-get purge --auto-remove unsloth
sudo dpkg --purge --force-depends unsloth
sudo dpkg --purge --force-all unsloth
sudo apt-get autoremove
sudo apt-get autoclean
