sudo npm uninstall -g @google/gemini-cli
sudo apt-get purge -y nodejs
sudo apt-get autoremove -y
sudo rm /etc/apt/sources.list.d/nodesource.list
sudo apt update
rm -rf ~/.npm
rm -rf ~/.node-gyp
sudo rm -rf /root/.npm
sudo apt-get purge --auto-remove nodejs node npm nvm
sudo apt-get remove nodejs node npm nvm
sudo apt-get purge nodejs node npm nvm
sudo apt remove --purge nodejs node npm nvm
sudo dpkg --purge --force-all nodejs node npm nvm
sudo apt purge nodejs*
sudo apt purge node*
sudo apt purge npm*
sudo apt purge nvm*
sudo apt-get autoremove
sudo apt-get autoclean
