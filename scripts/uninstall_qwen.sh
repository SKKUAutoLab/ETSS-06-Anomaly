sudo npm uninstall -g @qwen-code/qwen-code
rm -rf ~/.qwen
npm cache clean --force
rm -rf "$(npm config get cache)/_npx"
sudo apt-get remove qwen-code
sudo apt-get remove qwen-code*
sudo apt-get remove qwen*
sudo apt-get purge qwen-code
sudo apt-get --purge autoremove qwen-code
sudo apt-get purge --auto-remove qwen-code
sudo dpkg --purge --force-depends qwen-code
sudo dpkg --purge --force-all qwen-code
sudo apt-get autoremove
sudo apt-get autoclean
