nvm deactivate
nvm uninstall 24
nvm uninstall --all
sudo rm -rf ~/.nvm
sed -i '/NVM_DIR/d' ~/.bashrc ~/.profile ~/.bash_profile 2>/dev/null || true
sed -i '/NVM_DIR/d' ~/.zshrc ~/.zprofile 2>/dev/null || true
sudo rm -rf ~/.npm
sudo rm -rf ~/.node-gyp
sudo rm -rf ~/.npmrc
sudo rm -rf ~/.nvm
sudo rm -rf /usr/local/bin/node
sudo rm -rf /usr/local/bin/npm
sudo rm -rf /usr/local/lib/node_modules
# clean nvm
sudo apt-get remove nvm
sudo apt-get remove nvm
sudo apt-get remove nvm
sudo apt-get purge nvm
sudo apt-get --purge autoremove nvm
sudo apt-get purge --auto-remove nvm
sudo dpkg --purge --force-depends nvm
sudo dpkg --purge --force-all nvm
# clean npm
sudo apt-get remove npm
sudo apt-get remove npm
sudo apt-get remove npm
sudo apt-get purge npm
sudo apt-get --purge autoremove npm
sudo apt-get purge --auto-remove npm
sudo dpkg --purge --force-depends npm
sudo dpkg --purge --force-all npm
# clean node
sudo apt-get remove node
sudo apt-get remove node
sudo apt-get remove node
sudo apt-get purge node
sudo apt-get --purge autoremove node
sudo apt-get purge --auto-remove node
sudo dpkg --purge --force-depends node
sudo dpkg --purge --force-all node
sudo apt-get autoremove
sudo apt-get autoclean
node -v
npm -v
nvm --version
find ~ -name "*node*" -o -name "*npm*" -o -name "*nvm*" 2>/dev/null
