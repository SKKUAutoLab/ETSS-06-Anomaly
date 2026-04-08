sudo rm -rf ~/.bun
sudo rm -rf ~/.cache/bun
sudo rm -rf /root/.bun
sudo rm -rf ~/.cache/bun 2>/dev/null || true
sudo apt-get remove bun
sudo apt-get remove bun*
sudo apt-get purge bun
sudo apt-get --purge autoremove bun
sudo apt-get purge --auto-remove bun
sudo dpkg --purge --force-depends bun
sudo dpkg --purge --force-all bun
sudo apt-get autoremove
sudo apt-get autoclean
sed -i '/BUN_INSTALL/d' ~/.bashrc
sed -i '/\.bun\/bin/d' ~/.bashrc
sed -i '/BUN_INSTALL/d' ~/.zshrc
sed -i '/\.bun\/bin/d' ~/.zshrc
source ~/.bashrc
