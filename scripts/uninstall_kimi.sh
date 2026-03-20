rm -rf ~/.kimi
sed -i '/kimi/d' ~/.bashrc ~/.profile 2>/dev/null || true
sed -i '/kimi/d' ~/.zshrc ~/.zprofile 2>/dev/null || true
uv tool uninstall kimi-cli
sudo apt-get remove kimi
sudo apt-get remove kimi*
sudo apt-get remove kimi*
sudo apt-get purge kimi
sudo apt-get --purge autoremove kimi
sudo apt-get purge --auto-remove kimi
sudo dpkg --purge --force-depends kimi
sudo dpkg --purge --force-all kimi
sudo apt-get autoremove
sudo apt-get autoclean
