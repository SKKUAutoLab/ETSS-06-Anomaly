rm -f ~/.local/bin/cursor ~/.local/bin/agent
rm -rf ~/.cursor
rm -rf ~/.config/Cursor
source ~/.bashrc
source ~/.zshrc
sudo apt-get purge --auto-remove cursor
sudo apt-get remove cursor
sudo apt-get purge cursor
sudo apt remove --purge cursor
sudo dpkg --purge --force-all cursor
sudo apt purge cursor*
sudo apt-get autoremove
sudo apt-get autoclean
