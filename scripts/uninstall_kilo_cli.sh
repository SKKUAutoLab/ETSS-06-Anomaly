kilo uninstall
sudo rm -f ~/.local/bin/kilo
sudo rm -f /usr/local/bin/kilo
sudo rm -f /usr/bin/kilo
sudo npm uninstall -g @kilocode/cli 2>/dev/null || true
sudo rm -rf ~/.kilo
sudo rm -rf ~/.config/kilo
sudo rm -rf ~/.cache/kilo
sudo rm -rf ~/.local/share/kilo
sudo rm -rf ~/.kilocode
sed -i '/kilo/d' ~/.bashrc ~/.profile ~/.bash_profile 2>/dev/null || true
sed -i '/kilo/d' ~/.zshrc ~/.zprofile 2>/dev/null || true
which kilo
kilo --version
find ~ -name "*kilo*" -o -name "*Kilo*" 2>/dev/null
sudo apt-get remove kilo
sudo apt-get remove kilo*
sudo apt-get purge kilo
sudo apt-get --purge autoremove kilo
sudo apt-get purge --auto-remove kilo
sudo dpkg --purge --force-depends kilo
sudo dpkg --purge --force-all kilo
sudo apt-get autoremove
sudo apt-get autoclean
