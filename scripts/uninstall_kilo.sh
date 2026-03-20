kilo uninstall
sudo npm uninstall -g @kilocode/cli
sudo rm -rf ~/.kilo
sudo rm -rf ~/.config/kilo
sudo rm -rf ~/.cache/kilo
sudo rm -rf ~/.local/share/kilo
sudo rm -f ~/.npm-global/bin/kilo* 2>/dev/null || true
sudo rm -f /usr/local/bin/kilo* 2>/dev/null || true
sudo npm cache clean --force
sed -i '/kilo/d' ~/.bashrc ~/.profile 2>/dev/null || true
sed -i '/kilo/d' ~/.zshrc ~/.zprofile 2>/dev/null || true
which kilo
kilo --version
find ~ -name "*kilo*" -o -name "*Kilo*" 2>/dev/null
