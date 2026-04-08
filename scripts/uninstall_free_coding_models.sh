npm uninstall -g free-coding-models
npm uninstall -g free-coding-models --force
sudo rm -rf ~/.free-coding-models
sudo rm -rf ~/.config/free-coding-models
sudo rm -rf ~/.local/share/free-coding-models
sudo rm -rf ~/.cache/free-coding-models
sudo rm -rf ~/.free-coding-models* 2>/dev/null || true
npm cache clean --force
ls ~/.local/bin/free-coding-models* 2>/dev/null || true
ls /usr/local/bin/free-coding-models* 2>/dev/null || true
sudo rm -f ~/.local/bin/free-coding-models*
sudo rm -f /usr/local/bin/free-coding-models*
