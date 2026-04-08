npm uninstall -g @gitlawb/openclaude
npm uninstall -g @gitlawb/openclaude --force
sudo rm -rf ~/.openclaude
sudo rm -rf ~/.config/openclaude
sudo rm -rf ~/.cache/openclaude
sudo rm -rf ~/.openclaude* 2>/dev/null || true
npm cache clean --force
ls ~/.local/bin/openclaude* 2>/dev/null || true
ls /usr/local/bin/openclaude* 2>/dev/null || true
sudo rm -f ~/.local/bin/openclaude*
sudo rm -f /usr/local/bin/openclaude*
