codex logout
sudo npm uninstall -g @openai/codex
sudo rm -rf ~/.codex
sudo rm -f ~/.npm-global/bin/codex* 2>/dev/null || true
sudo rm -f /usr/local/bin/codex* 2>/dev/null || true
sudo npm cache clean --force
sed -i '/codex/d' ~/.bashrc ~/.profile 2>/dev/null || true
sed -i '/codex/d' ~/.zshrc ~/.zprofile 2>/dev/null || true
which codex
codex --version
find ~ -name "*codex*" -o -name "*Codex*" 2>/dev/null
