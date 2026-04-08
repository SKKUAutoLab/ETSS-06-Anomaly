openclaw uninstall --dry-run
openclaw uninstall --all --yes --non-interactive
openclaw uninstall --all --yes
npm uninstall -g openclaw
sudo rm -rf ~/.openclaw
sudo rm -rf ~/.clawdbot
sudo rm -rf ~/.moltbot
sudo rm -rf ~/clawdbot
sudo rm -rf ~/.config/openclaw
sudo rm -rf ~/.local/share/openclaw
sudo rm -rf ~/.cache/openclaw
npm cache clean --force
sudo rm -f ~/.local/bin/openclaw
sudo rm -f /usr/local/bin/openclaw
sed -i '/openclaw completion/d' ~/.bashrc
sed -i '/openclaw completion/d' ~/.zshrc
systemctl --user stop openclaw-gateway 2>/dev/null || true
systemctl --user disable openclaw-gateway 2>/dev/null || true
sudo rm -f ~/.config/systemd/user/openclaw-gateway.service
systemctl --user daemon-reload
sudo apt-get remove openclaw
sudo apt-get remove openclaw*
sudo apt-get purge openclaw
sudo apt-get --purge autoremove openclaw
sudo apt-get purge --auto-remove openclaw
sudo dpkg --purge --force-depends openclaw
sudo dpkg --purge --force-all openclaw
sudo apt-get autoremove
sudo apt-get autoclean
