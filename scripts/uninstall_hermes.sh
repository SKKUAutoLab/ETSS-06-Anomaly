hermes uninstall --full --yes
hermes uninstall
sudo rm -f ~/.local/bin/hermes
sudo rm -rf ~/.hermes/hermes-agent
sudo rm -rf ~/.hermes
sudo rm -rf ~/.cache/hermes
sudo apt-get remove hermes-agent
sudo apt-get remove hermes-agent*
sudo apt-get remove hermes*
sudo apt-get purge hermes-agent
sudo apt-get --purge autoremove hermes-agent
sudo apt-get purge --auto-remove hermes-agent
sudo dpkg --purge --force-depends hermes-agent
sudo dpkg --purge --force-all hermes-agent
sudo apt-get autoremove
sudo apt-get autoclean
hermes gateway stop 2>/dev/null || true
systemctl --user stop hermes-gateway 2>/dev/null || true
systemctl --user disable hermes-gateway 2>/dev/null || true
sudo rm -f ~/.config/systemd/user/hermes-gateway.service
systemctl --user daemon-reload
