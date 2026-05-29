curl -fsSL https://raw.githubusercontent.com/brokechubb/cliproxyapi-installer/refs/heads/master/cliproxyapi-installer | bash -s -- uninstall
sudo systemctl stop cli-proxy-api 2>/dev/null
sudo systemctl disable cli-proxy-api 2>/dev/null
sudo rm -f /etc/systemd/system/cli-proxy-api.service
sudo systemctl daemon-reload
sudo rm -f /usr/local/bin/cli-proxy-api
sudo rm -rf ~/.cli-proxy-api
sudo rm -rf ~/cliproxyapi
sudo rm -rf /opt/cli-proxy-api
sudo rm -f ~/.config/cli-proxy-api*
which cli-proxy-api
