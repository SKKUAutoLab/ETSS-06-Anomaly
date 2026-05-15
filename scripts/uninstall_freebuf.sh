sudo npm uninstall -g freebuff
sudo rm -rf /usr/local/lib/node_modules/freebuff
sudo rm -rf /usr/lib/node_modules/freebuff
sudo rm -rf ~/.npm/_npx/*
sudo rm -rf ~/.npm/freebuff*
sudo rm -f /usr/local/bin/freebuff
sudo rm -f /usr/bin/freebuff
sudo npm cache clean --force
