sudo npm uninstall -g codebuff
sudo rm -rf /usr/local/lib/node_modules/codebuff
sudo rm -rf /usr/lib/node_modules/codebuff
sudo rm -rf ~/.npm/codebuff*
sudo rm -f /usr/local/bin/codebuff
sudo rm -f /usr/bin/codebuff
sudo npm cache clean --force
