npm uninstall -g @augmentcode/auggie
sudo rm -f /usr/local/bin/auggie
sudo rm -rf /usr/local/lib/node_modules/@augmentcode
npm cache clean --force
npm uninstall -g @augmentcode/auggie --force
which auggie
