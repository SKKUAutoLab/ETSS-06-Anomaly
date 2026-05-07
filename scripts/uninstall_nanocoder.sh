npm uninstall -g @nanocollective/nanocoder
sudo rm -rf /usr/local/lib/node_modules/@nanocollective/nanocoder
sudo rm -rf ~/.npm/_npx/*nanocoder*
sudo rm -f /usr/local/bin/nanocoder
npm cache clean --force
which nanocoder
