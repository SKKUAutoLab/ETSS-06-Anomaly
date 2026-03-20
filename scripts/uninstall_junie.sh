rm -f ~/.local/bin/junie
rm -rf ~/.local/share/junie
rm -rf ~/.junie
npm uninstall -g @jetbrains/junie-cli || true
which junie
junie --version
sudo apt-get remove junie
sudo apt-get remove junie*
sudo apt-get purge junie
sudo apt-get --purge autoremove junie
sudo apt-get purge --auto-remove junie
sudo dpkg --purge --force-depends junie
sudo dpkg --purge --force-all junie
sudo apt-get autoremove
sudo apt-get autoclean
