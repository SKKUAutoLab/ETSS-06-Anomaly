mimo uninstall --force
sudo apt-get remove mimo
sudo apt-get remove mimo*
sudo apt-get purge mimo
sudo apt-get --purge autoremove mimo
sudo apt-get purge --auto-remove mimo
sudo dpkg --purge --force-depends mimo
sudo dpkg --purge --force-all mimo
sudo apt-get autoremove
sudo apt-get autoclean
rm -rf ~/.mimocode
rm -f ~/.local/bin/mimo 2>/dev/null || true
rm -rf ~/.config/mimocode 2>/dev/null || true
rm -rf ~/.cache/mimocode 2>/dev/null || true
which mimo || echo "mimo command not found - good!"
mimo --version 2>/dev/null && echo "Still installed!" || echo "Successfully removed."
