rm -rf ~/.grok
rm -f ~/.local/bin/grok ~/.local/bin/agent
rm -f /usr/local/bin/grok /usr/local/bin/agent
sudo apt-get remove grok
sudo apt-get remove grok*
sudo apt-get purge grok
sudo apt-get --purge autoremove grok
sudo apt-get purge --auto-remove grok
sudo dpkg --purge --force-depends grok
sudo dpkg --purge --force-all grok
sudo apt-get autoremove
sudo apt-get autoclean
