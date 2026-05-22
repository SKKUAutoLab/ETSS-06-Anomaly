sudo rm -f ~/.local/bin/agy
sudo rm -f ~/.local/bin/antigravity
sudo rm -f /usr/local/bin/agy
sudo rm -f /usr/local/bin/antigravity
sudo rm -rf ~/.antigravity
sudo rm -rf ~/.config/antigravity
sudo rm -rf ~/.cache/antigravity
sudo rm -rf ~/.local/share/antigravity
sed -i '/antigravity/d' ~/.bashrc
sed -i '/agy/d' ~/.bashrc
source ~/.bashrc
which agy || echo "agy not found"
which antigravity || echo "antigravity not found"
sudo apt purge antigravity
sudo rm -f /etc/apt/sources.list.d/*antigravity*
sudo rm -f /usr/share/keyrings/*antigravity*
sudo apt update
