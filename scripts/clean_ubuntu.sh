# find -type f -name '*.git' -delete
# find -type f -name '*.gitignore' -delete
# find . -name __pycache__ -type d -exec rm -rf {} \;
# find . -name git -type d -exec rm -rf {} \;
# find . -name gitignore -type d -exec rm -rf {} \;
# find . -name .idea -type d -exec rm -rf {} \;
sudo rm -rf /tmp/*
sudo apt-get autoremove -y
sudo apt-get autoclean -y
sudo ap-get clean -y
sudo apt-get autopurge -y
dpkg -l | grep '^rc' | awk '{print $2}' | xargs -r sudo apt purge -y
rm -rf ~/.cache/thumbnails/*
rm -rf ~/.cache/*
sudo apt-get autoremove --purge
sudo rm -rf /var/crash/*
sudo rm -rf /var/lib/snapd/cache/*
sudo apt purge $(dpkg -l | awk '/^rc/ {print $2}')
sudo rm -rf /var/cache/apt/archives/*
rm -rf ~/.local/share/Trash/*
sudo find /var/log -type f -name "*.log" -mtime +30 -delete
sudo find /var/log -type f -name "*.gz" -mtime +60 -delete
sudo find /tmp -type f -atime +7 -delete
sudo snap list --all | awk '/disabled/{print $1, $3}' | while read snapname revision; do sudo snap remove "$snapname" --revision="$revision"; done
sudo rm -rf /var/cache/snapd/*
sudo journalctl --vacuum-time=7d
sudo sync && echo 1 | sudo tee /proc/sys/vm/drop_caches
sudo sync && echo 2 | sudo tee /proc/sys/vm/drop_caches
sudo sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo swapoff -a && sudo swapon -a
# sudo journalctl --vacuum-size=100M
# dpkg --list | grep linux-image
# sudo apt purge linux-image-5.15.0-73-generic
# sudo apt autoremove --purge $(dpkg -l 'linux-image-*' | grep '^ii' | awk '{print $2}' | grep -v $(uname -r))
