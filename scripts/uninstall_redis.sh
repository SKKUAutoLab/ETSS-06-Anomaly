sudo systemctl stop redis-server
sudo systemctl disable redis-server
sudo apt-get purge --auto-remove redis redis-server redis-tools
sudo apt-get purge --auto-remove redis-*
sudo rm -rf /etc/redis/
sudo rm -rf /var/lib/redis/
sudo rm -rf /var/log/redis/
sudo rm -f /etc/apt/sources.list.d/redis.list
sudo rm -f /usr/share/keyrings/redis-archive-keyring.gpg
sudo apt-get update
sudo apt-get clean
sudo apt-get autoclean
