rm -rf miniconda3
rm -rf ~/miniconda3
sudo rm -rf /opt/miniconda3
rm -rf ~/.condarc ~/.conda ~/.continuum
sudo -E /opt/miniconda3/uninstall.sh --remove-caches --remove-config-files user --remove-user-data
~/miniconda3/uninstall.sh --remove-caches --remove-config-files user --remove-user-data
sudo -E /opt/miniconda3/uninstall.sh
~/miniconda3/uninstall.sh
rm -r ~/miniconda/
rm -rf ~/.anaconda_backup
rm -rf ~/opt/anaconda3
