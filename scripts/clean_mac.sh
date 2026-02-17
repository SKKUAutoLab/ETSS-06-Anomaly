sudo rm -rf ~/.Trash/*
sudo rm -rf ~/Library/Caches/*
du -sh /Library/Caches ~/Library/Caches /var/log /private/var/tmp /private/tmp
sudo rm -rf /Library/Caches/* ~/Library/Logs/* /var/log/* /private/var/tmp/* /private/tmp/*
sudo rm -rf ~/Library/Caches/* ~/Library/Logs/* ~/.Trash/* \
&& sudo rm -rf /Library/Caches/* /var/log/* /private/var/tmp/* /private/tmp/*
sudo rm -rf /private/var/log/*
sudo purge
tmutil thinlocalsnapshots / 9999999999999 4 # or tmutil deletelocalsnapshots 2026-02-10-123456
brew update && brew upgrade && brew cleanup && brew autoremove && brew cleanup --prune=all
brew cleanup -s
rm -rf "$(brew --cache)"
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder
rm -rf ~/Library/Application\ Support/MobileSync/Backup/*
rm -rf ~/Library/Developer/Xcode/DerivedData/*
rm -rf ~/Library/Developer/Xcode/Archives/*
rm -rf ~/Library/Developer/Xcode/iOS\ DeviceSupport/*
sudo rm -rf /private/var/vm/swapfile*
rm -rf ~/Library/Developer/CoreSimulator/Caches/*
sudo rm -rf /private/var/folders/*
rm -rf ~/Library/Caches/CocoaPods
rm -rf /Library/Logs/DiagnosticReports/old
sudo rm -rf /Volumes/*/.Trashes/*
sudo atsutil databases -remove
sudo rm -rf /tmp/*

### prevent sleepimage from coming back ###
# sudo rm /private/var/vm/sleepimage
# sudo pmset -a hibernatemode 0
# sudo rm /private/var/vm/sleepimage
# sudo touch /private/var/vm/sleepimage
# sudo chflags uchg /private/var/vm/sleepimage
### prevent sleepimage from coming back ###

### uninstall homebrew ###
# /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/uninstall.sh)"
# sudo rm -rf /opt/homebrew
# sudo rm -rf /etc/paths.d/homebrew
# sudo rm -rf ~/Library/Caches/Homebrew/
# sudo rm -rf ~/Library/Logs/Homebrew/
### uninstall homebrew ###

# docker system prune -a --volumes
# npm cache clean --force
# pip cache purge
# yarn cache clean