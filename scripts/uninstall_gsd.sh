npx get-shit-done-cc@latest --uninstall
npx get-shit-done-cc@latest --claude --global --uninstall
npx get-shit-done-cc@latest --claude --local --uninstall
npx get-shit-done-cc --global --uninstall
sudo rm -rf ~/.npm/_npx
sudo npm cache clean --force
sudo rm -rf ~/.claude/skills/gsd-*
sudo rm -rf ~/.codex/skills/gsd-*
sudo rm -rf ~/.config/opencode/gsd-*
sudo rm -rf ~/.gemini/skills/gsd-*
sudo rm -f ~/.claude/gsd-file-manifest.json
sudo rm -f ~/.codex/gsd-file-manifest.json
npm uninstall -g get-shit-done-cc 2>/dev/null || true
