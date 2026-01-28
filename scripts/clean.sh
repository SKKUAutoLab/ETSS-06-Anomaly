find -type f -name '*.git' -delete
find -type f -name '*.gitignore' -delete
find . -name __pycache__ -type d -exec rm -rf {} \;
find . -name git -type d -exec rm -rf {} \;
find . -name gitignore -type d -exec rm -rf {} \;
rm -rf /tmp/*
