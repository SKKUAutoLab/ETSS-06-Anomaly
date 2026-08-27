cd data
huggingface-cli download KDliang/CrashChat --repo-type dataset --include "CrashChat-resized/*" --local-dir ./
mv CrashChat-resized videos
huggingface-cli download KDliang/CrashChat --repo-type dataset --include "CrashChat-resized_02/*" --local-dir ./
mv CrashChat-resized_02/* videos/
rm -rf CrashChat-resized_02
cd ..
