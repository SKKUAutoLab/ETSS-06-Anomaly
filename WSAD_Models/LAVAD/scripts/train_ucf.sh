bash scripts/00_extract_frames.sh
python delete_frames.py
python delete_annotations.py
rm -rf datasets/ucf_crime/annotations/test.txt
mv datasets/ucf_crime/annotations/cut_test.txt datasets/ucf_crime/annotations/test.txt
bash scripts/01_caption.sh
bash scripts/02_create_index.sh
bash scripts/03_clean_captions.sh
bash scripts/04_query_llm.sh
bash scripts/05_create_summary_index.sh
bash scripts/06_refine_anomaly_scores.sh
