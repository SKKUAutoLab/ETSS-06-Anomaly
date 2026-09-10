set -e
DATA_DIR="data/MM-AU"
cd "$DATA_DIR"
echo "Processing CAP-DATA..."
mkdir -p CAP-DATA_temp
for dir in CAP-DATA_chunks/*/; do
    prefix=$(basename "$dir")
    echo "  Merging $prefix..."
    cat "$dir"*.part_* | tar -xzf - -C CAP-DATA_temp/
done
mv CAP-DATA_temp/CAP-DATA CAP-DATA
mv cap_text_annotations.xls CAP-DATA/
rm -rf CAP-DATA_temp
echo "Cleaning up CAP-DATA chunks..."
rm -rf CAP-DATA_chunks
echo "Processing DADA-2000..."
mkdir -p DADA-DATA_temp
echo "  Merging DADA-2000..."
cat DADA-2000_chunks/DADA2000.part_* | tar -xzf - -C DADA-DATA_temp/
mv DADA-DATA_temp/* DADA-DATA/ 2>/dev/null || mv DADA-DATA_temp/DADA-2000 DADA-DATA
mv dada_text_annotations.xlsx DADA-DATA/
rm -rf DADA-DATA_temp
echo "Cleaning up DADA-2000 chunks..."
rm -rf DADA-2000_chunks
echo "Done! Dataset organized as per org.txt"
python process_dataset.py
