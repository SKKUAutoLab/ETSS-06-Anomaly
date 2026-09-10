mkdir -p data/MM-AU
hf download JeffreyChou/MM-AU --repo-type dataset --local-dir data/MM-AU --include "CAP-DATA_chunks/**"
hf download JeffreyChou/MM-AU --repo-type dataset --local-dir data/MM-AU --include "DADA-2000_chunks/**"
