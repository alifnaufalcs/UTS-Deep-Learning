#!/bin/bash
# Script satu perintah reproduksi UTS Deep Learning
set -e

# Berpindah ke root repositori jika dipanggil dari folder kode/
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

echo "======================================================================"
echo "UTS DEEP LEARNING - RUN ALL PIPELINE REPRODUCTION (Seed 42)"
echo "======================================================================"
python kode/run_all.py --seed 42 "$@"
