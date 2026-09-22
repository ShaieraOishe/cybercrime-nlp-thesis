#!/usr/bin/env bash
set -e

echo "========================================================="
echo "   Running All 7 Cybercrime Categories In One Command   "
echo "========================================================="

# Ensure virtual environment is active
source .venv/bin/activate

# Execute master dataset builder
python -m src.data.build_unified_dataset "$@"

echo "[✓] All 7 categories successfully generated, split, and prepared for retrieval!"
