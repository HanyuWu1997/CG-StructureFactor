#!/usr/bin/env bash
set -euo pipefail

python3 cg_structure_factor.py \
  --data-path ./TEST \
  --qmax 1.20 \
  --bin-size 0.005 \
  --peak-qmax 0.60 \
  --arch cuda
