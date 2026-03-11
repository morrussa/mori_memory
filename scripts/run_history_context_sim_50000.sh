#!/usr/bin/env bash
set -euo pipefail

# Standard behavior:
# - module/config.lua defaults for variant weights / recency / ratios
# - weighted multi-variant selection
# - effective history budget fixed to 1200 tokens
#   (roughly the previous 4800-char simulation window, but now token-based)
python3 scripts/history_context_variant_sim.py \
  --runs 50000 \
  --seed 42 \
  --history-pairs 12 \
  --input-token-budget 12000 \
  --history-budget-tokens 1200 \
  --recency-decay 0.90 \
  --weight-full 1.00 \
  --weight-slight 0.72 \
  --weight-heavy 0.40 \
  --weight-none 0.00 \
  --ratio-slight 0.65 \
  --ratio-heavy 0.30 \
  --selection-mode weighted
