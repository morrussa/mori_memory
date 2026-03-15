#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_SO="${ROOT_DIR}/module/simdc_math.so"
SRC_C="${ROOT_DIR}/native/simdc_math/simd_math.c"

mkdir -p "${ROOT_DIR}/module"

CC_BIN="${CC:-gcc}"

"${CC_BIN}" -O3 -shared -fPIC -mavx2 -mfma -o "${OUT_SO}" "${SRC_C}" -lm

echo "built ${OUT_SO}"
