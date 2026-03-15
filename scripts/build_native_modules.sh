#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${ROOT_DIR}/scripts/build_simdc_math.sh"
"${ROOT_DIR}/scripts/build_hnsw_module.sh"

