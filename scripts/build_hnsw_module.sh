#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_SO="${ROOT_DIR}/module/mori_hnsw.so"
MANIFEST_PATH="${ROOT_DIR}/native/hnsw/rust/Cargo.toml"
TARGET_DIR="${ROOT_DIR}/target"
RUST_SO="${TARGET_DIR}/release/libmori_hnsw_ffi.so"

mkdir -p "${ROOT_DIR}/module"

CARGO_TARGET_DIR="${TARGET_DIR}" cargo build --release --manifest-path "${MANIFEST_PATH}"
cp "${RUST_SO}" "${OUT_SO}"

echo "built ${OUT_SO}"
