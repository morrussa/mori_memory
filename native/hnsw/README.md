HNSW Native Layout

- Rust core: `native/hnsw/rust`
- Python binding implementation: `native/hnsw/python/mori_hnsw.py`
- Lua entrypoint: `module/hnsw.lua`
- ABI contract: `native/hnsw/ABI.md`
- Shared library output: `module/mori_hnsw.so`
- Build command: `./scripts/build_hnsw_module.sh`
- Python test: `python3 scripts/test_hnsw.py`
- Lua test: `luajit scripts/test_hnsw.lua`

Compatibility notes:

- Root `mori_hnsw.py` is only a shim for old imports.
- The Rust implementation is the source of truth for ABI.
