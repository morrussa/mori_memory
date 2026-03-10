#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

from mori_hnsw import HNSWIndex


def main() -> None:
    index = HNSWIndex.create(dim=4, max_elements=32, ef_search=32)
    index.add(101, [1.0, 0.0, 0.0, 0.0])
    index.add(102, [0.0, 1.0, 0.0, 0.0])
    index.add(103, [0.0, 0.0, 1.0, 0.0])

    hits = index.search([0.99, 0.01, 0.0, 0.0], 2)
    assert hits, "expected search hits"
    assert hits[0].label == 101, f"unexpected best label: {hits[0]}"

    with tempfile.TemporaryDirectory(prefix="mori_hnsw_py_") as tmpdir:
        path = os.path.join(tmpdir, "index.bin")
        index.save(path)
        index.close()

        reloaded = HNSWIndex.load(path, dim=4, ef_search=32)
        hits2 = reloaded.search([0.98, 0.02, 0.0, 0.0], 2)
        assert hits2 and hits2[0].label == 101, f"unexpected reload result: {hits2}"
        print("python-hnsw-ok")

    ip_index = HNSWIndex.create(dim=4, max_elements=16, space="ip", ef_search=16)
    ip_index.add(301, [2.0, 0.0, 0.0, 0.0])
    ip_index.add(302, [1.0, 0.0, 0.0, 0.0])
    ip_hits = ip_index.search([3.0, 0.0, 0.0, 0.0], 2)
    assert ip_hits and ip_hits[0].label == 301, f"unexpected ip result: {ip_hits}"
    assert abs(ip_hits[0].similarity - 6.0) < 1e-5, f"unexpected ip similarity: {ip_hits[0]}"
    with tempfile.TemporaryDirectory(prefix="mori_hnsw_ip_py_") as tmpdir:
        path = os.path.join(tmpdir, "index.bin")
        ip_index.save(path)
        ip_index.close()

        ip_reloaded = HNSWIndex.load(path, dim=4, space="ip", ef_search=16)
        ip_hits2 = ip_reloaded.search([3.0, 0.0, 0.0, 0.0], 2)
        assert ip_hits2 and ip_hits2[0].label == 301, f"unexpected reloaded ip result: {ip_hits2}"
        assert abs(ip_hits2[0].similarity - 6.0) < 1e-5, f"unexpected reloaded ip similarity: {ip_hits2[0]}"
        ip_reloaded.close()


if __name__ == "__main__":
    main()
