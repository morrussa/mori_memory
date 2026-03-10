from __future__ import annotations

import ctypes
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


SPACE_L2 = 0
SPACE_INNER_PRODUCT = 1
SPACE_COSINE = 2

_SPACE_NAMES = {
    "l2": SPACE_L2,
    "ip": SPACE_INNER_PRODUCT,
    "inner_product": SPACE_INNER_PRODUCT,
    "cosine": SPACE_COSINE,
}


@dataclass
class SearchResult:
    label: int
    distance: float
    similarity: float


def _default_lib_path() -> Path:
    return Path(__file__).resolve().parents[3] / "module" / "mori_hnsw.so"


def _space_value(space: str | int) -> int:
    if isinstance(space, int):
        return int(space)
    key = str(space or "cosine").strip().lower()
    if key not in _SPACE_NAMES:
        raise ValueError(f"unsupported space: {space}")
    return _SPACE_NAMES[key]


def _load_library(path: Optional[os.PathLike[str] | str] = None) -> ctypes.CDLL:
    lib_path = Path(path) if path else _default_lib_path()
    lib = ctypes.CDLL(os.fspath(lib_path))
    lib.mori_hnsw_global_last_error.restype = ctypes.c_char_p
    lib.mori_hnsw_create.argtypes = [
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.mori_hnsw_create.restype = ctypes.c_void_p
    lib.mori_hnsw_load.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_int,
    ]
    lib.mori_hnsw_load.restype = ctypes.c_void_p
    lib.mori_hnsw_destroy.argtypes = [ctypes.c_void_p]
    lib.mori_hnsw_destroy.restype = None
    lib.mori_hnsw_last_error.argtypes = [ctypes.c_void_p]
    lib.mori_hnsw_last_error.restype = ctypes.c_char_p
    lib.mori_hnsw_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    lib.mori_hnsw_save.restype = ctypes.c_int
    lib.mori_hnsw_add.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_float)]
    lib.mori_hnsw_add.restype = ctypes.c_int
    lib.mori_hnsw_mark_deleted.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.mori_hnsw_mark_deleted.restype = ctypes.c_int
    lib.mori_hnsw_unmark_deleted.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.mori_hnsw_unmark_deleted.restype = ctypes.c_int
    lib.mori_hnsw_resize.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mori_hnsw_resize.restype = ctypes.c_int
    lib.mori_hnsw_set_ef.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    lib.mori_hnsw_set_ef.restype = ctypes.c_int
    lib.mori_hnsw_search.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint64),
        ctypes.POINTER(ctypes.c_float),
    ]
    lib.mori_hnsw_search.restype = ctypes.c_size_t
    lib.mori_hnsw_dim.argtypes = [ctypes.c_void_p]
    lib.mori_hnsw_dim.restype = ctypes.c_size_t
    lib.mori_hnsw_capacity.argtypes = [ctypes.c_void_p]
    lib.mori_hnsw_capacity.restype = ctypes.c_size_t
    lib.mori_hnsw_count.argtypes = [ctypes.c_void_p]
    lib.mori_hnsw_count.restype = ctypes.c_size_t
    lib.mori_hnsw_deleted_count.argtypes = [ctypes.c_void_p]
    lib.mori_hnsw_deleted_count.restype = ctypes.c_size_t
    lib.mori_hnsw_space.argtypes = [ctypes.c_void_p]
    lib.mori_hnsw_space.restype = ctypes.c_int
    return lib


def _decode_error(raw: Optional[bytes]) -> str:
    return raw.decode("utf-8", errors="replace") if raw else ""


def _vector_buffer(vector: Iterable[float], dim: int) -> ctypes.Array[ctypes.c_float]:
    values = [float(x) for x in vector]
    if len(values) != dim:
        raise ValueError(f"expected vector dim {dim}, got {len(values)}")
    return (ctypes.c_float * dim)(*values)


def _similarity_from_distance(space: int, distance: float) -> float:
    if space == SPACE_L2:
        return -distance
    if space == SPACE_INNER_PRODUCT:
        if distance <= 0.0:
            return math.inf
        if distance >= 1.0:
            return -math.inf
        return math.log((1.0 - distance) / distance)
    return 1.0 - distance


class HNSWIndex:
    def __init__(self, handle: int, lib: ctypes.CDLL, space: int, dim: int):
        self._handle = ctypes.c_void_p(handle)
        self._lib = lib
        self.space = int(space)
        self.dim = int(dim)

    @classmethod
    def create(
        cls,
        dim: int,
        max_elements: int,
        *,
        space: str | int = "cosine",
        m: int = 16,
        ef_construction: int = 200,
        ef_search: Optional[int] = None,
        random_seed: int = 100,
        allow_replace_deleted: bool = False,
        lib_path: Optional[os.PathLike[str] | str] = None,
    ) -> "HNSWIndex":
        lib = _load_library(lib_path)
        space_value = _space_value(space)
        handle = lib.mori_hnsw_create(
            space_value,
            int(dim),
            int(max_elements),
            int(m),
            int(ef_construction),
            int(random_seed),
            1 if allow_replace_deleted else 0,
        )
        if not handle:
            raise RuntimeError(_decode_error(lib.mori_hnsw_global_last_error()) or "unknown error")
        index = cls(handle, lib, space_value, dim)
        if ef_search:
            index.set_ef(ef_search)
        return index

    @classmethod
    def load(
        cls,
        path: os.PathLike[str] | str,
        *,
        dim: int,
        space: str | int = "cosine",
        max_elements: int = 0,
        allow_replace_deleted: bool = False,
        ef_search: Optional[int] = None,
        lib_path: Optional[os.PathLike[str] | str] = None,
    ) -> "HNSWIndex":
        lib = _load_library(lib_path)
        space_value = _space_value(space)
        handle = lib.mori_hnsw_load(
            os.fsencode(path),
            space_value,
            int(dim),
            int(max_elements),
            1 if allow_replace_deleted else 0,
        )
        if not handle:
            raise RuntimeError(_decode_error(lib.mori_hnsw_global_last_error()) or "unknown error")
        index = cls(handle, lib, space_value, dim)
        if ef_search:
            index.set_ef(ef_search)
        return index

    def close(self) -> None:
        if self._handle:
            self._lib.mori_hnsw_destroy(self._handle)
            self._handle = ctypes.c_void_p()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _raise_last_error(self) -> None:
        raise RuntimeError(_decode_error(self._lib.mori_hnsw_last_error(self._handle)) or "unknown error")

    @property
    def count(self) -> int:
        return int(self._lib.mori_hnsw_count(self._handle))

    @property
    def deleted_count(self) -> int:
        return int(self._lib.mori_hnsw_deleted_count(self._handle))

    @property
    def capacity(self) -> int:
        return int(self._lib.mori_hnsw_capacity(self._handle))

    def set_ef(self, ef: int) -> None:
        if not self._lib.mori_hnsw_set_ef(self._handle, int(ef)):
            self._raise_last_error()

    def resize(self, new_max_elements: int) -> None:
        if not self._lib.mori_hnsw_resize(self._handle, int(new_max_elements)):
            self._raise_last_error()

    def add(self, label: int, vector: Iterable[float]) -> None:
        buf = _vector_buffer(vector, self.dim)
        if not self._lib.mori_hnsw_add(self._handle, int(label), buf):
            self._raise_last_error()

    def mark_deleted(self, label: int) -> None:
        if not self._lib.mori_hnsw_mark_deleted(self._handle, int(label)):
            self._raise_last_error()

    def unmark_deleted(self, label: int) -> None:
        if not self._lib.mori_hnsw_unmark_deleted(self._handle, int(label)):
            self._raise_last_error()

    def save(self, path: os.PathLike[str] | str) -> None:
        if not self._lib.mori_hnsw_save(self._handle, os.fsencode(path)):
            self._raise_last_error()

    def search(self, vector: Iterable[float], k: int) -> List[SearchResult]:
        k = int(k)
        if k <= 0:
            return []
        query = _vector_buffer(vector, self.dim)
        labels = (ctypes.c_uint64 * k)()
        distances = (ctypes.c_float * k)()
        found = int(self._lib.mori_hnsw_search(self._handle, query, k, labels, distances))
        out: List[SearchResult] = []
        for idx in range(found):
            dist = float(distances[idx])
            sim = _similarity_from_distance(self.space, dist)
            out.append(SearchResult(label=int(labels[idx]), distance=dist, similarity=sim))
        return out
