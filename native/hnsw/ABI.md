Mori HNSW C ABI

This document defines the shared-library interface exported by `module/mori_hnsw.so`.

Source of truth:

- Rust implementation: `native/hnsw/rust/src/lib.rs`
- Lua consumer: `module/hnsw.lua`
- Python consumer: `native/hnsw/python/mori_hnsw.py`

Opaque type

```c
typedef struct mori_hnsw_index mori_hnsw_index;
```

Space constants

```c
enum {
    MORI_HNSW_SPACE_L2 = 0,
    MORI_HNSW_SPACE_INNER_PRODUCT = 1,
    MORI_HNSW_SPACE_COSINE = 2
};
```

Distance semantics

- `L2`: returned `distance` is Euclidean distance.
- `COSINE`: returned `distance` is cosine distance in `[0, 2]` style, currently exposed to bindings as `1 - cosine_similarity`.
- `INNER_PRODUCT`: internal Rust distance is a monotonic positive transform of the dot product so it fits `hnsw_rs` constraints. Bindings must treat returned `distance` as internal-only and reconstruct user-facing similarity themselves.

Current binding similarity rules

- Python and Lua convert `distance` to `similarity` like this:
- `L2`: `similarity = -distance`
- `COSINE`: `similarity = 1 - distance`
- `INNER_PRODUCT`: `similarity = log((1 - distance) / distance)`

Error model

- Functions that return pointers return `NULL` on failure.
- Functions that return `int` use `1` for success and `0` for failure.
- `mori_hnsw_search` returns the number of filled results. On failure it returns `0` and sets an error string.
- Error strings are owned by the library and must not be freed by callers.
- Error pointers stay valid until the next API call mutates the corresponding error slot.

Error functions

```c
const char* mori_hnsw_global_last_error(void);
const char* mori_hnsw_last_error(const mori_hnsw_index* index);
```

- Global error is used for create/load/null-handle failures.
- Per-index error is used for operations on a valid handle.

Lifecycle

```c
mori_hnsw_index* mori_hnsw_create(
    int space,
    size_t dim,
    size_t max_elements,
    size_t m,
    size_t ef_construction,
    size_t random_seed,
    int allow_replace_deleted
);

mori_hnsw_index* mori_hnsw_load(
    const char* path,
    int space,
    size_t dim,
    size_t max_elements,
    int allow_replace_deleted
);

void mori_hnsw_destroy(mori_hnsw_index* index);
```

Create constraints

- `dim > 0`
- `max_elements > 0`
- `1 <= m <= 256`
- `space` must be one of the constants above
- `random_seed` is currently accepted for ABI compatibility but ignored
- `allow_replace_deleted` is currently accepted for ABI compatibility but ignored

Load constraints

- `path` must be valid UTF-8
- `dim > 0`
- `space` must match the index that was dumped
- if `max_elements == 0`, runtime capacity becomes at least the number of loaded points
- `allow_replace_deleted` is ignored

Persistence

```c
int mori_hnsw_save(mori_hnsw_index* index, const char* path);
```

- `path` is treated as a basename, not as a single binary file.
- Save writes:
- `<path>.hnsw.graph`
- `<path>.hnsw.data`
- Load expects the same basename.

Mutation

```c
int mori_hnsw_add(mori_hnsw_index* index, uint64_t label, const float* vector);
int mori_hnsw_mark_deleted(mori_hnsw_index* index, uint64_t label);
int mori_hnsw_unmark_deleted(mori_hnsw_index* index, uint64_t label);
int mori_hnsw_resize(mori_hnsw_index* index, size_t new_max_elements);
int mori_hnsw_set_ef(mori_hnsw_index* index, size_t ef);
```

Mutation constraints

- `vector` must point to exactly `dim` contiguous `float32` values
- labels are passed as `uint64_t`, but the Rust core currently stores them through `usize`
- `ef > 0`

Unsupported operations

- `mori_hnsw_mark_deleted`
- `mori_hnsw_unmark_deleted`
- `mori_hnsw_resize`

These currently return `0` with an explanatory error string because `hnsw_rs` does not provide matching behavior for this integration.

Search

```c
size_t mori_hnsw_search(
    mori_hnsw_index* index,
    const float* vector,
    size_t k,
    uint64_t* labels_out,
    float* distances_out
);
```

Search behavior

- `k == 0` returns `0` without error.
- `labels_out` must be non-null if `k > 0`.
- `distances_out` may be null; labels will still be written.
- The function writes up to `k` results and returns the actual count.
- Effective search breadth is `max(k, previously_set_ef)`.

Vector preprocessing

- `L2`: vectors are used as-is.
- `INNER_PRODUCT`: vectors are used as-is.
- `COSINE`: vectors are normalized to unit L2 norm during both insert and search.

Metadata

```c
size_t mori_hnsw_dim(const mori_hnsw_index* index);
size_t mori_hnsw_capacity(const mori_hnsw_index* index);
size_t mori_hnsw_count(const mori_hnsw_index* index);
size_t mori_hnsw_deleted_count(const mori_hnsw_index* index);
int mori_hnsw_space(const mori_hnsw_index* index);
```

Metadata notes

- `mori_hnsw_deleted_count` always returns `0` in the current implementation.
- `mori_hnsw_capacity` is the configured capacity tracked by the wrapper.
- `mori_hnsw_space` returns the original space enum for the handle.

Compatibility policy

- Preserve symbol names and argument order unless all bindings are updated together.
- Treat this file as the contract for non-Rust callers.
- When behavior changes, update this file in the same change as the Rust implementation.
