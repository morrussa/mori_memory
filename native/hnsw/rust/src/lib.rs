use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock, RwLock};

use hnsw_rs::api::AnnT;
use hnsw_rs::prelude::{DistDot, DistL2, DistPtr, Hnsw, HnswIo};

const MORI_HNSW_SPACE_L2: i32 = 0;
const MORI_HNSW_SPACE_INNER_PRODUCT: i32 = 1;
const MORI_HNSW_SPACE_COSINE: i32 = 2;

type HnswL2 = Hnsw<'static, f32, DistL2>;
type HnswDot = Hnsw<'static, f32, DistDot>;
type HnswIp = Hnsw<'static, f32, DistPtr<f32, f32>>;

enum InnerIndex {
    L2(Mutex<HnswL2>),
    Ip(Mutex<HnswIp>),
    Dot(Mutex<HnswDot>),
}

pub struct MoriHnswIndex {
    space_kind: i32,
    normalize: bool,
    dim: usize,
    capacity: usize,
    ef_search: AtomicUsize,
    last_error: RwLock<CString>,
    inner: InnerIndex,
}

static GLOBAL_ERROR: OnceLock<RwLock<CString>> = OnceLock::new();

fn global_error() -> &'static RwLock<CString> {
    GLOBAL_ERROR.get_or_init(|| RwLock::new(cstring("")))
}

fn cstring(message: &str) -> CString {
    let sanitized = message.replace('\0', " ");
    CString::new(sanitized).unwrap_or_else(|_| CString::new("invalid error").unwrap())
}

fn set_global_error(message: &str) {
    if let Ok(mut guard) = global_error().write() {
        *guard = cstring(message);
    }
}

impl MoriHnswIndex {
    fn new(space_kind: i32, dim: usize, capacity: usize, inner: InnerIndex, normalize: bool) -> Self {
        MoriHnswIndex {
            space_kind,
            normalize,
            dim,
            capacity,
            ef_search: AtomicUsize::new(64),
            last_error: RwLock::new(cstring("")),
            inner,
        }
    }

    fn clear_error(&self) {
        if let Ok(mut guard) = self.last_error.write() {
            *guard = cstring("");
        }
        set_global_error("");
    }

    fn set_error(&self, message: &str) {
        if let Ok(mut guard) = self.last_error.write() {
            *guard = cstring(message);
        }
        set_global_error(message);
    }

    fn vector_from_ptr(&self, vector: *const f32) -> Result<Vec<f32>, String> {
        if vector.is_null() {
            return Err("vector is null".to_string());
        }
        let slice = unsafe { std::slice::from_raw_parts(vector, self.dim) };
        if !self.normalize {
            return Ok(slice.to_vec());
        }
        let mut out = slice.to_vec();
        let norm = out
            .iter()
            .map(|v| (*v as f64) * (*v as f64))
            .sum::<f64>()
            .sqrt();
        if norm > 0.0 {
            for item in &mut out {
                *item = (*item as f64 / norm) as f32;
            }
        }
        Ok(out)
    }

    fn current_count(&self) -> usize {
        match &self.inner {
            InnerIndex::L2(index) => index.lock().map(|idx| idx.get_nb_point()).unwrap_or(0),
            InnerIndex::Ip(index) => index.lock().map(|idx| idx.get_nb_point()).unwrap_or(0),
            InnerIndex::Dot(index) => index.lock().map(|idx| idx.get_nb_point()).unwrap_or(0),
        }
    }
}

fn catch_result<T, F>(f: F) -> Result<T, String>
where
    F: FnOnce() -> Result<T, String>,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .map_err(|_| "panic inside mori_hnsw".to_string())?
}

fn validate_space(space: i32) -> Result<bool, String> {
    match space {
        MORI_HNSW_SPACE_L2 => Ok(false),
        MORI_HNSW_SPACE_INNER_PRODUCT => Ok(false),
        MORI_HNSW_SPACE_COSINE => Ok(true),
        _ => Err("unsupported space".to_string()),
    }
}

fn compute_max_layer(max_elements: usize) -> usize {
    let _ = max_elements;
    16
}

fn split_dump_path(raw: &str) -> Result<(PathBuf, String), String> {
    if raw.is_empty() {
        return Err("path is empty".to_string());
    }
    let path = PathBuf::from(raw);
    let basename = path
        .file_name()
        .map(|name| name.to_string_lossy().to_string())
        .ok_or_else(|| "path has no basename".to_string())?;
    let dir = path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    Ok((dir, basename))
}

unsafe fn lift_loaded_hnsw_l2(index: Hnsw<'_, f32, DistL2>) -> HnswL2 {
    // HnswIo defaults to `ReloadOptions::default()`, which disables mmap.
    // In this mode the loaded index owns all vector data, so extending the
    // lifetime to 'static is sound for the boxed handle we return here.
    unsafe { std::mem::transmute::<Hnsw<'_, f32, DistL2>, HnswL2>(index) }
}

unsafe fn lift_loaded_hnsw_dot(index: Hnsw<'_, f32, DistDot>) -> HnswDot {
    // See the comment in `lift_loaded_hnsw_l2`.
    unsafe { std::mem::transmute::<Hnsw<'_, f32, DistDot>, HnswDot>(index) }
}

unsafe fn lift_loaded_hnsw_ip(index: Hnsw<'_, f32, DistPtr<f32, f32>>) -> HnswIp {
    // See the comment in `lift_loaded_hnsw_l2`.
    unsafe { std::mem::transmute::<Hnsw<'_, f32, DistPtr<f32, f32>>, HnswIp>(index) }
}

fn c_path(path: *const c_char) -> Result<String, String> {
    if path.is_null() {
        return Err("path is empty".to_string());
    }
    let raw = unsafe { CStr::from_ptr(path) };
    raw.to_str()
        .map(|text| text.to_string())
        .map_err(|_| "path is not valid UTF-8".to_string())
}

fn build_l2(max_elements: usize, m: usize, ef_construction: usize) -> HnswL2 {
    Hnsw::<f32, DistL2>::new(
        m,
        max_elements,
        compute_max_layer(max_elements),
        ef_construction.max(m),
        DistL2 {},
    )
}

fn build_dot(max_elements: usize, m: usize, ef_construction: usize) -> HnswDot {
    Hnsw::<f32, DistDot>::new(
        m,
        max_elements,
        compute_max_layer(max_elements),
        ef_construction.max(m),
        DistDot {},
    )
}

fn inner_product_distance(lhs: &[f32], rhs: &[f32]) -> f32 {
    let dot = lhs
        .iter()
        .zip(rhs.iter())
        .fold(0.0f64, |acc, (a, b)| acc + f64::from(*a) * f64::from(*b));
    // hnsw_rs assumes all distances are non-negative. Map inner product to a
    // strictly monotonic positive distance so ranking still matches max-dot.
    (1.0 / (1.0 + dot.exp())) as f32
}

fn build_ip(max_elements: usize, m: usize, ef_construction: usize) -> HnswIp {
    Hnsw::<f32, DistPtr<f32, f32>>::new(
        m,
        max_elements,
        compute_max_layer(max_elements),
        ef_construction.max(m),
        DistPtr::<f32, f32>::new(inner_product_distance),
    )
}

fn not_supported(handle: *mut MoriHnswIndex, message: &str) -> i32 {
    if handle.is_null() {
        set_global_error(message);
    } else {
        unsafe { (&*handle).set_error(message) };
    }
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_global_last_error() -> *const c_char {
    match global_error().read() {
        Ok(guard) => guard.as_ptr(),
        Err(_) => c"unknown error".as_ptr(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_last_error(index: *const MoriHnswIndex) -> *const c_char {
    if index.is_null() {
        return c"".as_ptr();
    }
    let handle = unsafe { &*index };
    match handle.last_error.read() {
        Ok(guard) => guard.as_ptr(),
        Err(_) => c"unknown error".as_ptr(),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_create(
    space: i32,
    dim: usize,
    max_elements: usize,
    m: usize,
    ef_construction: usize,
    _random_seed: usize,
    _allow_replace_deleted: i32,
) -> *mut MoriHnswIndex {
    match catch_result(|| {
        let normalize = validate_space(space)?;
        if dim == 0 {
            return Err("dim must be > 0".to_string());
        }
        if max_elements == 0 {
            return Err("max_elements must be > 0".to_string());
        }
        if m == 0 || m > 256 {
            return Err("m must be between 1 and 256 for hnsw_rs".to_string());
        }

        let inner = match space {
            MORI_HNSW_SPACE_L2 => InnerIndex::L2(Mutex::new(build_l2(max_elements, m, ef_construction))),
            MORI_HNSW_SPACE_INNER_PRODUCT => InnerIndex::Ip(Mutex::new(build_ip(max_elements, m, ef_construction))),
            MORI_HNSW_SPACE_COSINE => {
                InnerIndex::Dot(Mutex::new(build_dot(max_elements, m, ef_construction)))
            }
            _ => return Err("unsupported space".to_string()),
        };

        let handle = MoriHnswIndex::new(space, dim, max_elements, inner, normalize);
        handle
            .ef_search
            .store((m * 2).max(32), Ordering::Relaxed);
        Ok(Box::into_raw(Box::new(handle)))
    }) {
        Ok(ptr) => {
            set_global_error("");
            ptr
        }
        Err(message) => {
            set_global_error(&message);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_load(
    path: *const c_char,
    space: i32,
    dim: usize,
    max_elements: usize,
    _allow_replace_deleted: i32,
) -> *mut MoriHnswIndex {
    match catch_result(|| {
        let normalize = validate_space(space)?;
        if dim == 0 {
            return Err("dim must be > 0".to_string());
        }
        let path_string = c_path(path)?;
        let (dir, basename) = split_dump_path(&path_string)?;
        let mut loader = HnswIo::new(dir.as_path(), &basename);

        let handle = match space {
            MORI_HNSW_SPACE_L2 => {
                let loaded = loader
                    .load_hnsw::<f32, DistL2>()
                    .map_err(|err| err.to_string())?;
                let index: HnswL2 = unsafe { lift_loaded_hnsw_l2(loaded) };
                let actual_dim = index.get_point_indexation().get_data_dimension();
                if actual_dim != dim {
                    return Err(format!("dimension mismatch: dump has {}, caller passed {}", actual_dim, dim));
                }
                MoriHnswIndex::new(
                    space,
                    actual_dim,
                    max_elements.max(index.get_nb_point()),
                    InnerIndex::L2(Mutex::new(index)),
                    normalize,
                )
            }
            MORI_HNSW_SPACE_INNER_PRODUCT => {
                let loaded = loader
                    .load_hnsw_with_dist::<f32, DistPtr<f32, f32>>(DistPtr::<f32, f32>::new(inner_product_distance))
                    .map_err(|err| err.to_string())?;
                let index: HnswIp = unsafe { lift_loaded_hnsw_ip(loaded) };
                let actual_dim = index.get_point_indexation().get_data_dimension();
                if actual_dim != dim {
                    return Err(format!("dimension mismatch: dump has {}, caller passed {}", actual_dim, dim));
                }
                MoriHnswIndex::new(
                    space,
                    actual_dim,
                    max_elements.max(index.get_nb_point()),
                    InnerIndex::Ip(Mutex::new(index)),
                    normalize,
                )
            }
            MORI_HNSW_SPACE_COSINE => {
                let loaded = loader
                    .load_hnsw::<f32, DistDot>()
                    .map_err(|err| err.to_string())?;
                let index: HnswDot = unsafe { lift_loaded_hnsw_dot(loaded) };
                let actual_dim = index.get_point_indexation().get_data_dimension();
                if actual_dim != dim {
                    return Err(format!("dimension mismatch: dump has {}, caller passed {}", actual_dim, dim));
                }
                MoriHnswIndex::new(
                    space,
                    actual_dim,
                    max_elements.max(index.get_nb_point()),
                    InnerIndex::Dot(Mutex::new(index)),
                    normalize,
                )
            }
            _ => return Err("unsupported space".to_string()),
        };

        handle.ef_search.store(64, Ordering::Relaxed);
        Ok(Box::into_raw(Box::new(handle)))
    }) {
        Ok(ptr) => {
            set_global_error("");
            ptr
        }
        Err(message) => {
            set_global_error(&message);
            std::ptr::null_mut()
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_destroy(index: *mut MoriHnswIndex) {
    if !index.is_null() {
        unsafe {
            drop(Box::from_raw(index));
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_save(index: *mut MoriHnswIndex, path: *const c_char) -> i32 {
    if index.is_null() {
        set_global_error("index is null");
        return 0;
    }
    let handle = unsafe { &*index };
    match catch_result(|| {
        let path_string = c_path(path)?;
        let (dir, basename) = split_dump_path(&path_string)?;
        std::fs::create_dir_all(&dir).map_err(|err| err.to_string())?;
        match &handle.inner {
            InnerIndex::L2(inner) => inner
                .lock()
                .map_err(|_| "index lock poisoned".to_string())?
                .file_dump(dir.as_path(), &basename)
                .map_err(|err| err.to_string())?,
            InnerIndex::Ip(inner) => inner
                .lock()
                .map_err(|_| "index lock poisoned".to_string())?
                .file_dump(dir.as_path(), &basename)
                .map_err(|err| err.to_string())?,
            InnerIndex::Dot(inner) => inner
                .lock()
                .map_err(|_| "index lock poisoned".to_string())?
                .file_dump(dir.as_path(), &basename)
                .map_err(|err| err.to_string())?,
        };
        Ok(())
    }) {
        Ok(()) => {
            handle.clear_error();
            1
        }
        Err(message) => {
            handle.set_error(&message);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_add(index: *mut MoriHnswIndex, label: u64, vector: *const f32) -> i32 {
    if index.is_null() {
        set_global_error("index is null");
        return 0;
    }
    let handle = unsafe { &*index };
    match catch_result(|| {
        let data = handle.vector_from_ptr(vector)?;
        match &handle.inner {
            InnerIndex::L2(inner) => {
                let mut guard = inner.lock().map_err(|_| "index lock poisoned".to_string())?;
                guard.set_searching_mode(false);
                guard.insert((&data, label as usize));
            }
            InnerIndex::Ip(inner) => {
                let mut guard = inner.lock().map_err(|_| "index lock poisoned".to_string())?;
                guard.set_searching_mode(false);
                guard.insert((&data, label as usize));
            }
            InnerIndex::Dot(inner) => {
                let mut guard = inner.lock().map_err(|_| "index lock poisoned".to_string())?;
                guard.set_searching_mode(false);
                guard.insert((&data, label as usize));
            }
        }
        Ok(())
    }) {
        Ok(()) => {
            handle.clear_error();
            1
        }
        Err(message) => {
            handle.set_error(&message);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_mark_deleted(index: *mut MoriHnswIndex, _label: u64) -> i32 {
    not_supported(index, "mark_deleted is not supported by hnsw_rs")
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_unmark_deleted(index: *mut MoriHnswIndex, _label: u64) -> i32 {
    not_supported(index, "unmark_deleted is not supported by hnsw_rs")
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_resize(index: *mut MoriHnswIndex, _new_max_elements: usize) -> i32 {
    not_supported(index, "resize is not supported by hnsw_rs")
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_set_ef(index: *mut MoriHnswIndex, ef: usize) -> i32 {
    if index.is_null() {
        set_global_error("index is null");
        return 0;
    }
    let handle = unsafe { &*index };
    if ef == 0 {
        handle.set_error("ef must be > 0");
        return 0;
    }
    handle.ef_search.store(ef, Ordering::Relaxed);
    handle.clear_error();
    1
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_search(
    index: *mut MoriHnswIndex,
    vector: *const f32,
    k: usize,
    labels_out: *mut u64,
    distances_out: *mut f32,
) -> usize {
    if index.is_null() {
        set_global_error("index is null");
        return 0;
    }
    let handle = unsafe { &*index };
    match catch_result(|| {
        if k == 0 {
            return Ok(Vec::new());
        }
        if labels_out.is_null() {
            return Err("labels_out is null".to_string());
        }
        let data = handle.vector_from_ptr(vector)?;
        let ef = handle.ef_search.load(Ordering::Relaxed).max(k);
        let results = match &handle.inner {
            InnerIndex::L2(inner) => {
                let mut guard = inner.lock().map_err(|_| "index lock poisoned".to_string())?;
                guard.set_searching_mode(true);
                guard.search(&data, k, ef)
            }
            InnerIndex::Ip(inner) => {
                let mut guard = inner.lock().map_err(|_| "index lock poisoned".to_string())?;
                guard.set_searching_mode(true);
                guard.search(&data, k, ef)
            }
            InnerIndex::Dot(inner) => {
                let mut guard = inner.lock().map_err(|_| "index lock poisoned".to_string())?;
                guard.set_searching_mode(true);
                guard.search(&data, k, ef)
            }
        };
        Ok(results)
    }) {
        Ok(results) => {
            for (idx, neighbour) in results.iter().enumerate() {
                unsafe {
                    *labels_out.add(idx) = neighbour.d_id as u64;
                    if !distances_out.is_null() {
                        *distances_out.add(idx) = neighbour.distance;
                    }
                }
            }
            handle.clear_error();
            results.len()
        }
        Err(message) => {
            handle.set_error(&message);
            0
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_dim(index: *const MoriHnswIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    unsafe { (*index).dim }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_capacity(index: *const MoriHnswIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    unsafe { (*index).capacity }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_count(index: *const MoriHnswIndex) -> usize {
    if index.is_null() {
        return 0;
    }
    unsafe { (*index).current_count() }
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_deleted_count(_index: *const MoriHnswIndex) -> usize {
    0
}

#[unsafe(no_mangle)]
pub extern "C" fn mori_hnsw_space(index: *const MoriHnswIndex) -> i32 {
    if index.is_null() {
        return -1;
    }
    unsafe { (*index).space_kind }
}
