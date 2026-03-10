local ffi = require("ffi")

ffi.cdef[[
typedef struct mori_hnsw_index mori_hnsw_index;

enum {
    MORI_HNSW_SPACE_L2 = 0,
    MORI_HNSW_SPACE_INNER_PRODUCT = 1,
    MORI_HNSW_SPACE_COSINE = 2
};

const char* mori_hnsw_global_last_error(void);
const char* mori_hnsw_last_error(const mori_hnsw_index* index);

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
int mori_hnsw_save(mori_hnsw_index* index, const char* path);
int mori_hnsw_add(mori_hnsw_index* index, uint64_t label, const float* vector);
int mori_hnsw_mark_deleted(mori_hnsw_index* index, uint64_t label);
int mori_hnsw_unmark_deleted(mori_hnsw_index* index, uint64_t label);
int mori_hnsw_resize(mori_hnsw_index* index, size_t new_max_elements);
int mori_hnsw_set_ef(mori_hnsw_index* index, size_t ef);
size_t mori_hnsw_search(
    mori_hnsw_index* index,
    const float* vector,
    size_t k,
    uint64_t* labels_out,
    float* distances_out
);
size_t mori_hnsw_dim(const mori_hnsw_index* index);
size_t mori_hnsw_capacity(const mori_hnsw_index* index);
size_t mori_hnsw_count(const mori_hnsw_index* index);
size_t mori_hnsw_deleted_count(const mori_hnsw_index* index);
int mori_hnsw_space(const mori_hnsw_index* index);
]]

local M = {
    SPACE_L2 = 0,
    SPACE_INNER_PRODUCT = 1,
    SPACE_COSINE = 2,
}

local lib = ffi.load("./module/mori_hnsw.so")

local Index = {}
Index.__index = Index

local function last_error(handle)
    if handle ~= nil then
        local raw = lib.mori_hnsw_last_error(handle)
        if raw ~= nil then
            local text = ffi.string(raw)
            if text ~= "" then
                return text
            end
        end
    end
    local global_raw = lib.mori_hnsw_global_last_error()
    if global_raw == nil then
        return "unknown error"
    end
    local text = ffi.string(global_raw)
    if text == "" then
        return "unknown error"
    end
    return text
end

local function vector_buffer(vec, dim)
    if type(vec) ~= "table" then
        error("vector must be a Lua table")
    end
    if #vec ~= dim then
        error(string.format("expected vector dim %d, got %d", dim, #vec))
    end
    local buf = ffi.new("float[?]", dim)
    for i = 1, dim do
        buf[i - 1] = tonumber(vec[i]) or 0.0
    end
    return buf
end

local function similarity_from_distance(space, distance)
    if space == M.SPACE_L2 then
        return -distance
    end
    if space == M.SPACE_INNER_PRODUCT then
        if distance <= 0.0 then
            return math.huge
        end
        if distance >= 1.0 then
            return -math.huge
        end
        return math.log((1.0 - distance) / distance)
    end
    return 1.0 - distance
end

local function wrap_handle(handle)
    local obj = {
        _handle = ffi.gc(handle, lib.mori_hnsw_destroy),
        dim = tonumber(lib.mori_hnsw_dim(handle)) or 0,
        space = tonumber(lib.mori_hnsw_space(handle)) or M.SPACE_COSINE,
    }
    return setmetatable(obj, Index)
end

function M.new(opts)
    opts = type(opts) == "table" and opts or {}
    local handle = lib.mori_hnsw_create(
        tonumber(opts.space) or M.SPACE_COSINE,
        tonumber(opts.dim) or 0,
        tonumber(opts.max_elements) or 0,
        tonumber(opts.m) or 16,
        tonumber(opts.ef_construction) or 200,
        tonumber(opts.random_seed) or 100,
        opts.allow_replace_deleted and 1 or 0
    )
    if handle == nil then
        return nil, last_error(nil)
    end
    local obj = wrap_handle(handle)
    if opts.ef_search and opts.ef_search > 0 then
        local ok, err = obj:set_ef(opts.ef_search)
        if not ok then
            obj:close()
            return nil, err
        end
    end
    return obj
end

function M.load(path, opts)
    opts = type(opts) == "table" and opts or {}
    local handle = lib.mori_hnsw_load(
        tostring(path or ""),
        tonumber(opts.space) or M.SPACE_COSINE,
        tonumber(opts.dim) or 0,
        tonumber(opts.max_elements) or 0,
        opts.allow_replace_deleted and 1 or 0
    )
    if handle == nil then
        return nil, last_error(nil)
    end
    local obj = wrap_handle(handle)
    if opts.ef_search and opts.ef_search > 0 then
        local ok, err = obj:set_ef(opts.ef_search)
        if not ok then
            obj:close()
            return nil, err
        end
    end
    return obj
end

function Index:close()
    if self._handle ~= nil then
        ffi.gc(self._handle, nil)
        lib.mori_hnsw_destroy(self._handle)
        self._handle = nil
    end
end

function Index:count()
    return tonumber(lib.mori_hnsw_count(self._handle)) or 0
end

function Index:deleted_count()
    return tonumber(lib.mori_hnsw_deleted_count(self._handle)) or 0
end

function Index:capacity()
    return tonumber(lib.mori_hnsw_capacity(self._handle)) or 0
end

function Index:set_ef(ef)
    local ok = lib.mori_hnsw_set_ef(self._handle, tonumber(ef) or 0)
    if ok == 0 then
        return nil, last_error(self._handle)
    end
    return true
end

function Index:resize(new_max_elements)
    local ok = lib.mori_hnsw_resize(self._handle, tonumber(new_max_elements) or 0)
    if ok == 0 then
        return nil, last_error(self._handle)
    end
    return true
end

function Index:add(label, vec)
    local buf = vector_buffer(vec, self.dim)
    local ok = lib.mori_hnsw_add(self._handle, tonumber(label) or 0, buf)
    if ok == 0 then
        return nil, last_error(self._handle)
    end
    return true
end

function Index:mark_deleted(label)
    local ok = lib.mori_hnsw_mark_deleted(self._handle, tonumber(label) or 0)
    if ok == 0 then
        return nil, last_error(self._handle)
    end
    return true
end

function Index:unmark_deleted(label)
    local ok = lib.mori_hnsw_unmark_deleted(self._handle, tonumber(label) or 0)
    if ok == 0 then
        return nil, last_error(self._handle)
    end
    return true
end

function Index:save(path)
    local ok = lib.mori_hnsw_save(self._handle, tostring(path or ""))
    if ok == 0 then
        return nil, last_error(self._handle)
    end
    return true
end

function Index:search(vec, k)
    local want = math.max(0, math.floor(tonumber(k) or 0))
    if want <= 0 then
        return {}
    end
    local query = vector_buffer(vec, self.dim)
    local labels = ffi.new("uint64_t[?]", want)
    local distances = ffi.new("float[?]", want)
    local found = tonumber(lib.mori_hnsw_search(self._handle, query, want, labels, distances)) or 0
    local out = {}
    for i = 0, found - 1 do
        local distance = tonumber(distances[i]) or 0.0
        out[#out + 1] = {
            label = tonumber(labels[i]) or 0,
            distance = distance,
            similarity = similarity_from_distance(self.space, distance),
        }
    end
    return out
end

return M
