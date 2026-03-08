-- memory.lua (V3 shard storage)

local M = {}

local tool = require("module.tool")
local config = require("module.config")
local cluster = require("module.memory.cluster")
local adaptive = require("module.memory.adaptive")
local memtypes = require("module.memory.types")
local ffi = require("ffi")
local saver = require("module.memory.saver")
local persistence = require("module.persistence")

local STORAGE_CFG = (config.settings or {}).storage_v3 or {}
local ROOT_DIR = STORAGE_CFG.root or "memory/v3"
local SHARD_DIR = ROOT_DIR .. "/shards"
local MANIFEST_PATH = ROOT_DIR .. "/manifest.txt"
local INDEX_PATH = ROOT_DIR .. "/memory_index.bin"

local INDEX_MAGIC = "MID3"
local INDEX_VERSION = 2
local SHARD_MAGIC = "SHD3"
local SHARD_VERSION = 1

local VECTOR_DIM = tonumber((config.settings or {}).dim) or 1024
local MAX_MEMORY = tonumber(STORAGE_CFG.max_memory) or 200000
local FAST_SCAN_TOPK = math.max(64, (tonumber((((config.settings or {}).ai_query or {}).max_memory)) or 5) * 8)

M.memories = {}

local next_line = 1
local preload_io_count = 0
local current_turn = 0

M._cluster_cache = {}
M._cluster_cache_order = {}
M._dirty_shards = {}
M._dirty_index = false
M._loaded_v3 = false

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function normalize_line(line)
    local n = tonumber(line)
    if not n then return nil end
    n = math.floor(n)
    if n < 1 or n >= next_line then return nil end
    return n
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function topic_key_for_turn(turn)
    local ok_topic, topic = pcall(require, "module.memory.topic")
    if not ok_topic or not topic then
        return nil
    end

    local ti = topic.get_topic_for_turn and topic.get_topic_for_turn(turn) or nil
    if not ti then
        local anchor = topic.get_topic_anchor and topic.get_topic_anchor(turn) or nil
        return anchor
    end

    if ti.is_active then
        local start_turn = topic.active_topic and topic.active_topic.start
        if start_turn then
            return "S:" .. tostring(start_turn)
        end
        local anchor = topic.get_topic_anchor and topic.get_topic_anchor(turn) or nil
        return anchor
    end

    if ti.topic_idx and topic.topics and topic.topics[ti.topic_idx] then
        local rec = topic.topics[ti.topic_idx]
        if rec and rec.start then
            return "S:" .. tostring(rec.start)
        end
    end

    local anchor = topic.get_topic_anchor and topic.get_topic_anchor(turn) or nil
    return anchor
end

local function get_cache_cap()
    local cfg = (config.settings or {}).storage_v3 or {}
    return math.max(1, tonumber(cfg.cluster_cache_cap) or 24)
end

local function lru_enabled()
    local cfg = (config.settings or {}).storage_v3 or {}
    return cfg.enable_lru ~= false
end

local function shard_path(cid)
    return string.format("%s/cluster_%d.bin", SHARD_DIR, tonumber(cid) or -1)
end

local function reset_runtime_state()
    M.memories = {}
    next_line = 1
    preload_io_count = 0
    current_turn = 0

    M.topic_index = {}
    M._cluster_cache = {}
    M._cluster_cache_order = {}
    M._dirty_shards = {}
    M._dirty_index = false
    M._loaded_v3 = false
end

local function topic_index_add(topic_anchor, line)
    if not topic_anchor or topic_anchor == "" then return end
    local bucket = M.topic_index[topic_anchor]
    if not bucket then
        bucket = {}
        M.topic_index[topic_anchor] = bucket
    end
    bucket[line] = true
end

local function topic_index_remove(topic_anchor, line)
    if not topic_anchor or topic_anchor == "" then return end
    local bucket = M.topic_index[topic_anchor]
    if not bucket then return end
    bucket[line] = nil
    if not next(bucket) then
        M.topic_index[topic_anchor] = nil
    end
end

local function touch_cache_order(cid)
    if not lru_enabled() then return end
    local order = M._cluster_cache_order
    for i = #order, 1, -1 do
        if order[i] == cid then
            table.remove(order, i)
            break
        end
    end
    order[#order + 1] = cid
end

local function parse_manifest()
    local f = io.open(MANIFEST_PATH, "r")
    if not f then return nil end

    local out = {}
    for line in f:lines() do
        local k, v = tostring(line or ""):match("^([%w_]+)%s*=%s*(.+)$")
        if k and v then
            out[k] = v
        end
    end
    f:close()
    return out
end

local function write_manifest()
    local count = next_line - 1
    return persistence.write_atomic(MANIFEST_PATH, "w", function(f)
        local rows = {
            "version=V3",
            "index_magic=" .. INDEX_MAGIC,
            "index_version=" .. tostring(INDEX_VERSION),
            "shard_magic=" .. SHARD_MAGIC,
            "shard_version=" .. tostring(SHARD_VERSION),
            "dim=" .. tostring(VECTOR_DIM),
            "next_line=" .. tostring(next_line),
            "count=" .. tostring(count),
            "current_turn=" .. tostring(current_turn),
        }
        for _, row in ipairs(rows) do
            local ok_w, err_w = f:write(row .. "\n")
            if not ok_w then
                return false, err_w
            end
        end
        return true
    end)
end

local function build_index_record(line, mem)
    local topic_anchor = tostring(mem.topic_anchor or "")
    local turns = mem.turns or {}
    local kind_code = memtypes.code_for(mem.type or mem.kind)
    local topic_len = #topic_anchor
    local turn_count = #turns
    local record_size = 4 + 4 + 4 + 4 + 2 + 2 + topic_len + 2 + (turn_count * 4)

    local buf = ffi.new("uint8_t[?]", record_size)
    local base = 0

    ffi.cast("uint32_t*", buf + base)[0] = record_size
    base = base + 4

    ffi.cast("uint32_t*", buf + base)[0] = line
    base = base + 4

    ffi.cast("uint32_t*", buf + base)[0] = math.max(0, math.floor(tonumber(mem.heat) or 0))
    base = base + 4

    ffi.cast("int32_t*", buf + base)[0] = math.floor(tonumber(mem.cluster_id) or -1)
    base = base + 4

    ffi.cast("uint16_t*", buf + base)[0] = math.floor(tonumber(kind_code) or 0)
    base = base + 2

    ffi.cast("uint16_t*", buf + base)[0] = topic_len
    base = base + 2

    if topic_len > 0 then
        ffi.copy(buf + base, topic_anchor, topic_len)
        base = base + topic_len
    end

    ffi.cast("uint16_t*", buf + base)[0] = turn_count
    base = base + 2

    if turn_count > 0 then
        local ptr = ffi.cast("uint32_t*", buf + base)
        for i = 1, turn_count do
            ptr[i - 1] = math.max(0, math.floor(tonumber(turns[i]) or 0))
        end
    end

    return ffi.string(buf, record_size)
end

local function save_index_binary()
    return persistence.write_atomic(INDEX_PATH, "wb", function(f)
        local function write_chunk(chunk)
            local ok_w, err_w = f:write(chunk)
            if not ok_w then
                return false, err_w
            end
            return true
        end

        local ok1, err1 = write_chunk(INDEX_MAGIC)
        if not ok1 then return false, err1 end

        local header = ffi.new("uint32_t[4]", INDEX_VERSION, next_line - 1, VECTOR_DIM, next_line)
        local ok2, err2 = write_chunk(ffi.string(header, 16))
        if not ok2 then return false, err2 end

        for line = 1, next_line - 1 do
            local mem = M.memories[line]
            if mem then
                local rec = build_index_record(line, mem)
                local okr, errr = write_chunk(rec)
                if not okr then return false, errr end
            end
        end

        return true
    end)
end

local function load_index_binary()
    local f = io.open(INDEX_PATH, "rb")
    if not f then
        return false, "index_missing"
    end

    local data = f:read("*a")
    f:close()

    if (not data) or #data < 20 or data:sub(1, 4) ~= INDEX_MAGIC then
        return false, "index_magic_invalid"
    end

    local p = ffi.cast("const uint8_t*", data)
    local header = ffi.cast("const uint32_t*", p + 4)
    local version = tonumber(header[0]) or 0
    local count = tonumber(header[1]) or 0
    local dim = tonumber(header[2]) or VECTOR_DIM
    local next_line_header = tonumber(header[3]) or (count + 1)

    if version ~= 1 and version ~= INDEX_VERSION then
        return false, "index_version_mismatch"
    end

    VECTOR_DIM = math.max(1, dim)
    M.memories = {}
    M.topic_index = {}

    local offset = 20
    local loaded = 0
    while offset + 4 <= #data and loaded < count do
        local rec_len = tonumber(ffi.cast("const uint32_t*", p + offset)[0]) or 0
        if rec_len < 20 or (offset + rec_len) > #data then
            break
        end

        local base = offset + 4
        local line = tonumber(ffi.cast("const uint32_t*", p + base)[0]) or 0
        base = base + 4

        local heat = tonumber(ffi.cast("const uint32_t*", p + base)[0]) or 0
        base = base + 4

        local cluster_id = tonumber(ffi.cast("const int32_t*", p + base)[0]) or -1
        base = base + 4

        local kind_code = 0
        if version >= 2 then
            if base + 2 > offset + rec_len then break end
            kind_code = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 0
            base = base + 2
        end

        local topic_len = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 0
        base = base + 2

        local topic_anchor = ""
        if topic_len > 0 then
            if base + topic_len > offset + rec_len then break end
            topic_anchor = ffi.string(p + base, topic_len)
            base = base + topic_len
        end

        if base + 2 > offset + rec_len then break end
        local turn_count = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 0
        base = base + 2

        local turns = {}
        local turn_bytes = turn_count * 4
        if base + turn_bytes > offset + rec_len then break end
        if turn_count > 0 then
            local tptr = ffi.cast("const uint32_t*", p + base)
            for i = 0, turn_count - 1 do
                turns[#turns + 1] = tonumber(tptr[i]) or 0
            end
        end

        if line > 0 then
            M.memories[line] = {
                turns = turns,
                heat = heat,
                cluster_id = cluster_id,
                type = memtypes.kind_for_code(kind_code, memtypes.DEFAULT_KIND),
                topic_anchor = topic_anchor,
            }
            topic_index_add(topic_anchor, line)
            loaded = loaded + 1
        end

        offset = offset + rec_len
    end

    next_line = math.max(next_line_header, loaded + 1)
    return true
end

local function save_cluster_shard(cid)
    cid = tonumber(cid)
    if not cid or cid <= 0 then
        return true
    end

    local shard = M._cluster_cache[cid]
    if not shard then
        return true
    end

    local lines = {}
    for line, vec in pairs(shard.ptr_by_line or {}) do
        if line and vec and vec.__ptr then
            lines[#lines + 1] = tonumber(line)
        end
    end
    table.sort(lines)

    local path = shard_path(cid)
    local ok, err = persistence.write_atomic(path, "wb", function(f)
        local function write_chunk(chunk)
            local ok_w, err_w = f:write(chunk)
            if not ok_w then
                return false, err_w
            end
            return true
        end

        local ok1, err1 = write_chunk(SHARD_MAGIC)
        if not ok1 then return false, err1 end

        local header = ffi.new("uint32_t[4]", SHARD_VERSION, cid, #lines, VECTOR_DIM)
        local ok2, err2 = write_chunk(ffi.string(header, 16))
        if not ok2 then return false, err2 end

        local rec_size = 4 + VECTOR_DIM * 4
        for _, line in ipairs(lines) do
            local vec = shard.ptr_by_line[line]
            if vec and vec.__ptr then
                local buf = ffi.new("uint8_t[?]", rec_size)
                ffi.cast("uint32_t*", buf)[0] = line
                local vptr = ffi.cast("float*", buf + 4)
                for i = 1, VECTOR_DIM do
                    vptr[i - 1] = vec.__ptr[i - 1]
                end
                local ok_r, err_r = write_chunk(ffi.string(buf, rec_size))
                if not ok_r then return false, err_r end
            end
        end
        return true
    end)

    if not ok then
        return false, err
    end

    M._dirty_shards[cid] = nil
    return true
end

local function enforce_lru_capacity()
    if not lru_enabled() then return end
    local cap = get_cache_cap()
    local order = M._cluster_cache_order

    while #order > cap do
        local evict_cid = table.remove(order, 1)
        if evict_cid then
            if M._dirty_shards[evict_cid] then
                local ok, err = save_cluster_shard(evict_cid)
                if not ok then
                    print(string.format("[Memory][WARN] shard %d eviction save failed: %s", evict_cid, tostring(err)))
                    order[#order + 1] = evict_cid
                    break
                end
            end
            M._cluster_cache[evict_cid] = nil
        end
    end
end

local function load_cluster_shard(cid)
    cid = tonumber(cid)
    if not cid or cid <= 0 then
        return { ptr_by_line = {} }
    end

    local cached = M._cluster_cache[cid]
    if cached then
        touch_cache_order(cid)
        return cached
    end

    local shard = { ptr_by_line = {}, raw_blob = nil, raw_ptr = nil }
    local path = shard_path(cid)
    local f = io.open(path, "rb")
    if f then
        local data = f:read("*a")
        f:close()

        if data and #data >= 20 and data:sub(1, 4) == SHARD_MAGIC then
            shard.raw_blob = data
            shard.raw_ptr = ffi.cast("const uint8_t*", shard.raw_blob)
            local p = shard.raw_ptr
            local header = ffi.cast("const uint32_t*", p + 4)
            local version = tonumber(header[0]) or 0
            local file_cid = tonumber(header[1]) or -1
            local count = tonumber(header[2]) or 0
            local dim = tonumber(header[3]) or VECTOR_DIM

            if version == SHARD_VERSION and file_cid == cid and dim == VECTOR_DIM then
                local offset = 20
                local rec_size = 4 + VECTOR_DIM * 4
                for _ = 1, count do
                    if offset + rec_size > #data then break end
                    local line = tonumber(ffi.cast("const uint32_t*", p + offset)[0]) or -1
                    local vptr = ffi.cast("const float*", p + offset + 4)
                    if line > 0 then
                        shard.ptr_by_line[line] = { __ptr = vptr, __dim = VECTOR_DIM }
                    end
                    offset = offset + rec_size
                end
            end
        end
    end

    M._cluster_cache[cid] = shard
    touch_cache_order(cid)
    enforce_lru_capacity()
    return shard
end

local function record_dirty_index()
    M._dirty_index = true
    saver.mark_dirty()
end

local function ensure_storage_dirs()
    ensure_dir(ROOT_DIR)
    ensure_dir(SHARD_DIR)
end

local function cleanup_legacy_files()
    for _, path in ipairs({
        "memory/memory.bin",
        "memory/clusters.bin",
    }) do
        if tool.file_exists(path) then
            os.remove(path)
        end
    end
end

function M.begin_turn(turn)
    current_turn = math.max(0, math.floor(tonumber(turn) or 0))
    preload_io_count = 0
end

function M.get_preload_io_count()
    return preload_io_count
end

function M.reserve_preload_io(requested, max_per_turn)
    local req = math.max(0, math.floor(tonumber(requested) or 0))
    local cap = math.max(0, math.floor(tonumber(max_per_turn) or 0))
    if req <= 0 or cap <= 0 then
        return 0
    end
    local room = math.max(0, cap - preload_io_count)
    local granted = math.min(room, req)
    preload_io_count = preload_io_count + granted
    return granted
end

function M.get_total_lines()
    return next_line - 1
end

function M.get_turns(line)
    local idx = normalize_line(line)
    if not idx then return {} end
    local mem = M.memories[idx]
    if not mem then return {} end
    return shallow_copy_array(mem.turns)
end

function M.get_topic_anchor(line)
    local idx = normalize_line(line)
    if not idx then return nil end
    local mem = M.memories[idx]
    if not mem then return nil end
    return mem.topic_anchor
end

function M.get_memory_type(line)
    local idx = normalize_line(line)
    if not idx then return memtypes.DEFAULT_KIND end
    local mem = M.memories[idx]
    if not mem then return memtypes.DEFAULT_KIND end
    return memtypes.normalize(mem.type or mem.kind, memtypes.DEFAULT_KIND)
end

function M.set_memory_type(line, kind)
    local idx = normalize_line(line)
    if not idx then return false end
    local mem = M.memories[idx]
    if not mem then return false end

    local normalized = memtypes.normalize(kind, memtypes.DEFAULT_KIND)
    if mem.type == normalized then
        return true
    end

    mem.type = normalized
    record_dirty_index()
    return true
end

function M.iter_topic_lines(topic_anchor, only_cold)
    local out = {}
    local bucket = M.topic_index[tostring(topic_anchor or "")]
    if not bucket then
        return out
    end

    local need_cold = only_cold == true
    for line, _ in pairs(bucket) do
        local idx = normalize_line(line)
        if idx then
            if (not need_cold) or (M.get_heat_by_index(idx) <= 0) then
                out[#out + 1] = idx
            end
        end
    end
    return out
end

function M.get_cluster_id(line)
    local idx = normalize_line(line)
    if not idx then return nil end
    local mem = M.memories[idx]
    if not mem then return nil end
    return tonumber(mem.cluster_id)
end

function M.set_cluster_id(line, cluster_id)
    local idx = normalize_line(line)
    if not idx then return false end
    local mem = M.memories[idx]
    if not mem then return false end

    local cid = math.floor(tonumber(cluster_id) or -1)
    if mem.cluster_id == cid then
        return true
    end

    mem.cluster_id = cid
    record_dirty_index()
    return true
end

function M.store_vector(line, cluster_id, vec)
    local idx = normalize_line(line)
    if not idx then return false end
    local cid = tonumber(cluster_id)
    if not cid or cid <= 0 then return false end

    local shard = load_cluster_shard(cid)
    local wrapped = tool.to_ptr_vec(vec)
    if (not wrapped) or (not wrapped.__ptr) then
        return false
    end
    if (tonumber(wrapped.__dim) or 0) ~= VECTOR_DIM then
        local fixed = ffi.new("float[?]", VECTOR_DIM)
        local src = wrapped.__ptr
        local src_dim = tonumber(wrapped.__dim) or 0
        for i = 0, VECTOR_DIM - 1 do
            fixed[i] = (i < src_dim) and src[i] or 0.0
        end
        wrapped = { __ptr = fixed, __dim = VECTOR_DIM }
    end
    shard.ptr_by_line[idx] = wrapped

    M._dirty_shards[cid] = true
    M.set_cluster_id(idx, cid)
    saver.mark_dirty()
    return true
end

function M.get_cluster_shard(cluster_id)
    local cid = tonumber(cluster_id)
    if not cid or cid <= 0 then
        return nil
    end
    return load_cluster_shard(cid)
end

function M.return_mem_vec(line)
    local idx = normalize_line(line)
    if not idx then return nil end
    local mem = M.memories[idx]
    if not mem then return nil end

    local cid = tonumber(mem.cluster_id)
    if not cid or cid <= 0 then return nil end

    local shard = load_cluster_shard(cid)
    local vec = shard.ptr_by_line[idx]
    if not vec then
        return nil
    end
    return vec
end

function M.get_heat_by_index(line)
    local idx = normalize_line(line)
    if not idx then return 0 end
    local mem = M.memories[idx]
    if not mem then return 0 end
    return math.max(0, tonumber(mem.heat) or 0)
end

function M.set_heat(line, new_heat)
    local idx = normalize_line(line)
    if not idx then return false end

    local mem = M.memories[idx]
    if not mem then return false end

    local old_val = math.max(0, math.floor(tonumber(mem.heat) or 0))
    local val = math.max(0, math.floor(tonumber(new_heat) or 0))
    if old_val == val then
        return true
    end

    mem.heat = val
    if cluster and cluster.on_memory_heat_change then
        cluster.on_memory_heat_change(idx, old_val, val)
    end
    record_dirty_index()
    return true
end

function M.append_turn(line, turn)
    local idx = normalize_line(line)
    if not idx then return false end
    local mem = M.memories[idx]
    if not mem then return false end
    mem.turns = mem.turns or {}
    table.insert(mem.turns, 1, math.max(0, math.floor(tonumber(turn) or 0)))
    record_dirty_index()
    return true
end

function M.find_similar_all_fast(query_vec_table, max_results)
    local use_full_sort = false
    if max_results == nil then
        max_results = FAST_SCAN_TOPK
    end
    max_results = tonumber(max_results) or FAST_SCAN_TOPK
    if max_results <= 0 then
        use_full_sort = true
    end

    local results = {}
    local topk = {}
    local total = next_line - 1

    for i = 1, total do
        local h = M.get_heat_by_index(i)
        if h > 0 then
            local mem_vec = M.return_mem_vec(i)
            if mem_vec and #mem_vec > 0 then
                local score = tonumber(tool.cosine_similarity(query_vec_table, mem_vec)) or 0.0
                if score > 0.5 then
                    if use_full_sort then
                        table.insert(results, { index = i, similarity = score })
                    else
                        if #topk < max_results then
                            table.insert(topk, { index = i, similarity = score })
                            local j = #topk
                            while j > 1 and topk[j].similarity < topk[j - 1].similarity do
                                topk[j], topk[j - 1] = topk[j - 1], topk[j]
                                j = j - 1
                            end
                        elseif score > topk[1].similarity then
                            topk[1] = { index = i, similarity = score }
                            local j = 1
                            while j < #topk and topk[j].similarity > topk[j + 1].similarity do
                                topk[j], topk[j + 1] = topk[j + 1], topk[j]
                                j = j + 1
                            end
                        end
                    end
                end
            end
        end
    end

    if not use_full_sort then
        for i = #topk, 1, -1 do
            results[#results + 1] = topk[i]
        end
        return results
    end

    table.sort(results, function(a, b)
        return a.similarity > b.similarity
    end)
    return results
end

function M.iterate_all()
    local i = 0
    return function()
        i = i + 1
        if i >= next_line then return nil end
        local mem = M.memories[i]
        if not mem then
            return {
                heat = 0,
                turns = {},
                topic_anchor = nil,
                vec = nil,
            }
        end
        return {
            heat = tonumber(mem.heat) or 0,
            turns = shallow_copy_array(mem.turns),
            type = memtypes.normalize(mem.type or mem.kind, memtypes.DEFAULT_KIND),
            topic_anchor = mem.topic_anchor,
            vec = M.return_mem_vec(i),
        }
    end
end

function M.load()
    reset_runtime_state()
    ensure_storage_dirs()

    local manifest = parse_manifest()
    if not manifest or tostring(manifest.version or "") ~= "V3" then
        local has_legacy_raw = tool.file_exists("memory/memory.bin")
            or tool.file_exists("memory/clusters.bin")
            or tool.file_exists("memory/history.txt")
        if has_legacy_raw then
            print("[Memory] 未检测到 V3 manifest，保留 legacy raw 文件；当前以内存空状态启动")
        else
            print("[Memory] V3 manifest 不存在，初始化空记忆池")
        end
        M._loaded_v3 = false
        return
    end

    local manifest_dim = tonumber(manifest.dim)
    if manifest_dim and manifest_dim > 0 then
        VECTOR_DIM = manifest_dim
    end

    local ok, err = load_index_binary()
    if not ok then
        print(string.format("[Memory] 加载 memory_index.bin 失败: %s，初始化空记忆池", tostring(err)))
        reset_runtime_state()
        ensure_storage_dirs()
        return
    end

    local parsed_next = tonumber(manifest.next_line)
    if parsed_next and parsed_next > next_line then
        next_line = math.floor(parsed_next)
    end

    local parsed_turn = tonumber(manifest.current_turn)
    if parsed_turn and parsed_turn >= 0 then
        current_turn = math.floor(parsed_turn)
    end

    M._loaded_v3 = true
    print(string.format("[Memory] V3 加载完成: %d 条记忆，Dim=%d", next_line - 1, VECTOR_DIM))
end

function M.save_index_to_disk()
    ensure_storage_dirs()

    local ok_manifest, err_manifest = write_manifest()
    if not ok_manifest then
        return false, err_manifest
    end

    local ok_index, err_index = save_index_binary()
    if not ok_index then
        return false, err_index
    end

    M._dirty_index = false
    return true
end

function M.save_dirty_shards()
    ensure_storage_dirs()

    local pending = {}
    for cid, dirty in pairs(M._dirty_shards) do
        if dirty then
            pending[#pending + 1] = tonumber(cid)
        end
    end
    table.sort(pending)

    for _, cid in ipairs(pending) do
        local ok, err = save_cluster_shard(cid)
        if not ok then
            return false, err
        end
    end

    return true
end

function M.save_to_disk()
    local ok_i, err_i = M.save_index_to_disk()
    if not ok_i then return false, err_i end
    local ok_s, err_s = M.save_dirty_shards()
    if not ok_s then return false, err_s end
    return true
end

function M.add_memory(vec, turn, opts)
    local heat_mod = require("module.memory.heat")
    opts = type(opts) == "table" and opts or {}

    if type(vec) ~= "table" or #vec <= 0 then
        return nil, "invalid_vector"
    end

    local mem_kind = memtypes.normalize(opts.type or opts.kind, memtypes.DEFAULT_KIND)

    local new_heat = tonumber((config.settings.heat or {}).new_memory_heat) or 43000
    local merge_limit = adaptive.get_merge_limit((config.settings or {}).merge_limit or 0.95)
    local cluster_sim_th = tonumber((((config.settings or {}).cluster or {}).cluster_sim)) or 0.72

    local best_id, best_sim = cluster.find_best_cluster(vec, {
        super_topn = (((config.settings or {}).cluster or {}).supercluster_topn_add),
    })

    local sim_results = {}
    if best_id and best_sim and best_sim >= cluster_sim_th then
        sim_results = cluster.find_sim_in_cluster(vec, best_id, {
            only_hot = true,
            max_results = FAST_SCAN_TOPK,
        })
    end

    if #sim_results > 0 then
        local top = sim_results[1]
        if top and (tonumber(top.similarity) or 0) >= merge_limit then
            local target = tonumber(top.index)
            if target and M.get_heat_by_index(target) > 0 then
                local ok_turn = M.append_turn(target, turn)
                if ok_turn then
                    local merged_topic_key = topic_key_for_turn(turn)
                    if merged_topic_key then
                        M.update_topic_anchor(target, merged_topic_key)
                    end
                    M.set_memory_type(target, mem_kind)
                    M.set_heat(target, M.get_heat_by_index(target) + new_heat)
                    heat_mod.sync_line(target)
                    heat_mod.normalize_heat()
                    saver.mark_dirty()
                    print(string.format("[Memory] 合并 -> 行 %d (sim=%.4f)", target, tonumber(top.similarity) or 0))
                    return target
                end
            end
        end
    end

    if next_line > MAX_MEMORY then
        print(string.format("[Memory][WARN] 记忆池已满（MAX_MEMORY=%d），跳过写入", MAX_MEMORY))
        return nil, "capacity_reached"
    end

    local new_line = next_line
    local topic_anchor = topic_key_for_turn(turn)
    M.memories[new_line] = {
        turns = { math.max(0, math.floor(tonumber(turn) or 0)) },
        heat = 0,
        cluster_id = -1,
        type = mem_kind,
        topic_anchor = topic_anchor,
    }
    topic_index_add(topic_anchor, new_line)

    next_line = next_line + 1
    record_dirty_index()

    local cid = cluster.add(vec, new_line)
    if cid then
        M.set_cluster_id(new_line, cid)
        M.store_vector(new_line, cid, vec)
    end

    heat_mod.add_new_with_cluster_cap(new_line, vec)
    if M.get_heat_by_index(new_line) <= 0 then
        cluster.mark_cold(new_line)
    end

    print(string.format("[Memory] 新建 -> 行 %d", new_line))
    return new_line
end

function M.update_topic_anchor(line, new_anchor)
    local idx = normalize_line(line)
    if not idx then return false end
    local mem = M.memories[idx]
    if not mem then return false end

    local old = mem.topic_anchor
    new_anchor = tostring(new_anchor or "")
    if old == new_anchor then
        return true
    end

    topic_index_remove(old, idx)
    mem.topic_anchor = new_anchor
    topic_index_add(new_anchor, idx)
    record_dirty_index()
    return true
end

return M
