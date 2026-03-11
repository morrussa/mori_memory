local ffi = require("ffi")
local tool = require("module.tool")

local M = {}

local INDEX_MAGIC = "MID3"
local SHARD_MAGIC = "SHD3"
local SHARD_VERSION = 1
local TOPIC_PREDICTOR_VERSION = "TPR2"

local function trim(text)
    return tostring(text or ""):match("^%s*(.-)%s*$")
end

local function parse_manifest(root)
    local f = io.open(root .. "/manifest.txt", "r")
    if not f then
        return nil
    end
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

local function load_index(root)
    local path = root .. "/memory_index.bin"
    local f = io.open(path, "rb")
    if not f then
        return nil, "missing_index"
    end
    local data = f:read("*a")
    f:close()
    if (not data) or #data < 20 or data:sub(1, 4) ~= INDEX_MAGIC then
        return nil, "invalid_index_magic"
    end

    local p = ffi.cast("const uint8_t*", data)
    local header = ffi.cast("const uint32_t*", p + 4)
    local version = tonumber(header[0]) or 0
    local count = tonumber(header[1]) or 0
    local dim = tonumber(header[2]) or 0
    local next_line = tonumber(header[3]) or (count + 1)
    if version < 1 then
        return nil, "invalid_index_version"
    end

    local memories = {}
    local offset = 20
    local loaded = 0
    while offset + 4 <= #data and loaded < count do
        local rec_len = tonumber(ffi.cast("const uint32_t*", p + offset)[0]) or 0
        if rec_len < 18 or offset + rec_len > #data then
            break
        end

        local base = offset + 4
        local line = tonumber(ffi.cast("const uint32_t*", p + base)[0]) or 0
        base = base + 4

        if version <= 2 then
            base = base + 4
        end

        local cluster_id = tonumber(ffi.cast("const int32_t*", p + base)[0]) or -1
        base = base + 4

        if version >= 2 then
            base = base + 2
        end

        local topic_len = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 0
        base = base + 2

        local topic_anchor = ""
        if topic_len > 0 then
            topic_anchor = ffi.string(p + base, topic_len)
            base = base + topic_len
        end

        local turn_count = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 0
        base = base + 2
        local turns = {}
        local turn_ptr = ffi.cast("const uint32_t*", p + base)
        for i = 0, turn_count - 1 do
            turns[#turns + 1] = tonumber(turn_ptr[i]) or 0
        end

        if line > 0 then
            memories[line] = {
                line = line,
                turns = turns,
                cluster_id = cluster_id,
                topic_anchor = topic_anchor,
            }
            loaded = loaded + 1
        end

        offset = offset + rec_len
    end

    return {
        memories = memories,
        dim = dim,
        next_line = next_line,
    }
end

local function load_shard(root, cid, dim)
    cid = tonumber(cid)
    if not cid or cid <= 0 then
        return {}
    end
    local path = string.format("%s/shards/cluster_%d.bin", root, cid)
    local f = io.open(path, "rb")
    if not f then
        return {}
    end
    local data = f:read("*a")
    f:close()
    if (not data) or #data < 20 or data:sub(1, 4) ~= SHARD_MAGIC then
        return {}
    end

    local out = {}
    local p = ffi.cast("const uint8_t*", data)
    local header = ffi.cast("const uint32_t*", p + 4)
    local version = tonumber(header[0]) or 0
    local file_cid = tonumber(header[1]) or -1
    local count = tonumber(header[2]) or 0
    local vector_dim = tonumber(header[3]) or dim
    if version ~= SHARD_VERSION or file_cid ~= cid or vector_dim <= 0 then
        return {}
    end

    local offset = 20
    local rec_size = 4 + vector_dim * 4
    for _ = 1, count do
        if offset + rec_size > #data then
            break
        end
        local line = tonumber(ffi.cast("const uint32_t*", p + offset)[0]) or -1
        local vec = {}
        local vptr = ffi.cast("const float*", p + offset + 4)
        for i = 0, vector_dim - 1 do
            vec[i + 1] = tonumber(vptr[i]) or 0.0
        end
        if line > 0 then
            out[line] = vec
        end
        offset = offset + rec_size
    end
    return out
end

function M.load_memory_v3(root)
    root = trim(root)
    if root == "" then
        return nil, "missing_root"
    end
    local manifest = parse_manifest(root)
    if not manifest or tostring(manifest.version or "") ~= "V3" then
        return nil, "missing_manifest"
    end

    local index, err = load_index(root)
    if not index then
        return nil, err
    end

    local dim = tonumber(index.dim) or tonumber(manifest.dim) or 0
    local shard_cache = {}
    for line, mem in pairs(index.memories or {}) do
        local cid = tonumber((mem or {}).cluster_id) or -1
        if cid > 0 then
            if not shard_cache[cid] then
                shard_cache[cid] = load_shard(root, cid, dim)
            end
            mem.vec = shard_cache[cid][line]
        end
    end

    return {
        memories = index.memories or {},
        dim = dim,
        next_line = tonumber(index.next_line) or 1,
        current_turn = tonumber(manifest.current_turn) or 0,
    }
end

function M.load_topic_predictor(path)
    path = trim(path)
    if path == "" then
        return nil, "missing_path"
    end
    local f = io.open(path, "r")
    if not f then
        return nil, "missing_file"
    end
    local header = f:read("*l")
    if header ~= TOPIC_PREDICTOR_VERSION then
        f:close()
        return nil, "version_mismatch"
    end

    local out = {
        topic_memory = {},
        memory_next = {},
        topic_transition = {},
        stats = {},
    }

    for line in f:lines() do
        local kind, a, b, c = tostring(line or ""):match("^([A-Z_]+)\t([^\t]*)\t([^\t]*)\t([^\t]*)$")
        if kind == "TOPIC" then
            local anchor = trim(a)
            local mem_idx = tonumber(b)
            local score = tonumber(c)
            if anchor ~= "" and mem_idx and score and score > 0 then
                out.topic_memory[anchor] = out.topic_memory[anchor] or {}
                out.topic_memory[anchor][mem_idx] = score
            end
        elseif kind == "NEXT" then
            local src = tonumber(a)
            local dst = tonumber(b)
            local score = tonumber(c)
            if src and dst and score and score > 0 then
                out.memory_next[src] = out.memory_next[src] or {}
                out.memory_next[src][dst] = score
            end
        elseif kind == "TRANS" then
            local src = trim(a)
            local dst = trim(b)
            local score = tonumber(c)
            if src ~= "" and dst ~= "" and score and score > 0 then
                out.topic_transition[src] = out.topic_transition[src] or {}
                out.topic_transition[src][dst] = score
            end
        elseif kind == "STAT" then
            out.stats[tostring(a)] = tonumber(b) or 0
        end
    end

    f:close()
    return out
end

function M.load_legacy_adaptive(path)
    path = trim(path)
    if path == "" then
        return nil
    end
    local f = io.open(path, "r")
    if not f then
        return nil
    end
    local raw = f:read("*a")
    f:close()
    return raw
end

return M
