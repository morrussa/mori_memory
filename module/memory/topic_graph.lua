local ffi = require("ffi")
local tool = require("module.tool")
local config = require("module.config")
local history = require("module.memory.history")
local topic = require("module.memory.topic")
local persistence = require("module.persistence")
local util = require("mori_memory.util")
local deep_artmap = require("module.memory.topic_graph_deep_artmap")
local topic_hnsw_mod = require("module.memory.topic_graph_hnsw")
local legacy_import = require("module.memory.legacy_v3_import")

local M = {}

local STATE_VERSION = "TG1"
local VECTOR_MAGIC = "TGV1"
local EMPTY_SCOPE_BUCKET = "__scope_empty__"

local function tg_cfg()
    return ((config.settings or {}).topic_graph or {})
end

local function ai_cfg()
    return ((config.settings or {}).ai_query or {})
end

local function storage_root()
    local storage = (tg_cfg().storage or {})
    return tostring(storage.root or "memory/v4/topic_graph")
end

local function state_path()
    return storage_root() .. "/state.lua"
end

local function vectors_path()
    return storage_root() .. "/vectors.bin"
end

local function hnsw_root()
    return storage_root() .. "/hnsw"
end

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function trim(text)
    return tostring(text or ""):match("^%s*(.-)%s*$")
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function copy_vec(vec)
    local out = {}
    for i = 1, #(vec or {}) do
        out[i] = tonumber(vec[i]) or 0.0
    end
    return out
end

local function normalize(vec)
    local sum = 0.0
    for i = 1, #(vec or {}) do
        local v = tonumber(vec[i]) or 0.0
        sum = sum + v * v
    end
    if sum <= 0.0 then
        return copy_vec(vec)
    end
    local scale = 1.0 / math.sqrt(sum)
    local out = {}
    for i = 1, #(vec or {}) do
        out[i] = (tonumber(vec[i]) or 0.0) * scale
    end
    return out
end

local function safe_similarity(a, b)
    if type(a) ~= "table" or #a <= 0 or type(b) ~= "table" or #b <= 0 then
        return 0.0
    end
    return tonumber(tool.cosine_similarity(a, b)) or 0.0
end

local function average_vectors(vectors)
    local count = 0
    local acc = {}
    for _, vec in ipairs(vectors or {}) do
        if type(vec) == "table" and #vec > 0 then
            count = count + 1
            for i = 1, #vec do
                acc[i] = (acc[i] or 0.0) + (tonumber(vec[i]) or 0.0)
            end
        end
    end
    if count <= 0 then
        return {}
    end
    for i = 1, #acc do
        acc[i] = acc[i] / count
    end
    return normalize(acc)
end

local function sort_numeric_keys(map)
    local out = {}
    for k in pairs(map or {}) do
        local n = tonumber(k)
        if n then
            out[#out + 1] = n
        end
    end
    table.sort(out)
    return out
end

local function sort_string_keys(map)
    local out = {}
    for k in pairs(map or {}) do
        local s = trim(k)
        if s ~= "" then
            out[#out + 1] = s
        end
    end
    table.sort(out)
    return out
end

local function unique_sorted_numbers(items)
    local seen = {}
    local out = {}
    for _, item in ipairs(items or {}) do
        local n = tonumber(item)
        if n and n > 0 and not seen[n] then
            seen[n] = true
            out[#out + 1] = n
        end
    end
    table.sort(out)
    return out
end

local function flow_key_from_opts(opts)
    opts = type(opts) == "table" and opts or {}
    return trim(opts.flow_key or opts.flow or "")
end

local function actor_key_from_opts(opts)
    opts = type(opts) == "table" and opts or {}
    return trim(opts.actor_key or "")
end

local function thread_key_from_opts(opts)
    opts = type(opts) == "table" and opts or {}
    return trim(opts.thread_key or "")
end

local function normalize_memory_scope(memory_scope)
    memory_scope = trim(memory_scope):lower()
    if memory_scope == "thread" or memory_scope == "actor" then
        return memory_scope
    end
    return "scope"
end

local function ensure_runtime_state(flow_key)
    flow_key = trim(flow_key)
    if flow_key == "" then
        local rt = (M.state or {}).runtime
        if type(rt) ~= "table" then
            rt = { last_anchor = "", last_selected = {}, last_turn = 0 }
            if type(M.state) == "table" then
                M.state.runtime = rt
            end
        end
        rt.last_anchor = trim(rt.last_anchor or "")
        rt.last_selected = unique_sorted_numbers(rt.last_selected or {})
        rt.last_turn = tonumber(rt.last_turn) or 0
        return rt
    end

    M.runtime = type(M.runtime) == "table" and M.runtime or {}
    M.runtime.flow_runtime = type(M.runtime.flow_runtime) == "table" and M.runtime.flow_runtime or {}
    local bucket = M.runtime.flow_runtime
    local rt = bucket[flow_key]
    if type(rt) ~= "table" then
        rt = { last_anchor = "", last_selected = {}, last_turn = 0 }
        bucket[flow_key] = rt
    end
    rt.last_anchor = trim(rt.last_anchor or "")
    rt.last_selected = unique_sorted_numbers(rt.last_selected or {})
    rt.last_turn = tonumber(rt.last_turn) or 0
    return rt
end

local function gc_flow_runtime(current_turn)
    M.runtime = type(M.runtime) == "table" and M.runtime or {}
    M.runtime.flow_runtime = type(M.runtime.flow_runtime) == "table" and M.runtime.flow_runtime or {}
    M.runtime.flow_memory_next = type(M.runtime.flow_memory_next) == "table" and M.runtime.flow_memory_next or {}

    local ttl = math.max(8, math.floor(tonumber(tg_cfg().flow_runtime_ttl) or 128))
    local cap = math.max(16, math.floor(tonumber(tg_cfg().flow_runtime_cap) or 256))
    local cur_turn = math.max(0, math.floor(tonumber(current_turn) or 0))
    local rows = {}

    for flow_key, rt in pairs(M.runtime.flow_runtime) do
        local last_turn = math.max(0, math.floor(tonumber((rt or {}).last_turn) or 0))
        if cur_turn > 0 and last_turn > 0 and (cur_turn - last_turn) > ttl then
            M.runtime.flow_runtime[flow_key] = nil
            M.runtime.flow_memory_next[flow_key] = nil
        else
            rows[#rows + 1] = {
                key = flow_key,
                last_turn = last_turn,
            }
        end
    end

    if #rows <= cap then
        return
    end

    table.sort(rows, function(a, b)
        if (a.last_turn or 0) ~= (b.last_turn or 0) then
            return (a.last_turn or 0) < (b.last_turn or 0)
        end
        return tostring(a.key) < tostring(b.key)
    end)
    for i = 1, #rows - cap do
        local flow_key = rows[i].key
        M.runtime.flow_runtime[flow_key] = nil
        M.runtime.flow_memory_next[flow_key] = nil
    end
end

local FACET_PUNCT = {
    [" "] = true, ["\t"] = true, ["\r"] = true, ["\n"] = true,
    [","] = true, ["."] = true, ["!"] = true, ["?"] = true, [":"] = true, [";"] = true,
    ["("] = true, [")"] = true, ["["] = true, ["]"] = true, ["{"] = true, ["}"] = true,
    ["-"] = true, ["_"] = true, ["/"] = true, ["\\"] = true, ["'"] = true, ['"'] = true,
    ["`"] = true, ["~"] = true, ["@"] = true, ["#"] = true, ["$"] = true, ["%"] = true,
    ["^"] = true, ["&"] = true, ["*"] = true, ["+"] = true, ["="] = true, ["<"] = true,
    [">"] = true, ["|"] = true,
    ["，"] = true, ["。"] = true, ["！"] = true, ["？"] = true, ["："] = true, ["；"] = true,
    ["（"] = true, ["）"] = true, ["【"] = true, ["】"] = true, ["《"] = true, ["》"] = true,
    ["“"] = true, ["”"] = true, ["‘"] = true, ["’"] = true, ["、"] = true, ["…"] = true,
}

local function utf8_chars(text)
    local out = {}
    for ch in tostring(text or ""):gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        out[#out + 1] = ch
    end
    return out
end

local function add_token_weight(map, token, weight)
    token = trim(token):lower()
    weight = tonumber(weight) or 0.0
    if token == "" or weight <= 0.0 then
        return
    end
    map[token] = (tonumber(map[token]) or 0.0) + weight
end

local function flush_ascii_token(buf, weights)
    local token = table.concat(buf or "")
    if #token >= 2 then
        add_token_weight(weights, token, 1.0 + math.min(0.6, (#token - 2) * 0.08))
    end
    for i = #buf, 1, -1 do
        buf[i] = nil
    end
end

local function flush_cjk_run(run, weights)
    local n = #(run or {})
    if n <= 0 then
        return
    end
    if n == 1 then
        add_token_weight(weights, run[1], 0.45)
    else
        for i = 1, n - 1 do
            add_token_weight(weights, (run[i] or "") .. (run[i + 1] or ""), 1.0)
        end
        for i = 1, n - 2 do
            add_token_weight(weights, (run[i] or "") .. (run[i + 1] or "") .. (run[i + 2] or ""), 0.45)
        end
    end
    for i = n, 1, -1 do
        run[i] = nil
    end
end

local function extract_text_facet_weights(text)
    local weights = {}
    local ascii_buf = {}
    local cjk_run = {}
    for _, ch in ipairs(utf8_chars(text)) do
        local byte = string.byte(ch)
        if byte and byte < 128 then
            if ch:match("[%w]") then
                flush_cjk_run(cjk_run, weights)
                ascii_buf[#ascii_buf + 1] = ch:lower()
            else
                flush_ascii_token(ascii_buf, weights)
                flush_cjk_run(cjk_run, weights)
            end
        else
            flush_ascii_token(ascii_buf, weights)
            if FACET_PUNCT[ch] then
                flush_cjk_run(cjk_run, weights)
            else
                cjk_run[#cjk_run + 1] = ch
            end
        end
    end
    flush_ascii_token(ascii_buf, weights)
    flush_cjk_run(cjk_run, weights)
    return weights
end

local function facet_slot_cap()
    return math.max(1, math.floor(tonumber(tg_cfg().facet_slots) or 6))
end

local function normalize_facet_rows(raw_rows, cap)
    cap = math.max(1, math.floor(tonumber(cap) or facet_slot_cap()))
    local rows = {}
    local total = 0.0
    if type(raw_rows) == "table" and type(raw_rows[1]) == "table" then
        for _, row in ipairs(raw_rows) do
            local facet_id = tonumber((row or {}).id or (row or {}).facet_id or row[1])
            local weight = tonumber((row or {}).weight or row[2]) or 0.0
            if facet_id and facet_id > 0 and weight > 0.0 then
                rows[#rows + 1] = { id = facet_id, weight = weight }
            end
        end
    else
        for facet_id, weight in pairs(raw_rows or {}) do
            local id = tonumber(facet_id)
            local w = tonumber(weight) or 0.0
            if id and id > 0 and w > 0.0 then
                rows[#rows + 1] = { id = id, weight = w }
            end
        end
    end
    if #rows <= 0 then
        return {}
    end
    table.sort(rows, function(a, b)
        if (a.weight or 0.0) ~= (b.weight or 0.0) then
            return (a.weight or 0.0) > (b.weight or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)
    local out = {}
    for i = 1, math.min(cap, #rows) do
        total = total + rows[i].weight
        out[#out + 1] = {
            id = rows[i].id,
            weight = rows[i].weight,
        }
    end
    total = math.max(1e-6, total)
    for i = 1, #out do
        out[i].weight = out[i].weight / total
    end
    return out
end

local function facet_rows_to_map(rows)
    local out = {}
    if type(rows) == "table" and type(rows[1]) == "table" then
        for _, row in ipairs(rows or {}) do
            local facet_id = tonumber((row or {}).id or (row or {}).facet_id or row[1])
            local weight = tonumber((row or {}).weight or row[2]) or 0.0
            if facet_id and facet_id > 0 and weight > 0.0 then
                out[facet_id] = math.max(tonumber(out[facet_id]) or 0.0, weight)
            end
        end
    else
        for facet_id, weight in pairs(rows or {}) do
            local id = tonumber(facet_id)
            local w = tonumber(weight) or 0.0
            if id and id > 0 and w > 0.0 then
                out[id] = math.max(tonumber(out[id]) or 0.0, w)
            end
        end
    end
    return out
end

local function ensure_facet_id(token)
    token = trim(token):lower()
    if token == "" then
        return nil
    end
    M.state.facet_vocab = type(M.state.facet_vocab) == "table" and M.state.facet_vocab or {}
    local facet_id = tonumber(M.state.facet_vocab[token])
    if facet_id and facet_id > 0 then
        return facet_id
    end
    facet_id = math.max(1, math.floor(tonumber(M.state.next_facet_id) or 1))
    M.state.facet_vocab[token] = facet_id
    M.state.next_facet_id = facet_id + 1
    return facet_id
end

local function build_facet_rows_from_text(text, create_missing)
    local weights = extract_text_facet_weights(text)
    local mapped = {}
    for token, weight in pairs(weights) do
        local facet_id = tonumber(((M.state or {}).facet_vocab or {})[token])
        if (not facet_id or facet_id <= 0) and create_missing then
            facet_id = ensure_facet_id(token)
        end
        if facet_id and facet_id > 0 then
            mapped[facet_id] = (tonumber(mapped[facet_id]) or 0.0) + (tonumber(weight) or 0.0)
        end
    end
    return normalize_facet_rows(mapped, facet_slot_cap())
end

local function merge_facet_rows(base_rows, extra_rows)
    local merged = facet_rows_to_map(base_rows or {})
    for facet_id, weight in pairs(facet_rows_to_map(extra_rows or {})) do
        merged[facet_id] = (tonumber(merged[facet_id]) or 0.0) + (tonumber(weight) or 0.0)
    end
    return normalize_facet_rows(merged, facet_slot_cap())
end

local function memory_facet_rows(mem)
    mem = type(mem) == "table" and mem or {}
    local rows = normalize_facet_rows(mem.facets or {}, facet_slot_cap())
    if #rows <= 0 and trim(mem.text) ~= "" then
        rows = build_facet_rows_from_text(mem.text, false)
    end
    return rows
end

local function memory_facet_map(mem_id)
    local mem = M.state.memories[tonumber(mem_id)]
    return facet_rows_to_map(memory_facet_rows(mem))
end

local function build_query_facet_map(user_input)
    return facet_rows_to_map(build_facet_rows_from_text(user_input, false))
end

local function topic_facet_cover(mem_ids)
    local cover = {}
    for _, mem_id in ipairs(mem_ids or {}) do
        for facet_id, weight in pairs(memory_facet_map(mem_id)) do
            cover[facet_id] = math.max(tonumber(cover[facet_id]) or 0.0, tonumber(weight) or 0.0)
        end
    end
    return cover
end

local function mark_dirty()
    M.dirty = true
    local ok, saver = pcall(require, "module.memory.saver")
    if ok and saver and saver.mark_dirty then
        saver.mark_dirty()
    end
end

local function default_state()
    return {
        version = STATE_VERSION,
        next_line = 1,
        next_facet_id = 1,
        current_turn = 0,
        facet_vocab = {},
        memories = {},
        topic_nodes = {},
        topic_edges = {},
        memory_next = {},
        runtime = {
            last_anchor = "",
            last_selected = {},
            last_turn = 0,
        },
        legacy = {
            imported = false,
            adaptive_raw = nil,
        },
    }
end

local function hnsw_cfg()
    local cfg = tg_cfg().topic_hnsw or {}
    return {
        enabled = cfg.enabled ~= false,
        max_elements = math.max(128, tonumber(cfg.max_elements) or 2048),
        m = math.max(4, tonumber(cfg.m) or 16),
        ef_construction = math.max(16, tonumber(cfg.ef_construction) or 80),
        ef_search = math.max(8, tonumber(cfg.ef_search) or 32),
        k = math.max(1, tonumber(cfg.k) or 48),
    }
end

local function default_runtime()
    return {
        loaded_topics = {},
        loaded_order = {},
        hnsw = topic_hnsw_mod.new(hnsw_cfg()),
        last_decay_turn = 0,
        flow_runtime = {},
        flow_memory_next = {},
        scope_index = {},
    }
end

local function reset_state()
    M.state = default_state()
    M.runtime = default_runtime()
    M.dirty = false
end

local function scope_bucket_key(scope_key)
    scope_key = trim(scope_key)
    if scope_key == "" then
        return EMPTY_SCOPE_BUCKET
    end
    return scope_key
end

local function anchor_scope_key(anchor)
    anchor = trim(anchor)
    if anchor == "" then
        return ""
    end
    local scope_key = anchor:match("^(.-)|[ASC]:%d+$")
    return trim(scope_key or "")
end

local function topic_scope_matches(anchor, scope_key)
    local anchor_scope = anchor_scope_key(anchor)
    scope_key = trim(scope_key)
    if anchor_scope == "" or scope_key == "" then
        return anchor_scope == scope_key
    end
    return anchor_scope == scope_key
end

local function retrieve_scope_key(current_anchor, opts)
    opts = type(opts) == "table" and opts or {}
    local explicit_scope = trim(opts.scope_key or "")
    if explicit_scope ~= "" then
        return explicit_scope
    end
    local anchor_scope = anchor_scope_key(current_anchor)
    if anchor_scope ~= "" then
        return anchor_scope
    end
    local flow_key = trim(flow_key_from_opts(opts))
    local flow_scope = flow_key:match("^(.-)%$seg:%d+$")
    return trim(flow_scope or "")
end

local function ensure_scope_index_bucket(scope_key)
    M.runtime = type(M.runtime) == "table" and M.runtime or default_runtime()
    M.runtime.scope_index = type(M.runtime.scope_index) == "table" and M.runtime.scope_index or {}

    local bucket_key = scope_bucket_key(scope_key)
    local bucket = M.runtime.scope_index[bucket_key]
    if type(bucket) ~= "table" then
        bucket = {
            ids = {},
            set = {},
        }
        M.runtime.scope_index[bucket_key] = bucket
    else
        bucket.ids = type(bucket.ids) == "table" and bucket.ids or {}
        bucket.set = type(bucket.set) == "table" and bucket.set or {}
        for _, mem_id in ipairs(bucket.ids) do
            local n = tonumber(mem_id)
            if n and n > 0 then
                bucket.set[n] = true
            end
        end
    end
    return bucket
end

local function index_memory_scope(mem_id, scope_key)
    mem_id = tonumber(mem_id)
    if not mem_id or mem_id <= 0 then
        return false
    end
    local bucket = ensure_scope_index_bucket(scope_key)
    if bucket.set[mem_id] then
        return false
    end
    bucket.ids[#bucket.ids + 1] = mem_id
    bucket.set[mem_id] = true
    return true
end

local function rebuild_scope_index()
    M.runtime = type(M.runtime) == "table" and M.runtime or default_runtime()
    M.runtime.scope_index = {}
    for line = 1, math.max(0, tonumber(M.state.next_line) or 1) - 1 do
        local mem = M.state.memories[line]
        if mem then
            index_memory_scope(line, (mem or {}).scope_key)
        end
    end
end

local function merge_scope_matches(mem, incoming_scope_key)
    local mem_scope_key = trim((mem or {}).scope_key)
    incoming_scope_key = trim(incoming_scope_key)

    if mem_scope_key == "" or incoming_scope_key == "" then
        return mem_scope_key == incoming_scope_key
    end
    return mem_scope_key == incoming_scope_key
end

local function memory_scope_of(mem)
    return normalize_memory_scope((mem or {}).memory_scope)
end

local function memory_visible_for_keys(mem, scope_key, actor_key, thread_key)
    if type(mem) ~= "table" then
        return false
    end
    if not merge_scope_matches(mem, scope_key) then
        return false
    end

    local memory_scope = memory_scope_of(mem)
    if memory_scope == "thread" then
        local mem_thread_key = trim((mem or {}).thread_key or "")
        thread_key = trim(thread_key or "")
        return mem_thread_key ~= "" and thread_key ~= "" and mem_thread_key == thread_key
    end
    if memory_scope == "actor" then
        local mem_actor_key = trim((mem or {}).actor_key or "")
        actor_key = trim(actor_key or "")
        return mem_actor_key ~= "" and actor_key ~= "" and mem_actor_key == actor_key
    end
    return true
end

local function merge_memory_matches(mem, incoming_scope_key, incoming_actor_key, incoming_thread_key, incoming_memory_scope)
    if not merge_scope_matches(mem, incoming_scope_key) then
        return false
    end

    local existing_scope = memory_scope_of(mem)
    local target_scope = normalize_memory_scope(incoming_memory_scope)
    if existing_scope ~= target_scope then
        return false
    end

    if target_scope == "thread" then
        local mem_thread_key = trim((mem or {}).thread_key or "")
        incoming_thread_key = trim(incoming_thread_key or "")
        return mem_thread_key ~= "" and incoming_thread_key ~= "" and mem_thread_key == incoming_thread_key
    end
    if target_scope == "actor" then
        local mem_actor_key = trim((mem or {}).actor_key or "")
        incoming_actor_key = trim(incoming_actor_key or "")
        return mem_actor_key ~= "" and incoming_actor_key ~= "" and mem_actor_key == incoming_actor_key
    end
    return true
end

local function candidate_memory_ids_for_merge(scope_key)
    local bucket = ensure_scope_index_bucket(scope_key)
    return bucket.ids or {}
end

local function ensure_topic_node(anchor)
    anchor = trim(anchor)
    if anchor == "" then
        return nil
    end
    local node = M.state.topic_nodes[anchor]
    if type(node) ~= "table" then
        node = {
            anchor = anchor,
            centroid = {},
            memory_ids = {},
            memory_set = {},
            memory_prior = {},
            local_state = deep_artmap.new_state(),
            last_turn = 0,
        }
        M.state.topic_nodes[anchor] = node
    else
        node.anchor = anchor
        node.memory_ids = unique_sorted_numbers(node.memory_ids)
        node.memory_set = node.memory_set or {}
        for _, mem_id in ipairs(node.memory_ids) do
            node.memory_set[mem_id] = true
        end
        node.memory_prior = type(node.memory_prior) == "table" and node.memory_prior or {}
        node.local_state = deep_artmap.ensure_state(node.local_state)
        node.last_turn = math.max(0, math.floor(tonumber(node.last_turn) or 0))
    end
    return node
end

local function dominant_topic(mem)
    local best_anchor, best_score
    for anchor, score in pairs((mem or {}).origin_topics or {}) do
        local s = tonumber(score) or 0.0
        if s > 0 and ((not best_anchor) or s > best_score) then
            best_anchor = tostring(anchor)
            best_score = s
        end
    end
    if best_anchor then
        return best_anchor
    end
    return trim((mem or {}).topic_anchor)
end

local function update_memory_topic_anchor(mem)
    mem.topic_anchor = dominant_topic(mem) or ""
end

local function recompute_topic_centroid(anchor)
    local node = ensure_topic_node(anchor)
    if not node then
        return {}
    end
    local vectors = {}
    for _, mem_id in ipairs(node.memory_ids or {}) do
        local mem = M.state.memories[tonumber(mem_id)]
        if mem and type(mem.vec) == "table" and #mem.vec > 0 then
            vectors[#vectors + 1] = mem.vec
        end
    end
    node.centroid = average_vectors(vectors)
    node.last_turn = math.max(node.last_turn or 0, tonumber(M.state.current_turn) or 0)
    M.runtime.hnsw:update(anchor, node.centroid)
    return node.centroid
end

local function edge_bucket(src)
    src = trim(src)
    if src == "" then
        return nil
    end
    local bucket = M.state.topic_edges[src]
    if type(bucket) ~= "table" then
        bucket = {}
        M.state.topic_edges[src] = bucket
    end
    return bucket
end

local function ensure_edge(src, dst)
    src = trim(src)
    dst = trim(dst)
    if src == "" or dst == "" or src == dst then
        return nil
    end
    local bucket = edge_bucket(src)
    local edge = bucket[dst]
    if type(edge) ~= "table" then
        edge = {
            transition = 0.0,
            recall = 0.0,
            adopt = 0.0,
            support = 0,
            last_turn = 0,
        }
        bucket[dst] = edge
    end
    return edge
end

local function prune_weight_map(bucket, cap)
    bucket = type(bucket) == "table" and bucket or {}
    cap = math.max(1, math.floor(tonumber(cap) or 64))
    local items = {}
    for key, score in pairs(bucket) do
        local s = tonumber(score)
        if s and s > 0 then
            items[#items + 1] = { key = key, score = s }
        end
    end
    table.sort(items, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return tostring(a.key) < tostring(b.key)
    end)
    local out = {}
    for i = 1, math.min(cap, #items) do
        out[items[i].key] = items[i].score
    end
    return out
end

local function boost_edge(src, dst, kind, amount, turn)
    local edge = ensure_edge(src, dst)
    if not edge then
        return
    end
    local key = tostring(kind or "transition")
    local field = (key == "adopt" and "adopt") or (key == "recall" and "recall") or "transition"
    edge[field] = clamp((tonumber(edge[field]) or 0.0) + math.max(0.0, tonumber(amount) or 0.0), 0.0, 8.0)
    edge.support = (tonumber(edge.support) or 0) + 1
    edge.last_turn = math.max(tonumber(edge.last_turn) or 0, math.floor(tonumber(turn) or 0))
end

local function bind_memory_to_topic(mem_id, anchor, vec, turn, amount, opts)
    mem_id = tonumber(mem_id)
    anchor = trim(anchor)
    if not mem_id or mem_id <= 0 or anchor == "" then
        return nil
    end
    local mem = M.state.memories[mem_id]
    if not mem then
        return nil
    end
    opts = type(opts) == "table" and opts or {}
    local node = ensure_topic_node(anchor)
    local existing_topic_weight = tonumber((type(mem.origin_topics) == "table" and mem.origin_topics[anchor]) or 0.0) or 0.0
    local already_bound = node.memory_set[mem_id] == true or existing_topic_weight > 0.0
    if not node.memory_set[mem_id] then
        node.memory_set[mem_id] = true
        node.memory_ids[#node.memory_ids + 1] = mem_id
        table.sort(node.memory_ids)
    end
    amount = math.max(0.0, tonumber(amount) or 1.0)
    node.memory_prior[mem_id] = (tonumber(node.memory_prior[mem_id]) or 0.0) + amount
    node.memory_prior = prune_weight_map(node.memory_prior, math.max(16, tonumber(tg_cfg().topic_cap) or 64))
    mem.origin_topics = type(mem.origin_topics) == "table" and mem.origin_topics or {}
    mem.origin_topics[anchor] = (tonumber(mem.origin_topics[anchor]) or 0.0) + amount
    update_memory_topic_anchor(mem)
    if type(vec) == "table" and #vec > 0 then
        local local_cfg = tg_cfg().deep_artmap or {}
        local inserted = nil
        local reinforced = false
        if already_bound then
            reinforced = deep_artmap.reinforce_memory(node.local_state, mem_id, "adopt", turn, {
                recall_lr = tonumber(tg_cfg().recall_lr) or 0.10,
                adopt_lr = tonumber(tg_cfg().adopt_lr) or 0.18,
            })
        end
        if not reinforced then
            local art_turn = math.max(0, math.floor(tonumber((node.local_state or {}).last_bundle_turn) or 0)) + 1
            inserted = deep_artmap.add_memory(node.local_state, vec, mem_id, art_turn, {
                category_vigilance = tonumber(local_cfg.category_vigilance) or 0.88,
                category_beta = tonumber(local_cfg.category_beta) or 0.28,
                bundle_vigilance = tonumber(local_cfg.bundle_vigilance) or 0.82,
                bundle_beta = tonumber(local_cfg.bundle_beta) or 0.18,
                temporal_link_window = tonumber(local_cfg.temporal_link_window) or 8,
                exemplars = tonumber(local_cfg.exemplars) or 12,
            })
        end
        if inserted and tonumber(inserted.category_id) then
            mem.cluster_id = tonumber(inserted.category_id) or tonumber(mem.cluster_id) or -1
        elseif opts.preserve_cluster_id and tonumber(mem.cluster_id) then
            mem.cluster_id = tonumber(mem.cluster_id)
        end
    end
    recompute_topic_centroid(anchor)
    mark_dirty()
    return node
end

local function rebuild_runtime_views()
    M.runtime = default_runtime()
    M.runtime.last_decay_turn = tonumber(M.state.current_turn) or 0
    rebuild_scope_index()
    local centroids = {}
    for anchor in pairs(M.state.topic_nodes or {}) do
        local node = ensure_topic_node(anchor)
        if type(node.centroid) ~= "table" or #node.centroid <= 0 then
            recompute_topic_centroid(anchor)
        end
        if type(node.centroid) == "table" and #node.centroid > 0 then
            centroids[anchor] = node.centroid
        end
    end
    M.runtime.hnsw:set_centroids(centroids)
    M.runtime.hnsw:load(hnsw_root())
end

local function current_anchor_for_turn(turn)
    if topic.get_stable_anchor then
        local anchor = topic.get_stable_anchor(turn)
        if trim(anchor) ~= "" then
            return trim(anchor)
        end
    end
    if topic.get_topic_anchor then
        return trim(topic.get_topic_anchor(turn))
    end
    return ""
end

local function export_state()
    local memories = {}
    for line = 1, math.max(0, tonumber(M.state.next_line) or 1) - 1 do
        local mem = M.state.memories[line]
        if mem then
            memories[line] = {
                turns = shallow_copy_array(mem.turns),
                topic_anchor = trim(mem.topic_anchor),
                cluster_id = tonumber(mem.cluster_id) or -1,
                origin_topics = prune_weight_map(mem.origin_topics or {}, 16),
                text = tostring(mem.text or ""),
                facets = normalize_facet_rows(mem.facets or {}, facet_slot_cap()),
                type = tostring(mem.type or "fact"),
                source = tostring(mem.source or ""),
                actor_key = tostring(mem.actor_key or ""),
                scope_key = tostring(mem.scope_key or ""),
                thread_key = tostring(mem.thread_key or ""),
                segment_key = tostring(mem.segment_key or ""),
                memory_scope = normalize_memory_scope(mem.memory_scope),
            }
        end
    end

    local topic_nodes = {}
    for _, anchor in ipairs(sort_string_keys(M.state.topic_nodes)) do
        local node = ensure_topic_node(anchor)
        topic_nodes[anchor] = {
            anchor = anchor,
            centroid = copy_vec(node.centroid),
            memory_ids = unique_sorted_numbers(node.memory_ids),
            memory_prior = prune_weight_map(node.memory_prior, math.max(16, tonumber(tg_cfg().topic_cap) or 64)),
            local_state = {
                categories = node.local_state.categories or {},
                bundles = node.local_state.bundles or {},
                memory_to_category = node.local_state.memory_to_category or {},
                next_category_id = tonumber(node.local_state.next_category_id) or 1,
                next_bundle_id = tonumber(node.local_state.next_bundle_id) or 1,
                last_bundle_id = tonumber(node.local_state.last_bundle_id) or -1,
                last_bundle_turn = tonumber(node.local_state.last_bundle_turn) or 0,
            },
            last_turn = tonumber(node.last_turn) or 0,
        }
    end

    local topic_edges = {}
    for _, src in ipairs(sort_string_keys(M.state.topic_edges)) do
        topic_edges[src] = {}
        for _, dst in ipairs(sort_string_keys(M.state.topic_edges[src])) do
            local edge = M.state.topic_edges[src][dst]
            topic_edges[src][dst] = {
                transition = tonumber(edge.transition) or 0.0,
                recall = tonumber(edge.recall) or 0.0,
                adopt = tonumber(edge.adopt) or 0.0,
                support = tonumber(edge.support) or 0,
                last_turn = tonumber(edge.last_turn) or 0,
            }
        end
    end

    local memory_next = {}
    for _, src in ipairs(sort_numeric_keys(M.state.memory_next)) do
        memory_next[src] = prune_weight_map(M.state.memory_next[src], math.max(8, tonumber(tg_cfg().next_cap) or 32))
    end

    return {
        version = STATE_VERSION,
        next_line = tonumber(M.state.next_line) or 1,
        next_facet_id = tonumber(M.state.next_facet_id) or 1,
        current_turn = tonumber(M.state.current_turn) or 0,
        facet_vocab = type(M.state.facet_vocab) == "table" and M.state.facet_vocab or {},
        memories = memories,
        topic_nodes = topic_nodes,
        topic_edges = topic_edges,
        memory_next = memory_next,
        runtime = {
            last_anchor = trim(((M.state or {}).runtime or {}).last_anchor),
            last_selected = unique_sorted_numbers(((M.state or {}).runtime or {}).last_selected),
            last_turn = tonumber((((M.state or {}).runtime or {}).last_turn)) or 0,
        },
        legacy = ((M.state or {}).legacy or {}),
    }
end

local function save_state_table()
    ensure_dir(storage_root())
    return persistence.write_atomic(state_path(), "w", function(f)
        return f:write(util.encode_lua_value(export_state(), 0))
    end)
end

local function save_vectors()
    ensure_dir(storage_root())
    return persistence.write_atomic(vectors_path(), "wb", function(f)
        local count = 0
        for line = 1, math.max(0, tonumber(M.state.next_line) or 1) - 1 do
            local mem = M.state.memories[line]
            if mem and type(mem.vec) == "table" and #mem.vec > 0 then
                count = count + 1
            end
        end
        local ok_magic, err_magic = f:write(VECTOR_MAGIC)
        if not ok_magic then
            return false, err_magic
        end
        local header = ffi.new("uint32_t[1]", count)
        local ok_head, err_head = f:write(ffi.string(header, 4))
        if not ok_head then
            return false, err_head
        end
        for line = 1, math.max(0, tonumber(M.state.next_line) or 1) - 1 do
            local mem = M.state.memories[line]
            if mem and type(mem.vec) == "table" and #mem.vec > 0 then
                local line_buf = ffi.new("uint32_t[1]", line)
                local ok_line, err_line = f:write(ffi.string(line_buf, 4))
                if not ok_line then
                    return false, err_line
                end
                local vec_bin = tool.vector_to_bin(mem.vec)
                local ok_vec, err_vec = f:write(vec_bin)
                if not ok_vec then
                    return false, err_vec
                end
            end
        end
        return true
    end)
end

local function load_vectors()
    local f = io.open(vectors_path(), "rb")
    if not f then
        return {}
    end
    local data = f:read("*a")
    f:close()
    if (not data) or #data < 8 or data:sub(1, 4) ~= VECTOR_MAGIC then
        return {}
    end
    local p = ffi.cast("const uint8_t*", data)
    local count = tonumber(ffi.cast("const uint32_t*", p + 4)[0]) or 0
    local out = {}
    local offset = 8
    for _ = 1, count do
        if offset + 4 > #data then
            break
        end
        local line = tonumber(ffi.cast("const uint32_t*", p + offset)[0]) or 0
        local vec, used = tool.bin_to_vector(data, offset + 4)
        if line > 0 and type(vec) == "table" and #vec > 0 then
            out[line] = vec
        end
        used = tonumber(used) or 0
        if used <= 0 then
            break
        end
        offset = offset + 4 + used
    end
    return out
end

local function load_state_table()
    local f = io.open(state_path(), "r")
    if not f then
        return nil, "missing_state"
    end
    local raw = f:read("*a")
    f:close()
    local parsed, err = util.parse_lua_table_literal(raw or "")
    if type(parsed) ~= "table" then
        return nil, err or "invalid_state"
    end
    return parsed
end

local function adopt_loaded_state(parsed)
    reset_state()
    parsed = type(parsed) == "table" and parsed or {}
    M.state.version = tostring(parsed.version or STATE_VERSION)
    M.state.next_line = math.max(1, math.floor(tonumber(parsed.next_line) or 1))
    M.state.next_facet_id = math.max(1, math.floor(tonumber(parsed.next_facet_id) or 1))
    M.state.current_turn = math.max(0, math.floor(tonumber(parsed.current_turn) or 0))
    M.state.facet_vocab = type(parsed.facet_vocab) == "table" and parsed.facet_vocab or {}
    M.state.runtime = type(parsed.runtime) == "table" and parsed.runtime or M.state.runtime
    M.state.legacy = type(parsed.legacy) == "table" and parsed.legacy or M.state.legacy
    local vectors = load_vectors()

    for line_str, meta in pairs(parsed.memories or {}) do
        local line = tonumber(line_str)
        if line and line > 0 then
            M.state.memories[line] = {
                turns = shallow_copy_array((meta or {}).turns),
                topic_anchor = trim((meta or {}).topic_anchor),
                cluster_id = tonumber((meta or {}).cluster_id) or -1,
                origin_topics = type((meta or {}).origin_topics) == "table" and (meta or {}).origin_topics or {},
                text = tostring((meta or {}).text or ""),
                facets = normalize_facet_rows((meta or {}).facets or {}, facet_slot_cap()),
                type = tostring((meta or {}).type or "fact"),
                source = tostring((meta or {}).source or ""),
                actor_key = tostring((meta or {}).actor_key or ""),
                scope_key = tostring((meta or {}).scope_key or ""),
                thread_key = tostring((meta or {}).thread_key or ""),
                segment_key = tostring((meta or {}).segment_key or ""),
                memory_scope = normalize_memory_scope((meta or {}).memory_scope),
                vec = vectors[line] or {},
            }
            update_memory_topic_anchor(M.state.memories[line])
        end
    end

    for anchor, node in pairs(parsed.topic_nodes or {}) do
        local ensured = ensure_topic_node(anchor)
        ensured.centroid = copy_vec((node or {}).centroid)
        ensured.memory_ids = unique_sorted_numbers((node or {}).memory_ids)
        ensured.memory_set = {}
        for _, mem_id in ipairs(ensured.memory_ids) do
            ensured.memory_set[mem_id] = true
        end
        ensured.memory_prior = type((node or {}).memory_prior) == "table" and (node or {}).memory_prior or {}
        ensured.local_state = deep_artmap.ensure_state((node or {}).local_state)
        ensured.last_turn = tonumber((node or {}).last_turn) or 0
    end

    for src, bucket in pairs(parsed.topic_edges or {}) do
        M.state.topic_edges[src] = {}
        for dst, edge in pairs(bucket or {}) do
            M.state.topic_edges[src][dst] = {
                transition = tonumber((edge or {}).transition) or 0.0,
                recall = tonumber((edge or {}).recall) or 0.0,
                adopt = tonumber((edge or {}).adopt) or 0.0,
                support = tonumber((edge or {}).support) or 0,
                last_turn = tonumber((edge or {}).last_turn) or 0,
            }
        end
    end

    for src, bucket in pairs(parsed.memory_next or {}) do
        M.state.memory_next[tonumber(src) or src] = type(bucket) == "table" and bucket or {}
    end

    rebuild_runtime_views()
    M.dirty = false
    return true
end

local function import_legacy()
    reset_state()
    local legacy_root = tostring(((config.settings or {}).storage_v3 or {}).root or "memory/v3")
    local legacy_cfg = (tg_cfg().legacy or {})
    local predictor_path = tostring(legacy_cfg.topic_predictor_path or "memory/topic_predictor.txt")
    local adaptive_path = tostring(legacy_cfg.adaptive_state_path or "memory/adaptive_state.txt")
    local legacy_mem = legacy_import.load_memory_v3(legacy_root)
    if not legacy_mem then
        M.state.legacy.imported = false
        return true
    end

    M.state.next_line = math.max(1, tonumber(legacy_mem.next_line) or 1)
    M.state.current_turn = math.max(0, tonumber(legacy_mem.current_turn) or 0)

    for line = 1, M.state.next_line - 1 do
        local mem = legacy_mem.memories[line]
        if mem then
            M.state.memories[line] = {
                turns = shallow_copy_array(mem.turns),
                topic_anchor = trim(mem.topic_anchor),
                cluster_id = tonumber(mem.cluster_id) or -1,
                origin_topics = {},
                text = "",
                facets = {},
                type = "fact",
                vec = copy_vec(mem.vec or {}),
            }
            if trim(mem.topic_anchor) ~= "" then
                bind_memory_to_topic(line, trim(mem.topic_anchor), M.state.memories[line].vec, (mem.turns or {})[#(mem.turns or {})] or 0, 1.0, {
                    preserve_cluster_id = true,
                })
            end
        end
    end

    local predictor_state = legacy_import.load_topic_predictor(predictor_path)
    if predictor_state then
        for anchor, bucket in pairs(predictor_state.topic_memory or {}) do
            local node = ensure_topic_node(anchor)
            for mem_id, score in pairs(bucket or {}) do
                local line = tonumber(mem_id)
                local mem = line and M.state.memories[line] or nil
                if mem then
                    bind_memory_to_topic(line, anchor, mem.vec, (mem.turns or {})[#(mem.turns or {})] or 0, tonumber(score) or 0.0, {
                        preserve_cluster_id = true,
                    })
                    node.memory_prior[line] = math.max(tonumber(node.memory_prior[line]) or 0.0, tonumber(score) or 0.0)
                end
            end
        end
        for src, bucket in pairs(predictor_state.memory_next or {}) do
            M.state.memory_next[tonumber(src) or src] = prune_weight_map(bucket, math.max(8, tonumber(tg_cfg().next_cap) or 32))
        end
        for src, bucket in pairs(predictor_state.topic_transition or {}) do
            for dst, score in pairs(bucket or {}) do
                boost_edge(src, dst, "transition", tonumber(score) or 0.0, M.state.current_turn)
            end
        end
    end

    M.state.legacy.imported = true
    M.state.legacy.adaptive_raw = legacy_import.load_legacy_adaptive(adaptive_path)
    rebuild_runtime_views()
    mark_dirty()
    return true
end

local function save_all()
    local ok_state, err_state = save_state_table()
    if not ok_state then
        return false, err_state
    end
    local ok_vec, err_vec = save_vectors()
    if not ok_vec then
        return false, err_vec
    end
    local ok_hnsw, err_hnsw = M.runtime.hnsw:save(hnsw_root())
    if not ok_hnsw then
        return false, err_hnsw
    end
    M.dirty = false
    return true
end

local function load_or_import()
    local parsed = load_state_table()
    if parsed then
        return adopt_loaded_state(parsed)
    end
    local ok = import_legacy()
    if ok and M.state.legacy.imported == true then
        local save_ok, save_err = save_all()
        if not save_ok then
            print(string.format("[TopicGraph][WARN] legacy import saved failed: %s", tostring(save_err)))
        end
    end
    return true
end

local function get_topic_families(anchors)
    local threshold = clamp(tonumber(tg_cfg().family_similarity) or 0.84, -1.0, 1.0)
    local family_of = {}
    local families = {}
    local next_family = 1
    for _, anchor in ipairs(anchors or {}) do
        local node = M.state.topic_nodes[anchor]
        local assigned = nil
        for _, prev_anchor in ipairs(anchors or {}) do
            if prev_anchor == anchor then
                break
            end
            if family_of[prev_anchor] then
                local prev = M.state.topic_nodes[prev_anchor]
                local sim = safe_similarity((node or {}).centroid, (prev or {}).centroid)
                if sim >= threshold then
                    assigned = family_of[prev_anchor]
                    break
                end
            end
        end
        if not assigned then
            assigned = next_family
            next_family = next_family + 1
        end
        family_of[anchor] = assigned
        families[assigned] = families[assigned] or {}
        families[assigned][#families[assigned] + 1] = anchor
    end
    return family_of, families
end

local function mark_topics_loaded(topics, turn)
    local cap = math.max(1, tonumber(tg_cfg().loaded_cap) or 24)
    for _, anchor in ipairs(topics or {}) do
        anchor = trim(anchor)
        if anchor ~= "" then
            M.runtime.loaded_topics[anchor] = math.max(0, math.floor(tonumber(turn) or 0))
            for i = #M.runtime.loaded_order, 1, -1 do
                if M.runtime.loaded_order[i] == anchor then
                    table.remove(M.runtime.loaded_order, i)
                    break
                end
            end
            M.runtime.loaded_order[#M.runtime.loaded_order + 1] = anchor
        end
    end
    while #M.runtime.loaded_order > cap do
        local removed = table.remove(M.runtime.loaded_order, 1)
        if removed then
            M.runtime.loaded_topics[removed] = nil
        end
    end
end

local function is_topic_loaded(anchor)
    return M.runtime.loaded_topics[trim(anchor)] ~= nil
end

local function semantic_topic_scores(query_vec, current_anchor, scope_key)
    local scores = {}
    local seen = {}
    local hcfg = hnsw_cfg()
    for _, hit in ipairs(M.runtime.hnsw:search(query_vec, math.max(hcfg.k, tonumber(tg_cfg().seed_topics) or 2)) or {}) do
        local anchor = trim(hit.anchor)
        if anchor ~= "" and topic_scope_matches(anchor, scope_key) then
            local score = (tonumber(tg_cfg().query_semantic_weight) or 1.0) * (tonumber(hit.similarity) or 0.0)
            if anchor == current_anchor then
                score = score + (tonumber(tg_cfg().current_topic_bonus) or 0.12)
            end
            scores[anchor] = math.max(tonumber(scores[anchor]) or -1e9, score)
            seen[anchor] = true
        end
    end
    for anchor, node in pairs(M.state.topic_nodes or {}) do
        if not seen[anchor] and topic_scope_matches(anchor, scope_key) then
            local score = (tonumber(tg_cfg().query_semantic_weight) or 1.0) * safe_similarity(query_vec, node.centroid)
            if anchor == current_anchor then
                score = score + (tonumber(tg_cfg().current_topic_bonus) or 0.12)
            end
            scores[anchor] = math.max(tonumber(scores[anchor]) or -1e9, score)
        end
    end
    return scores
end

local function expand_topic_candidates(seed_scores, scope_key)
    local bridge_topk = math.max(1, tonumber(tg_cfg().bridge_topk) or 8)
    local max_hops = math.max(0, tonumber(tg_cfg().max_bridge_hops) or 2)
    local min_bridge = math.max(0.0, tonumber(tg_cfg().min_bridge_score) or 0.08)
    local bridge_weight = math.max(0.0, tonumber(tg_cfg().bridge_weight) or 0.55)
    local resident_bonus = math.max(0.0, tonumber(tg_cfg().resident_bonus) or 0.08)

    local scores = {}
    local frontier = {}
    local via_bridge = {}
    for anchor, score in pairs(seed_scores or {}) do
        anchor = trim(anchor)
        if anchor ~= "" and topic_scope_matches(anchor, scope_key) then
            scores[anchor] = tonumber(score) or 0.0
            frontier[#frontier + 1] = { anchor = anchor, score = tonumber(score) or 0.0 }
        end
    end

    for _ = 1, max_hops do
        table.sort(frontier, function(a, b)
            return (a.score or 0.0) > (b.score or 0.0)
        end)
        local next_frontier = {}
        for i = 1, math.min(bridge_topk, #frontier) do
            local src = frontier[i].anchor
            local bucket = M.state.topic_edges[src] or {}
            local ranked = {}
            for dst, edge in pairs(bucket) do
                local edge_score = math.max(
                    tonumber((edge or {}).transition) or 0.0,
                    tonumber((edge or {}).recall) or 0.0,
                    tonumber((edge or {}).adopt) or 0.0
                )
                if edge_score >= min_bridge and topic_scope_matches(dst, scope_key) then
                    ranked[#ranked + 1] = {
                        anchor = trim(dst),
                        score = edge_score,
                    }
                end
            end
            table.sort(ranked, function(a, b)
                return (a.score or 0.0) > (b.score or 0.0)
            end)
            for j = 1, math.min(bridge_topk, #ranked) do
                local dst = ranked[j].anchor
                local score = (frontier[i].score or 0.0) * 0.60 + bridge_weight * (ranked[j].score or 0.0)
                if is_topic_loaded(dst) then
                    score = score + resident_bonus
                end
                if (scores[dst] or -1e9) < score then
                    if scores[dst] == nil then
                        via_bridge[dst] = true
                    end
                    scores[dst] = score
                    next_frontier[#next_frontier + 1] = { anchor = dst, score = score }
                end
            end
        end
        frontier = next_frontier
        if #frontier <= 0 then
            break
        end
    end

    return scores, sort_string_keys(via_bridge)
end

local function preload_topics(candidate_scores, turn, scope_key)
    local ranked = {}
    local load_budget = math.max(0, math.floor(tonumber(tg_cfg().load_budget) or 4))
    for anchor, score in pairs(candidate_scores or {}) do
        anchor = trim(anchor)
        if anchor ~= "" and topic_scope_matches(anchor, scope_key) then
            ranked[#ranked + 1] = {
                anchor = anchor,
                score = tonumber(score) or 0.0,
            }
        end
    end
    table.sort(ranked, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return tostring(a.anchor) < tostring(b.anchor)
    end)

    local available = {}
    local used_budget = 0
    for _, row in ipairs(ranked) do
        local anchor = trim(row.anchor)
        if anchor ~= "" then
            if is_topic_loaded(anchor) then
                available[#available + 1] = anchor
            elseif used_budget < load_budget then
                used_budget = used_budget + 1
                available[#available + 1] = anchor
            end
        end
    end
    if #available > 0 then
        mark_topics_loaded(available, turn)
    end
    return available
end

local function select_memories_for_context(memory_ids, query_vec, query_map, budget, semantic_scores, local_scores, group_ids)
    budget = math.max(1, math.floor(tonumber(budget) or 1))
    semantic_scores = type(semantic_scores) == "table" and semantic_scores or {}
    local_scores = type(local_scores) == "table" and local_scores or {}
    group_ids = type(group_ids) == "table" and group_ids or {}

    local unique_ids = unique_sorted_numbers(memory_ids or {})
    if #unique_ids <= 0 then
        return {}
    end

    local local_max = 0.0
    for _, score in pairs(local_scores) do
        local s = math.max(0.0, tonumber(score) or 0.0)
        if s > local_max then
            local_max = s
        end
    end
    local query_total_weight = 0.0
    for _, weight in pairs(query_map or {}) do
        query_total_weight = query_total_weight + math.max(0.0, tonumber(weight) or 0.0)
    end
    query_total_weight = math.max(1e-6, query_total_weight)
    local saturation_threshold = clamp(
        tonumber(tg_cfg().context_similarity_saturation_threshold) or 0.95,
        -1.0,
        0.999
    )

    local cache = {}
    for _, mem_id in ipairs(unique_ids) do
        local mem = M.state.memories[tonumber(mem_id)] or {}
        cache[mem_id] = {
            semantic = math.max(0.0, tonumber(semantic_scores[mem_id]) or safe_similarity(query_vec, mem.vec)),
            local_score = (local_max > 1e-6) and (math.max(0.0, tonumber(local_scores[mem_id]) or 0.0) / local_max) or 0.0,
            map = memory_facet_map(mem_id),
            group_id = tonumber(group_ids[mem_id]) or -1,
            vec = mem.vec,
        }
    end

    local remaining = {}
    local selected = {}
    local selected_ids = {}
    local best_facet_cover = {}
    local group_counts = {}
    for _, mem_id in ipairs(unique_ids) do
        remaining[mem_id] = true
    end
    for facet_id in pairs(query_map or {}) do
        best_facet_cover[facet_id] = 0.0
    end

    while next(remaining) and #selected < budget do
        local best_mem = nil
        local best_score = -1e9
        for _, mem_id in ipairs(unique_ids) do
            if remaining[mem_id] then
                local info = cache[mem_id]
                local mem_map = info.map or {}
                local coverage_gain = 0.0
                if next(query_map or {}) and next(mem_map or {}) then
                    for facet_id, query_weight in pairs(query_map or {}) do
                        local current = tonumber(best_facet_cover[facet_id]) or 0.0
                        local updated = tonumber(mem_map[facet_id]) or 0.0
                        if updated > current then
                            coverage_gain = coverage_gain + (tonumber(query_weight) or 0.0) * (updated - current)
                        end
                    end
                    coverage_gain = coverage_gain / query_total_weight
                end

                local max_pair_excess = 0.0
                for _, selected_id in ipairs(selected_ids) do
                    local sim = safe_similarity(info.vec, ((cache[selected_id] or {}).vec))
                    if sim > saturation_threshold then
                        max_pair_excess = math.max(
                            max_pair_excess,
                            (sim - saturation_threshold) / math.max(1e-6, 1.0 - saturation_threshold)
                        )
                    end
                end

                local group_id = tonumber(info.group_id) or -1
                local group_repeat = (group_id >= 0) and (tonumber(group_counts[group_id]) or 0) or 0
                local group_bonus = (group_id >= 0 and group_repeat <= 0) and 0.08 or 0.0
                local group_penalty = (group_repeat > 0) and (0.05 * (group_repeat / (group_repeat + 1.0))) or 0.0
                local semantic = tonumber(info.semantic) or 0.0
                local local_score = tonumber(info.local_score) or 0.0
                local context_score
                if next(query_map or {}) then
                    context_score = coverage_gain
                        + 0.50 * semantic
                        + 0.22 * local_score
                        + group_bonus
                        - 0.14 * max_pair_excess
                        - group_penalty
                else
                    context_score = 0.72 * semantic
                        + 0.28 * local_score
                        + group_bonus
                        - 0.14 * max_pair_excess
                        - group_penalty
                end
                if context_score > best_score then
                    best_score = context_score
                    best_mem = mem_id
                end
            end
        end

        if not best_mem then
            break
        end

        selected[#selected + 1] = {
            mem_idx = best_mem,
            score = best_score,
        }
        selected_ids[#selected_ids + 1] = best_mem
        remaining[best_mem] = nil

        local info = cache[best_mem] or {}
        for facet_id in pairs(query_map or {}) do
            best_facet_cover[facet_id] = math.max(
                tonumber(best_facet_cover[facet_id]) or 0.0,
                tonumber(((info.map or {})[facet_id])) or 0.0
            )
        end
        local group_id = tonumber(info.group_id) or -1
        if group_id >= 0 then
            group_counts[group_id] = (tonumber(group_counts[group_id]) or 0) + 1
        end
    end

    return selected
end

local function select_topics_for_context(topic_rows, query_map)
    if #(topic_rows or {}) <= 0 then
        return {}
    end

    local max_topics = math.max(1, tonumber(tg_cfg().max_return_topics) or 4)
    local query_total_weight = 0.0
    for _, weight in pairs(query_map or {}) do
        query_total_weight = query_total_weight + math.max(0.0, tonumber(weight) or 0.0)
    end
    query_total_weight = math.max(1e-6, query_total_weight)
    local family_limit = math.max(1, tonumber(tg_cfg().family_member_limit) or 1)

    local selected = {}
    local selected_set = {}
    local best_facet_cover = {}
    local family_counts = {}
    for facet_id in pairs(query_map or {}) do
        best_facet_cover[facet_id] = 0.0
    end

    while #selected < max_topics do
        local best_anchor = nil
        local best_score = -1e9
        local best_cover = nil
        local best_family = 0
        for _, row in ipairs(topic_rows or {}) do
            local anchor = trim((row or {}).anchor)
            if anchor ~= "" and not selected_set[anchor] then
                local family_id = tonumber((row or {}).family_id) or 0
                local family_repeat = (family_id > 0) and (tonumber(family_counts[family_id]) or 0) or 0
                if family_id <= 0 or family_repeat < family_limit or #selected <= 0 then
                    local topic_cover = type((row or {}).topic_cover) == "table" and (row or {}).topic_cover or {}
                    local coverage_gain = 0.0
                    if next(query_map or {}) and next(topic_cover or {}) then
                        for facet_id, query_weight in pairs(query_map or {}) do
                            local current = tonumber(best_facet_cover[facet_id]) or 0.0
                            local updated = tonumber(topic_cover[facet_id]) or 0.0
                            if updated > current then
                                coverage_gain = coverage_gain + (tonumber(query_weight) or 0.0) * (updated - current)
                            end
                        end
                        coverage_gain = coverage_gain / query_total_weight
                    end

                    local peak_sim = math.max(0.0, tonumber((row or {}).peak_sim) or 0.0)
                    local mean_sim = math.max(0.0, tonumber((row or {}).mean_sim) or 0.0)
                    local route_score = math.max(0.0, math.min(1.0, tonumber((row or {}).route_score) or 0.0))
                    local semantic_support = math.min(1.0, 0.55 * peak_sim + 0.20 * mean_sim + 0.25 * route_score)
                    local novelty_bonus = (family_id > 0 and family_repeat <= 0) and 0.10 or 0.0
                    local repeat_penalty = (family_repeat > 0) and (0.06 * (family_repeat / (family_repeat + 1.0))) or 0.0
                    local context_score = coverage_gain + 0.55 * semantic_support + novelty_bonus - repeat_penalty
                    if context_score > best_score then
                        best_anchor = anchor
                        best_score = context_score
                        best_cover = topic_cover
                        best_family = family_id
                    end
                end
            end
        end

        if not best_anchor then
            break
        end

        selected[#selected + 1] = {
            anchor = best_anchor,
            score = best_score,
        }
        selected_set[best_anchor] = true
        if type(best_cover) == "table" then
            for facet_id, weight in pairs(best_cover) do
                best_facet_cover[facet_id] = math.max(
                    tonumber(best_facet_cover[facet_id]) or 0.0,
                    tonumber(weight) or 0.0
                )
            end
        end
        if best_family > 0 then
            family_counts[best_family] = (tonumber(family_counts[best_family]) or 0) + 1
        end
    end

    if #selected > 0 then
        return selected
    end

    local fallback = {}
    for _, row in ipairs(topic_rows or {}) do
        local anchor = trim((row or {}).anchor)
        if anchor ~= "" then
            fallback[#fallback + 1] = {
                anchor = anchor,
                score = tonumber((row or {}).total_score) or 0.0,
            }
        end
    end
    table.sort(fallback, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return tostring(a.anchor) < tostring(b.anchor)
    end)
    while #fallback > max_topics do
        table.remove(fallback)
    end
    return fallback
end

local function one_turn_momentum_scores(query_vec, current_anchor, current_turn, opts)
    local scores = {}

    opts = type(opts) == "table" and opts or {}
    local flow_key = flow_key_from_opts(opts)
    local scope_key = retrieve_scope_key(current_anchor, opts)
    local actor_key = actor_key_from_opts(opts)
    local thread_key = thread_key_from_opts(opts)
    local rt = ensure_runtime_state(flow_key)
    local memory_next = M.state.memory_next
    if flow_key ~= "" then
        M.runtime = type(M.runtime) == "table" and M.runtime or {}
        M.runtime.flow_memory_next = type(M.runtime.flow_memory_next) == "table" and M.runtime.flow_memory_next or {}
        memory_next = M.runtime.flow_memory_next[flow_key]
        if type(memory_next) ~= "table" then
            memory_next = {}
            M.runtime.flow_memory_next[flow_key] = memory_next
        end
    end
    local cur_turn = math.floor(tonumber(current_turn) or 0)
    local gap = cur_turn - (tonumber(rt.last_turn) or 0)

    if flow_key == "" then
        if gap ~= 1 then
            return scores
        end
    else
        if gap <= 0 then
            return scores
        end
    end

    if trim(rt.last_anchor) ~= trim(current_anchor) then
        return scores
    end

    local recent_weight = math.max(0.0, tonumber(tg_cfg().recent_weight) or 0.55)
    local gap_factor = 1.0
    if flow_key ~= "" and gap > 1 then
        gap_factor = 1.0 / math.sqrt(gap)
    end
    for _, prev_mem in ipairs(rt.last_selected or {}) do
        local bucket = memory_next[tonumber(prev_mem)] or {}
        for mem_id, weight in pairs(bucket) do
            local mem = M.state.memories[tonumber(mem_id)]
            if mem
                and memory_visible_for_keys(mem, scope_key, actor_key, thread_key)
                and type(mem.vec) == "table"
                and #mem.vec > 0
            then
                scores[tonumber(mem_id)] = math.max(
                    tonumber(scores[tonumber(mem_id)]) or -1e9,
                    safe_similarity(query_vec, mem.vec) + recent_weight * gap_factor * (tonumber(weight) or 0.0)
                )
            end
        end
    end
    return scores
end

local function retrieve_topic_evidence(query_vec, candidate_scores, available_topics, current_anchor, current_turn, query_map, opts)
    local per_topic_evidence = math.max(1, tonumber(tg_cfg().per_topic_evidence) or 3)
    local local_cfg = tg_cfg().deep_artmap or {}
    local family_of = get_topic_families(available_topics)
    local momentum_scores = one_turn_momentum_scores(query_vec, current_anchor, current_turn, opts)
    local scope_key = retrieve_scope_key(current_anchor, opts)
    local actor_key = actor_key_from_opts(opts)
    local thread_key = thread_key_from_opts(opts)
    local evidence_topics = {}
    local evidence_memories = {}
    local evidence_by_topic = {}
    local evidence_rows = {}
    local local_signals = {}
    local topic_rows = {}

    local function visible_memory(mem_id)
        local mem = M.state.memories[tonumber(mem_id)]
        if memory_visible_for_keys(mem, scope_key, actor_key, thread_key) then
            return mem
        end
        return nil
    end

    for _, anchor in ipairs(available_topics or {}) do
        local node = ensure_topic_node(anchor)
        if node and #(node.memory_ids or {}) > 0 then
            local candidates, local_debug = deep_artmap.collect_candidates(
                node.local_state,
                query_vec,
                function(mem_id)
                    local mem = visible_memory(mem_id)
                    return mem and mem.vec or nil
                end,
                function(mem_id)
                    local mem = visible_memory(mem_id)
                    return memory_facet_rows(mem)
                end,
                {
                    query_bundles = tonumber(local_cfg.query_bundles) or 2,
                    query_margin = tonumber(local_cfg.query_margin) or 0.06,
                    neighbor_topk = tonumber(local_cfg.neighbor_topk) or 2,
                    query_categories = tonumber(local_cfg.query_categories) or 4,
                    exemplars = tonumber(local_cfg.exemplars) or 12,
                    bundle_vigilance = tonumber(local_cfg.bundle_vigilance) or 0.82,
                    bundle_prior_weight = tonumber(local_cfg.bundle_prior_weight) or 0.18,
                    saturation_threshold = tonumber(tg_cfg().context_similarity_saturation_threshold) or 0.95,
                    max_results = per_topic_evidence * 4,
                    query_map = query_map,
                }
            )

            local local_score_map = {}
            local local_group_map = {}
            local local_category_map = {}
            local local_bundle_map = {}
            local candidate_ids = {}
            local seen_ids = {}
            for _, item in ipairs(candidates or {}) do
                local mem_id = tonumber((item or {}).mem_idx)
                local mem = visible_memory(mem_id)
                if mem_id and mem_id > 0 and mem then
                    if not seen_ids[mem_id] then
                        seen_ids[mem_id] = true
                        candidate_ids[#candidate_ids + 1] = mem_id
                    end
                    local_score_map[mem_id] = math.max(
                        tonumber(local_score_map[mem_id]) or -1e9,
                        tonumber((item or {}).score) or 0.0
                    )
                    local_group_map[mem_id] = tonumber((item or {}).bundle_id) or 0
                    local_category_map[mem_id] = tonumber((item or {}).category_id) or 0
                    local_bundle_map[mem_id] = tonumber((item or {}).bundle_id) or 0
                end
            end

            if trim(anchor) == trim(current_anchor) then
                for mem_id, score in pairs(momentum_scores) do
                    if not seen_ids[mem_id] then
                        seen_ids[mem_id] = true
                        candidate_ids[#candidate_ids + 1] = mem_id
                    end
                    local_score_map[mem_id] = math.max(tonumber(local_score_map[mem_id]) or -1e9, tonumber(score) or 0.0)
                end
            end

            if #candidate_ids <= 0 then
                for _, mem_id in ipairs(node.memory_ids or {}) do
                    mem_id = tonumber(mem_id)
                    local mem = visible_memory(mem_id)
                    if mem_id and mem_id > 0 and mem and not seen_ids[mem_id] then
                        seen_ids[mem_id] = true
                        candidate_ids[#candidate_ids + 1] = mem_id
                    end
                    if mem then
                        local_score_map[mem_id] = safe_similarity(query_vec, mem.vec)
                        local_group_map[mem_id] = tonumber(mem.cluster_id) or 0
                        local_category_map[mem_id] = tonumber(mem.cluster_id) or 0
                        local_bundle_map[mem_id] = 0
                    end
                end
            end

            local semantic_map = {}
            for _, mem_id in ipairs(candidate_ids) do
                local mem = visible_memory(mem_id)
                semantic_map[mem_id] = safe_similarity(query_vec, mem and mem.vec)
            end

            local ranked_local = select_memories_for_context(
                candidate_ids,
                query_vec,
                query_map,
                per_topic_evidence,
                semantic_map,
                local_score_map,
                local_group_map
            )
            local best_memories = {}
            local best_rows = {}
            local peak_sim = 0.0
            local mean_sim = 0.0
            for _, item in ipairs(ranked_local) do
                local mem_id = tonumber(item.mem_idx) or 0
                if mem_id > 0 then
                    local sim = tonumber(semantic_map[mem_id]) or 0.0
                    best_memories[#best_memories + 1] = mem_id
                    best_rows[#best_rows + 1] = {
                        mem_idx = mem_id,
                        score = tonumber(item.score) or 0.0,
                        anchor = anchor,
                        category_id = tonumber(local_category_map[mem_id]) or 0,
                        bundle_id = tonumber(local_bundle_map[mem_id]) or 0,
                    }
                    peak_sim = math.max(peak_sim, sim)
                    mean_sim = mean_sim + sim
                end
            end
            mean_sim = (#best_memories > 0) and (mean_sim / #best_memories) or 0.0

            local evidence_score = math.max(tonumber((local_debug or {}).score) or 0.0, peak_sim) + 0.15 * mean_sim
            local total_score = (tonumber(candidate_scores[anchor]) or 0.0) + evidence_score
            evidence_topics[#evidence_topics + 1] = anchor
            for _, mem_id in ipairs(best_memories) do
                evidence_memories[#evidence_memories + 1] = mem_id
            end
            evidence_by_topic[anchor] = best_memories
            evidence_rows[anchor] = best_rows
            local_signals[anchor] = {
                bundle_ids = shallow_copy_array(((local_debug or {}).bundle_ids) or {}),
                category_ids = shallow_copy_array(((local_debug or {}).category_ids) or {}),
                route = tostring(((local_debug or {}).route) or ""),
            }
            topic_rows[#topic_rows + 1] = {
                anchor = anchor,
                total_score = total_score,
                route_score = tonumber(candidate_scores[anchor]) or 0.0,
                peak_sim = peak_sim,
                mean_sim = mean_sim,
                topic_cover = topic_facet_cover(best_memories),
                family_id = tonumber(family_of[anchor]) or 0,
            }
        end
    end

    local ranked_topics = select_topics_for_context(topic_rows, query_map)
    local selected_topics = {}
    for _, row in ipairs(ranked_topics or {}) do
        selected_topics[#selected_topics + 1] = trim((row or {}).anchor)
    end
    return selected_topics, {
        evidence_topics = evidence_topics,
        evidence_memories = unique_sorted_numbers(evidence_memories),
        evidence_by_topic = evidence_by_topic,
        ranked_topics = ranked_topics,
        local_signals = local_signals,
        topic_rows = topic_rows,
    }, evidence_rows
end

local function ranked_memories_from_evidence(selected_topics, evidence_rows)
    local merged = {}
    for _, anchor in ipairs(selected_topics or {}) do
        for _, item in ipairs((evidence_rows or {})[anchor] or {}) do
            local mem_id = tonumber((item or {}).mem_idx)
            if mem_id and mem_id > 0 then
                local current = merged[mem_id]
                if not current or (tonumber(item.score) or 0.0) > (tonumber(current.score) or -1e9) then
                    merged[mem_id] = {
                        mem_idx = mem_id,
                        score = tonumber(item.score) or 0.0,
                        anchor = trim((item or {}).anchor),
                        category_id = tonumber((item or {}).category_id) or 0,
                        bundle_id = tonumber((item or {}).bundle_id) or 0,
                    }
                end
            end
        end
    end
    local ranked = {}
    for _, item in pairs(merged) do
        ranked[#ranked + 1] = item
    end
    table.sort(ranked, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.mem_idx or 0) < (b.mem_idx or 0)
    end)
    return ranked
end

local function representative_turn_rows(rows, cap)
    cap = math.max(1, math.floor(tonumber(cap) or 2))
    local ranked = {}
    local seen_turns = {}
    for _, item in ipairs(rows or {}) do
        local mem = M.state.memories[tonumber((item or {}).mem_idx)]
        local best_turn = 0
        for _, turn in ipairs((mem or {}).turns or {}) do
            local t = tonumber(turn) or 0
            if t > best_turn then
                best_turn = t
            end
        end
        if best_turn > 0 and not seen_turns[best_turn] then
            seen_turns[best_turn] = true
            ranked[#ranked + 1] = {
                turn = best_turn,
                score = tonumber((item or {}).score) or 0.0,
                mem_idx = tonumber((item or {}).mem_idx) or 0,
            }
        end
    end
    table.sort(ranked, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.turn or 0) > (b.turn or 0)
    end)
    while #ranked > cap do
        table.remove(ranked)
    end
    return ranked
end

local function topic_summary_fallback(rows)
    local turn_rows = representative_turn_rows(rows, 2)
    if #turn_rows <= 0 then
        return ""
    end

    local parts = {}
    for _, row in ipairs(turn_rows) do
        local entry = history.get_by_turn(row.turn)
        if entry then
            local user_text, ai_text = history.parse_entry(entry)
            local user_compact = util.utf8_take(trim(user_text), 80)
            local ai_compact = util.utf8_take(trim(ai_text), 100)
            if user_compact ~= "" and ai_compact ~= "" then
                parts[#parts + 1] = string.format(
                    "第%d轮：用户提到%s；助手回应%s",
                    row.turn,
                    user_compact,
                    ai_compact
                )
            elseif user_compact ~= "" then
                parts[#parts + 1] = string.format("第%d轮：用户提到%s", row.turn, user_compact)
            elseif ai_compact ~= "" then
                parts[#parts + 1] = string.format("第%d轮：助手回应%s", row.turn, ai_compact)
            end
        end
    end

    if #parts <= 0 then
        return ""
    end
    return "当前活跃 topic 尚未生成闭环摘要。代表片段：" .. table.concat(parts, "；")
end

local function topic_context_from_selection(selected_topics, evidence_rows)
    local context_lines = {}
    for i, anchor in ipairs(selected_topics or {}) do
        anchor = trim(anchor)
        if anchor ~= "" then
            local fp = topic.get_topic_fingerprint and topic.get_topic_fingerprint(anchor) or {}
            local summary = trim((fp or {}).summary)
            if summary == "" then
                summary = topic_summary_fallback((evidence_rows or {})[anchor] or {})
            end
            if summary ~= "" then
                local label = (((fp or {}).is_active) == true) and "当前主题" or "相关主题"
                local extra = {}
                if tonumber((fp or {}).start) then
                    extra[#extra + 1] = string.format("起始轮次=%d", tonumber(fp.start) or 0)
                end
                if tonumber((fp or {}).memory_count) and tonumber(fp.memory_count) > 0 then
                    extra[#extra + 1] = string.format("memory=%d", tonumber(fp.memory_count) or 0)
                end
                local head = string.format("%s%d", label, i)
                if #extra > 0 then
                    head = head .. "（" .. table.concat(extra, "，") .. "）"
                end
                context_lines[#context_lines + 1] = head .. "\n概况：" .. summary
            end
        end
    end

    if #context_lines <= 0 then
        return ""
    end
    return "【相关主题】\n" .. table.concat(context_lines, "\n\n")
end

local function turns_from_memories(ranked_memories)
    local limit = math.max(1, tonumber(tg_cfg().retrieve_max_turns) or tonumber(ai_cfg().max_turns) or 10)
    local turn_best = {}
    local turn_mem = {}
    for _, item in ipairs(ranked_memories or {}) do
        local mem = M.state.memories[tonumber(item.mem_idx)]
        for _, turn in ipairs((mem or {}).turns or {}) do
            local t = tonumber(turn)
            if t and t > 0 then
                local score = tonumber(item.score) or 0.0
                if (turn_best[t] or -1e9) < score then
                    turn_best[t] = score
                    turn_mem[t] = tonumber(item.mem_idx)
                end
            end
        end
    end
    local ranked_turns = {}
    for turn, score in pairs(turn_best) do
        ranked_turns[#ranked_turns + 1] = {
            turn = tonumber(turn) or 0,
            score = tonumber(score) or 0.0,
            mem_idx = tonumber(turn_mem[turn]) or 0,
        }
    end
    table.sort(ranked_turns, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.turn or 0) > (b.turn or 0)
    end)
    local selected_turns = {}
    local fragments = {}
    local context_lines = {}
    for i = 1, math.min(limit, #ranked_turns) do
        local item = ranked_turns[i]
        selected_turns[#selected_turns + 1] = item.turn
        local entry = history.get_by_turn(item.turn)
        if entry then
            local user_text, ai_text = history.parse_entry(entry)
            local fragment_text = string.format("第%d轮 用户：%s\n助手：%s", item.turn, tostring(user_text or ""), tostring(ai_text or ""))
            fragments[#fragments + 1] = {
                turn = item.turn,
                mem_idx = item.mem_idx,
                user = tostring(user_text or ""),
                assistant = tostring(ai_text or ""),
                text = fragment_text,
                source = "topic_graph",
                score = tonumber(item.score) or 0.0,
            }
            context_lines[#context_lines + 1] = fragment_text
        end
    end
    local context = ""
    if #context_lines > 0 then
        context = "【相关记忆】\n" .. table.concat(context_lines, "\n\n")
    end
    return selected_turns, fragments, context
end

local function apply_decay(turn)
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    local last = math.max(0, math.floor(tonumber(M.runtime.last_decay_turn) or 0))
    if turn <= 0 or turn <= last then
        return
    end
    local decay = clamp(tonumber(tg_cfg().decay) or 0.995, 0.0, 1.0)
    local delta = turn - last
    if delta <= 0 or decay >= 0.999999 then
        M.runtime.last_decay_turn = turn
        return
    end
    local factor = decay ^ delta
    local bundle_decay = clamp(tonumber((((tg_cfg().deep_artmap) or {}).bundle_decay)) or decay, 0.0, 1.0)
    local bundle_factor = bundle_decay ^ delta
    for src, bucket in pairs(M.state.topic_edges or {}) do
        for dst, edge in pairs(bucket or {}) do
            edge.transition = (tonumber(edge.transition) or 0.0) * factor
            edge.recall = (tonumber(edge.recall) or 0.0) * factor
            edge.adopt = (tonumber(edge.adopt) or 0.0) * factor
            if (edge.transition + edge.recall + edge.adopt) < 1e-4 then
                bucket[dst] = nil
            end
        end
        if not next(bucket) then
            M.state.topic_edges[src] = nil
        end
    end
    for src, bucket in pairs(M.state.memory_next or {}) do
        for dst, weight in pairs(bucket or {}) do
            local v = (tonumber(weight) or 0.0) * factor
            if v < 1e-4 then
                bucket[dst] = nil
            else
                bucket[dst] = v
            end
        end
        if not next(bucket) then
            M.state.memory_next[src] = nil
        end
    end
    for _, node in pairs(M.state.topic_nodes or {}) do
        deep_artmap.decay(node.local_state, bundle_factor)
    end
    M.runtime.last_decay_turn = turn
    mark_dirty()
end

function M.load()
    local ok, err = load_or_import()
    if not ok then
        return nil, err
    end
    return true
end

function M.save_to_disk()
    return save_all()
end

function M.begin_turn(turn)
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    gc_flow_runtime(turn)
    if turn > (tonumber(M.state.current_turn) or 0) then
        M.state.current_turn = turn
        apply_decay(turn)
        mark_dirty()
    end
    return true
end

function M.get_total_lines()
    return math.max(0, (tonumber(M.state.next_line) or 1) - 1)
end

function M.get_turns(line)
    local mem = M.state.memories[tonumber(line)]
    return shallow_copy_array((mem or {}).turns)
end

function M.get_topic_anchor(line)
    return trim(((M.state.memories[tonumber(line)] or {}).topic_anchor))
end

function M.get_cluster_id(line)
    return tonumber(((M.state.memories[tonumber(line)] or {}).cluster_id)) or -1
end

function M.return_mem_vec(line)
    local mem = M.state.memories[tonumber(line)]
    if not mem or type(mem.vec) ~= "table" or #mem.vec <= 0 then
        return nil
    end
    return mem.vec
end

function M.iterate_all()
    local idx = 0
    local max_idx = M.get_total_lines()
    return function()
        idx = idx + 1
        while idx <= max_idx do
            local mem = M.state.memories[idx]
            if mem then
                return {
                    turns = shallow_copy_array(mem.turns),
                    cluster_id = tonumber(mem.cluster_id) or -1,
                    topic_anchor = trim(mem.topic_anchor),
                    text = tostring(mem.text or ""),
                    facets = normalize_facet_rows(mem.facets or {}, facet_slot_cap()),
                    vec = mem.vec,
                }
            end
            idx = idx + 1
        end
        return nil
    end
end

function M.iter_topic_lines(anchor)
    local node = ensure_topic_node(anchor)
    if not node then
        return {}
    end
    return unique_sorted_numbers(node.memory_ids)
end

function M.store_vector(line, cluster_id, vec)
    line = tonumber(line)
    if not line or line <= 0 or type(vec) ~= "table" or #vec <= 0 then
        return nil, "invalid_args"
    end
    local mem = M.state.memories[line]
    if not mem then
        return nil, "missing_memory"
    end
    mem.vec = normalize(vec)
    mem.cluster_id = tonumber(cluster_id) or tonumber(mem.cluster_id) or -1
    local anchors = {}
    if type(mem.origin_topics) == "table" then
        for anchor in pairs(mem.origin_topics) do
            anchor = trim(anchor)
            if anchor ~= "" then
                anchors[#anchors + 1] = anchor
            end
        end
    end
    if #anchors <= 0 and trim(mem.topic_anchor) ~= "" then
        anchors[1] = trim(mem.topic_anchor)
    end
    for _, anchor in ipairs(anchors) do
        recompute_topic_centroid(anchor)
    end
    mark_dirty()
    return true
end

function M.find_similar_all_fast(query_vec, max_results)
    local ranked = {}
    local limit = math.max(1, math.floor(tonumber(max_results) or 8))
    for line = 1, M.get_total_lines() do
        local mem = M.state.memories[line]
        if mem and type(mem.vec) == "table" and #mem.vec > 0 then
            ranked[#ranked + 1] = {
                index = line,
                similarity = safe_similarity(query_vec, mem.vec),
            }
        end
    end
    table.sort(ranked, function(a, b)
        if (a.similarity or 0.0) ~= (b.similarity or 0.0) then
            return (a.similarity or 0.0) > (b.similarity or 0.0)
        end
        return (a.index or 0) < (b.index or 0)
    end)
    while #ranked > limit do
        table.remove(ranked)
    end
    return ranked
end

function M.add_memory(vec, turn, opts)
    opts = type(opts) == "table" and opts or {}
    vec = normalize(vec or {})
    if #vec <= 0 then
        return nil, "invalid_vector"
    end
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    local anchor = trim(opts.topic_anchor or current_anchor_for_turn(turn))
    local text = tostring(opts.text or opts.fact or "")
    local facets = normalize_facet_rows(opts.facets or {}, facet_slot_cap())
    local actor_key = trim(opts.actor_key or "")
    local scope_key = trim(opts.scope_key or "")
    local thread_key = trim(opts.thread_key or "")
    local segment_key = trim(opts.segment_key or "")
    local memory_scope = normalize_memory_scope(opts.memory_scope)
    local turns = unique_sorted_numbers(opts.turns or { turn })
    if #turns <= 0 then
        turns = { turn }
    end
    if #facets <= 0 and text ~= "" then
        facets = build_facet_rows_from_text(text, true)
    end
    local merge_limit = tonumber(config.settings.merge_limit) or 0.95

    local best_line, best_sim = nil, -1.0
    for _, line in ipairs(candidate_memory_ids_for_merge(scope_key)) do
        local mem = M.state.memories[line]
        if mem
            and merge_memory_matches(mem, scope_key, actor_key, thread_key, memory_scope)
            and type(mem.vec) == "table"
            and #mem.vec > 0
        then
            local sim = safe_similarity(vec, mem.vec)
            if sim > best_sim then
                best_line = line
                best_sim = sim
            end
        end
    end

    if best_line and best_sim >= merge_limit and opts.allow_merge ~= false then
        local mem = M.state.memories[best_line]
        mem.turns = unique_sorted_numbers((function()
            local merged_turns = {}
            for _, existing_turn in ipairs(mem.turns or {}) do
                merged_turns[#merged_turns + 1] = existing_turn
            end
            for _, incoming_turn in ipairs(turns) do
                merged_turns[#merged_turns + 1] = incoming_turn
            end
            return merged_turns
        end)())
        if #mem.turns <= 0 then
            mem.turns = shallow_copy_array(turns)
        end
        if text ~= "" and trim(mem.text) == "" then
            mem.text = text
        end
        if trim(mem.source or "") == "" and trim(opts.source or "") ~= "" then
            mem.source = tostring(opts.source or "")
        end
        if trim(mem.actor_key or "") == "" and actor_key ~= "" then
            mem.actor_key = actor_key
        end
        if trim(mem.scope_key or "") == "" and scope_key ~= "" then
            mem.scope_key = scope_key
        end
        if trim(mem.thread_key or "") == "" and thread_key ~= "" then
            mem.thread_key = thread_key
        end
        if trim(mem.segment_key or "") == "" and segment_key ~= "" then
            mem.segment_key = segment_key
        end
        mem.memory_scope = memory_scope_of(mem)
        mem.facets = merge_facet_rows(mem.facets or {}, facets)
        if anchor ~= "" then
            bind_memory_to_topic(best_line, anchor, mem.vec, turn, 1.0, { preserve_cluster_id = true })
        end
        mark_dirty()
        return best_line
    end

    local line = math.max(1, tonumber(M.state.next_line) or 1)
    M.state.next_line = line + 1
    M.state.memories[line] = {
        turns = turns,
        topic_anchor = anchor,
        cluster_id = -1,
        origin_topics = {},
        text = text,
        facets = facets,
        type = tostring(opts.kind or "fact"),
        source = tostring(opts.source or ""),
        actor_key = actor_key,
        scope_key = scope_key,
        thread_key = thread_key,
        segment_key = segment_key,
        memory_scope = memory_scope,
        vec = vec,
    }
    index_memory_scope(line, scope_key)
    if anchor ~= "" then
        bind_memory_to_topic(line, anchor, vec, turn, 1.0)
    end
    mark_dirty()
    return line
end

function M.update_topic_anchor(line, new_anchor)
    line = tonumber(line)
    if not line or line <= 0 then
        return false
    end
    local mem = M.state.memories[line]
    if not mem then
        return false
    end
    new_anchor = trim(new_anchor)
    if new_anchor == "" then
        return false
    end
    bind_memory_to_topic(line, new_anchor, mem.vec, (mem.turns or {})[#(mem.turns or {})] or 0, 1.0, {
        preserve_cluster_id = true,
    })
    mark_dirty()
    return true
end

function M.reset_flow(flow_key)
    flow_key = trim(flow_key)
    if flow_key == "" then
        local rt = ensure_runtime_state("")
        rt.last_anchor = ""
        rt.last_selected = {}
        rt.last_turn = 0
        return true
    end
    M.runtime = type(M.runtime) == "table" and M.runtime or {}
    M.runtime.flow_runtime = type(M.runtime.flow_runtime) == "table" and M.runtime.flow_runtime or {}
    M.runtime.flow_memory_next = type(M.runtime.flow_memory_next) == "table" and M.runtime.flow_memory_next or {}
    M.runtime.flow_runtime[flow_key] = nil
    M.runtime.flow_memory_next[flow_key] = nil
    return true
end

function M.observe_turn(turn, current_anchor, opts)
    opts = type(opts) == "table" and opts or {}
    local flow_key = flow_key_from_opts(opts)
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    current_anchor = trim(current_anchor)
    if current_anchor == "" then
        return false
    end
    M.begin_turn(turn)
    local rt = ensure_runtime_state(flow_key)
    local prev_anchor = trim(rt.last_anchor or "")
    local prev_turn = tonumber(rt.last_turn) or 0
    if flow_key == "" and prev_anchor ~= "" and prev_anchor ~= current_anchor and prev_turn > 0 then
        local prev_scope = prev_anchor:match("^(.-)|") or ""
        local cur_scope = current_anchor:match("^(.-)|") or ""
        if prev_scope ~= cur_scope and (prev_scope ~= "" or cur_scope ~= "") then
            -- scope boundary: do not learn cross-scope transitions
        else
            boost_edge(prev_anchor, current_anchor, "transition", tonumber(tg_cfg().transition_lr) or 0.12, turn)
        end
    end
    return true
end

function M.retrieve(query_vec, current_anchor, current_turn, opts)
    opts = type(opts) == "table" and opts or {}
    local flow_key = flow_key_from_opts(opts)
    local scope_key = retrieve_scope_key(current_anchor, opts)
    query_vec = normalize(query_vec or {})
    current_anchor = trim(current_anchor)
    current_turn = math.max(0, math.floor(tonumber(current_turn) or 0))
    M.begin_turn(current_turn)

    if #query_vec <= 0 or M.get_total_lines() <= 0 then
        return {
            context = "",
            topic_anchor = current_anchor,
            predicted_topics = {},
            predicted_memories = {},
            predicted_nodes = {},
            selected_turns = {},
            selected_memories = {},
            fragments = {},
            adopted_memories = {},
            bridge_topics = {},
            candidate_topics = {},
            local_signals = {},
            topic_debug = {},
        }
    end

    local query_map = build_query_facet_map(opts.user_input or "")
    local seed_scores = semantic_topic_scores(query_vec, current_anchor, scope_key)
    local expanded_scores, bridge_topics
    if flow_key ~= "" then
        expanded_scores = seed_scores
        bridge_topics = {}
    else
        expanded_scores, bridge_topics = expand_topic_candidates(seed_scores, scope_key)
    end
    local available_topics = preload_topics(expanded_scores, current_turn, scope_key)
    local selected_topics, topic_debug, evidence_rows = retrieve_topic_evidence(
        query_vec,
        expanded_scores,
        available_topics,
        current_anchor,
        current_turn,
        query_map,
        opts
    )
    local ranked_memories = ranked_memories_from_evidence(selected_topics, evidence_rows)
    local max_memories = math.max(1, tonumber(tg_cfg().retrieve_max_memories) or tonumber(ai_cfg().max_memory) or 8)
    while #ranked_memories > max_memories do
        table.remove(ranked_memories)
    end

    local selected_memories = {}
    for _, item in ipairs(ranked_memories) do
        selected_memories[#selected_memories + 1] = tonumber(item.mem_idx)
    end
    selected_memories = unique_sorted_numbers(selected_memories)

    local memory_anchors = {}
    for _, item in ipairs(ranked_memories) do
        local mem_id = tonumber((item or {}).mem_idx) or 0
        if mem_id > 0 then
            local anchor = trim((item or {}).anchor)
            if anchor ~= "" then
                memory_anchors[mem_id] = anchor
            end
        end
    end

    local selected_turns, fragments, memory_fallback_context = turns_from_memories(ranked_memories)
    local context = topic_context_from_selection(selected_topics, evidence_rows)
    if context == "" then
        context = memory_fallback_context
    end
    return {
        context = context,
        topic_anchor = current_anchor,
        predicted_topics = shallow_copy_array(selected_topics),
        predicted_memories = shallow_copy_array(selected_memories),
        predicted_nodes = {},
        selected_turns = shallow_copy_array(selected_turns),
        selected_memories = shallow_copy_array(selected_memories),
        memory_anchors = memory_anchors,
        fragments = fragments,
        adopted_memories = {},
        local_signals = type((topic_debug or {}).local_signals) == "table" and (topic_debug or {}).local_signals or {},
        topic_debug = topic_debug,
        candidate_topics = expanded_scores,
        bridge_topics = shallow_copy_array(bridge_topics or {}),
    }
end

function M.observe_feedback(current_anchor, recall_state, adopted_memories, turn, opts)
    opts = type(opts) == "table" and opts or {}
    local flow_key = flow_key_from_opts(opts)
    current_anchor = trim(current_anchor)
    recall_state = type(recall_state) == "table" and recall_state or {}
    local memory_anchors = type(recall_state.memory_anchors) == "table" and recall_state.memory_anchors or {}
    adopted_memories = unique_sorted_numbers(adopted_memories)
    local retrieved = unique_sorted_numbers(recall_state.selected_memories)
    local observed = unique_sorted_numbers((function()
        local tmp = {}
        for _, v in ipairs(retrieved) do tmp[#tmp + 1] = v end
        for _, v in ipairs(adopted_memories) do tmp[#tmp + 1] = v end
        return tmp
    end)())
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    M.begin_turn(turn)
    local rt = ensure_runtime_state(flow_key)

    local topic_cap = math.max(16, tonumber(tg_cfg().topic_cap) or 64)
    local next_cap = math.max(8, tonumber(tg_cfg().next_cap) or 32)
    local recall_lr = math.max(0.0, tonumber(tg_cfg().recall_lr) or 0.10)
    local adopt_lr = math.max(recall_lr, tonumber(tg_cfg().adopt_lr) or 0.18)
    local local_cfg = tg_cfg().deep_artmap or {}
    local bundle_recall_lr = math.max(0.0, tonumber(local_cfg.bundle_recall_lr) or recall_lr)
    local bundle_adopt_lr = math.max(0.0, tonumber(local_cfg.bundle_adopt_lr) or adopt_lr)
    local bundle_edge_feedback = math.max(0.0, tonumber(local_cfg.bundle_edge_feedback) or 0.14)

    if current_anchor ~= "" then
        local node = M.state.topic_nodes[current_anchor]
        if node or #adopted_memories > 0 then
            node = ensure_topic_node(current_anchor)
            for _, mem_id in ipairs(retrieved) do
                node.memory_prior[mem_id] = (tonumber(node.memory_prior[mem_id]) or 0.0) + recall_lr
            end
            for _, mem_id in ipairs(adopted_memories) do
                node.memory_prior[mem_id] = (tonumber(node.memory_prior[mem_id]) or 0.0) + adopt_lr
            end
            node.memory_prior = prune_weight_map(node.memory_prior, topic_cap)
        end
    end

    local prev_selected = unique_sorted_numbers(rt.last_selected or {})
    local prev_anchor = trim(rt.last_anchor or "")
    local memory_next = M.state.memory_next
    if flow_key ~= "" then
        M.runtime = type(M.runtime) == "table" and M.runtime or {}
        M.runtime.flow_memory_next = type(M.runtime.flow_memory_next) == "table" and M.runtime.flow_memory_next or {}
        memory_next = M.runtime.flow_memory_next[flow_key]
        if type(memory_next) ~= "table" then
            memory_next = {}
            M.runtime.flow_memory_next[flow_key] = memory_next
        end
    end
    if #prev_selected > 0 and #observed > 0 then
        for _, src in ipairs(prev_selected) do
            local bucket = memory_next[src] or {}
            for _, dst in ipairs(retrieved) do
                if src ~= dst then
                    bucket[dst] = (tonumber(bucket[dst]) or 0.0) + recall_lr
                end
            end
            for _, dst in ipairs(adopted_memories) do
                if src ~= dst then
                    bucket[dst] = (tonumber(bucket[dst]) or 0.0) + adopt_lr
                end
            end
            memory_next[src] = prune_weight_map(bucket, next_cap)
        end
    end

    local predicted_topics = {}
    for _, anchor in ipairs(recall_state.predicted_topics or {}) do
        anchor = trim(anchor)
        if anchor ~= "" then
            predicted_topics[#predicted_topics + 1] = anchor
        end
    end
    if #predicted_topics <= 0 then
        local seen = {}
        for _, mem_id in ipairs(observed) do
            local anchor = trim(memory_anchors[mem_id] or M.get_topic_anchor(mem_id))
            if anchor ~= "" and not seen[anchor] then
                seen[anchor] = true
                predicted_topics[#predicted_topics + 1] = anchor
            end
        end
    end

    local local_signals = type(recall_state.local_signals) == "table" and recall_state.local_signals or {}
    local current_signal = type(local_signals[current_anchor]) == "table" and local_signals[current_anchor] or {}
    local signal_bundle_ids = unique_sorted_numbers((current_signal or {}).bundle_ids)

    if current_anchor ~= "" and flow_key == "" then
        for _, anchor in ipairs(predicted_topics) do
            if anchor ~= current_anchor then
                boost_edge(current_anchor, anchor, "recall", recall_lr, turn)
            end
        end
        for _, mem_id in ipairs(adopted_memories) do
            local mem_anchor = ""
            if flow_key == "" then
                mem_anchor = trim(M.get_topic_anchor(mem_id))
            else
                mem_anchor = current_anchor
            end
            if mem_anchor ~= "" and mem_anchor ~= current_anchor then
                boost_edge(current_anchor, mem_anchor, "adopt", adopt_lr, turn)
            end
        end
    end

    if current_anchor ~= "" and #signal_bundle_ids > 0 then
        local node = ensure_topic_node(current_anchor)
        if node then
            local predicted_hit = false
            for _, anchor in ipairs(predicted_topics) do
                if trim(anchor) == current_anchor then
                    predicted_hit = true
                    break
                end
            end
            if predicted_hit then
                for _, bundle_id in ipairs(signal_bundle_ids) do
                    deep_artmap.reinforce_bundle(node.local_state, bundle_id, "recall", turn, {
                        recall_lr = bundle_recall_lr,
                        adopt_lr = bundle_adopt_lr,
                    })
                end
                if bundle_edge_feedback > 0.0 then
                    for i = 1, #signal_bundle_ids do
                        for j = i + 1, #signal_bundle_ids do
                            deep_artmap.link_bundles(node.local_state, signal_bundle_ids[i], signal_bundle_ids[j], bundle_edge_feedback)
                        end
                    end
                end
            end
            if trim(predicted_topics[1]) == current_anchor then
                deep_artmap.reinforce_bundle(node.local_state, signal_bundle_ids[1], "adopt", turn, {
                    recall_lr = bundle_recall_lr,
                    adopt_lr = bundle_adopt_lr,
                })
                for i = 2, #signal_bundle_ids do
                    deep_artmap.reinforce_bundle(node.local_state, signal_bundle_ids[i], "adopt", turn, {
                        recall_lr = bundle_recall_lr,
                        adopt_lr = bundle_adopt_lr * 0.35,
                    })
                end
            end
        end
    end

    for _, mem_id in ipairs(retrieved) do
        local anchor = trim(memory_anchors[mem_id] or M.get_topic_anchor(mem_id))
        local node = ensure_topic_node(anchor)
        if node then
            deep_artmap.reinforce_memory(node.local_state, mem_id, "recall", turn, {
                recall_lr = recall_lr,
                adopt_lr = adopt_lr,
            })
        end
    end
    for _, mem_id in ipairs(adopted_memories) do
        local anchor = current_anchor
        local node = ensure_topic_node(anchor)
        if node then
            deep_artmap.reinforce_memory(node.local_state, mem_id, "adopt", turn, {
                recall_lr = recall_lr,
                adopt_lr = adopt_lr,
            })
        end
    end

    rt.last_anchor = current_anchor
    rt.last_selected = observed
    rt.last_turn = turn
    mark_dirty()
    return true
end

reset_state()

return M
