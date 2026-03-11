local ffi = require("ffi")
local tool = require("module.tool")
local config = require("module.config")
local history = require("module.memory.history")
local topic = require("module.memory.topic")
local persistence = require("module.persistence")
local util = require("module.graph.util")
local deep_artmap = require("module.memory.topic_graph_deep_artmap")
local topic_hnsw_mod = require("module.memory.topic_graph_hnsw")
local legacy_import = require("module.memory.legacy_v3_import")

local M = {}

local STATE_VERSION = "TG1"
local VECTOR_MAGIC = "TGV1"

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
        current_turn = 0,
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
    }
end

local function reset_state()
    M.state = default_state()
    M.runtime = default_runtime()
    M.dirty = false
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
        local inserted = deep_artmap.add_memory(node.local_state, vec, mem_id, turn, {
            category_vigilance = tonumber(local_cfg.category_vigilance) or 0.88,
            category_beta = tonumber(local_cfg.category_beta) or 0.28,
            bundle_vigilance = tonumber(local_cfg.bundle_vigilance) or 0.82,
            bundle_beta = tonumber(local_cfg.bundle_beta) or 0.18,
        })
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
                type = tostring(mem.type or "fact"),
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
        current_turn = tonumber(M.state.current_turn) or 0,
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
    M.state.current_turn = math.max(0, math.floor(tonumber(parsed.current_turn) or 0))
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
                type = tostring((meta or {}).type or "fact"),
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

local function select_seed_topics(query_vec, current_anchor)
    local scores = {}
    local seen = {}
    local hcfg = hnsw_cfg()
    for _, hit in ipairs(M.runtime.hnsw:search(query_vec, math.max(hcfg.k, tonumber(tg_cfg().seed_topics) or 2)) or {}) do
        local anchor = trim(hit.anchor)
        if anchor ~= "" then
            local score = (tonumber(tg_cfg().query_semantic_weight) or 1.0) * (tonumber(hit.similarity) or 0.0)
            if anchor == current_anchor then
                score = score + (tonumber(tg_cfg().current_topic_bonus) or 0.12)
            end
            scores[anchor] = math.max(tonumber(scores[anchor]) or -1e9, score)
            seen[anchor] = true
        end
    end
    for anchor, node in pairs(M.state.topic_nodes or {}) do
        if not seen[anchor] then
            local score = (tonumber(tg_cfg().query_semantic_weight) or 1.0) * safe_similarity(query_vec, node.centroid)
            if anchor == current_anchor then
                score = score + (tonumber(tg_cfg().current_topic_bonus) or 0.12)
            end
            scores[anchor] = math.max(tonumber(scores[anchor]) or -1e9, score)
        end
    end
    return scores
end

local function expand_topic_candidates(seed_scores)
    local bridge_topk = math.max(1, tonumber(tg_cfg().bridge_topk) or 8)
    local max_hops = math.max(0, tonumber(tg_cfg().max_bridge_hops) or 2)
    local min_bridge = math.max(0.0, tonumber(tg_cfg().min_bridge_score) or 0.08)
    local bridge_weight = math.max(0.0, tonumber(tg_cfg().bridge_weight) or 0.55)
    local resident_bonus = math.max(0.0, tonumber(tg_cfg().resident_bonus) or 0.08)

    local scores = {}
    local frontier = {}
    for anchor, score in pairs(seed_scores or {}) do
        scores[anchor] = tonumber(score) or 0.0
        frontier[#frontier + 1] = { anchor = anchor, score = tonumber(score) or 0.0 }
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
                if edge_score >= min_bridge then
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

    return scores
end

local function select_topics(candidate_scores, current_anchor)
    local ranked = {}
    for anchor, score in pairs(candidate_scores or {}) do
        if trim(anchor) ~= "" then
            ranked[#ranked + 1] = {
                anchor = trim(anchor),
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
    local anchor_order = {}
    for _, item in ipairs(ranked) do
        anchor_order[#anchor_order + 1] = item.anchor
    end
    local family_of = get_topic_families(anchor_order)
    local family_limit = math.max(1, tonumber(tg_cfg().family_member_limit) or 1)
    local self_penalty = math.max(0.0, tonumber(tg_cfg().self_excite_penalty) or 0.14)
    local revisit_penalty = math.max(0.0, tonumber(tg_cfg().family_revisit_penalty) or 0.10)
    local escape_bonus = math.max(0.0, tonumber(tg_cfg().family_escape_bonus) or 0.06)
    local max_topics = math.max(1, tonumber(tg_cfg().max_return_topics) or 4)

    local out = {}
    local family_counts = {}
    local last_family = nil
    for _, item in ipairs(ranked) do
        local anchor = item.anchor
        local family_id = tonumber(family_of[anchor]) or 0
        local score = tonumber(item.score) or 0.0
        if anchor == current_anchor then
            score = score - self_penalty
        end
        if family_counts[family_id] and family_counts[family_id] >= family_limit then
            score = score - revisit_penalty
        end
        if last_family and last_family ~= family_id then
            score = score + escape_bonus
        end
        if score > -1e9 and ((not family_counts[family_id]) or family_counts[family_id] < family_limit or #out <= 0) then
            out[#out + 1] = anchor
            family_counts[family_id] = (family_counts[family_id] or 0) + 1
            last_family = family_id
            if #out >= max_topics then
                break
            end
        end
    end
    return out
end

local function memory_candidates_for_topics(selected_topics, query_vec, current_anchor, current_turn)
    local out = {}
    local debug = {}
    local per_topic_evidence = math.max(1, tonumber(tg_cfg().per_topic_evidence) or 3)
    local recent_weight = math.max(0.0, tonumber(tg_cfg().recent_weight) or 0.55)
    local prior_weight = math.max(0.0, tonumber(tg_cfg().topic_prior_weight) or 1.00)

    local function note(mem_id, score, anchor, category_id, bundle_id)
        mem_id = tonumber(mem_id)
        if not mem_id or mem_id <= 0 then
            return
        end
        local current = out[mem_id]
        if current == nil or score > current.score then
            out[mem_id] = {
                mem_idx = mem_id,
                score = score,
                anchor = anchor,
                category_id = tonumber(category_id) or 0,
                bundle_id = tonumber(bundle_id) or 0,
            }
        end
    end

    if tonumber((((M.state or {}).runtime or {}).last_turn)) == (math.floor(tonumber(current_turn) or 0) - 1)
        and trim((((M.state or {}).runtime or {}).last_anchor)) == trim(current_anchor) then
        for _, prev_mem in ipairs((((M.state or {}).runtime or {}).last_selected) or {}) do
            local bucket = M.state.memory_next[tonumber(prev_mem)] or {}
            for mem_id, weight in pairs(bucket) do
                local mem = M.state.memories[tonumber(mem_id)]
                local sim = safe_similarity(query_vec, mem and mem.vec)
                note(mem_id, sim + recent_weight * (tonumber(weight) or 0.0), trim((mem or {}).topic_anchor), 0, 0)
            end
        end
    end

    for _, anchor in ipairs(selected_topics or {}) do
        local node = ensure_topic_node(anchor)
        local candidates, local_debug = deep_artmap.collect_candidates(node.local_state, query_vec, function(mem_id)
            local mem = M.state.memories[tonumber(mem_id)]
            return mem and mem.vec or nil
        end, {
            query_bundles = tonumber((((tg_cfg().deep_artmap) or {}).query_bundles)) or 2,
            query_margin = tonumber((((tg_cfg().deep_artmap) or {}).query_margin)) or 0.06,
            neighbor_topk = tonumber((((tg_cfg().deep_artmap) or {}).neighbor_topk)) or 2,
            max_results = per_topic_evidence * 2,
        })
        for _, item in ipairs(candidates or {}) do
            local prior = tonumber((node.memory_prior or {})[tonumber(item.mem_idx)]) or 0.0
            local resident = is_topic_loaded(anchor) and (tonumber(tg_cfg().resident_bonus) or 0.08) or 0.0
            note(item.mem_idx, (tonumber(item.score) or 0.0) + prior_weight * prior + resident, anchor, item.category_id, item.bundle_id)
        end
        debug[anchor] = local_debug

        if #(candidates or {}) <= 0 then
            for _, mem_id in ipairs(node.memory_ids or {}) do
                local mem = M.state.memories[tonumber(mem_id)]
                local score = safe_similarity(query_vec, mem and mem.vec) + prior_weight * (tonumber((node.memory_prior or {})[tonumber(mem_id)]) or 0.0)
                note(mem_id, score, anchor, tonumber((mem or {}).cluster_id) or 0, 0)
            end
        end
    end

    local ranked = {}
    for _, item in pairs(out) do
        ranked[#ranked + 1] = item
    end
    table.sort(ranked, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.mem_idx or 0) < (b.mem_idx or 0)
    end)
    return ranked, debug
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
        deep_artmap.decay(node.local_state, factor)
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
    if trim(mem.topic_anchor) ~= "" then
        recompute_topic_centroid(mem.topic_anchor)
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
    local merge_limit = tonumber(config.settings.merge_limit) or 0.95

    local best_line, best_sim = nil, -1.0
    for line = 1, M.get_total_lines() do
        local mem = M.state.memories[line]
        if mem and type(mem.vec) == "table" and #mem.vec > 0 then
            local sim = safe_similarity(vec, mem.vec)
            if sim > best_sim then
                best_line = line
                best_sim = sim
            end
        end
    end

    if best_line and best_sim >= merge_limit and opts.allow_merge ~= false then
        local mem = M.state.memories[best_line]
        local last_turn = tonumber((mem.turns or {})[#(mem.turns or {})]) or -1
        if last_turn ~= turn then
            mem.turns[#mem.turns + 1] = turn
        end
        if anchor ~= "" then
            bind_memory_to_topic(best_line, anchor, mem.vec, turn, 1.0, { preserve_cluster_id = true })
        end
        mark_dirty()
        return best_line
    end

    local line = math.max(1, tonumber(M.state.next_line) or 1)
    M.state.next_line = line + 1
    M.state.memories[line] = {
        turns = { turn },
        topic_anchor = anchor,
        cluster_id = -1,
        origin_topics = {},
        type = tostring(opts.kind or "fact"),
        vec = vec,
    }
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

function M.observe_turn(turn, current_anchor)
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    current_anchor = trim(current_anchor)
    if current_anchor == "" then
        return false
    end
    M.begin_turn(turn)
    ensure_topic_node(current_anchor)
    local prev_anchor = trim((((M.state or {}).runtime or {}).last_anchor))
    local prev_turn = tonumber((((M.state or {}).runtime or {}).last_turn)) or 0
    if prev_anchor ~= "" and prev_anchor ~= current_anchor and prev_turn > 0 then
        boost_edge(prev_anchor, current_anchor, "transition", tonumber(tg_cfg().transition_lr) or 0.12, turn)
    end
    return true
end

function M.retrieve(query_vec, current_anchor, current_turn, _opts)
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
        }
    end

    local seed_scores = select_seed_topics(query_vec, current_anchor)
    local expanded_scores = expand_topic_candidates(seed_scores)
    local selected_topics = select_topics(expanded_scores, current_anchor)
    mark_topics_loaded(selected_topics, current_turn)

    local ranked_memories, topic_debug = memory_candidates_for_topics(selected_topics, query_vec, current_anchor, current_turn)
    local max_memories = math.max(1, tonumber(tg_cfg().retrieve_max_memories) or tonumber(ai_cfg().max_memory) or 8)
    while #ranked_memories > max_memories do
        table.remove(ranked_memories)
    end

    local selected_memories = {}
    for _, item in ipairs(ranked_memories) do
        selected_memories[#selected_memories + 1] = tonumber(item.mem_idx)
    end
    selected_memories = unique_sorted_numbers(selected_memories)

    local selected_turns, fragments, context = turns_from_memories(ranked_memories)
    return {
        context = context,
        topic_anchor = current_anchor,
        predicted_topics = shallow_copy_array(selected_topics),
        predicted_memories = shallow_copy_array(selected_memories),
        predicted_nodes = {},
        selected_turns = shallow_copy_array(selected_turns),
        selected_memories = shallow_copy_array(selected_memories),
        fragments = fragments,
        adopted_memories = {},
        topic_debug = topic_debug,
    }
end

function M.observe_feedback(current_anchor, recall_state, adopted_memories, turn)
    current_anchor = trim(current_anchor)
    recall_state = type(recall_state) == "table" and recall_state or {}
    adopted_memories = unique_sorted_numbers(adopted_memories)
    local retrieved = unique_sorted_numbers(recall_state.selected_memories)
    local observed = unique_sorted_numbers((function()
        local tmp = {}
        for _, v in ipairs(retrieved) do tmp[#tmp + 1] = v end
        for _, v in ipairs(adopted_memories) do tmp[#tmp + 1] = v end
        return tmp
    end)())
    turn = math.max(0, math.floor(tonumber(turn) or 0))

    local topic_cap = math.max(16, tonumber(tg_cfg().topic_cap) or 64)
    local next_cap = math.max(8, tonumber(tg_cfg().next_cap) or 32)
    local recall_lr = math.max(0.0, tonumber(tg_cfg().recall_lr) or 0.10)
    local adopt_lr = math.max(recall_lr, tonumber(tg_cfg().adopt_lr) or 0.18)

    if current_anchor ~= "" then
        local node = ensure_topic_node(current_anchor)
        for _, mem_id in ipairs(retrieved) do
            node.memory_prior[mem_id] = (tonumber(node.memory_prior[mem_id]) or 0.0) + recall_lr
        end
        for _, mem_id in ipairs(adopted_memories) do
            node.memory_prior[mem_id] = (tonumber(node.memory_prior[mem_id]) or 0.0) + adopt_lr
        end
        node.memory_prior = prune_weight_map(node.memory_prior, topic_cap)
    end

    local prev_selected = unique_sorted_numbers((((M.state or {}).runtime or {}).last_selected) or {})
    local prev_anchor = trim((((M.state or {}).runtime or {}).last_anchor))
    if #prev_selected > 0 and #observed > 0 then
        for _, src in ipairs(prev_selected) do
            local bucket = M.state.memory_next[src] or {}
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
            M.state.memory_next[src] = prune_weight_map(bucket, next_cap)
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
            local anchor = trim(M.get_topic_anchor(mem_id))
            if anchor ~= "" and not seen[anchor] then
                seen[anchor] = true
                predicted_topics[#predicted_topics + 1] = anchor
            end
        end
    end

    if current_anchor ~= "" then
        for _, anchor in ipairs(predicted_topics) do
            if anchor ~= current_anchor then
                boost_edge(current_anchor, anchor, "recall", recall_lr, turn)
            end
        end
        for _, mem_id in ipairs(adopted_memories) do
            local mem_anchor = trim(M.get_topic_anchor(mem_id))
            if mem_anchor ~= "" and mem_anchor ~= current_anchor then
                boost_edge(current_anchor, mem_anchor, "adopt", adopt_lr, turn)
            end
        end
    elseif prev_anchor ~= "" and current_anchor ~= "" and prev_anchor ~= current_anchor then
        boost_edge(prev_anchor, current_anchor, "transition", tonumber(tg_cfg().transition_lr) or 0.12, turn)
    end

    for _, mem_id in ipairs(retrieved) do
        local anchor = trim(M.get_topic_anchor(mem_id))
        local node = ensure_topic_node(anchor)
        if node then
            deep_artmap.reinforce_memory(node.local_state, mem_id, "recall", turn, {
                recall_lr = recall_lr,
                adopt_lr = adopt_lr,
            })
        end
    end
    for _, mem_id in ipairs(adopted_memories) do
        local anchor = trim(M.get_topic_anchor(mem_id))
        local node = ensure_topic_node(anchor)
        if node then
            deep_artmap.reinforce_memory(node.local_state, mem_id, "adopt", turn, {
                recall_lr = recall_lr,
                adopt_lr = adopt_lr,
            })
        end
    end

    M.state.runtime.last_anchor = current_anchor
    M.state.runtime.last_selected = observed
    M.state.runtime.last_turn = turn
    mark_dirty()
    return true
end

reset_state()

return M
