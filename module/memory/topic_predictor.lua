local M = {}

local config = require("module.config")
local persistence = require("module.persistence")

local STATE_FILE = "memory/topic_predictor.txt"
local VERSION = "TPR1"

M.state = nil
M.runtime = nil
M.dirty = false

local function ai_cfg()
    return ((config.settings or {}).ai_query or {})
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function sorted_keys(map)
    local keys = {}
    for k in pairs(map or {}) do
        keys[#keys + 1] = k
    end
    table.sort(keys, function(a, b)
        if type(a) == "number" and type(b) == "number" then
            return a < b
        end
        return tostring(a) < tostring(b)
    end)
    return keys
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function add_weight(bucket, key, delta)
    if not bucket or key == nil then return end
    local d = tonumber(delta) or 0.0
    if d == 0.0 then return end
    local next_v = (tonumber(bucket[key]) or 0.0) + d
    if next_v <= 1e-9 then
        bucket[key] = nil
        return
    end
    bucket[key] = next_v
end

local function prune_bucket(bucket, max_items)
    if type(bucket) ~= "table" then
        return
    end
    local limit = math.max(1, math.floor(tonumber(max_items) or 1))
    local items = {}
    for key, score in pairs(bucket) do
        items[#items + 1] = { key = key, score = tonumber(score) or 0.0 }
    end
    table.sort(items, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return tostring(a.key) < tostring(b.key)
    end)
    while #items > limit do
        local dropped = table.remove(items)
        if dropped then
            bucket[dropped.key] = nil
        end
    end
end

local function normalize_anchor(topic_anchor)
    if type(topic_anchor) == "number" then
        local topic = require("module.memory.topic")
        return topic.get_stable_anchor and topic.get_stable_anchor(topic_anchor) or nil
    end

    local topic = require("module.memory.topic")
    local fp = topic.get_topic_fingerprint and topic.get_topic_fingerprint(topic_anchor) or nil
    if fp and fp.key then
        return fp.key
    end
    local key = tostring(topic_anchor or ""):match("^%s*(.-)%s*$")
    if key == "" then
        return nil
    end
    return key
end

local function unique_sorted_memories(memories)
    local seen = {}
    local out = {}
    for _, mem_idx in ipairs(memories or {}) do
        local idx = tonumber(mem_idx)
        if idx and idx > 0 and not seen[idx] then
            seen[idx] = true
            out[#out + 1] = idx
        end
    end
    table.sort(out)
    return out
end

local function mark_dirty()
    M.dirty = true
    local ok_saver, saver = pcall(require, "module.memory.saver")
    if ok_saver and saver and saver.mark_dirty then
        saver.mark_dirty()
    end
end

local function default_state()
    return {
        topic_memory = {},
        memory_next = {},
        topic_transition = {},
        observe_count = 0,
        predict_count = 0,
    }
end

local function default_runtime()
    return {
        last_anchor = nil,
        last_selected = {},
    }
end

function M.reset_defaults()
    M.state = default_state()
    M.runtime = default_runtime()
    M.dirty = false
end

function M.load()
    M.reset_defaults()

    local f = io.open(STATE_FILE, "r")
    if not f then
        return
    end

    local header = f:read("*l")
    if header ~= VERSION then
        f:close()
        return
    end

    for line in f:lines() do
        local kind, a, b, c = tostring(line or ""):match("^([A-Z_]+)\t([^\t]*)\t([^\t]*)\t([^\t]*)$")
        if kind == "TOPIC" then
            local anchor = normalize_anchor(a)
            local mem_idx = tonumber(b)
            local score = tonumber(c)
            if anchor and mem_idx and score and score > 0 then
                local bucket = M.state.topic_memory[anchor] or {}
                bucket[mem_idx] = score
                M.state.topic_memory[anchor] = bucket
            end
        elseif kind == "NEXT" then
            local src = tonumber(a)
            local dst = tonumber(b)
            local score = tonumber(c)
            if src and dst and score and score > 0 then
                local bucket = M.state.memory_next[src] or {}
                bucket[dst] = score
                M.state.memory_next[src] = bucket
            end
        elseif kind == "TRANS" then
            local prev_anchor = normalize_anchor(a)
            local next_anchor = normalize_anchor(b)
            local score = tonumber(c)
            if prev_anchor and next_anchor and score and score > 0 then
                local bucket = M.state.topic_transition[prev_anchor] or {}
                bucket[next_anchor] = score
                M.state.topic_transition[prev_anchor] = bucket
            end
        elseif kind == "STAT" then
            if a == "observe_count" then
                M.state.observe_count = tonumber(b) or 0
            elseif a == "predict_count" then
                M.state.predict_count = tonumber(b) or 0
            end
        end
    end

    f:close()
    M.dirty = false
end

function M.save_to_disk()
    if not M.dirty then
        return true
    end

    local ok, err = persistence.write_atomic(STATE_FILE, "w", function(f)
        local ok_w, err_w = f:write(VERSION .. "\n")
        if not ok_w then
            return false, err_w
        end

        local stats = {
            { "observe_count", tonumber(M.state.observe_count) or 0 },
            { "predict_count", tonumber(M.state.predict_count) or 0 },
        }
        for _, item in ipairs(stats) do
            local ok_s, err_s = f:write(string.format("STAT\t%s\t%.10f\t0\n", item[1], item[2]))
            if not ok_s then
                return false, err_s
            end
        end

        for _, anchor in ipairs(sorted_keys(M.state.topic_memory)) do
            local bucket = M.state.topic_memory[anchor] or {}
            for _, mem_idx in ipairs(sorted_keys(bucket)) do
                local ok_t, err_t = f:write(string.format("TOPIC\t%s\t%d\t%.10f\n", tostring(anchor), tonumber(mem_idx), tonumber(bucket[mem_idx]) or 0.0))
                if not ok_t then
                    return false, err_t
                end
            end
        end

        for _, src in ipairs(sorted_keys(M.state.memory_next)) do
            local bucket = M.state.memory_next[src] or {}
            for _, dst in ipairs(sorted_keys(bucket)) do
                local ok_n, err_n = f:write(string.format("NEXT\t%d\t%d\t%.10f\n", tonumber(src), tonumber(dst), tonumber(bucket[dst]) or 0.0))
                if not ok_n then
                    return false, err_n
                end
            end
        end

        for _, prev_anchor in ipairs(sorted_keys(M.state.topic_transition)) do
            local bucket = M.state.topic_transition[prev_anchor] or {}
            for _, next_anchor in ipairs(sorted_keys(bucket)) do
                local ok_tr, err_tr = f:write(string.format("TRANS\t%s\t%s\t%.10f\n", tostring(prev_anchor), tostring(next_anchor), tonumber(bucket[next_anchor]) or 0.0))
                if not ok_tr then
                    return false, err_tr
                end
            end
        end
        return true
    end)

    if not ok then
        return false, err
    end

    M.dirty = false
    return true
end

function M.observe(topic_anchor, selected_memories, opts)
    opts = type(opts) == "table" and opts or {}

    local anchor = normalize_anchor(topic_anchor)
    local selected = unique_sorted_memories(selected_memories)
    local cfg = ai_cfg()
    local topic_cap = math.max(8, math.floor(tonumber(cfg.topic_activation_topic_cap) or 64))
    local next_cap = math.max(8, math.floor(tonumber(cfg.topic_activation_next_cap) or 32))
    local trans_cap = math.max(4, math.floor(tonumber(cfg.topic_activation_transition_cap) or 12))
    local cross_weight = clamp(tonumber(cfg.topic_activation_cross_topic_next_weight) or 0.45, 0.0, 1.0)

    if anchor and #selected > 0 then
        local topic_bucket = M.state.topic_memory[anchor] or {}
        for _, mem_idx in ipairs(selected) do
            add_weight(topic_bucket, mem_idx, 1.0)
        end
        prune_bucket(topic_bucket, topic_cap)
        M.state.topic_memory[anchor] = topic_bucket
    end

    local prev_anchor = normalize_anchor(opts.previous_anchor or M.runtime.last_anchor)
    local prev_selected = unique_sorted_memories(opts.previous_selected or M.runtime.last_selected)
    if #prev_selected > 0 and #selected > 0 then
        local pair_weight = (prev_anchor and anchor and prev_anchor ~= anchor) and cross_weight or 1.0
        for _, src in ipairs(prev_selected) do
            local bucket = M.state.memory_next[src] or {}
            for _, dst in ipairs(selected) do
                if src ~= dst then
                    add_weight(bucket, dst, pair_weight)
                end
            end
            prune_bucket(bucket, next_cap)
            M.state.memory_next[src] = bucket
        end
    end

    if prev_anchor and anchor and prev_anchor ~= anchor then
        local trans_bucket = M.state.topic_transition[prev_anchor] or {}
        add_weight(trans_bucket, anchor, 1.0)
        prune_bucket(trans_bucket, trans_cap)
        M.state.topic_transition[prev_anchor] = trans_bucket
    end

    if anchor then
        M.runtime.last_anchor = anchor
    end
    if #selected > 0 then
        M.runtime.last_selected = selected
    end

    M.state.observe_count = (tonumber(M.state.observe_count) or 0) + 1
    mark_dirty()
end

function M.predict(topic_anchor, opts)
    opts = type(opts) == "table" and opts or {}

    local anchor = normalize_anchor(topic_anchor)
    local cfg = ai_cfg()
    local topic_weight = math.max(0.0, tonumber(cfg.topic_activation_topic_weight) or 1.00)
    local chain_weight = math.max(0.0, tonumber(cfg.topic_activation_chain_weight) or 0.65)
    local resident_weight = math.max(0.0, tonumber(cfg.topic_activation_resident_weight) or 0.28)
    local recent_weight = math.max(0.0, tonumber(cfg.topic_activation_recent_weight) or 0.55)
    local query_weight = math.max(0.0, tonumber(cfg.topic_activation_query_weight) or 0.30)
    local min_score = math.max(0.0, tonumber(cfg.topic_activation_min_score) or 0.08)
    local memory_topn = math.max(1, math.floor(tonumber(opts.memory_topn) or tonumber(cfg.topic_activation_memory_topn) or 12))
    local chain_topn = math.max(1, math.floor(tonumber(opts.chain_topn) or tonumber(cfg.topic_activation_chain_topn) or 2))
    local chain_min_score = tonumber(opts.chain_min_score) or tonumber(cfg.topic_activation_chain_min_score) or 0.18

    local scores = {}
    local topic = require("module.memory.topic")
    local store = require("module.memory.store")
    local ghsom = require("module.memory.ghsom")

    if anchor then
        local same_topic = M.state.topic_memory[anchor] or {}
        for mem_idx, score in pairs(same_topic) do
            add_weight(scores, tonumber(mem_idx), (tonumber(score) or 0.0) * topic_weight)
        end

        if store.iter_topic_lines then
            for _, mem_idx in ipairs(store.iter_topic_lines(anchor, false) or {}) do
                add_weight(scores, tonumber(mem_idx), resident_weight)
            end
        end

        if topic.get_topic_chain then
            local chain = topic.get_topic_chain(anchor, {
                topn = chain_topn,
                min_score = chain_min_score,
            })
            for _, item in ipairs(chain or {}) do
                local neighbor_bucket = M.state.topic_memory[tostring(item.key or "")] or {}
                local scale = (tonumber(item.score) or 0.0) * chain_weight
                for mem_idx, score in pairs(neighbor_bucket) do
                    add_weight(scores, tonumber(mem_idx), (tonumber(score) or 0.0) * scale)
                end
                if store.iter_topic_lines then
                    for _, mem_idx in ipairs(store.iter_topic_lines(item.key, false) or {}) do
                        add_weight(scores, tonumber(mem_idx), resident_weight * scale)
                    end
                end
            end
        end
    end

    local recent_selected = unique_sorted_memories(opts.recent_selected or M.runtime.last_selected)
    for _, src in ipairs(recent_selected) do
        local bucket = M.state.memory_next[src] or {}
        for dst, score in pairs(bucket) do
            add_weight(scores, tonumber(dst), (tonumber(score) or 0.0) * recent_weight)
        end
    end

    if type(opts.query_vec) == "table" and #opts.query_vec > 0 then
        for mem_idx, base in pairs(scores) do
            local mem_vec = store.return_mem_vec and store.return_mem_vec(mem_idx) or nil
            if mem_vec then
                local sim = tonumber(require("module.tool").cosine_similarity(opts.query_vec, mem_vec)) or 0.0
                add_weight(scores, mem_idx, math.max(0.0, sim) * query_weight)
            else
                scores[mem_idx] = base
            end
        end
    end

    local ranked = {}
    for mem_idx, score in pairs(scores) do
        if (tonumber(score) or 0.0) >= min_score then
            ranked[#ranked + 1] = {
                mem_idx = tonumber(mem_idx),
                score = tonumber(score) or 0.0,
            }
        end
    end
    table.sort(ranked, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.mem_idx or 0) < (b.mem_idx or 0)
    end)

    while #ranked > memory_topn do
        table.remove(ranked)
    end

    local lines = {}
    local memory_scores = {}
    local node_scores = {}
    for _, item in ipairs(ranked) do
        lines[#lines + 1] = item.mem_idx
        memory_scores[item.mem_idx] = item.score
        local cid = ghsom.get_node_for_line and ghsom.get_node_for_line(item.mem_idx) or nil
        if cid then
            node_scores[cid] = (tonumber(node_scores[cid]) or 0.0) + item.score
        end
    end

    local node_max = 0.0
    for _, score in pairs(node_scores) do
        if score > node_max then
            node_max = score
        end
    end
    if node_max > 0 then
        for cid, score in pairs(node_scores) do
            node_scores[cid] = score / node_max
        end
    end

    M.state.predict_count = (tonumber(M.state.predict_count) or 0) + 1
    mark_dirty()

    return {
        topic_anchor = anchor,
        lines = lines,
        memory_scores = memory_scores,
        node_scores = node_scores,
        recent_selected = shallow_copy_array(recent_selected),
    }
end

function M.snapshot()
    local topic_n = 0
    for _ in pairs((M.state or {}).topic_memory or {}) do
        topic_n = topic_n + 1
    end
    return {
        topic_count = topic_n,
        observe_count = tonumber((M.state or {}).observe_count) or 0,
        predict_count = tonumber((M.state or {}).predict_count) or 0,
        last_anchor = tostring(((M.runtime or {}).last_anchor) or ""),
        last_selected = shallow_copy_array(((M.runtime or {}).last_selected) or {}),
    }
end

M.reset_defaults()

return M
