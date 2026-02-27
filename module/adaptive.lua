local M = {}

local config = require("module.config")
local persistence = require("module.persistence")
local tool = require("module.tool")

local STATE_FILE = "memory/adaptive_state.txt"
local VERSION = "ADPT1"

M.state = nil
M.dirty = false

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function ai_cfg()
    return ((config.settings or {}).ai_query or {})
end

local function default_learned_gate()
    local aq = ai_cfg()
    local start_gate = tonumber(aq.learning_min_sim_gate_start) or tonumber(aq.min_sim_gate) or 0.58
    local end_gate = tonumber(aq.min_sim_gate) or start_gate
    local g = math.min(start_gate * 0.78, end_gate * 0.62)
    return clamp(g, 0.05, 0.95)
end

local function default_merge_limit()
    return tonumber((config.settings or {}).merge_limit) or 0.95
end

local function make_default_state()
    return {
        learned_min_gate = default_learned_gate(),
        online_merge_limit = default_merge_limit(),
        cluster_route_score = {},
        cluster_route_seen = {},
        refinement_events = 0,
        persistent_explore_events = 0,
        persistent_explore_cluster_probes = 0,
        persistent_explore_turn_hits = 0,
        uncertain_recency_applied = 0,
        uncertain_low_top1_count = 0,
        uncertain_small_gap_count = 0,
        uncertain_sparse_count = 0,
        cold_rescue_enqueued = 0,
        cold_rescue_executed = 0,
        topic_lift_attempted = 0,
        topic_lift_executed = 0,
        topic_cache_unload_count = 0,
        topic_cache_selected_turns_total = 0,
    }
end

local function quantile(values, q)
    local n = #values
    if n <= 0 then return nil end
    table.sort(values)
    if n == 1 then return values[1] end
    local qq = clamp(tonumber(q) or 0.5, 0.0, 1.0)
    local pos = 1 + (n - 1) * qq
    local lo = math.floor(pos)
    local hi = math.ceil(pos)
    if lo == hi then return values[lo] end
    local t = pos - lo
    return values[lo] * (1 - t) + values[hi] * t
end

local function refinement_progress(turn)
    local aq = ai_cfg()
    if aq.refinement_enabled ~= true then return 0.0 end
    local start_t = math.max(0, tonumber(aq.refinement_start_turn) or 200)
    local full_t = math.max(start_t + 1, tonumber(aq.learning_full_turns) or 12000)
    local cur = tonumber(turn) or 0
    if cur <= start_t then return 0.0 end
    if cur >= full_t then return 1.0 end
    return (cur - start_t) / (full_t - start_t)
end

local SCALAR_KEYS = {
    "learned_min_gate",
    "online_merge_limit",
    "refinement_events",
    "persistent_explore_events",
    "persistent_explore_cluster_probes",
    "persistent_explore_turn_hits",
    "uncertain_recency_applied",
    "uncertain_low_top1_count",
    "uncertain_small_gap_count",
    "uncertain_sparse_count",
    "cold_rescue_enqueued",
    "cold_rescue_executed",
    "topic_lift_attempted",
    "topic_lift_executed",
    "topic_cache_unload_count",
    "topic_cache_selected_turns_total",
}

local function scalar_key_set()
    local out = {}
    for _, k in ipairs(SCALAR_KEYS) do
        out[k] = true
    end
    return out
end

local SCALAR_KEY_SET = scalar_key_set()

function M.reset_defaults()
    M.state = make_default_state()
    M.dirty = false
end

function M.mark_dirty()
    M.dirty = true
end

function M.load()
    M.reset_defaults()
    if not tool.file_exists(STATE_FILE) then
        print("[Adaptive] adaptive_state.txt 不存在，使用默认状态")
        return
    end

    local f = io.open(STATE_FILE, "r")
    if not f then
        print("[Adaptive] adaptive_state.txt 打开失败，使用默认状态")
        return
    end

    local header = f:read("*l")
    if header ~= VERSION then
        f:close()
        print("[Adaptive] adaptive_state.txt 版本不匹配，已回退默认状态")
        return
    end

    for line in f:lines() do
        line = tostring(line or ""):gsub("^%s*(.-)%s*$", "%1")
        if line ~= "" then
            local k, v = line:match("^([%w_]+)%s*=%s*(.+)$")
            if k and v then
                if k == "route" then
                    local cid_s, score_s, seen_s = v:match("^([%-%d]+),([%+%-]?[%d%.eE]+),([%+%-]?[%d%.eE]+)$")
                    local cid = tonumber(cid_s)
                    local score = tonumber(score_s)
                    local seen = tonumber(seen_s)
                    if cid and score then
                        M.state.cluster_route_score[cid] = score
                        M.state.cluster_route_seen[cid] = seen or 0
                    end
                elseif SCALAR_KEY_SET[k] then
                    local n = tonumber(v)
                    if n ~= nil then
                        M.state[k] = n
                    end
                end
            end
        end
    end

    f:close()
    M.dirty = false
    local route_n = 0
    for _ in pairs(M.state.cluster_route_score) do
        route_n = route_n + 1
    end
    print(string.format("[Adaptive] 状态加载完成: route_scores=%d", route_n))
end

function M.save_to_disk()
    if not M.dirty then return true end

    local ok, err = persistence.write_atomic(STATE_FILE, "w", function(f)
        local w_ok, w_err = f:write(VERSION .. "\n")
        if not w_ok then
            return false, w_err
        end

        for _, k in ipairs(SCALAR_KEYS) do
            local val = tonumber(M.state[k]) or 0
            local ok_line, err_line = f:write(string.format("%s=%.10f\n", k, val))
            if not ok_line then
                return false, err_line
            end
        end

        local ids = {}
        for cid, score in pairs(M.state.cluster_route_score) do
            local seen = tonumber(M.state.cluster_route_seen[cid]) or 0
            if math.abs(tonumber(score) or 0) > 1e-6 or seen > 0 then
                ids[#ids + 1] = tonumber(cid)
            end
        end
        table.sort(ids)

        for _, cid in ipairs(ids) do
            local score = tonumber(M.state.cluster_route_score[cid]) or 0
            local seen = tonumber(M.state.cluster_route_seen[cid]) or 0
            local ok_route, err_route = f:write(string.format("route=%d,%.10f,%.2f\n", cid, score, seen))
            if not ok_route then
                return false, err_route
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

function M.get_min_sim_gate(base_gate)
    local aq = ai_cfg()
    if aq.refinement_enabled ~= true then
        return tonumber(base_gate) or tonumber(aq.min_sim_gate) or 0.58
    end
    return tonumber(M.state.learned_min_gate) or tonumber(base_gate) or tonumber(aq.min_sim_gate) or 0.58
end

function M.get_merge_limit(base_limit)
    local aq = ai_cfg()
    if aq.refinement_enabled ~= true then
        return tonumber(base_limit) or default_merge_limit()
    end
    return tonumber(M.state.online_merge_limit) or tonumber(base_limit) or default_merge_limit()
end

function M.get_route_score(cluster_id)
    cluster_id = tonumber(cluster_id)
    if not cluster_id then return 0.0 end
    return tonumber(M.state.cluster_route_score[cluster_id]) or 0.0
end

function M.add_counter(name, delta)
    if not name then return end
    local d = tonumber(delta) or 1
    local cur = tonumber(M.state[name]) or 0
    M.state[name] = cur + d
    M.dirty = true
end

function M.snapshot()
    local out = {}
    for _, k in ipairs(SCALAR_KEYS) do
        out[k] = tonumber(M.state[k]) or 0
    end
    local route_n = 0
    local route_abs = 0
    for _, score in pairs(M.state.cluster_route_score) do
        route_n = route_n + 1
        route_abs = route_abs + math.abs(tonumber(score) or 0)
    end
    out.route_score_count = route_n
    out.route_score_abs_mean = route_n > 0 and (route_abs / route_n) or 0
    return out
end

function M.update_after_recall(event)
    local aq = ai_cfg()
    if aq.refinement_enabled ~= true then return end

    local turn = tonumber((event or {}).turn) or 0
    if turn < (tonumber(aq.refinement_start_turn) or 200) then
        return
    end

    local samples = (event and event.candidate_samples) or {}
    if #samples == 0 then return end

    local prog = refinement_progress(turn)
    local route_lr = (tonumber(aq.refinement_route_lr) or 0.10) * (0.18 + 0.82 * prog)
    local gate_lr = (tonumber(aq.refinement_gate_lr) or 0.08) * (prog ^ 1.6)
    local merge_lr = (tonumber(aq.refinement_merge_lr) or 0.05) * (0.12 + 0.88 * prog)
    if route_lr <= 0 and gate_lr <= 0 and merge_lr <= 0 then return end

    local pos_set = {}
    local neg_set = {}
    for _, mem_idx in ipairs((event and event.positive_memories) or {}) do
        pos_set[tonumber(mem_idx)] = true
    end
    for _, mem_idx in ipairs((event and event.negative_memories) or {}) do
        neg_set[tonumber(mem_idx)] = true
    end

    local pos_sims, neg_sims = {}, {}
    local clu_pos, clu_neg = {}, {}

    for _, s in ipairs(samples) do
        local mem_idx = tonumber(s.mem_idx)
        local cid = tonumber(s.cid)
        if mem_idx and cid then
            local sim = tonumber(s.sim) or 0
            local eff = math.max(0, tonumber(s.effective) or sim)
            if pos_set[mem_idx] then
                pos_sims[#pos_sims + 1] = sim
                clu_pos[cid] = (clu_pos[cid] or 0) + math.max(1e-6, eff)
            elseif neg_set[mem_idx] then
                neg_sims[#neg_sims + 1] = sim
                clu_neg[cid] = (clu_neg[cid] or 0) + math.max(1e-6, eff)
            end
        end
    end

    local route_decay = 1.0 - math.min(0.12, route_lr * 0.22)
    local touched = {}
    for cid in pairs(clu_pos) do touched[cid] = true end
    for cid in pairs(clu_neg) do touched[cid] = true end
    for cid in pairs(touched) do
        local p_w = clu_pos[cid] or 0
        local n_w = clu_neg[cid] or 0
        local total = p_w + n_w
        if total > 0 then
            local signal = (p_w - n_w) / total
            local prev = tonumber(M.state.cluster_route_score[cid]) or 0
            local nxt = clamp(prev * route_decay + route_lr * signal, -2.0, 2.0)
            M.state.cluster_route_score[cid] = nxt
            M.state.cluster_route_seen[cid] = (tonumber(M.state.cluster_route_seen[cid]) or 0) + 1
        end
    end

    local hits_all = tonumber((event or {}).hits_all) or 0
    local gate_target = tonumber(M.state.learned_min_gate) or default_learned_gate()
    if #pos_sims > 0 and #neg_sims > 0 then
        local pos_floor = quantile(pos_sims, 0.20)
        local neg_ceil = quantile(neg_sims, 0.80)
        if pos_floor and neg_ceil then
            gate_target = 0.5 * (pos_floor + neg_ceil)
        end
    elseif #neg_sims > 0 then
        gate_target = gate_target + ((hits_all <= 0) and 0.035 or 0.018)
    elseif #pos_sims > 0 then
        gate_target = gate_target - ((hits_all <= 0) and 0.020 or 0.008)
    end

    local start_gate = tonumber(aq.learning_min_sim_gate_start) or tonumber(aq.min_sim_gate) or 0.58
    local end_gate = tonumber(aq.min_sim_gate) or start_gate
    local gate_lo = math.max(0.05, math.min(start_gate, end_gate) - 0.22)
    local gate_hi = math.min(0.90, end_gate + 0.08)
    gate_target = clamp(gate_target, gate_lo, gate_hi)
    local old_gate = tonumber(M.state.learned_min_gate) or default_learned_gate()
    M.state.learned_min_gate = old_gate + gate_lr * (gate_target - old_gate)

    local pos_n = #pos_sims
    local neg_n = #neg_sims
    local merge_base = default_merge_limit()
    local merge_target = merge_base
    local neg_ratio = neg_n / math.max(1, pos_n + neg_n)
    if neg_ratio >= 0.66 then
        merge_target = merge_base - 0.055
    elseif neg_ratio >= 0.52 then
        merge_target = merge_base - 0.030
    elseif hits_all <= 0 and pos_n <= 0 then
        merge_target = merge_base + 0.020
    elseif pos_n > neg_n * 1.4 then
        merge_target = merge_base + 0.010
    end

    local merge_lo = math.max(0.50, merge_base - 0.10)
    local merge_hi = math.min(0.995, merge_base + 0.03)
    merge_target = clamp(merge_target, merge_lo, merge_hi)
    local old_merge = tonumber(M.state.online_merge_limit) or merge_base
    M.state.online_merge_limit = old_merge + merge_lr * (merge_target - old_merge)

    M.state.refinement_events = (tonumber(M.state.refinement_events) or 0) + 1
    M.dirty = true
end

M.reset_defaults()

return M
