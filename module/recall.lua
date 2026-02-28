-- recall.lua
-- 召回主链：learning/refinement/persistent-explore/topic-cache/cold-rescue

local M = {}

local tool = require("module.tool")
local memory = require("module.memory")
local history = require("module.history")
local config = require("module.config")
local cluster = require("module.cluster")
local topic = require("module.topic")
local adaptive = require("module.adaptive")
local heat = require("module.heat")

local ANXIETY_SENTENCES = {
    "我很焦虑", "我急死了", "快点", "急急急", "我现在很着急"
}

local HELP_CRY_SENTENCES = {
    "救命", "帮帮我", "求助", "救救我", "help"
}

local PAST_TALK_SENTENCES = {
    "之前", "过去", "曾经", "上一次", "以前", "recall", "remember"
}

local anxiety_vec = nil
local help_cry_vec = nil
local past_talk_vec = nil

M._last_topic_anchor = nil
M._prev_query_vec = nil
M._same_topic_streak = 0
M._streak_sim_sum = 0.0
M._streak_sim_count = 0
M._topic_cache_anchor = nil
M._topic_cache_mem = {}
M.metrics = {
    query_turns = 0,
    empty_query_count = 0,
    hits_same_sum = 0,
    recall_recent_sum = 0.0,
    sim_ops = {},
}

local function ai_cfg()
    return ((config.settings or {}).ai_query or {})
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function percentile(values, q)
    local n = #values
    if n <= 0 then return 0.0 end
    local arr = {}
    for i = 1, n do arr[i] = tonumber(values[i]) or 0.0 end
    table.sort(arr)
    if n == 1 then return arr[1] end
    local qq = clamp(tonumber(q) or 0.5, 0.0, 1.0)
    local pos = 1 + (n - 1) * qq
    local lo = math.floor(pos)
    local hi = math.ceil(pos)
    if lo == hi then return arr[lo] end
    local t = pos - lo
    return arr[lo] * (1 - t) + arr[hi] * t
end

function M.reset_metrics()
    M.metrics = {
        query_turns = 0,
        empty_query_count = 0,
        hits_same_sum = 0,
        recall_recent_sum = 0.0,
        sim_ops = {},
    }
end

function M.get_metrics_snapshot()
    local qn = math.max(1, tonumber(M.metrics.query_turns) or 0)
    local p95 = percentile(M.metrics.sim_ops or {}, 0.95)
    return {
        query_turns = tonumber(M.metrics.query_turns) or 0,
        hits_same = tonumber(M.metrics.hits_same_sum) or 0,
        empty_query_rate = (tonumber(M.metrics.empty_query_count) or 0) / qn,
        target_recall_recent = (tonumber(M.metrics.recall_recent_sum) or 0.0) / qn,
        p95_sim_ops = p95,
    }
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function shuffle_inplace(arr)
    for i = #arr, 2, -1 do
        local j = math.random(i)
        arr[i], arr[j] = arr[j], arr[i]
    end
end

local function topic_relation(current_info, ti, sim_th)
    if not current_info or not ti then
        return "cross"
    end

    if ti.is_active and current_info.is_active then
        return "same"
    end

    if (not ti.is_active) and (not current_info.is_active)
        and ti.topic_idx and current_info.topic_idx
        and ti.topic_idx == current_info.topic_idx then
        return "same"
    end

    if ti.centroid and current_info.centroid then
        local ts = tool.cosine_similarity(ti.centroid, current_info.centroid)
        if ts >= sim_th then
            return "near"
        end
    end

    return "cross"
end

local function topic_relation_for_turn(turn, current_info, sim_th)
    local ti = topic.get_topic_for_turn and topic.get_topic_for_turn(turn) or nil
    return topic_relation(current_info, ti, sim_th)
end

local function compute_average_vector(sentences)
    if #sentences == 0 then return {} end
    local sum = nil
    for _, text in ipairs(sentences) do
        local vec = tool.get_embedding(text)
        if sum == nil then
            sum = vec
        else
            for i = 1, #vec do
                sum[i] = sum[i] + vec[i]
            end
        end
    end
    local norm = 0
    for i = 1, #sum do
        norm = norm + sum[i] * sum[i]
    end
    norm = math.sqrt(norm)
    if norm > 0 then
        for i = 1, #sum do
            sum[i] = sum[i] / norm
        end
    end
    return sum
end

function M.init_all_sentiment_vectors()
    anxiety_vec = compute_average_vector(ANXIETY_SENTENCES)
    help_cry_vec = compute_average_vector(HELP_CRY_SENTENCES)
    past_talk_vec = compute_average_vector(PAST_TALK_SENTENCES)
    return anxiety_vec, help_cry_vec, past_talk_vec
end

local function compute_recall_score(user_input, user_vec)
    local score = 0
    local cfg = ai_cfg()

    local past_keywords = {"之前", "上次", "以前", "过去", "曾经", "recall", "remember"}
    for _, kw in ipairs(past_keywords) do
        if user_input:find(kw, 1, true) then
            score = score + (cfg.history_search_bonus or 0)
            break
        end
    end

    local tech_keywords = {"代码", "函数", "API", "算法", "编程", "Python", "Lua", "配置", "参数"}
    for _, kw in ipairs(tech_keywords) do
        if user_input:find(kw, 1, true) then
            score = score + (cfg.technical_term_bonus or 0)
            break
        end
    end

    if #user_input >= (cfg.length_limit or 20) then
        score = score + (cfg.length_bonus or 0)
    end

    if anxiety_vec and #anxiety_vec > 0 and user_vec and #user_vec > 0 then
        score = score + tool.cosine_similarity(user_vec, anxiety_vec) * (cfg.anxiety_multi or 0)
    end
    if help_cry_vec and #help_cry_vec > 0 and user_vec and #user_vec > 0 then
        score = score + tool.cosine_similarity(user_vec, help_cry_vec) * (cfg.help_cry_multi or 0)
    end
    if past_talk_vec and #past_talk_vec > 0 and user_vec and #user_vec > 0 then
        score = score + tool.cosine_similarity(user_vec, past_talk_vec) * (cfg.past_talk_multi or 0)
    end

    return score
end

local function cold_start_strength(turn)
    local cfg = ai_cfg()
    if cfg.cold_start_enabled ~= true then
        return 0.0
    end
    local end_t = math.max(1, tonumber(cfg.cold_start_turns) or tonumber(cfg.learning_full_turns) or 1600)
    local t = math.max(1, tonumber(turn) or 1)
    if t >= end_t then
        return 0.0
    end
    local s = 1.0 - ((t - 1) / math.max(1, end_t - 1))
    return clamp(s, 0.0, 1.0)
end

local function need_recall(user_input, user_vec, current_turn)
    user_vec = user_vec or tool.get_embedding_query(user_input)
    local score = compute_recall_score(user_input, user_vec)
    local cfg = ai_cfg()
    local threshold = cfg.recall_base or 5.3

    local cs = cold_start_strength(current_turn)
    if cs > 0 then
        local min_scale = clamp(tonumber(cfg.cold_start_recall_base_scale) or 0.82, 0.30, 1.00)
        local scale = 1.0 - cs * (1.0 - min_scale)
        threshold = threshold * scale
    end

    print(string.format(
        "[Recall] 回忆分数 = %.2f (阈值 %.2f, cold_start=%.2f)",
        score, threshold, cs
    ))
    return score >= threshold
end

local function normalize_vec(vec)
    local norm = 0.0
    for i = 1, #vec do
        local v = tonumber(vec[i]) or 0.0
        vec[i] = v
        norm = norm + v * v
    end
    norm = math.sqrt(norm)
    if norm > 0 then
        for i = 1, #vec do
            vec[i] = vec[i] / norm
        end
    end
    return vec
end

local function random_unit_like(ref_vec)
    local out = {}
    local n = #ref_vec
    if n <= 0 then return out end
    for i = 1, n do
        out[i] = math.random() * 2.0 - 1.0
    end
    return normalize_vec(out)
end

local function lerp(a, b, t)
    return a + (b - a) * t
end

local function learning_progress(turn)
    local cfg = ai_cfg()
    if cfg.learning_curve_enabled ~= true then
        return 1.0
    end
    local warm = math.max(0, tonumber(cfg.learning_warmup_turns) or 500)
    local full = math.max(warm + 1, tonumber(cfg.learning_full_turns) or 12000)
    local t = tonumber(turn) or 0
    if t <= warm then return 0.0 end
    if t >= full then return 1.0 end
    return (t - warm) / (full - warm)
end

local function refinement_progress(turn)
    local cfg = ai_cfg()
    if cfg.refinement_enabled ~= true then
        return 0.0
    end
    local start_t = math.max(0, tonumber(cfg.refinement_start_turn) or 200)
    local full_t = math.max(start_t + 1, tonumber(cfg.learning_full_turns) or 12000)
    local t = tonumber(turn) or 0
    if t <= start_t then return 0.0 end
    if t >= full_t then return 1.0 end
    return (t - start_t) / (full_t - start_t)
end

local function effective_probe_clusters(turn)
    local cfg = ai_cfg()
    local start_k = math.max(1, tonumber(cfg.refinement_probe_clusters_start) or 8)
    local end_k = math.max(1, tonumber(cfg.refinement_probe_clusters_end) or 2)
    if cfg.refinement_enabled ~= true then
        return end_k
    end
    local rp = refinement_progress(turn)
    return math.max(1, math.floor(lerp(start_k, end_k, rp) + 0.5))
end

local function effective_retrieval_knobs(turn)
    local cfg = ai_cfg()
    local prog = learning_progress(turn)
    local min_gate = lerp(tonumber(cfg.learning_min_sim_gate_start) or 0.42, tonumber(cfg.min_sim_gate) or 0.58, prog)
    local power = lerp(tonumber(cfg.learning_power_suppress_start) or 1.15, tonumber(cfg.power_suppress) or 1.8, prog)
    local cross = lerp(tonumber(cfg.learning_topic_cross_quota_start) or 0.48, tonumber(cfg.topic_cross_quota_ratio) or 0.25, prog)
    local kw_weight = lerp(tonumber(cfg.learning_keyword_weight_start) or 0.78, tonumber(cfg.keyword_weight) or 0.55, prog)
    local max_memory = math.floor(lerp(tonumber(cfg.learning_max_memory_start) or 3, tonumber(cfg.max_memory) or 5, prog) + 0.5)
    local max_turns = math.floor(lerp(tonumber(cfg.learning_max_turns_start) or 14, tonumber(cfg.max_turns) or 10, prog) + 0.5)
    local super_q = math.floor(lerp(tonumber(cfg.learning_super_topn_query_start) or 2, tonumber(((config.settings or {}).cluster or {}).supercluster_topn_query) or 4, prog) + 0.5)

    if cfg.refinement_enabled == true then
        min_gate = adaptive.get_min_sim_gate(min_gate)
    end

    local cs = cold_start_strength(turn)
    if cs > 0 then
        local gate_drop = math.max(0.0, tonumber(cfg.cold_start_min_gate_drop) or 0.04)
        local power_drop = math.max(0.0, tonumber(cfg.cold_start_power_drop) or 0.18)
        local mem_boost = math.max(0, tonumber(cfg.cold_start_max_memory_boost) or 2)
        local turns_boost = math.max(0, tonumber(cfg.cold_start_max_turns_boost) or 2)
        local probe_boost = math.max(0, tonumber(cfg.cold_start_probe_clusters_boost) or 2)

        min_gate = min_gate - gate_drop * cs
        power = power - power_drop * cs
        max_memory = max_memory + math.floor(mem_boost * cs + 0.5)
        max_turns = max_turns + math.floor(turns_boost * cs + 0.5)
        super_q = super_q + math.floor(probe_boost * cs + 0.5)
    end

    return {
        progress = prog,
        min_sim_gate = clamp(min_gate, 0.05, 0.95),
        power_suppress = math.max(1.0, power),
        topic_cross_quota_ratio = clamp(cross, 0.0, 0.5),
        keyword_weight = math.max(0.0, kw_weight),
        max_memory = math.max(1, max_memory),
        max_turns = math.max(1, max_turns),
        supercluster_topn_query = math.max(1, super_q),
        probe_clusters = math.max(1, effective_probe_clusters(turn)),
    }
end

local function top_probe_clusters(vec, probe_clusters, super_topn, turn)
    local cfg = ai_cfg()
    local cand, ops0 = cluster.super_candidate_clusters(vec, super_topn)
    if #cand <= 0 then
        return {}, ops0
    end

    local route_scale = 0.0
    if cfg.refinement_enabled == true and (tonumber(turn) or 0) >= (tonumber(cfg.refinement_start_turn) or 200) then
        route_scale = (tonumber(cfg.refinement_route_bias_scale) or 0.08) * refinement_progress(turn)
    end

    local scored = {}
    local ops = ops0
    for _, cid in ipairs(cand) do
        local clu = cluster.clusters[cid]
        if clu and clu.centroid then
            local sim = tool.cosine_similarity(vec, clu.centroid)
            ops = ops + 1
            local adjusted = sim
            if route_scale > 0 then
                adjusted = adjusted + route_scale * adaptive.get_route_score(cid)
            end
            scored[#scored + 1] = { cid = cid, adjusted = adjusted }
        end
    end

    if #scored <= 0 then
        return {}, ops
    end

    table.sort(scored, function(a, b) return a.adjusted > b.adjusted end)
    local k = math.max(1, math.min(tonumber(probe_clusters) or 1, #scored))
    local out = {}
    for i = 1, k do
        out[#out + 1] = scored[i].cid
    end

    if cfg.refinement_enabled == true and #scored > k then
        local rp = refinement_progress(turn)
        local explore_n = math.floor((1.0 - rp) * k * 0.75 + 0.5)
        if explore_n > 0 then
            local remain = {}
            for i = k + 1, #scored do
                remain[#remain + 1] = scored[i].cid
            end
            shuffle_inplace(remain)
            for i = 1, math.min(explore_n, #remain) do
                out[#out + 1] = remain[i]
            end
        end
    end

    return out, ops
end

local function unload_topic_cache()
    if next(M._topic_cache_mem) ~= nil then
        adaptive.add_counter("topic_cache_unload_count", 1)
    end
    M._topic_cache_mem = {}
    M._topic_cache_anchor = nil
end

local function update_topic_stability(current_anchor, query_vec)
    local cfg = ai_cfg()
    local stable_warm = math.max(1, tonumber(cfg.stable_warmup_turns) or 6)
    local stable_sim = clamp(tonumber(cfg.stable_min_pair_sim) or 0.72, 0.0, 1.0)

    if current_anchor and current_anchor == M._last_topic_anchor then
        M._same_topic_streak = M._same_topic_streak + 1
        if M._prev_query_vec and #M._prev_query_vec > 0 and query_vec and #query_vec > 0 then
            local sim = tool.cosine_similarity(M._prev_query_vec, query_vec)
            M._streak_sim_sum = M._streak_sim_sum + sim
            M._streak_sim_count = M._streak_sim_count + 1
        end
    else
        M._same_topic_streak = 1
        M._streak_sim_sum = 0.0
        M._streak_sim_count = 0
    end

    M._last_topic_anchor = current_anchor
    M._prev_query_vec = shallow_copy_array(query_vec or {})

    local avg_pair = 1.0
    if M._streak_sim_count > 0 then
        avg_pair = M._streak_sim_sum / M._streak_sim_count
    end

    return (M._same_topic_streak >= stable_warm) and (avg_pair >= stable_sim)
end

local function memory_matches_current_topic(mem_idx, current_info, sim_th)
    local mem = memory.memories[mem_idx]
    if not mem or not mem.turns then return false end
    for _, t in ipairs(mem.turns) do
        if topic_relation_for_turn(t, current_info, sim_th) == "same" then
            return true
        end
    end
    return false
end

local function topic_random_lift(turn, current_anchor, current_info, stable_ready)
    local cfg = ai_cfg()
    if not current_anchor or not current_info then return end
    if not stable_ready then return end

    local interval = math.max(1, tonumber(cfg.topic_random_lift_interval) or 3)
    if interval > 1 and (turn % interval) ~= 0 then return end

    local prob = clamp(tonumber(cfg.topic_random_lift_prob) or 0.85, 0.0, 1.0)
    if math.random() > prob then return end

    local sim_th = tonumber(cfg.topic_sim_threshold) or 0.70
    local only_cold = cfg.topic_random_lift_only_cold ~= false

    local candidates = {}
    for idx = 1, memory.get_total_lines() do
        if memory_matches_current_topic(idx, current_info, sim_th) then
            if (not only_cold) or (not heat.is_hot(idx)) then
                candidates[#candidates + 1] = idx
            end
        end
    end
    if #candidates <= 0 then return end

    adaptive.add_counter("topic_lift_attempted", 1)
    shuffle_inplace(candidates)

    local take_n = math.max(1, tonumber(cfg.topic_random_lift_count) or 2)
    local picked = {}
    for i = 1, math.min(take_n, #candidates) do
        picked[#picked + 1] = candidates[i]
    end
    if #picked <= 0 then return end

    M._topic_cache_anchor = current_anchor
    M._topic_cache_mem = {}
    for _, idx in ipairs(picked) do
        M._topic_cache_mem[idx] = true
    end
    adaptive.add_counter("topic_lift_executed", #picked)
end

local function build_query_vectors(user_vec, keyword_weight)
    local cfg = ai_cfg()
    local out = {
        { vec = user_vec, weight = 1.0, is_primary = true }
    }

    local num = math.max(0, math.floor(tonumber(cfg.keyword_queries) or 0))
    if num <= 0 or type(user_vec) ~= "table" or #user_vec <= 0 then
        return out
    end

    local mix = clamp(tonumber(cfg.keyword_noise_mix) or 0.20, 0.0, 0.95)
    local kw_weight = tonumber(keyword_weight) or tonumber(cfg.keyword_weight) or 0.55
    for _ = 1, num do
        local noise = random_unit_like(user_vec)
        local qv = {}
        for i = 1, #user_vec do
            local base = tonumber(user_vec[i]) or 0.0
            local nv = tonumber(noise[i]) or 0.0
            qv[i] = (1.0 - mix) * base + mix * nv
        end
        qv = normalize_vec(qv)
        out[#out + 1] = {
            vec = qv,
            weight = kw_weight,
            is_primary = false,
        }
    end
    return out
end

local function to_sorted_pairs(turn_best)
    local out = {}
    for turn, score in pairs(turn_best) do
        out[#out + 1] = { turn = turn, score = score }
    end
    table.sort(out, function(a, b) return a.score > b.score end)
    return out
end

local function collect_candidate_samples(mem_best_sim, mem_best_cluster, mem_best_effective, limit)
    local items = {}
    for mem_idx, sim in pairs(mem_best_sim) do
        items[#items + 1] = { mem_idx = mem_idx, sim = sim }
    end
    table.sort(items, function(a, b) return a.sim > b.sim end)

    local out = {}
    for i = 1, math.min(limit, #items) do
        local mem_idx = items[i].mem_idx
        out[#out + 1] = {
            mem_idx = mem_idx,
            cid = tonumber(mem_best_cluster[mem_idx]) or -1,
            sim = tonumber(items[i].sim) or 0.0,
            effective = tonumber(mem_best_effective[mem_idx]) or tonumber(items[i].sim) or 0.0,
        }
    end
    return out
end

local function memory_topic_counts(mem_idx)
    local out = {}
    local mem = memory.memories[mem_idx]
    if not mem or not mem.turns then return out end
    for _, t in ipairs(mem.turns) do
        local tid = topic.get_topic_id_for_turn and topic.get_topic_id_for_turn(t) or nil
        if tid then
            out[tid] = (out[tid] or 0) + 1
        end
    end
    return out
end

local function dominant_memory_topic_id(mem_idx)
    local counts = memory_topic_counts(mem_idx)
    local best_id = nil
    local best_n = -1
    for tid, n in pairs(counts) do
        if n > best_n then
            best_id = tid
            best_n = n
        end
    end
    return best_id
end

local function topic_ids_same_band(a, b, sim_th)
    if not a or not b then return false end
    if a == b then return true end

    local ai = topic.get_topic_info_by_id and topic.get_topic_info_by_id(a) or nil
    local bi = topic.get_topic_info_by_id and topic.get_topic_info_by_id(b) or nil
    if not ai or not bi then return false end

    if ai.is_active and bi.is_active then return true end
    if (not ai.is_active) and (not bi.is_active)
        and ai.topic_idx and bi.topic_idx
        and ai.topic_idx == bi.topic_idx then
        return true
    end

    if ai.centroid and bi.centroid then
        local ts = tool.cosine_similarity(ai.centroid, bi.centroid)
        return ts >= sim_th
    end
    return false
end

local function apply_refinement(
    turn,
    hits_all,
    candidate_samples,
    selected_memories,
    evidence_memories,
    target_topic_id,
    current_info,
    sim_th,
    min_gate
)
    if ai_cfg().refinement_enabled ~= true then return end
    if not current_info or not target_topic_id then return end

    local evidence_set = {}
    local selected_set = {}
    for _, mem_idx in ipairs(evidence_memories or {}) do
        evidence_set[tonumber(mem_idx)] = true
    end
    for _, mem_idx in ipairs(selected_memories or {}) do
        selected_set[tonumber(mem_idx)] = true
    end

    local pos_set = {}
    local neg_set = {}
    for _, s in ipairs(candidate_samples or {}) do
        local mem_idx = tonumber(s.mem_idx)
        local sim = tonumber(s.sim) or 0.0
        if mem_idx and mem_idx > 0 and memory.memories[mem_idx] then
            local is_pos = evidence_set[mem_idx] == true
            if not is_pos then
                local cnt = memory_topic_counts(mem_idx)[target_topic_id] or 0
                if cnt > 0 and selected_set[mem_idx] and sim >= min_gate then
                    is_pos = true
                end
            end

            local is_neg = false
            if not is_pos then
                local dom = dominant_memory_topic_id(mem_idx)
                if dom and topic_ids_same_band(dom, target_topic_id, sim_th) then
                    if selected_set[mem_idx] or sim >= (min_gate * 0.92) then
                        is_neg = true
                    end
                end
            end

            if is_pos then
                pos_set[mem_idx] = true
            elseif is_neg then
                neg_set[mem_idx] = true
            end
        end
    end

    local pos = {}
    local neg = {}
    for idx in pairs(pos_set) do pos[#pos + 1] = idx end
    for idx in pairs(neg_set) do
        if not pos_set[idx] then
            neg[#neg + 1] = idx
        end
    end

    adaptive.update_after_recall({
        turn = turn,
        hits_all = hits_all,
        candidate_samples = candidate_samples,
        positive_memories = pos,
        negative_memories = neg,
    })
end

local function retrieve(user_input, user_vec, current_turn, current_info, current_anchor, opts)
    if memory.get_total_lines() <= 0 then
        return ""
    end
    opts = opts or {}
    local freeze_mode = opts.freeze_mode == true

    local cfg = ai_cfg()
    local knobs = effective_retrieval_knobs(current_turn)
    local min_gate = tonumber(knobs.min_sim_gate) or (cfg.min_sim_gate or 0.58)
    local power = tonumber(knobs.power_suppress) or (cfg.power_suppress or 1.80)
    local max_memory = math.max(1, tonumber(knobs.max_memory) or 5)
    local max_turns = math.max(1, tonumber(knobs.max_turns) or 10)
    local query_topn = math.max(1, tonumber(knobs.supercluster_topn_query) or 4)
    local probe_clusters = math.max(1, tonumber(knobs.probe_clusters) or 2)
    local cross_quota_ratio = clamp(tonumber(knobs.topic_cross_quota_ratio) or 0.25, 0.0, 0.5)
    local keyword_weight = math.max(0.0, tonumber(knobs.keyword_weight) or 0.55)
    local sim_th = tonumber(cfg.topic_sim_threshold) or 0.70

    local per_cluster_limit = math.max(2, tonumber(cfg.refinement_probe_per_cluster_limit) or 12)
    local base_scan_limit = math.max(per_cluster_limit, max_memory)
    local persistent_cap = math.max(1, tonumber(cfg.persistent_explore_candidate_cap) or 32)
    local sim_ops = 0

    local persistent_extra_budget = 0
        if cfg.persistent_explore_enabled == true and cluster.cluster_count() > 1 then
        local eps = clamp(tonumber(cfg.persistent_explore_epsilon) or 0.01, 0.0, 1.0)
        local periodic = math.max(0, tonumber(cfg.persistent_explore_period_turns) or 0)
        local trigger = false
        if eps > 0 and math.random() < eps then trigger = true end
        if periodic > 0 and (current_turn % periodic) == 0 then trigger = true end
        if trigger then
            persistent_extra_budget = math.max(1, tonumber(cfg.persistent_explore_extra_clusters) or 1)
            if not freeze_mode then
                adaptive.add_counter("persistent_explore_events", 1)
            end
        end
    end

    local turn_best = {}
    local turn_src = {}
    local turn_mem = {}
    local mem_best_sim = {}
    local mem_best_cluster = {}
    local mem_best_effective = {}
    local persistent_probe_clusters_query = {}

    local qvecs = build_query_vectors(user_vec, keyword_weight)
    local primary = qvecs[1]
    if (not primary) or type(primary.vec) ~= "table" or #primary.vec == 0 then
        return ""
    end

    local function push_turn_score(mem_idx, effective, src_label)
        local mem_data = memory.memories[mem_idx]
        if not mem_data or not mem_data.turns then return end
        for _, t in ipairs(mem_data.turns) do
            if (not turn_best[t]) or effective > turn_best[t] then
                turn_best[t] = effective
                turn_src[t] = src_label or "hot"
                turn_mem[t] = mem_idx
            end
        end
    end

    local function update_mem_best(mem_idx, sim, weight, cid)
        local sim_pos = math.max(0.0, sim)
        local prev_sim = mem_best_sim[mem_idx]
        if (not prev_sim) or sim > prev_sim then
            mem_best_sim[mem_idx] = sim
            mem_best_cluster[mem_idx] = tonumber(cid) or (cluster.get_cluster_id_for_line(mem_idx) or -1)
            mem_best_effective[mem_idx] = (sim_pos ^ power) * weight
        end
    end

    for q_idx, q in ipairs(qvecs) do
        local qv = q.vec
        local weight = tonumber(q.weight) or 1.0
        if type(qv) ~= "table" or #qv <= 0 then
            goto continue_query
        end

        local cluster_ids, ops = top_probe_clusters(qv, probe_clusters, query_topn, current_turn)
        sim_ops = sim_ops + (tonumber(ops) or 0)
        local persistent_probe_clusters_vec = {}

        if q_idx == 1 and persistent_extra_budget > 0 and #cluster_ids > 0 then
            local picked = {}
            for _, cid in ipairs(cluster_ids) do picked[cid] = true end
            local pool = {}
            for _, cid in ipairs(cluster.get_cluster_ids()) do
                if not picked[cid] then
                    pool[#pool + 1] = cid
                end
            end
            if #pool > 0 then
                shuffle_inplace(pool)
                local take = math.min(persistent_extra_budget, #pool)
                for i = 1, take do
                    local cid = pool[i]
                    cluster_ids[#cluster_ids + 1] = cid
                    persistent_probe_clusters_vec[cid] = true
                    persistent_probe_clusters_query[cid] = true
                end
                if not freeze_mode then
                    adaptive.add_counter("persistent_explore_cluster_probes", take)
                end
            end
        end

        local scanned_any = false
        for _, cid in ipairs(cluster_ids) do
            local scan_limit = base_scan_limit
            local src_label = "hot"
            if persistent_probe_clusters_vec[cid] then
                scan_limit = math.min(scan_limit, persistent_cap)
                src_label = "explore"
            end

            local sim_results, ops2 = cluster.find_sim_in_cluster(qv, cid, {
                only_hot = true,
                max_results = scan_limit,
            })
            sim_ops = sim_ops + (tonumber(ops2) or #sim_results)
            if #sim_results > 0 then
                scanned_any = true
            end

            for _, mem in ipairs(sim_results) do
                local mem_idx = tonumber(mem.index)
                local sim = tonumber(mem.similarity) or 0.0
                if mem_idx then
                    update_mem_best(mem_idx, sim, weight, cid)
                    if sim < min_gate then
                        break
                    end
                    local effective = (sim ^ power) * weight
                    push_turn_score(mem_idx, effective, src_label)
                end
            end
        end

        if not scanned_any then
            local fallback, ops2 = memory.find_similar_all_fast(qv, max_memory * 2)
            sim_ops = sim_ops + (tonumber(ops2) or #fallback)
            for _, mem in ipairs(fallback or {}) do
                local mem_idx = tonumber(mem.index)
                local sim = tonumber(mem.similarity) or 0.0
                if mem_idx then
                    local cid = cluster.get_cluster_id_for_line(mem_idx) or -1
                    update_mem_best(mem_idx, sim, weight, cid)
                    if sim < min_gate then
                        break
                    end
                    local effective = (sim ^ power) * weight
                    push_turn_score(mem_idx, effective, "hot")
                end
            end
        end

        if current_anchor and M._topic_cache_anchor == current_anchor and next(M._topic_cache_mem) ~= nil then
            local cache_scored = {}
            for mem_idx in pairs(M._topic_cache_mem) do
                local mem_vec = memory.return_mem_vec(mem_idx)
                if mem_vec then
                    local sim = tool.cosine_similarity(qv, mem_vec)
                    cache_scored[#cache_scored + 1] = { mem_idx = mem_idx, sim = sim }
                end
            end
            sim_ops = sim_ops + #cache_scored
            table.sort(cache_scored, function(a, b) return a.sim > b.sim end)
            for _, item in ipairs(cache_scored) do
                local sim = tonumber(item.sim) or 0.0
                local mem_idx = tonumber(item.mem_idx)
                if mem_idx then
                    local cid = cluster.get_cluster_id_for_line(mem_idx) or -1
                    update_mem_best(mem_idx, sim, weight, cid)
                    if sim < min_gate then
                        break
                    end
                    local effective = (sim ^ power) * weight * (tonumber(cfg.topic_cache_weight) or 1.02)
                    push_turn_score(mem_idx, effective, "cache")
                end
            end
        end

        ::continue_query::
    end

    local sample_limit = math.max(1, tonumber(cfg.refinement_sample_mem_topk) or 48)
    local candidate_samples = collect_candidate_samples(mem_best_sim, mem_best_cluster, mem_best_effective, sample_limit)
    local ranked = to_sorted_pairs(turn_best)

    if #ranked <= 0 then
        if not freeze_mode then
            heat.enqueue_cold_rescue(user_vec, current_turn, current_info, min_gate)
        end
        M.metrics.query_turns = (tonumber(M.metrics.query_turns) or 0) + 1
        M.metrics.empty_query_count = (tonumber(M.metrics.empty_query_count) or 0) + 1
        M.metrics.sim_ops[#M.metrics.sim_ops + 1] = sim_ops
        return ""
    end

    local selected = {}
    if cfg.use_topic_buckets ~= true or not current_info then
        for i = 1, math.min(max_turns, #ranked) do
            selected[#selected + 1] = ranked[i]
        end

        if cfg.refinement_enabled == true then
            local rp = refinement_progress(current_turn)
            local explore_slots = math.floor(max_turns * 0.55 * (1.0 - rp) + 0.5)
            if explore_slots > 0 and #ranked > max_turns then
                local core_n = math.max(1, max_turns - explore_slots)
                local core = {}
                for i = 1, math.min(core_n, #ranked) do
                    core[#core + 1] = ranked[i]
                end
                local pool = {}
                for i = core_n + 1, math.min(#ranked, max_turns * 6) do
                    pool[#pool + 1] = ranked[i]
                end
                if #pool > 0 then
                    shuffle_inplace(pool)
                    selected = {}
                    for _, it in ipairs(core) do selected[#selected + 1] = it end
                    local choose_n = math.min(explore_slots, #pool)
                    for i = 1, choose_n do
                        selected[#selected + 1] = pool[i]
                    end
                    table.sort(selected, function(a, b) return a.score > b.score end)
                    while #selected > max_turns do
                        table.remove(selected)
                    end
                end
            end
        end
    else
        local same, near, cross = {}, {}, {}
        for _, item in ipairs(ranked) do
            local rel = topic_relation_for_turn(item.turn, current_info, sim_th)
            if rel == "same" then
                same[#same + 1] = item
            elseif rel == "near" then
                near[#near + 1] = item
            else
                cross[#cross + 1] = item
            end
        end

        local reserved_cross = math.min(#cross, math.floor(max_turns * cross_quota_ratio))
        local in_topic_budget = max_turns - reserved_cross
        local used = {}
        local function append_until(src, limit)
            for _, item in ipairs(src) do
                if #selected >= limit then break end
                if not used[item.turn] then
                    used[item.turn] = true
                    selected[#selected + 1] = item
                end
            end
        end

        append_until(same, in_topic_budget)
        append_until(near, in_topic_budget)
        append_until(cross, max_turns)
        if #selected < max_turns then
            append_until(near, max_turns)
            append_until(same, max_turns)
            append_until(cross, max_turns)
        end
        table.sort(selected, function(a, b) return a.score > b.score end)
        while #selected > max_turns do
            table.remove(selected)
        end
    end

    local selected_turns = {}
    local selected_memories = {}
    local selected_mem_set = {}
    local evidence_memories = {}
    local evidence_set = {}
    local target_topic_id = topic.get_topic_id_for_turn and topic.get_topic_id_for_turn(current_turn) or nil
    local cache_contrib = 0

    for _, item in ipairs(selected) do
        local t = tonumber(item.turn)
        if t then
            selected_turns[#selected_turns + 1] = t
            local mem_idx = turn_mem[t]
            if mem_idx and not selected_mem_set[mem_idx] then
                selected_mem_set[mem_idx] = true
                selected_memories[#selected_memories + 1] = mem_idx
            end
            if target_topic_id and mem_idx and (topic.get_topic_id_for_turn and topic.get_topic_id_for_turn(t) == target_topic_id) then
                if not evidence_set[mem_idx] then
                    evidence_set[mem_idx] = true
                    evidence_memories[#evidence_memories + 1] = mem_idx
                end
            end
            if turn_src[t] == "cache" then
                cache_contrib = cache_contrib + 1
            end
        end
    end

    if cache_contrib > 0 then
        if not freeze_mode then
            adaptive.add_counter("topic_cache_selected_turns_total", cache_contrib)
        end
    end

    local persistent_hits = 0
    if next(persistent_probe_clusters_query) ~= nil then
        for _, mem_idx in ipairs(selected_memories) do
            local cid = cluster.get_cluster_id_for_line(mem_idx)
            if cid and persistent_probe_clusters_query[cid] then
                persistent_hits = persistent_hits + 1
            end
        end
    else
        for _, t in ipairs(selected_turns) do
            if turn_src[t] == "explore" then
                persistent_hits = persistent_hits + 1
            end
        end
    end
    if persistent_hits > 0 then
        if not freeze_mode then
            adaptive.add_counter("persistent_explore_turn_hits", persistent_hits)
        end
    end

    local hits_all = 0
    for _, t in ipairs(selected_turns) do
        if current_info and topic_relation_for_turn(t, current_info, sim_th) == "same" then
            hits_all = hits_all + 1
        end
    end

    local relevant_window = math.max(1, tonumber(cfg.relevant_window) or 40)
    local left = math.max(1, current_turn - relevant_window)
    local recent_total = 0
    local hits_recent = 0
    if current_info then
        for t = left, current_turn - 1 do
            if topic_relation_for_turn(t, current_info, sim_th) == "same" then
                recent_total = recent_total + 1
            end
        end
        for _, t in ipairs(selected_turns) do
            if t >= left and topic_relation_for_turn(t, current_info, sim_th) == "same" then
                hits_recent = hits_recent + 1
            end
        end
    end
    local recall_recent = (recent_total > 0) and (hits_recent / recent_total) or 0.0

    if not freeze_mode then
        apply_refinement(
            current_turn,
            hits_all,
            candidate_samples,
            selected_memories,
            evidence_memories,
            target_topic_id,
            current_info,
            sim_th,
            min_gate
        )
    end

    local need_rescue = (#selected_turns == 0)
    if not need_rescue and (cfg.cold_rescue_on_empty_only ~= true) and hits_all <= 0 then
        need_rescue = true
    end
    if need_rescue and not freeze_mode then
        heat.enqueue_cold_rescue(user_vec, current_turn, current_info, min_gate)
    end

    M.metrics.query_turns = (tonumber(M.metrics.query_turns) or 0) + 1
    if #selected_turns == 0 then
        M.metrics.empty_query_count = (tonumber(M.metrics.empty_query_count) or 0) + 1
    end
    M.metrics.hits_same_sum = (tonumber(M.metrics.hits_same_sum) or 0) + hits_all
    M.metrics.recall_recent_sum = (tonumber(M.metrics.recall_recent_sum) or 0.0) + recall_recent
    M.metrics.sim_ops[#M.metrics.sim_ops + 1] = sim_ops

    print(string.format(
        "[Recall] 选中 %d 条 turn（hits_same=%d, gate=%.3f, max_turns=%d, sim_ops=%d）",
        #selected_turns, hits_all, min_gate, max_turns, sim_ops
    ))

    local memory_text_lines = {}
    for _, item in ipairs(selected) do
        local entry = history.get_by_turn(item.turn)
        if entry then
            local user_part, ai_part = history.parse_entry(entry)
            if user_part then
                memory_text_lines[#memory_text_lines + 1] =
                    string.format("第%d轮 用户：%s\n助手：%s", item.turn, user_part, ai_part)
            end
        end
    end

    if #memory_text_lines > 0 then
        return "【相关记忆】\n" .. table.concat(memory_text_lines, "\n\n")
    end
    return ""
end

function M.check_and_retrieve(user_input, user_vec, opts)
    opts = opts or {}
    user_vec = user_vec or tool.get_embedding_query(user_input)

    local current_turn = tonumber(opts.current_turn) or (history.get_turn() + 1)
    local current_anchor = opts.current_anchor
    if current_anchor == nil then
        current_anchor = topic.get_topic_anchor and topic.get_topic_anchor(current_turn) or nil
    end
    local current_info = opts.current_info
    if current_info == nil then
        current_info = topic.get_topic_for_turn and topic.get_topic_for_turn(current_turn) or nil
    end
    local freeze_mode = opts.freeze_mode == true

    if (not freeze_mode) and M._last_topic_anchor and current_anchor ~= M._last_topic_anchor then
        unload_topic_cache()
    end

    local stable_ready = update_topic_stability(current_anchor, user_vec)
    if not freeze_mode then
        topic_random_lift(current_turn, current_anchor, current_info, stable_ready)
    end

    if need_recall(user_input, user_vec, current_turn) then
        return retrieve(user_input, user_vec, current_turn, current_info, current_anchor, opts)
    end

    return ""
end

return M
