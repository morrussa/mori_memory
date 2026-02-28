-- recall.lua (V3)
-- simub-style retrieval chain: top-probe-clusters + soft-gate + expected-recall + smart-preload

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
M._consecutive_empty_count = 0
M._soft_gate_pass_count = 0

M._cluster_visit_counts = {}
M._cluster_hit_counts = {}
M._cluster_hit_rate_ema = {}

local function ai_cfg()
    return ((config.settings or {}).ai_query or {})
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function shuffle_inplace(arr)
    for i = #arr, 2, -1 do
        local j = math.random(i)
        arr[i], arr[j] = arr[j], arr[i]
    end
end

local function topic_key_for_turn(turn)
    local t = tonumber(turn)
    if not t then return nil end

    local ti = topic.get_topic_for_turn and topic.get_topic_for_turn(t) or nil
    if not ti then
        return topic.get_topic_anchor and topic.get_topic_anchor(t) or nil
    end

    if ti.is_active then
        local start_turn = topic.active_topic and topic.active_topic.start
        if start_turn then
            return "S:" .. tostring(start_turn)
        end
        return topic.get_topic_anchor and topic.get_topic_anchor(t) or nil
    end

    if ti.topic_idx and topic.topics and topic.topics[ti.topic_idx] then
        local rec = topic.topics[ti.topic_idx]
        if rec and rec.start then
            return "S:" .. tostring(rec.start)
        end
    end

    return topic.get_topic_anchor and topic.get_topic_anchor(t) or nil
end

local function topic_key_for_current(current_turn, current_info)
    if current_info and current_info.is_active then
        local start_turn = topic.active_topic and topic.active_topic.start
        if start_turn then
            return "S:" .. tostring(start_turn)
        end
    elseif current_info and current_info.topic_idx and topic.topics and topic.topics[current_info.topic_idx] then
        local rec = topic.topics[current_info.topic_idx]
        if rec and rec.start then
            return "S:" .. tostring(rec.start)
        end
    end
    return topic_key_for_turn(current_turn)
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

    local gate_floor = tonumber(cfg.min_gate_floor) or 0.25
    local gate_ceiling = tonumber(cfg.max_gate_ceiling) or 0.85
    min_gate = clamp(min_gate, gate_floor, gate_ceiling)

    return {
        min_sim_gate = min_gate,
        power_suppress = math.max(1.0, power),
        topic_cross_quota_ratio = clamp(cross, 0.0, 0.5),
        max_memory = math.max(1, max_memory),
        max_turns = math.max(1, max_turns),
        keyword_weight = kw_weight,
        supercluster_topn_query = math.max(1, super_q),
        probe_clusters = math.max(1, effective_probe_clusters(turn)),
    }
end

local function estimate_cluster_expected_recall(cid)
    local cfg = ai_cfg()
    local visits = tonumber(M._cluster_visit_counts[cid]) or 0.0
    local hits = tonumber(M._cluster_hit_counts[cid]) or 0.0
    if visits <= 0 then
        return 0.5
    end

    local ema = tonumber(M._cluster_hit_rate_ema[cid]) or 0.5
    local direct = hits / math.max(1e-6, visits)
    local alpha = clamp(tonumber(cfg.cluster_hit_rate_alpha) or 0.10, 0.01, 0.99)
    return alpha * direct + (1.0 - alpha) * ema
end

local function update_cluster_hit_rate(cid, hit)
    if not cid then return end
    local cfg = ai_cfg()
    local alpha = clamp(tonumber(cfg.cluster_hit_rate_alpha) or 0.10, 0.01, 0.99)

    M._cluster_visit_counts[cid] = (tonumber(M._cluster_visit_counts[cid]) or 0) + 1
    if hit then
        M._cluster_hit_counts[cid] = (tonumber(M._cluster_hit_counts[cid]) or 0) + 1
    end

    local visits = tonumber(M._cluster_visit_counts[cid]) or 0
    local hits = tonumber(M._cluster_hit_counts[cid]) or 0
    local current_rate = (visits > 0) and (hits / visits) or 0.0
    local old_ema = tonumber(M._cluster_hit_rate_ema[cid]) or 0.5
    M._cluster_hit_rate_ema[cid] = alpha * current_rate + (1.0 - alpha) * old_ema
end

local function soft_gate_filter(sim, min_gate)
    local cfg = ai_cfg()
    if sim >= min_gate then
        return true
    end
    if cfg.soft_gate_enabled ~= true then
        return false
    end

    local margin = clamp(tonumber(cfg.soft_gate_margin) or 0.10, 0.01, 0.95)
    local threshold = min_gate * (1.0 - margin)
    if sim < threshold then
        return false
    end

    local den = math.max(1e-6, min_gate - threshold)
    local prob = clamp((sim - threshold) / den, 0.0, 1.0)
    if math.random() < prob then
        M._soft_gate_pass_count = M._soft_gate_pass_count + 1
        return true
    end
    return false
end

local function adjust_gate_on_result(hit)
    local cfg = ai_cfg()
    local gate_floor = tonumber(cfg.min_gate_floor) or 0.25
    local gate_ceiling = tonumber(cfg.max_gate_ceiling) or 0.85
    if not adaptive.state then
        return
    end

    local cur = adaptive.get_min_sim_gate(tonumber(cfg.min_sim_gate) or 0.58)
    if hit then
        local boost = tonumber(cfg.hit_gate_boost) or 1.002
        local nxt = clamp(cur * boost, gate_floor, gate_ceiling)
        adaptive.state.learned_min_gate = nxt
        M._consecutive_empty_count = 0
    else
        M._consecutive_empty_count = M._consecutive_empty_count + 1
        local decay = tonumber(cfg.empty_gate_decay) or 0.98
        local decay_aggr = tonumber(cfg.empty_gate_decay_aggressive) or 0.95
        local used = decay
        if M._consecutive_empty_count >= 3 then
            used = decay_aggr
        end
        local nxt = clamp(cur * used, gate_floor, gate_ceiling)
        adaptive.state.learned_min_gate = nxt
    end
    adaptive.mark_dirty()
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

    local use_expected = cfg.expected_recall_enabled == true
    local bonus_scale = tonumber(cfg.route_score_bonus_scale) or 0.15

    local scored = {}
    local ops = ops0
    for _, cid in ipairs(cand) do
        local clu = cluster.clusters[cid]
        if clu and clu.centroid then
            local sim = tool.cosine_similarity(vec, clu.centroid)
            ops = ops + 1
            if use_expected then
                sim = sim + estimate_cluster_expected_recall(cid) * bonus_scale
            end
            if route_scale > 0 then
                sim = sim + route_scale * adaptive.get_route_score(cid)
            end
            scored[#scored + 1] = { cid = cid, adjusted = sim }
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

local function collect_topic_centroids(current_turn, current_key, current_info)
    local out = {}
    local seen = {}

    local function add_item(key, centroid)
        if not key or key == "" or not centroid or #centroid == 0 then return end
        if seen[key] then return end
        seen[key] = true
        out[#out + 1] = { key = key, centroid = centroid }
    end

    if current_key and current_info and current_info.centroid then
        add_item(current_key, current_info.centroid)
    end

    for _, rec in ipairs(topic.topics or {}) do
        local key = rec.start and ("S:" .. tostring(rec.start)) or nil
        add_item(key, rec.centroid)
    end

    return out
end

local function predict_topic_key(query_vec, current_turn, current_key, current_info)
    local candidates = collect_topic_centroids(current_turn, current_key, current_info)
    if #candidates <= 0 then
        return nil, 0.0
    end

    local qptr = tool.to_ptr_vec(query_vec) or query_vec
    local scored = {}
    for _, item in ipairs(candidates) do
        local sim = tool.cosine_similarity(qptr, item.centroid)
        scored[#scored + 1] = { key = item.key, sim = sim }
    end

    table.sort(scored, function(a, b) return a.sim > b.sim end)
    local best = scored[1]
    if not best then return nil, 0.0 end

    local margin = 0.0
    if #scored >= 2 then
        margin = (tonumber(best.sim) or 0.0) - (tonumber(scored[2].sim) or 0.0)
    end

    local conf = clamp((tonumber(best.sim) or 0.0) * (1.0 + margin), -1.0, 1.0)
    return best.key, conf
end

local function topic_hot_ratio(topic_key)
    local lines = memory.iter_topic_lines(topic_key, false)
    if #lines <= 0 then
        return 0.0
    end

    local hot = 0
    for _, line in ipairs(lines) do
        if memory.get_heat_by_index(line) > 0 then
            hot = hot + 1
        end
    end
    return hot / #lines
end

local function pick_top_lines_by_sim(query_vec, lines, budget)
    local qptr = tool.to_ptr_vec(query_vec) or query_vec
    local scored = {}
    for _, line in ipairs(lines) do
        local vec = memory.return_mem_vec(line)
        if vec then
            local sim = tool.cosine_similarity(qptr, vec)
            scored[#scored + 1] = { line = line, sim = sim }
        end
    end
    table.sort(scored, function(a, b) return a.sim > b.sim end)

    local out = {}
    for i = 1, math.min(budget, #scored) do
        out[#out + 1] = scored[i].line
    end
    return out
end

local function smart_preload_cold_memories(query_vec, current_turn, current_key, current_info, candidate_clusters)
    local cfg = ai_cfg()
    if cfg.smart_preload_enabled ~= true then
        return 0
    end

    local preload_budget = math.max(0, tonumber(cfg.preload_budget_per_query) or 5)
    local preload_max_io = math.max(0, tonumber(cfg.preload_max_io_per_turn) or 8)
    if preload_budget <= 0 or preload_max_io <= 0 then
        return 0
    end

    local io_granted = memory.reserve_preload_io(preload_budget, preload_max_io)
    if io_granted <= 0 then
        return 0
    end

    local hot_ratio_th = clamp(tonumber(cfg.preload_low_hot_ratio_threshold) or 0.15, 0.0, 1.0)
    local conf_th = tonumber(cfg.preload_topic_confidence) or 0.50

    local topics_to_preload = {}
    local topic_seen = {}

    local function try_add_topic(key)
        if not key or topic_seen[key] then return end
        local ratio = topic_hot_ratio(key)
        if ratio < hot_ratio_th then
            topic_seen[key] = true
            topics_to_preload[#topics_to_preload + 1] = key
        end
    end

    try_add_topic(current_key)

    if cfg.preload_use_vector_prediction ~= false then
        local pred_key, conf = predict_topic_key(query_vec, current_turn, current_key, current_info)
        if pred_key and conf >= conf_th then
            try_add_topic(pred_key)
        end
    end

    if #topics_to_preload <= 0 then
        return 0
    end

    local cluster_allow = {}
    for _, cid in ipairs(candidate_clusters or {}) do
        cluster_allow[tonumber(cid)] = true
    end

    local budget_left = io_granted
    local preloaded = 0
    local preload_heat = math.max(1, tonumber(cfg.preload_heat_amount) or 25000)

    for _, topic_key in ipairs(topics_to_preload) do
        if budget_left <= 0 then break end

        local candidates = memory.iter_topic_lines(topic_key, true)
        if #candidates > 0 then
            local local_lines = {}
            for _, line in ipairs(candidates) do
                local cid = cluster.get_cluster_id_for_line(line)
                if cid and cluster_allow[cid] then
                    local_lines[#local_lines + 1] = line
                end
            end

            if #local_lines > 0 then
                local picked = pick_top_lines_by_sim(query_vec, local_lines, budget_left)
                for _, line in ipairs(picked) do
                    if budget_left <= 0 then break end
                    local old_heat = memory.get_heat_by_index(line)
                    if old_heat <= 0 then
                        memory.set_heat(line, preload_heat)
                        heat.sync_line(line)
                        cluster.mark_hot(line)
                        preloaded = preloaded + 1
                        budget_left = budget_left - 1
                    end
                end
            end
        end
    end

    if preloaded > 0 then
        heat.normalize_heat()
    end

    return preloaded
end

local function unload_topic_cache()
    if next(M._topic_cache_mem) ~= nil then
        adaptive.add_counter("topic_cache_unload_count", 1)
    end
    M._topic_cache_mem = {}
    M._topic_cache_anchor = nil
end

local function update_topic_stability(current_anchor, query_vec)
    local stable_warm = math.max(1, tonumber(ai_cfg().stable_warmup_turns) or 6)
    local stable_sim = tonumber(ai_cfg().stable_min_pair_sim) or 0.72

    if current_anchor and current_anchor == M._last_topic_anchor then
        M._same_topic_streak = M._same_topic_streak + 1
        if M._prev_query_vec and query_vec and #query_vec > 0 then
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
    M._prev_query_vec = query_vec

    local avg_pair = 1.0
    if M._streak_sim_count > 0 then
        avg_pair = M._streak_sim_sum / M._streak_sim_count
    end

    return (M._same_topic_streak >= stable_warm) and (avg_pair >= stable_sim)
end

local function topic_random_lift(turn, current_anchor, stable_ready)
    local cfg = ai_cfg()
    if not current_anchor then return end
    if not stable_ready then return end

    local interval = math.max(1, tonumber(cfg.topic_random_lift_interval) or 3)
    if interval > 1 and (turn % interval) ~= 0 then return end

    local prob = clamp(tonumber(cfg.topic_random_lift_prob) or 0.85, 0.0, 1.0)
    if math.random() > prob then return end

    local only_cold = cfg.topic_random_lift_only_cold ~= false
    local candidates = memory.iter_topic_lines(current_anchor, only_cold)
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

local function apply_refinement(turn, hits_all, candidate_samples, selected_memories, current_info, sim_th)
    if ai_cfg().refinement_enabled ~= true then return end
    if not current_info then return end

    local pos_set = {}
    local neg_set = {}
    for _, mem_idx in ipairs(selected_memories or {}) do
        local turns = memory.get_turns(mem_idx)
        local has_same = false
        local has_near = false
        for _, t in ipairs(turns) do
            local rel = topic_relation_for_turn(t, current_info, sim_th)
            if rel == "same" then
                has_same = true
                break
            elseif rel == "near" then
                has_near = true
            end
        end
        if has_same then
            pos_set[mem_idx] = true
        elseif has_near then
            neg_set[mem_idx] = true
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

local function retrieve(user_input, user_vec, current_turn, current_info, current_anchor)
    if memory.get_total_lines() <= 0 then
        return ""
    end

    memory.begin_turn(current_turn)

    local cfg = ai_cfg()
    local knobs = effective_retrieval_knobs(current_turn)
    local min_gate = tonumber(knobs.min_sim_gate) or (cfg.min_sim_gate or 0.58)
    local power = tonumber(knobs.power_suppress) or (cfg.power_suppress or 1.80)
    local max_memory = math.max(1, tonumber(knobs.max_memory) or 5)
    local max_turns = math.max(1, tonumber(knobs.max_turns) or 10)
    local query_topn = math.max(1, tonumber(knobs.supercluster_topn_query) or 4)
    local probe_clusters = math.max(1, tonumber(knobs.probe_clusters) or 2)
    local cross_quota_ratio = clamp(tonumber(knobs.topic_cross_quota_ratio) or 0.25, 0.0, 0.5)
    local sim_th = tonumber(cfg.topic_sim_threshold) or 0.70

    local per_cluster_limit = math.max(2, tonumber(cfg.refinement_probe_per_cluster_limit) or 12)
    local base_scan_limit = math.max(per_cluster_limit, max_memory)
    local persistent_cap = math.max(1, tonumber(cfg.persistent_explore_candidate_cap) or 32)

    local current_topic_key = topic_key_for_current(current_turn, current_info)

    local turn_best = {}
    local turn_src = {}
    local turn_mem = {}
    local mem_best_sim = {}
    local mem_best_cluster = {}
    local mem_best_effective = {}
    local persistent_probe_clusters_query = {}

    local query_vectors = {
        { vec = user_vec, weight = 1.0, is_primary = true }
    }
    local primary = query_vectors[1]
    if (not primary) or type(primary.vec) ~= "table" or #primary.vec == 0 then
        return ""
    end
    local primary_qptr = tool.to_ptr_vec(primary.vec) or primary.vec

    local persistent_extra_budget = 0
    if cfg.persistent_explore_enabled == true and cluster.cluster_count() > 1 then
        local eps = clamp(tonumber(cfg.persistent_explore_epsilon) or 0.01, 0.0, 1.0)
        local periodic = math.max(0, tonumber(cfg.persistent_explore_period_turns) or 0)
        local trigger = false
        if eps > 0 and math.random() < eps then trigger = true end
        if periodic > 0 and (current_turn % periodic) == 0 then trigger = true end
        if trigger then
            persistent_extra_budget = math.max(1, tonumber(cfg.persistent_explore_extra_clusters) or 1)
            adaptive.add_counter("persistent_explore_events", 1)
        end
    end

    local cluster_ids_for_preload, _ = top_probe_clusters(primary_qptr, probe_clusters, query_topn, current_turn)
    local preloaded = smart_preload_cold_memories(primary_qptr, current_turn, current_topic_key, current_info, cluster_ids_for_preload)

    local function push_turn_score(mem_idx, effective, src_label)
        local turns = memory.get_turns(mem_idx)
        for _, t in ipairs(turns) do
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

    for q_idx, q in ipairs(query_vectors) do
        local qv = q.vec
        local qptr = tool.to_ptr_vec(qv) or qv
        local weight = q.weight or 1.0
        local cluster_ids, _ = top_probe_clusters(qptr, probe_clusters, query_topn, current_turn)
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
                adaptive.add_counter("persistent_explore_cluster_probes", take)
            end
        end

        for _, cid in ipairs(cluster_ids) do
            local scan_limit = base_scan_limit
            local src_label = "hot"
            if persistent_probe_clusters_vec[cid] then
                scan_limit = math.min(scan_limit, persistent_cap)
                src_label = "explore"
            end

            local sim_results = cluster.find_sim_in_cluster(qptr, cid, {
                only_hot = true,
                max_results = scan_limit,
            })

            for _, mem in ipairs(sim_results) do
                local mem_idx = mem.index
                local sim = mem.similarity
                update_mem_best(mem_idx, sim, weight, cid)

                if not soft_gate_filter(sim, min_gate) then
                    break
                end

                local effective = (sim ^ power) * weight
                push_turn_score(mem_idx, effective, src_label)
            end
        end

        if current_topic_key and M._topic_cache_anchor == current_topic_key and next(M._topic_cache_mem) ~= nil then
            local cache_scored = {}
            for mem_idx, _ in pairs(M._topic_cache_mem) do
                local mem_vec = memory.return_mem_vec(mem_idx)
                if mem_vec then
                    local sim = tool.cosine_similarity(qptr, mem_vec)
                    cache_scored[#cache_scored + 1] = { mem_idx = mem_idx, sim = sim }
                end
            end
            table.sort(cache_scored, function(a, b) return a.sim > b.sim end)
            for _, item in ipairs(cache_scored) do
                local sim = item.sim
                local mem_idx = item.mem_idx
                update_mem_best(mem_idx, sim, weight, cluster.get_cluster_id_for_line(mem_idx) or -1)
                if sim >= min_gate then
                    local effective = (sim ^ power) * weight * (tonumber(cfg.topic_cache_weight) or 1.02)
                    push_turn_score(mem_idx, effective, "cache")
                end
            end
        end
    end

    local sample_limit = math.max(1, tonumber(cfg.refinement_sample_mem_topk) or 48)
    local candidate_samples = collect_candidate_samples(mem_best_sim, mem_best_cluster, mem_best_effective, sample_limit)

    local selected = {}
    local ranked = to_sorted_pairs(turn_best)
    if #ranked <= 0 then
        adjust_gate_on_result(false)
        heat.enqueue_cold_rescue(user_vec, current_turn, current_info, min_gate)
        return ""
    end

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
                    local choose_n = math.min(explore_slots, #pool)
                    selected = {}
                    for _, it in ipairs(core) do selected[#selected + 1] = it end
                    for i = 1, choose_n do selected[#selected + 1] = pool[i] end
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
    local cache_contrib = 0
    for _, item in ipairs(selected) do
        selected_turns[#selected_turns + 1] = item.turn
        local mem_idx = turn_mem[item.turn]
        if mem_idx then selected_memories[#selected_memories + 1] = mem_idx end
        if turn_src[item.turn] == "cache" then
            cache_contrib = cache_contrib + 1
        end
    end

    if cache_contrib > 0 then
        adaptive.add_counter("topic_cache_selected_turns_total", cache_contrib)
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
        adaptive.add_counter("persistent_explore_turn_hits", persistent_hits)
    end

    local hits_all = 0
    if current_info then
        for _, t in ipairs(selected_turns) do
            if topic_relation_for_turn(t, current_info, sim_th) == "same" then
                hits_all = hits_all + 1
            end
        end
    end

    local hit_flag = hits_all > 0
    local clusters_seen = {}
    for _, sample in ipairs(candidate_samples) do
        local cid = tonumber(sample.cid)
        if cid and cid >= 0 and not clusters_seen[cid] then
            update_cluster_hit_rate(cid, hit_flag)
            clusters_seen[cid] = true
        end
    end

    adjust_gate_on_result(hit_flag)

    apply_refinement(current_turn, hits_all, candidate_samples, selected_memories, current_info, sim_th)

    local need_rescue = (#selected_turns == 0)
    if not need_rescue and (cfg.cold_rescue_on_empty_only ~= true) and hits_all <= 0 then
        need_rescue = true
    end
    if need_rescue then
        heat.enqueue_cold_rescue(user_vec, current_turn, current_info, min_gate)
    end

    print(string.format(
        "[Recall] 选中 %d 条 turn（hits_same=%d, gate=%.3f, max_turns=%d, preloaded=%d）",
        #selected_turns, hits_all, min_gate, max_turns, preloaded
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

function M.check_and_retrieve(user_input, user_vec)
    user_vec = user_vec or tool.get_embedding_query(user_input)

    local current_turn = history.get_turn() + 1
    local current_info = topic.get_topic_for_turn and topic.get_topic_for_turn(current_turn) or nil
    local current_key = topic_key_for_current(current_turn, current_info)

    if M._last_topic_anchor and current_key ~= M._last_topic_anchor then
        unload_topic_cache()
    end

    local stable_ready = update_topic_stability(current_key, user_vec)
    topic_random_lift(current_turn, current_key, stable_ready)

    if need_recall(user_input, user_vec, current_turn) then
        return retrieve(user_input, user_vec, current_turn, current_info, current_key)
    end

    return ""
end

return M
