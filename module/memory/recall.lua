-- recall.lua
-- 召回主链：learning/refinement/persistent-explore/topic-cache/cold-rescue

local M = {}

local tool = require("module.tool")
local memory = require("module.memory.store")
local history = require("module.memory.history")
local config = require("module.config")
local ghsom = require("module.memory.ghsom")
local topic = require("module.memory.topic")
local adaptive = require("module.memory.adaptive")
local predictor = require("module.memory.topic_predictor")

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

local function ai_cfg()
    return ((config.settings or {}).ai_query or {})
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

local function generate_search_keywords(user_input)
    if not py_pipeline then
        return {}
    end

    local prompt = string.format([[
用户提问：%s
请推测用户想要回忆起过去的哪些具体内容。输出3-5个最可能的原子事实关键词。
要求：
1. 关键词要具体（如"Python爬虫"、"讨厌香菜"），不要宽泛（如"代码"、"食物"）。
2. 严格输出Lua table格式：{"关键词1", "关键词2", ...}
3. 不要输出任何其他解释。

输出：
]], user_input)

    local messages = {
        { role = "system", content = "你是一个记忆检索助手，擅长生成精准的搜索关键词。" },
        { role = "user", content = prompt }
    }

    local params = {
        max_tokens = 256,
        temperature = 0.3,
        seed = 42
    }

    local result_str = py_pipeline:generate_chat_sync(messages, params)
    result_str = tool.remove_cot(result_str)

    local keywords = {}
    local max_keywords = tonumber(ai_cfg().max_keywords) or 4
    if max_keywords < 1 then max_keywords = 4 end

    result_str = tostring(result_str or ""):gsub("%s+", " ")
    result_str = result_str:match("^%s*(.-)%s*$")

    local parsed, err = tool.parse_lua_string_array_strict(result_str, {
        max_items = max_keywords,
        max_item_chars = 64,
        must_full = true,
        extract_first_on_fail = true,
    })
    if parsed then
        local seen = {}
        for _, kw in ipairs(parsed) do
            local k = tostring(kw or ""):match("^%s*(.-)%s*$")
            if k ~= "" and k ~= "无" and not seen[k] then
                seen[k] = true
                keywords[#keywords + 1] = k
            end
        end
    else
        print(string.format("[Recall] 关键词输出格式非法，已丢弃: %s", tostring(err)))
    end

    if #keywords > 0 then
        print("[Recall] LLM 生成的搜索关键词: " .. table.concat(keywords, ", "))
    end
    return keywords
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

local function merge_unique_results(dst, src, limit)
    local merged = {}
    local seen = {}
    for _, item in ipairs(dst or {}) do
        local idx = tonumber(item.index)
        if idx and not seen[idx] then
            seen[idx] = true
            merged[#merged + 1] = item
        end
    end
    for _, item in ipairs(src or {}) do
        local idx = tonumber(item.index)
        if idx and not seen[idx] then
            seen[idx] = true
            merged[#merged + 1] = item
        end
    end
    table.sort(merged, function(a, b)
        return (tonumber(a.similarity) or -1.0) > (tonumber(b.similarity) or -1.0)
    end)
    if limit and limit > 0 then
        while #merged > limit do
            table.remove(merged)
        end
    end
    return merged
end

local function build_query_vectors(user_input, user_vec, keyword_weight)
    local out = {
        { vec = user_vec, weight = 1.0, is_primary = true }
    }

    local keywords = generate_search_keywords(user_input)
    if #keywords <= 0 then
        return out
    end

    local ok_batch, kw_vecs = pcall(function()
        return tool.get_embeddings_query(keywords)
    end)

    if ok_batch and type(kw_vecs) == "table" and #kw_vecs > 0 then
        for i, kw_vec in ipairs(kw_vecs) do
            if type(kw_vec) == "table" and #kw_vec > 0 then
                out[#out + 1] = {
                    vec = kw_vec,
                    weight = keyword_weight,
                    is_primary = false,
                }
            else
                local kw = keywords[i]
                if kw and kw ~= "" then
                    local single_vec = tool.get_embedding_query(kw)
                    out[#out + 1] = {
                        vec = single_vec,
                        weight = keyword_weight,
                        is_primary = false,
                    }
                end
            end
        end
    else
        for _, kw in ipairs(keywords) do
            local kw_vec = tool.get_embedding_query(kw)
            out[#out + 1] = {
                vec = kw_vec,
                weight = keyword_weight,
                is_primary = false,
            }
        end
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

local function apply_refinement(turn, hits_all, candidate_samples, selected_memories, current_info, sim_th)
    if ai_cfg().refinement_enabled ~= true then return end
    if not current_info then return end

    local pos_set = {}
    local neg_set = {}
    for _, mem_idx in ipairs(selected_memories or {}) do
        local mem = memory.memories[mem_idx]
        if mem and mem.turns then
            local has_same = false
            local has_near = false
            for _, t in ipairs(mem.turns) do
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

local function retrieve(user_input, user_vec, current_turn, current_info, current_anchor, prediction)
    if memory.get_total_lines() <= 0 then
        return ""
    end

    local cfg = ai_cfg()
    local knobs = effective_retrieval_knobs(current_turn)
    local min_gate = tonumber(knobs.min_sim_gate) or (cfg.min_sim_gate or 0.58)
    local power = tonumber(knobs.power_suppress) or (cfg.power_suppress or 1.80)
    local max_memory = math.max(1, tonumber(knobs.max_memory) or 5)
    local max_turns = math.max(1, tonumber(knobs.max_turns) or 10)
    local probe_clusters = math.max(1, tonumber(knobs.probe_clusters) or 2)
    local cross_quota_ratio = clamp(tonumber(knobs.topic_cross_quota_ratio) or 0.25, 0.0, 0.5)
    local keyword_weight = math.max(0.0, tonumber(knobs.keyword_weight) or 0.55)
    local sim_th = tonumber(cfg.topic_sim_threshold) or 0.70

    local per_cluster_limit = math.max(2, tonumber(cfg.refinement_probe_per_cluster_limit) or 12)
    local base_scan_limit = math.max(per_cluster_limit, max_memory)

    local persistent_extra_budget = 0
    if cfg.persistent_explore_enabled == true and ghsom.node_count() > 1 then
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

    local turn_best = {}
    local turn_src = {}
    local turn_mem = {}
    local mem_best_sim = {}
    local mem_best_cluster = {}
    local mem_best_effective = {}
    local candidate_mem_set = {}
    local candidate_mem_src = {}
    local candidate_mem_cluster = {}
    local candidate_mem_list = {}
    local predicted_node_scores = type((prediction or {}).node_scores) == "table" and prediction.node_scores or {}
    local predicted_lines = type((prediction or {}).lines) == "table" and prediction.lines or {}

    local query_vectors = build_query_vectors(user_input, user_vec, keyword_weight)
    local primary = query_vectors[1]
    if (not primary) or type(primary.vec) ~= "table" or #primary.vec == 0 then
        return ""
    end

    local function mark_candidate(mem_idx, src_label, cid)
        mem_idx = tonumber(mem_idx)
        if not mem_idx then return end
        if not candidate_mem_set[mem_idx] then
            candidate_mem_set[mem_idx] = true
            candidate_mem_list[#candidate_mem_list + 1] = mem_idx
            candidate_mem_src[mem_idx] = src_label or "hot"
            candidate_mem_cluster[mem_idx] = tonumber(cid) or (ghsom.get_node_for_line(mem_idx) or -1)
        elseif src_label == "explore" then
            candidate_mem_src[mem_idx] = "explore"
        elseif src_label == "predict" and candidate_mem_src[mem_idx] ~= "explore" then
            candidate_mem_src[mem_idx] = "predict"
        end
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
            mem_best_cluster[mem_idx] = tonumber(cid) or (ghsom.get_node_for_line(mem_idx) or -1)
            mem_best_effective[mem_idx] = (sim_pos ^ power) * weight
        end
    end

    -- Stage A: topic 预测先分配 GHSOM 节点预算，再做 active-first 扫描。
    do
        local qv = primary.vec
        local weight = primary.weight or 1.0
        local total_scan_budget = math.max(base_scan_limit * math.max(1, probe_clusters), max_memory * 3)
        local per_node_floor = math.max(2, math.floor(base_scan_limit / math.max(1, probe_clusters)))
        local node_plan = ghsom.plan_probe_budget(qv, {
            max_nodes = probe_clusters + persistent_extra_budget,
            total_scan_budget = total_scan_budget,
            per_node_floor = per_node_floor,
            predicted_nodes = predicted_node_scores,
            prior_scale = tonumber(cfg.topic_activation_node_prior_scale) or 0.78,
            activation_bonus = tonumber(cfg.topic_activation_node_active_bonus) or 0.18,
        })
        local scanned_any = false
        for _, plan_item in ipairs(node_plan) do
            local cid = tonumber(plan_item.id)
            local scan_limit = math.max(1, tonumber(plan_item.scan_limit) or base_scan_limit)
            local src_label = ((tonumber(plan_item.prior) or 0.0) > 0.0) and "predict" or "hot"
            local sim_results = {}
            if plan_item.prefer_active == true then
                sim_results = ghsom.find_sim_in_node(qv, cid, {
                    only_hot = true,
                    max_results = scan_limit,
                })
                if #sim_results < scan_limit then
                    local fallback_results = ghsom.find_sim_in_node(qv, cid, {
                        only_hot = false,
                        max_results = scan_limit,
                    })
                    sim_results = merge_unique_results(sim_results, fallback_results, scan_limit)
                end
            else
                sim_results = ghsom.find_sim_in_node(qv, cid, {
                    only_hot = false,
                    max_results = scan_limit,
                })
            end
            if #sim_results > 0 then
                scanned_any = true
            end

            for _, mem in ipairs(sim_results) do
                local mem_idx = mem.index
                local sim = mem.similarity
                mark_candidate(mem_idx, src_label, cid)
                update_mem_best(mem_idx, sim, weight, cid)
                if sim >= min_gate then
                    local effective = (sim ^ power) * weight
                    push_turn_score(mem_idx, effective, src_label)
                end
            end
        end

        if not scanned_any then
            local fallback = memory.find_similar_all_fast(qv, max_memory * 2)
            for _, mem in ipairs(fallback) do
                local mem_idx = mem.index
                local sim = mem.similarity
                mark_candidate(mem_idx, "hot", ghsom.get_node_for_line(mem_idx) or -1)
                update_mem_best(mem_idx, sim, weight, ghsom.get_node_for_line(mem_idx) or -1)
                if sim >= min_gate then
                    local effective = (sim ^ power) * weight
                    push_turn_score(mem_idx, effective, "hot")
                end
            end
        end

        if #predicted_lines > 0 then
            local predict_gain = tonumber(cfg.topic_activation_predict_gain) or 1.06
            for _, mem_idx in ipairs(predicted_lines) do
                local mem_vec = memory.return_mem_vec(mem_idx)
                if mem_vec then
                    local sim = tool.cosine_similarity(qv, mem_vec)
                    local cid = ghsom.get_node_for_line(mem_idx) or -1
                    mark_candidate(mem_idx, "predict", cid)
                    update_mem_best(mem_idx, sim, weight, cid)
                    if sim >= min_gate then
                        local effective = (sim ^ power) * weight * predict_gain
                        push_turn_score(mem_idx, effective, "predict")
                    end
                end
            end
        end
    end

    -- Stage B: 关键词仅在候选池内重排；夹角过大自动降权或跳过，避免“错向量”拖偏
    do
        local kw_total = math.max(0, #query_vectors - 1)
        if kw_total > 0 and #candidate_mem_list > 0 then
            local align_reject = tonumber(cfg.keyword_align_reject) or 0.05
            local align_floor = tonumber(cfg.keyword_align_floor) or 0.20
            local align_gamma = tonumber(cfg.keyword_align_gamma) or 1.25
            local candidate_cap = math.max(96, max_turns * 24)
            local kw_used = 0
            local kw_skipped = 0

            if #candidate_mem_list > candidate_cap then
                table.sort(candidate_mem_list, function(a, b)
                    return (tonumber(mem_best_sim[a]) or -1) > (tonumber(mem_best_sim[b]) or -1)
                end)
                while #candidate_mem_list > candidate_cap do
                    local drop = table.remove(candidate_mem_list)
                    if drop then
                        candidate_mem_set[drop] = nil
                        candidate_mem_src[drop] = nil
                        candidate_mem_cluster[drop] = nil
                    end
                end
            end

            local candidate_vec = {}
            for _, mem_idx in ipairs(candidate_mem_list) do
                candidate_vec[mem_idx] = memory.return_mem_vec(mem_idx)
            end

            local function align_scale(aln)
                if aln <= align_floor then return 0.0 end
                local t = (aln - align_floor) / math.max(1e-6, (1.0 - align_floor))
                if t < 0 then t = 0 end
                if t > 1 then t = 1 end
                return t ^ align_gamma
            end

            for i = 2, #query_vectors do
                local q = query_vectors[i]
                local qv = q.vec
                if type(qv) == "table" and #qv > 0 then
                    local alignment = tool.cosine_similarity(primary.vec, qv)
                    if alignment < align_reject then
                        kw_skipped = kw_skipped + 1
                    else
                        local scale = align_scale(alignment)
                        if scale > 0 then
                            kw_used = kw_used + 1
                            local weight = (q.weight or keyword_weight) * scale
                            for _, mem_idx in ipairs(candidate_mem_list) do
                                local mem_vec = candidate_vec[mem_idx]
                                if mem_vec then
                                    local sim = tool.cosine_similarity(qv, mem_vec)
                                    update_mem_best(mem_idx, sim, weight, candidate_mem_cluster[mem_idx])
                                    if sim >= min_gate then
                                        local effective = (sim ^ power) * weight
                                        push_turn_score(mem_idx, effective, candidate_mem_src[mem_idx] or "hot")
                                    end
                                end
                            end
                        else
                            kw_skipped = kw_skipped + 1
                        end
                    end
                end
            end

            if kw_total > 0 then
                print(string.format(
                    "[Recall] keyword聚合重排: total=%d used=%d skipped=%d candidate_mem=%d",
                    kw_total, kw_used, kw_skipped, #candidate_mem_list
                ))
            end
        end
    end

    local sample_limit = math.max(1, tonumber(cfg.refinement_sample_mem_topk) or 48)
    local candidate_samples = collect_candidate_samples(mem_best_sim, mem_best_cluster, mem_best_effective, sample_limit)

    local selected = {}
    local ranked = to_sorted_pairs(turn_best)
    if #ranked <= 0 then
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
    local predict_contrib = 0
    for _, item in ipairs(selected) do
        selected_turns[#selected_turns + 1] = item.turn
        local mem_idx = turn_mem[item.turn]
        if mem_idx then selected_memories[#selected_memories + 1] = mem_idx end
        if turn_src[item.turn] == "predict" then
            predict_contrib = predict_contrib + 1
        end
    end
    if predict_contrib > 0 then
        adaptive.add_counter("topic_cache_selected_turns_total", predict_contrib)
    end

    local hits_all = 0
    if current_info then
        for _, t in ipairs(selected_turns) do
            if topic_relation_for_turn(t, current_info, sim_th) == "same" then
                hits_all = hits_all + 1
            end
        end
    end

    apply_refinement(current_turn, hits_all, candidate_samples, selected_memories, current_info, sim_th)
    if #selected_memories > 0 then
        ghsom.activate_lines(selected_memories, { mode = "append" })
    end
    predictor.observe(current_anchor, selected_memories)

    print(string.format(
        "[Recall] 选中 %d 条 turn（hits_same=%d, gate=%.3f, max_turns=%d）",
        #selected_turns, hits_all, min_gate, max_turns
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
    local current_anchor = topic.get_stable_anchor and topic.get_stable_anchor(current_turn) or nil
    local current_info = topic.get_topic_for_turn and topic.get_topic_for_turn(current_turn) or nil
    local prediction = predictor.predict(current_anchor, {
        query_vec = user_vec,
    })
    ghsom.activate_lines((prediction or {}).lines or {}, { mode = "replace" })

    if need_recall(user_input, user_vec, current_turn) then
        return retrieve(user_input, user_vec, current_turn, current_info, current_anchor, prediction)
    end

    return ""
end

return M
