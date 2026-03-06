-- recall.lua (V3)
-- simub-style retrieval chain: top-probe-clusters + soft-gate + expected-recall + smart-preload

local M = {}

local tool = require("module.tool")
local memory = require("module.memory.store")
local history = require("module.memory.history")
local config = require("module.config")
local cluster = require("module.memory.cluster")
local topic = require("module.memory.topic")
local adaptive = require("module.memory.adaptive")
local heat = require("module.memory.heat")

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
M._preload_cache_anchor = nil
M._preload_cache_clusters = {}
M._preload_cache_mem = {}
M._consecutive_empty_count = 0
M._soft_gate_pass_count = 0
M._last_recall_attempt_turn = nil

M._cluster_visit_counts = {}
M._cluster_hit_counts = {}
M._cluster_hit_rate_ema = {}

local RECALL_HISTORY_KEYWORDS = {
    "之前", "上次", "以前", "过去", "曾经", "recall", "remember", "earlier", "previously"
}

local RECALL_EXPLICIT_KEYWORDS = {
    "你还记得", "还记得", "记不记得", "我们聊过", "我说过", "我提过", "提到过",
    "回顾", "复盘", "历史记录", "旧记录", "之前提到", "上回", "上一次"
}

local RECALL_CONTEXT_KEYWORDS = {
    "继续", "接着", "延续", "刚才", "刚刚", "上面", "前面", "前文", "那个方案", "这件事"
}

local RECALL_NEW_TASK_KEYWORDS = {
    "帮我写", "写一个", "生成", "介绍", "解释", "翻译", "推荐", "教程", "是什么", "怎么"
}

local RECALL_SUPPRESS_KEYWORDS = {
    "不用回忆", "不需要回忆", "别回忆", "不要回忆", "无需回忆",
    "不查记忆", "别查记忆", "don't recall", "do not recall", "no memory search"
}

local TECH_KEYWORDS = {
    "代码", "函数", "api", "算法", "编程", "python", "lua", "配置", "参数"
}

local function ai_cfg()
    return ((config.settings or {}).ai_query or {})
end

local function graph_recall_cfg()
    return ((((config.settings or {}).graph or {}).recall) or {})
end

local function cluster_type_bonus()
    return tonumber((((config.settings or {}).memory_types or {}).cluster_type_bonus)) or 0.0
end

local function is_read_only(opts)
    return type(opts) == "table" and opts.read_only == true
end

local function derive_type_filters(opts)
    opts = opts or {}
    local allowed_types = memory.build_type_filter(opts.allowed_types)
    local blocked_types = memory.build_type_filter(opts.blocked_types)
    local preferred_type = memory.match_type_name(opts.preferred_type)

    if (not preferred_type) and allowed_types then
        local only_name = nil
        local count = 0
        for type_name in pairs(allowed_types) do
            only_name = type_name
            count = count + 1
            if count > 1 then
                only_name = nil
                break
            end
        end
        preferred_type = only_name
    end

    if preferred_type and blocked_types and blocked_types[preferred_type] then
        preferred_type = nil
    end

    return allowed_types, blocked_types, preferred_type
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function utf8_len(text)
    local s = tostring(text or "")
    local _, n = s:gsub("[^\128-\193]", "")
    return n
end

local function contains_any_keyword(text, text_lower, keywords)
    for _, kw in ipairs(keywords) do
        local token = tostring(kw or "")
        if token ~= "" then
            if text:find(token, 1, true) then
                return true, token
            end
            local token_lower = token:lower()
            if token_lower ~= token and text_lower:find(token_lower, 1, true) then
                return true, token
            end
        end
    end
    return false, nil
end

local function append_component(breakdown, name, delta)
    local v = tonumber(delta) or 0.0
    if v == 0.0 then
        return 0.0
    end
    breakdown[name] = (tonumber(breakdown[name]) or 0.0) + v
    return v
end

local function format_breakdown(breakdown)
    local keys = {}
    for k, _ in pairs(breakdown or {}) do
        keys[#keys + 1] = k
    end
    table.sort(keys)
    local parts = {}
    for _, k in ipairs(keys) do
        local v = tonumber(breakdown[k]) or 0.0
        parts[#parts + 1] = string.format("%s=%+.2f", k, v)
    end
    return table.concat(parts, ", ")
end

local function resolve_semantic_similarity(override_val, user_vec, target_vec, disable_embeddings)
    local ov = tonumber(override_val)
    if ov ~= nil then
        return clamp(ov, -1.0, 1.0)
    end
    if disable_embeddings then
        return 0.0
    end
    if user_vec and #user_vec > 0 and target_vec and #target_vec > 0 then
        local sim = tool.cosine_similarity(user_vec, target_vec)
        return clamp(sim, -1.0, 1.0)
    end
    return 0.0
end

local function shuffle_inplace(arr)
    for i = #arr, 2, -1 do
        local j = math.random(i)
        arr[i], arr[j] = arr[j], arr[i]
    end
end

local function normalize_vec(vec)
    if type(vec) ~= "table" or #vec == 0 then
        return {}
    end
    local out = {}
    local norm2 = 0.0
    for i = 1, #vec do
        local v = tonumber(vec[i]) or 0.0
        out[i] = v
        norm2 = norm2 + v * v
    end
    local norm = math.sqrt(norm2)
    if norm > 0.0 then
        for i = 1, #out do
            out[i] = out[i] / norm
        end
    end
    return out
end

local function query_vectors(base_vec, keyword_weight, opts)
    local cfg = ai_cfg()
    local out = {
        { vec = base_vec, weight = 1.0, is_primary = true }
    }
    if is_read_only(opts) then
        return out
    end
    local query_n = math.max(1, math.floor(tonumber(cfg.keyword_queries) or 2))
    if query_n <= 1 then
        return out
    end

    local base = normalize_vec(base_vec)
    local dim = #base
    if dim <= 0 then
        return out
    end
    local mix = clamp(tonumber(cfg.keyword_noise_mix) or 0.20, 0.0, 1.0)
    for _ = 2, query_n do
        local noise = {}
        for i = 1, dim do
            noise[i] = math.random() * 2.0 - 1.0
        end
        noise = normalize_vec(noise)
        local qv = {}
        for i = 1, dim do
            qv[i] = (1.0 - mix) * base[i] + mix * (noise[i] or 0.0)
        end
        qv = normalize_vec(qv)
        out[#out + 1] = { vec = qv, weight = keyword_weight or 1.0, is_primary = false }
    end
    return out
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

local function extract_recall_trigger_features(user_input, user_vec, opts)
    opts = opts or {}
    local text = tostring(user_input or "")
    local text_lower = text:lower()
    local override = opts.sim_override or {}
    local disable_embeddings = opts.disable_embeddings == true

    local has_history_keyword, _ = contains_any_keyword(text, text_lower, RECALL_HISTORY_KEYWORDS)
    local has_explicit_keyword, _ = contains_any_keyword(text, text_lower, RECALL_EXPLICIT_KEYWORDS)
    local has_context_keyword, _ = contains_any_keyword(text, text_lower, RECALL_CONTEXT_KEYWORDS)
    local has_new_task_keyword, _ = contains_any_keyword(text, text_lower, RECALL_NEW_TASK_KEYWORDS)
    local has_suppress_keyword, _ = contains_any_keyword(text, text_lower, RECALL_SUPPRESS_KEYWORDS)
    local has_tech_keyword, _ = contains_any_keyword(text, text_lower, TECH_KEYWORDS)

    local sim_anxiety = resolve_semantic_similarity(
        override.anxiety_sim or override.anxiety,
        user_vec,
        anxiety_vec,
        disable_embeddings
    )
    local sim_help_cry = resolve_semantic_similarity(
        override.help_cry_sim or override.help_cry or override.help,
        user_vec,
        help_cry_vec,
        disable_embeddings
    )
    local sim_past_talk = resolve_semantic_similarity(
        override.past_talk_sim or override.past_talk or override.past,
        user_vec,
        past_talk_vec,
        disable_embeddings
    )

    return {
        text = text,
        char_len = utf8_len(text),
        has_history_keyword = has_history_keyword,
        has_explicit_keyword = has_explicit_keyword,
        has_context_keyword = has_context_keyword,
        has_new_task_keyword = has_new_task_keyword,
        has_suppress_keyword = has_suppress_keyword,
        has_tech_keyword = has_tech_keyword,
        sim_anxiety = sim_anxiety,
        sim_help_cry = sim_help_cry,
        sim_past_talk = sim_past_talk,
    }
end

local function compute_recall_score(user_input, user_vec, opts)
    opts = opts or {}
    local cfg = ai_cfg()
    local features = extract_recall_trigger_features(user_input, user_vec, opts)
    local breakdown = {}
    local score = 0.0

    local explicit_signal = (features.has_history_keyword or features.has_explicit_keyword)
    local contextual_signal = (explicit_signal or features.has_context_keyword)

    if features.has_history_keyword then
        score = score + append_component(breakdown, "history_keyword_bonus", tonumber(cfg.history_search_bonus) or 0.0)
    end

    if features.has_explicit_keyword then
        score = score + append_component(
            breakdown,
            "explicit_recall_bonus",
            tonumber(cfg.explicit_recall_bonus) or 2.2
        )
    end

    if features.has_context_keyword then
        score = score + append_component(
            breakdown,
            "context_link_bonus",
            tonumber(cfg.context_link_bonus) or 1.1
        )
    end

    if features.has_tech_keyword then
        local base = tonumber(cfg.technical_term_bonus) or 0.0
        local no_intent_scale = clamp(tonumber(cfg.technical_bonus_scale_when_no_recall_intent) or 0.30, 0.0, 1.0)
        local scale = contextual_signal and 1.0 or no_intent_scale
        score = score + append_component(breakdown, "technical_term_bonus", base * scale)
    end

    local length_limit = math.max(1, math.floor(tonumber(cfg.length_limit) or 20))
    if features.char_len >= length_limit then
        local base = tonumber(cfg.length_bonus) or 0.0
        local no_intent_scale = clamp(tonumber(cfg.length_bonus_scale_when_no_recall_intent) or 0.35, 0.0, 1.0)
        local scale = contextual_signal and 1.0 or no_intent_scale
        score = score + append_component(breakdown, "length_bonus", base * scale)
    end

    local short_penalty_len = math.max(1, math.floor(tonumber(cfg.short_query_penalty_len) or 8))
    if features.char_len <= short_penalty_len and not contextual_signal then
        local penalty = math.abs(tonumber(cfg.short_query_penalty) or 0.65)
        score = score + append_component(breakdown, "short_query_penalty", -penalty)
    end

    if features.has_new_task_keyword and not contextual_signal then
        local penalty = math.abs(tonumber(cfg.fresh_task_penalty) or 1.6)
        score = score + append_component(breakdown, "fresh_task_penalty", -penalty)
    end

    if features.has_suppress_keyword then
        local penalty = math.abs(tonumber(cfg.suppress_recall_penalty) or 8.0)
        score = score + append_component(breakdown, "suppress_recall_penalty", -penalty)
    end

    score = score + append_component(
        breakdown,
        "anxiety_similarity",
        features.sim_anxiety * (tonumber(cfg.anxiety_multi) or 0.0)
    )
    score = score + append_component(
        breakdown,
        "help_cry_similarity",
        features.sim_help_cry * (tonumber(cfg.help_cry_multi) or 0.0)
    )

    local past_multi = tonumber(cfg.past_talk_multi) or 0.0
    if explicit_signal and past_multi < 0.0 then
        local explicit_scale = clamp(tonumber(cfg.past_talk_multi_explicit_scale) or 0.65, 0.0, 2.0)
        past_multi = math.abs(past_multi) * explicit_scale
    end
    score = score + append_component(breakdown, "past_talk_similarity", features.sim_past_talk * past_multi)

    return score, breakdown, features
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

local function compute_recall_threshold(current_turn, features)
    local cfg = ai_cfg()
    local threshold = tonumber(cfg.recall_base) or 5.3
    local cs = cold_start_strength(current_turn)

    if cs > 0 then
        local min_scale = clamp(tonumber(cfg.cold_start_recall_base_scale) or 0.82, 0.30, 1.00)
        local scale = 1.0 - cs * (1.0 - min_scale)
        threshold = threshold * scale
    end

    local explicit_signal = features and (features.has_history_keyword or features.has_explicit_keyword)
    local context_signal = features and (features.has_context_keyword == true)

    if explicit_signal then
        local scale = clamp(tonumber(cfg.explicit_recall_threshold_scale) or 0.72, 0.40, 1.00)
        threshold = threshold * scale
    elseif context_signal then
        local scale = clamp(tonumber(cfg.context_recall_threshold_scale) or 0.88, 0.50, 1.20)
        threshold = threshold * scale
    end

    local cooldown_applied = false
    local since_last = nil
    local cooldown_turns = math.max(0, math.floor(tonumber(cfg.recall_cooldown_turns) or 1))
    if cooldown_turns > 0 and M._last_recall_attempt_turn and not explicit_signal then
        since_last = (tonumber(current_turn) or 0) - (tonumber(M._last_recall_attempt_turn) or 0)
        if since_last > 0 and since_last <= cooldown_turns then
            local mult = math.max(1.0, tonumber(cfg.recall_cooldown_threshold_mult) or 1.18)
            threshold = threshold * mult
            cooldown_applied = true
        end
    end

    return threshold, cs, cooldown_applied, since_last
end

local function evaluate_recall_trigger(user_input, user_vec, current_turn, opts)
    opts = opts or {}
    local turn = math.max(1, tonumber(current_turn) or 1)
    if opts.disable_embeddings ~= true and (not user_vec or #user_vec == 0) then
        user_vec = tool.get_embedding_query(user_input)
    end

    local score, breakdown, features = compute_recall_score(user_input, user_vec, opts)
    local threshold, cs, cooldown_applied, since_last = compute_recall_threshold(turn, features)
    local explicit_signal = (features.has_history_keyword or features.has_explicit_keyword)
    local contextual_signal = (explicit_signal or features.has_context_keyword)

    local should_recall = (score >= threshold)
    if features.has_suppress_keyword then
        should_recall = false
    end

    return should_recall, {
        score = score,
        threshold = threshold,
        cold_start_strength = cs,
        cooldown_applied = cooldown_applied,
        since_last_recall = since_last,
        explicit_signal = explicit_signal,
        contextual_signal = contextual_signal,
        breakdown = breakdown,
        features = features,
        turn = turn,
    }
end

local function need_recall(user_input, user_vec, current_turn, opts)
    opts = opts or {}
    local should_recall, detail = evaluate_recall_trigger(user_input, user_vec, current_turn, opts)
    local cfg = ai_cfg()
    if opts.silent ~= true then
        print(string.format(
            "[Recall] 回忆分数 = %.2f (阈值 %.2f, cold_start=%.2f, explicit=%s, context=%s, cooldown=%s)",
            detail.score,
            detail.threshold,
            detail.cold_start_strength,
            detail.explicit_signal and "Y" or "N",
            detail.contextual_signal and "Y" or "N",
            detail.cooldown_applied and "Y" or "N"
        ))
        if cfg.recall_trigger_debug == true then
            print("[RecallTrigger] " .. format_breakdown(detail.breakdown))
        end
    end
    return should_recall, detail
end

function M.explain_recall_trigger(user_input, current_turn, opts)
    opts = opts or {}
    local silent = opts.silent
    if silent == nil then
        silent = true
    end
    local should_recall, detail = need_recall(
        user_input,
        opts.user_vec,
        current_turn or (history.get_turn() + 1),
        {
            disable_embeddings = opts.disable_embeddings,
            sim_override = opts.sim_override,
            silent = silent,
        }
    )
    detail.should_recall = should_recall
    return detail
end

function M.simulate_recall_trigger(cases, opts)
    opts = opts or {}
    local items = cases or {}
    local results = {}
    local tp, fp, tn, fn = 0, 0, 0, 0
    local labeled_count = 0
    local apply_state = (opts.apply_recall_state ~= false)

    local old_last = M._last_recall_attempt_turn
    if opts.reset_recall_state ~= false then
        M._last_recall_attempt_turn = nil
    end

    for idx, case in ipairs(items) do
        local input = tostring(case.input or case.text or "")
        local turn = math.max(1, tonumber(case.turn or case.current_turn or idx) or idx)
        local disable_embeddings = case.disable_embeddings
        if disable_embeddings == nil then
            if opts.disable_embeddings ~= nil then
                disable_embeddings = opts.disable_embeddings
            else
                disable_embeddings = true
            end
        end

        local should_recall, detail = evaluate_recall_trigger(
            input,
            case.user_vec,
            turn,
            {
                disable_embeddings = disable_embeddings,
                sim_override = case.sim_override or case.sim,
            }
        )

        if should_recall and apply_state then
            M._last_recall_attempt_turn = turn
        end

        local expected = nil
        if case.expected ~= nil then
            expected = (case.expected == true)
            labeled_count = labeled_count + 1
            if should_recall and expected then
                tp = tp + 1
            elseif should_recall and (not expected) then
                fp = fp + 1
            elseif (not should_recall) and expected then
                fn = fn + 1
            else
                tn = tn + 1
            end
        end

        local row = {
            idx = idx,
            id = case.id or ("case_" .. tostring(idx)),
            input = input,
            expected = expected,
            should_recall = should_recall,
            score = detail.score,
            threshold = detail.threshold,
            explicit_signal = detail.explicit_signal,
            contextual_signal = detail.contextual_signal,
            breakdown = detail.breakdown,
            features = detail.features,
            turn = turn,
        }
        results[#results + 1] = row

        if opts.print_details == true then
            local exp_str = "-"
            if expected ~= nil then
                exp_str = expected and "T" or "F"
            end
            print(string.format(
                "[RecallSim][%02d] id=%s turn=%d pred=%s exp=%s score=%.2f th=%.2f",
                idx,
                tostring(row.id),
                turn,
                should_recall and "T" or "F",
                exp_str,
                row.score,
                row.threshold
            ))
            if opts.print_breakdown == true then
                print("  " .. format_breakdown(row.breakdown))
            end
        end
    end

    if opts.reset_recall_state ~= false then
        M._last_recall_attempt_turn = old_last
    end

    local precision = ((tp + fp) > 0) and (tp / (tp + fp)) or 0.0
    local recall_rate = ((tp + fn) > 0) and (tp / (tp + fn)) or 0.0
    local accuracy = ((tp + tn + fp + fn) > 0) and ((tp + tn) / (tp + tn + fp + fn)) or 0.0
    local f1 = ((precision + recall_rate) > 0) and (2.0 * precision * recall_rate / (precision + recall_rate)) or 0.0

    return {
        results = results,
        metrics = {
            total = #results,
            labeled = labeled_count,
            tp = tp,
            fp = fp,
            tn = tn,
            fn = fn,
            precision = precision,
            recall = recall_rate,
            accuracy = accuracy,
            f1 = f1,
        }
    }
end

function M.reset_recall_trigger_state()
    M._last_recall_attempt_turn = nil
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

local function soft_gate_filter(sim, min_gate, opts)
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
    if is_read_only(opts) then
        return prob >= 0.5
    end
    if math.random() < prob then
        M._soft_gate_pass_count = M._soft_gate_pass_count + 1
        return true
    end
    return false
end

local function adjust_gate_on_result(hit, opts)
    if is_read_only(opts) then
        return
    end
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

local function top_probe_clusters(vec, probe_clusters, super_topn, turn, opts, preferred_type)
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
            if preferred_type and preferred_type ~= "" then
                sim = sim + cluster.get_cluster_type_affinity(cid, preferred_type) * cluster_type_bonus()
            end
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

    if (not is_read_only(opts)) and cfg.refinement_enabled == true and #scored > k then
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

local function smart_preload_cold_memories(query_vec, current_turn, current_key, current_info, candidate_clusters, opts)
    if is_read_only(opts) then
        return 0
    end
    _ = current_turn
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
    if budget_left <= 0 then
        return 0
    end
    local qptr = tool.to_ptr_vec(query_vec) or query_vec
    local preload_vec_cache = {}
    local function preload_mem_vec(mem_idx)
        local cached = preload_vec_cache[mem_idx]
        if cached == nil then
            local vec = memory.return_mem_vec(mem_idx)
            if vec then
                preload_vec_cache[mem_idx] = vec
                return vec
            end
            preload_vec_cache[mem_idx] = false
            return nil
        end
        if cached == false then
            return nil
        end
        return cached
    end
    local scored_clusters = {}
    local scored_seen = {}

    for _, topic_key in ipairs(topics_to_preload) do
        if budget_left <= 0 then break end

        local candidates = memory.iter_topic_lines(topic_key, false)
        if #candidates > 0 then
            local cluster_members = {}
            for _, line in ipairs(candidates) do
                local cid = cluster.get_cluster_id_for_line(line)
                if cid and cluster_allow[cid] then
                    local bucket = cluster_members[cid]
                    if not bucket then
                        bucket = {}
                        cluster_members[cid] = bucket
                    end
                    bucket[#bucket + 1] = line
                end
            end

            for cid, members in pairs(cluster_members) do
                if (not scored_seen[cid]) and (not M._preload_cache_clusters[cid]) then
                    local best_sim = -1.0
                    for _, mem_idx in ipairs(members) do
                        local mem_vec = preload_mem_vec(mem_idx)
                        if mem_vec then
                            local sim = tool.cosine_similarity(qptr, mem_vec)
                            if sim > best_sim then
                                best_sim = sim
                            end
                        end
                    end
                    if best_sim > -1.0 then
                        scored_seen[cid] = true
                        scored_clusters[#scored_clusters + 1] = { cid = cid, sim = best_sim }
                    end
                end
            end
        end
    end

    table.sort(scored_clusters, function(a, b) return a.sim > b.sim end)

    local total_lines = memory.get_total_lines()
    local loaded_clusters = 0
    for _, item in ipairs(scored_clusters) do
        if loaded_clusters >= budget_left then break end
        local cid = tonumber(item.cid)
        local clu = cid and cluster.clusters[cid] or nil
        if clu then
            M._preload_cache_clusters[cid] = true
            for _, raw in ipairs(clu.members or {}) do
                local idx = math.floor(tonumber(raw) or -1)
                if idx >= 1 and idx <= total_lines then
                    M._preload_cache_mem[idx] = true
                end
            end
            loaded_clusters = loaded_clusters + 1
        end
    end

    if loaded_clusters > 0 then
        M._preload_cache_anchor = current_key or topics_to_preload[1]
    end
    return loaded_clusters
end

local function unload_preload_cache()
    M._preload_cache_anchor = nil
    M._preload_cache_clusters = {}
    M._preload_cache_mem = {}
end

local function unload_topic_cache()
    if next(M._topic_cache_mem) ~= nil then
        adaptive.add_counter("topic_cache_unload_count", 1)
    end
    M._topic_cache_mem = {}
    M._topic_cache_anchor = nil
    unload_preload_cache()
end

local function update_topic_stability(current_anchor, query_vec, opts)
    local stable_warm = math.max(1, tonumber(ai_cfg().stable_warmup_turns) or 6)
    local stable_sim = tonumber(ai_cfg().stable_min_pair_sim) or 0.72

    if is_read_only(opts) then
        local same_topic_streak = 1
        local streak_sim_sum = 0.0
        local streak_sim_count = 0

        if current_anchor and current_anchor == M._last_topic_anchor then
            same_topic_streak = (tonumber(M._same_topic_streak) or 0) + 1
            if M._prev_query_vec and query_vec and #query_vec > 0 then
                local sim = tool.cosine_similarity(M._prev_query_vec, query_vec)
                streak_sim_sum = (tonumber(M._streak_sim_sum) or 0.0) + sim
                streak_sim_count = (tonumber(M._streak_sim_count) or 0) + 1
            else
                streak_sim_sum = tonumber(M._streak_sim_sum) or 0.0
                streak_sim_count = tonumber(M._streak_sim_count) or 0
            end
        end

        local avg_pair = 1.0
        if streak_sim_count > 0 then
            avg_pair = streak_sim_sum / streak_sim_count
        end
        return (same_topic_streak >= stable_warm) and (avg_pair >= stable_sim)
    end

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

local function topic_random_lift(turn, current_anchor, stable_ready, opts)
    if is_read_only(opts) then
        return
    end
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

local function to_sorted_pairs(turn_best, turn_mem)
    local out = {}
    for turn, score in pairs(turn_best) do
        out[#out + 1] = {
            turn = turn,
            score = score,
            mem_idx = turn_mem and turn_mem[turn] or nil,
        }
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

local function apply_refinement(turn, hits_all, candidate_samples, selected_memories, current_info, sim_th, opts)
    if is_read_only(opts) then return end
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

local function trim_text(s)
    return tostring(s or ""):gsub("^%s*(.-)%s*$", "%1")
end

local function compact_text(s, max_chars)
    local text = tostring(s or "")
    text = text:gsub("%s+", " ")
    text = trim_text(text)
    local limit = math.max(16, math.floor(tonumber(max_chars) or 96))
    if #text <= limit then
        return text
    end
    return text:sub(1, math.max(1, limit - 3)) .. "..."
end

local function sorted_unique_numbers(rows)
    local seen = {}
    local out = {}
    for _, raw in ipairs(rows or {}) do
        local n = math.floor(tonumber(raw) or -1)
        if n >= 1 and not seen[n] then
            seen[n] = true
            out[#out + 1] = n
        end
    end
    table.sort(out)
    return out
end

local function topic_meta_for_key(topic_key, current_turn)
    local key = tostring(topic_key or "")
    if key == "" then
        return nil
    end

    if key:match("^A:%d+$") then
        local start_turn = tonumber(key:match("^A:(%d+)$"))
        if start_turn then
            return {
                key = key,
                start_turn = start_turn,
                end_turn = math.max(start_turn, tonumber(current_turn) or start_turn),
                summary = "",
                is_active = true,
            }
        end
    end

    if key:match("^C:%d+$") then
        local idx = tonumber(key:match("^C:(%d+)$"))
        local rec = idx and (topic.topics or {})[idx] or nil
        if rec then
            return {
                key = key,
                start_turn = tonumber(rec.start) or 1,
                end_turn = tonumber(rec.end_) or tonumber(current_turn) or tonumber(rec.start) or 1,
                summary = tostring(rec.summary or ""),
                is_active = false,
            }
        end
    end

    if key:match("^S:%d+$") then
        local start_turn = tonumber(key:match("^S:(%d+)$"))
        if start_turn then
            local active_start = tonumber((topic.active_topic or {}).start)
            if active_start and start_turn == active_start then
                return {
                    key = key,
                    start_turn = start_turn,
                    end_turn = math.max(start_turn, tonumber(current_turn) or start_turn),
                    summary = "",
                    is_active = true,
                }
            end
            for _, rec in ipairs(topic.topics or {}) do
                if tonumber(rec.start) == start_turn then
                    return {
                        key = key,
                        start_turn = tonumber(rec.start) or start_turn,
                        end_turn = tonumber(rec.end_) or tonumber(current_turn) or start_turn,
                        summary = tostring(rec.summary or ""),
                        is_active = false,
                    }
                end
            end
        end
    end

    return {
        key = key,
        start_turn = 1,
        end_turn = math.max(1, tonumber(current_turn) or 1),
        summary = "",
        is_active = false,
    }
end

local function nearest_selected_distance(turn_value, selected_turns)
    local best = math.huge
    for _, selected_turn in ipairs(selected_turns or {}) do
        local dist = math.abs((tonumber(turn_value) or 0) - (tonumber(selected_turn) or 0))
        if dist < best then
            best = dist
        end
    end
    if best == math.huge then
        return 0
    end
    return best
end

local function expand_topic_turns(topic_key, selected_turns, current_turn, cfg)
    local max_turns = math.max(1, math.floor(tonumber((cfg or {}).compiled_context_max_turns_per_topic) or 4))
    local selected = sorted_unique_numbers(selected_turns)
    if #selected <= 0 then
        return {}
    end
    if tostring(topic_key or ""):match("^turn:%d+$") then
        while #selected > max_turns do
            table.remove(selected)
        end
        return selected
    end

    local meta = topic_meta_for_key(topic_key, current_turn)
    local neighbor_window = math.max(0, math.floor(tonumber((cfg or {}).compiled_context_neighbor_window) or 1))
    local used = {}
    local out = {}

    local function try_add(turn_value)
        local t = math.floor(tonumber(turn_value) or -1)
        if t < 1 or used[t] then
            return
        end
        if meta then
            if t < tonumber(meta.start_turn) or t > tonumber(meta.end_turn) then
                return
            end
        end
        if topic_key_for_turn(t) ~= topic_key then
            return
        end
        used[t] = true
        out[#out + 1] = t
    end

    for _, turn_value in ipairs(selected) do
        try_add(turn_value)
    end

    local candidates = {}
    for _, turn_value in ipairs(selected) do
        for delta = 1, neighbor_window do
            candidates[#candidates + 1] = turn_value - delta
            candidates[#candidates + 1] = turn_value + delta
        end
    end
    candidates = sorted_unique_numbers(candidates)
    table.sort(candidates, function(a, b)
        local da = nearest_selected_distance(a, selected)
        local db = nearest_selected_distance(b, selected)
        if da == db then
            return a > b
        end
        return da < db
    end)

    for _, turn_value in ipairs(candidates) do
        if #out >= max_turns then
            break
        end
        try_add(turn_value)
    end

    table.sort(out)
    while #out > max_turns do
        table.remove(out)
    end
    return out
end

local function build_topic_fallback_summary(turns, cfg)
    for _, turn_value in ipairs(turns or {}) do
        local entry = history.get_by_turn(turn_value)
        if entry then
            local user_part, _ = history.parse_entry(entry)
            local snippet = compact_text(user_part, tonumber((cfg or {}).compiled_context_summary_chars) or 96)
            if snippet ~= "" then
                return "主要围绕：" .. snippet
            end
        end
    end
    return ""
end

local function format_type_summary(type_counts)
    local items = {}
    for type_name, count in pairs(type_counts or {}) do
        local n = math.max(0, math.floor(tonumber(count) or 0))
        if n > 0 then
            items[#items + 1] = { name = tostring(type_name), count = n }
        end
    end
    table.sort(items, function(a, b)
        if a.count == b.count then
            return a.name < b.name
        end
        return a.count > b.count
    end)

    if #items <= 0 then
        return ""
    end

    local parts = {}
    for i = 1, math.min(3, #items) do
        parts[#parts + 1] = string.format("%s(%d)", items[i].name, items[i].count)
    end
    return table.concat(parts, ", ")
end

local function build_compiled_memory_context(selected, current_turn, cfg)
    local topic_groups = {}
    local topic_order = {}

    for _, item in ipairs(selected or {}) do
        local topic_key = topic_key_for_turn(item.turn) or ("turn:" .. tostring(item.turn))
        local bucket = topic_groups[topic_key]
        if not bucket then
            bucket = {
                key = topic_key,
                score = tonumber(item.score) or 0.0,
                turns = {},
                type_counts = {},
            }
            topic_groups[topic_key] = bucket
            topic_order[#topic_order + 1] = bucket
        else
            bucket.score = math.max(bucket.score, tonumber(item.score) or 0.0)
        end
        bucket.turns[#bucket.turns + 1] = item.turn
        if item.mem_idx then
            local type_name = memory.get_type_name(item.mem_idx)
            bucket.type_counts[type_name] = (bucket.type_counts[type_name] or 0) + 1
        end
    end

    table.sort(topic_order, function(a, b)
        if a.score == b.score then
            return tostring(a.key) < tostring(b.key)
        end
        return a.score > b.score
    end)

    local max_topics = math.max(1, math.floor(tonumber((cfg or {}).compiled_context_max_topics) or 3))
    local user_chars = math.max(24, math.floor(tonumber((cfg or {}).compiled_context_user_chars) or 72))
    local ai_chars = math.max(24, math.floor(tonumber((cfg or {}).compiled_context_ai_chars) or 96))
    local lines = { "【相关记忆摘要】" }

    for idx = 1, math.min(max_topics, #topic_order) do
        local bucket = topic_order[idx]
        local topic_turns = expand_topic_turns(bucket.key, bucket.turns, current_turn, cfg)
        local meta = topic_meta_for_key(bucket.key, current_turn)
        local summary = trim_text((meta or {}).summary or "")
        if summary == "" then
            summary = build_topic_fallback_summary(topic_turns, cfg)
        end

        lines[#lines + 1] = string.format("主题%d | 来源turn: %s", idx, table.concat(sorted_unique_numbers(bucket.turns), ", "))
        if summary ~= "" then
            lines[#lines + 1] = "摘要：" .. compact_text(summary, tonumber((cfg or {}).compiled_context_summary_chars) or 120)
        end
        local type_summary = format_type_summary(bucket.type_counts)
        if type_summary ~= "" then
            lines[#lines + 1] = "类型：" .. type_summary
        end
        lines[#lines + 1] = "线索："

        for _, turn_value in ipairs(topic_turns) do
            local entry = history.get_by_turn(turn_value)
            if entry then
                local user_part, ai_part = history.parse_entry(entry)
                local user_snippet = compact_text(user_part, user_chars)
                local ai_snippet = compact_text(ai_part, ai_chars)
                local detail = string.format("- 第%d轮：用户=%s", turn_value, user_snippet ~= "" and user_snippet or "(空)")
                if ai_snippet ~= "" then
                    detail = detail .. "；助手=" .. ai_snippet
                end
                lines[#lines + 1] = detail
            end
        end
    end

    if #lines <= 1 then
        return ""
    end
    return table.concat(lines, "\n")
end

local function retrieve(user_input, user_vec, current_turn, current_info, current_anchor, opts)
    opts = opts or {}
    local read_only = is_read_only(opts)
    if memory.get_total_lines() <= 0 then
        return ""
    end

    if not read_only then
        memory.begin_turn(current_turn)
    end

    local cfg = ai_cfg()
    local knobs = effective_retrieval_knobs(current_turn)
    local min_gate = tonumber(knobs.min_sim_gate) or (cfg.min_sim_gate or 0.58)
    local power = tonumber(knobs.power_suppress) or (cfg.power_suppress or 1.80)
    local max_memory = math.max(1, tonumber(knobs.max_memory) or 5)
    local max_turns = math.max(1, tonumber(knobs.max_turns) or 10)
    local query_topn = math.max(1, tonumber(knobs.supercluster_topn_query) or 4)
    local keyword_weight = tonumber(knobs.keyword_weight) or tonumber(cfg.keyword_weight) or 0.55
    local probe_clusters = math.max(1, tonumber(knobs.probe_clusters) or 2)
    local cross_quota_ratio = clamp(tonumber(knobs.topic_cross_quota_ratio) or 0.25, 0.0, 0.5)
    local memory_drop_sim = tonumber(cfg.memory_drop_sim)
    if memory_drop_sim == nil then memory_drop_sim = 0.60 end
    local sim_th = tonumber(cfg.topic_sim_threshold) or 0.70

    local per_cluster_limit = math.max(2, tonumber(cfg.refinement_probe_per_cluster_limit) or 12)
    local base_scan_limit = math.max(per_cluster_limit, max_memory)
    local persistent_cap = math.max(1, tonumber(cfg.persistent_explore_candidate_cap) or 32)
    local allowed_types, blocked_types, preferred_type = derive_type_filters(opts)

    local current_topic_key = topic_key_for_current(current_turn, current_info)

    local turn_best = {}
    local turn_src = {}
    local turn_mem = {}
    local mem_best_sim = {}
    local mem_best_cluster = {}
    local mem_best_effective = {}
    local mem_best_source = {}
    local persistent_probe_clusters_query = {}
    local keyword_perf_mode = tostring(cfg.keyword_perf_mode or "lossless")
    if keyword_perf_mode ~= "near_lossless" then
        keyword_perf_mode = "lossless"
    end
    local near_lossless_mode = (keyword_perf_mode == "near_lossless")
    local near_min_primary_candidates = math.max(1, math.floor(
        tonumber(cfg.near_lossless_min_primary_candidates) or (max_memory * 8)
    ))
    local near_secondary_cap = math.max(0, math.floor(tonumber(cfg.near_lossless_secondary_cap) or 0))
    local secondary_near_mode_used = false

    local mem_vec_cache = {}
    local mem_cluster_cache = {}
    local q_sim_cache = {}
    local primary_candidates = nil

    local function get_mem_vec_cached(mem_idx)
        local cached = mem_vec_cache[mem_idx]
        if cached == nil then
            local vec = memory.return_mem_vec(mem_idx)
            if vec then
                mem_vec_cache[mem_idx] = vec
                return vec
            end
            mem_vec_cache[mem_idx] = false
            return nil
        end
        if cached == false then
            return nil
        end
        return cached
    end

    local function get_mem_cluster_cached(mem_idx)
        local cid = mem_cluster_cache[mem_idx]
        if cid == nil then
            cid = cluster.get_cluster_id_for_line(mem_idx) or -1
            mem_cluster_cache[mem_idx] = cid
        end
        return cid
    end

    local function get_sim_cached(q_idx, qptr, mem_idx)
        local qcache = q_sim_cache[q_idx]
        if not qcache then
            qcache = {}
            q_sim_cache[q_idx] = qcache
        end

        local cached = qcache[mem_idx]
        if cached ~= nil then
            if cached == false then
                return nil
            end
            return cached
        end

        local mem_vec = get_mem_vec_cached(mem_idx)
        if not mem_vec then
            qcache[mem_idx] = false
            return nil
        end
        local sim = tool.cosine_similarity(qptr, mem_vec)
        qcache[mem_idx] = sim
        return sim
    end

    local query_vecs = query_vectors(user_vec, keyword_weight, opts)
    local primary = query_vecs[1]
    if (not primary) or type(primary.vec) ~= "table" or #primary.vec == 0 then
        return ""
    end
    local primary_qptr = tool.to_ptr_vec(primary.vec) or primary.vec

    local persistent_extra_budget = 0
    if (not read_only) and cfg.persistent_explore_enabled == true and cluster.cluster_count() > 1 then
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

    local cluster_ids_for_preload, _ = top_probe_clusters(
        primary_qptr, probe_clusters, query_topn, current_turn, opts, preferred_type
    )
    local preloaded_clusters = smart_preload_cold_memories(
        primary_qptr, current_turn, current_topic_key, current_info, cluster_ids_for_preload, opts
    )

    local function update_mem_best(mem_idx, sim, effective, cid, source)
        local prev_sim = mem_best_sim[mem_idx]
        if (not prev_sim) or sim > prev_sim then
            mem_best_sim[mem_idx] = sim
            mem_best_cluster[mem_idx] = tonumber(cid) or get_mem_cluster_cached(mem_idx)
            mem_best_effective[mem_idx] = effective
            mem_best_source[mem_idx] = source or "hot"
        end
    end

    local function type_filter_pass(mem_idx)
        if not allowed_types and not blocked_types then
            return true
        end
        return select(1, memory.type_matches(mem_idx, allowed_types, blocked_types))
    end

    local function build_primary_candidates()
        local items = {}
        for mem_idx, sim in pairs(mem_best_sim) do
            items[#items + 1] = {
                mem_idx = mem_idx,
                sim = tonumber(sim) or -1.0
            }
        end
        table.sort(items, function(a, b) return a.sim > b.sim end)

        local cap = #items
        if near_secondary_cap > 0 and near_secondary_cap < cap then
            cap = near_secondary_cap
        end

        local out = {}
        for i = 1, cap do
            out[#out + 1] = items[i].mem_idx
        end
        return out
    end

    for q_idx, q in ipairs(query_vecs) do
        local qv = q.vec
        local qptr = tool.to_ptr_vec(qv) or qv
        local weight = q.weight or 1.0
        local run_secondary_near = false
        if near_lossless_mode and q_idx > 1 then
            if not primary_candidates then
                primary_candidates = build_primary_candidates()
            end
            if #primary_candidates >= near_min_primary_candidates then
                run_secondary_near = true
                secondary_near_mode_used = true
            end
        end

        if run_secondary_near then
            for _, mem_idx in ipairs(primary_candidates) do
                local sim = get_sim_cached(q_idx, qptr, mem_idx)
                if sim then
                    local sim_pos = math.max(0.0, sim)
                    local effective = (sim_pos ^ power) * weight
                    local cid = mem_best_cluster[mem_idx] or get_mem_cluster_cached(mem_idx)
                    local src = mem_best_source[mem_idx] or "hot"
                    update_mem_best(mem_idx, sim, effective, cid, src)
                end
            end
        else
            local cluster_ids, _ = top_probe_clusters(
                qptr, probe_clusters, query_topn, current_turn, opts, preferred_type
            )
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

            local qcache = q_sim_cache[q_idx]
            if not qcache then
                qcache = {}
                q_sim_cache[q_idx] = qcache
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
                    allowed_types = allowed_types,
                    blocked_types = blocked_types,
                })

                for _, mem in ipairs(sim_results) do
                    local mem_idx = mem.index
                    local sim = mem.similarity
                    if qcache[mem_idx] == nil then
                        qcache[mem_idx] = sim
                    end

                    local sim_pos = math.max(0.0, sim)
                    local effective = (sim_pos ^ power) * weight
                    update_mem_best(mem_idx, sim, effective, cid, src_label)

                    if not soft_gate_filter(sim, min_gate, opts) then
                        break
                    end
                end
            end
        end

        if current_topic_key and M._topic_cache_anchor == current_topic_key and next(M._topic_cache_mem) ~= nil then
            for mem_idx, _ in pairs(M._topic_cache_mem) do
                if type_filter_pass(mem_idx) then
                    local sim = get_sim_cached(q_idx, qptr, mem_idx)
                    if sim then
                    local sim_pos = math.max(0.0, sim)
                    local effective = (sim_pos ^ power) * weight
                    if sim >= min_gate then
                        effective = effective * (tonumber(cfg.topic_cache_weight) or 1.02)
                    end
                    update_mem_best(mem_idx, sim, effective, get_mem_cluster_cached(mem_idx), "cache")
                    end
                end
            end
        end

        if next(M._preload_cache_mem) ~= nil then
            local preload_scored = {}
            for mem_idx, _ in pairs(M._preload_cache_mem) do
                if type_filter_pass(mem_idx) then
                    local sim = get_sim_cached(q_idx, qptr, mem_idx)
                    if sim then
                    preload_scored[#preload_scored + 1] = { mem_idx = mem_idx, sim = sim }
                    end
                end
            end
            table.sort(preload_scored, function(a, b) return a.sim > b.sim end)
            for _, item in ipairs(preload_scored) do
                local sim = item.sim
                if not soft_gate_filter(sim, min_gate, opts) then
                    break
                end
                local mem_idx = item.mem_idx
                local sim_pos = math.max(0.0, sim)
                local effective = (sim_pos ^ power) * weight
                update_mem_best(mem_idx, sim, effective, get_mem_cluster_cached(mem_idx), "preload_cluster")
            end
        end

        if q_idx == 1 and near_lossless_mode and not primary_candidates then
            primary_candidates = build_primary_candidates()
        end
    end

    local kept_memories = {}
    local dropped_by_memory_gate = 0
    for mem_idx, sim in pairs(mem_best_sim) do
        if sim >= memory_drop_sim then
            kept_memories[#kept_memories + 1] = mem_idx
            local effective = tonumber(mem_best_effective[mem_idx]) or 0.0
            if effective > 0.0 then
                local src_label = mem_best_source[mem_idx] or "hot"
                local turns = memory.get_turns(mem_idx)
                for _, t in ipairs(turns) do
                    if (not turn_best[t]) or effective > turn_best[t] then
                        turn_best[t] = effective
                        turn_src[t] = src_label
                        turn_mem[t] = mem_idx
                    end
                end
            end
        else
            dropped_by_memory_gate = dropped_by_memory_gate + 1
        end
    end

    local kept_best_sim = {}
    local kept_best_cluster = {}
    local kept_best_effective = {}
    for _, mem_idx in ipairs(kept_memories) do
        kept_best_sim[mem_idx] = mem_best_sim[mem_idx]
        kept_best_cluster[mem_idx] = mem_best_cluster[mem_idx]
        kept_best_effective[mem_idx] = mem_best_effective[mem_idx]
    end

    local sample_limit = math.max(1, tonumber(cfg.refinement_sample_mem_topk) or 48)
    local candidate_samples = collect_candidate_samples(kept_best_sim, kept_best_cluster, kept_best_effective, sample_limit)

    local selected = {}
    local ranked = to_sorted_pairs(turn_best, turn_mem)
    if #ranked <= 0 then
        print(string.format(
            "[Recall] 空召回（preloaded_clusters=%d, kept_memories=%d, dropped_by_memory_gate=%d, keyword_mode=%s）",
            preloaded_clusters, #kept_memories, dropped_by_memory_gate,
            secondary_near_mode_used and "near_lossless" or keyword_perf_mode
        ))
        adjust_gate_on_result(false, opts)
        if not read_only then
            heat.enqueue_cold_rescue(user_vec, current_turn, current_info, min_gate)
        end
        return ""
    end

    if cfg.use_topic_buckets ~= true or not current_info then
        for i = 1, math.min(max_turns, #ranked) do
            selected[#selected + 1] = ranked[i]
        end

        if (not read_only) and cfg.refinement_enabled == true then
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

    if (not read_only) and cache_contrib > 0 then
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
    if (not read_only) and persistent_hits > 0 then
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
            if not read_only then
                update_cluster_hit_rate(cid, hit_flag)
            end
            clusters_seen[cid] = true
        end
    end

    adjust_gate_on_result(hit_flag, opts)

    apply_refinement(current_turn, hits_all, candidate_samples, selected_memories, current_info, sim_th, opts)

    local need_rescue = (#selected_turns == 0)
    if not need_rescue and (cfg.cold_rescue_on_empty_only ~= true) and hits_all <= 0 then
        need_rescue = true
    end
    if need_rescue and (not read_only) then
        heat.enqueue_cold_rescue(user_vec, current_turn, current_info, min_gate)
    end

    print(string.format(
        "[Recall] 选中 %d 条 turn（hits_same=%d, gate=%.3f, max_turns=%d, preloaded_clusters=%d, kept_memories=%d, dropped_by_memory_gate=%d, keyword_mode=%s）",
        #selected_turns, hits_all, min_gate, max_turns, preloaded_clusters, #kept_memories, dropped_by_memory_gate,
        secondary_near_mode_used and "near_lossless" or keyword_perf_mode
    ))

    return build_compiled_memory_context(selected, current_turn, cfg)
end

function M.check_and_retrieve(user_input, user_vec, opts)
    opts = opts or {}
    local read_only = is_read_only(opts)
    local policy_decided = opts.policy_decided == true
    local legacy_trigger_enabled = graph_recall_cfg().legacy_trigger_enabled == true
    if opts.suppress == true then
        return ""
    end
    user_vec = user_vec or tool.get_embedding_query(user_input)

    local current_turn = history.get_turn() + 1
    local current_info = topic.get_topic_for_turn and topic.get_topic_for_turn(current_turn) or nil
    local current_key = topic_key_for_current(current_turn, current_info)

    if (not read_only) and M._last_topic_anchor and current_key ~= M._last_topic_anchor then
        unload_topic_cache()
    end

    local stable_ready = update_topic_stability(current_key, user_vec, opts)
    topic_random_lift(current_turn, current_key, stable_ready, opts)

    local should_recall = opts.force == true
    if (not should_recall) and (not policy_decided) and legacy_trigger_enabled then
        should_recall = select(1, need_recall(user_input, user_vec, current_turn, opts))
    end
    if should_recall then
        if not read_only then
            M._last_recall_attempt_turn = current_turn
        end
        return retrieve(user_input, user_vec, current_turn, current_info, current_key, opts)
    end

    return ""
end

return M
