-- module/experience/retriever.lua
-- 经验检索器：query-conditioned 候选召回 + relevance gate + joint rerank

local M = {}

local store = require("module.experience.store")
local adaptive = require("module.experience.adaptive")
local config = require("module.config")
local tool = require("module.tool")

local LANGUAGE_HINTS = {
    { pattern = "%f[%a]lua%f[%A]", language = "lua" },
    { pattern = "%f[%a]python%f[%A]", language = "python" },
    { pattern = "%f[%a]javascript%f[%A]", language = "javascript" },
    { pattern = "%f[%a]typescript%f[%A]", language = "typescript" },
    { pattern = "%f[%a]java%f[%A]", language = "java" },
    { pattern = "%f[%a]rust%f[%A]", language = "rust" },
    { pattern = "%f[%a]golang%f[%A]", language = "go" },
    { pattern = "%f[%a]go%f[%A]", language = "go" },
    { pattern = "%f[%a]cpp%f[%A]", language = "cpp" },
    { pattern = "%f[%a]csharp%f[%A]", language = "csharp" },
    { pattern = "%f[%a]php%f[%A]", language = "php" },
    { pattern = "%f[%a]ruby%f[%A]", language = "ruby" },
    { pattern = "%f[%a]swift%f[%A]", language = "swift" },
    { pattern = "%f[%a]kotlin%f[%A]", language = "kotlin" },
}

local STAGE_SCALES = { 1.0, 0.85, 0.70, 0.55 }
local STAGE_NAMES = { "strict", "relaxed", "wide", "backfill" }

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

-- ==================== 本地工具函数 ====================

local function serialize_context_signature(sig)
    if not sig or type(sig) ~= "table" then
        return ""
    end

    local parts = {}
    for k, v in pairs(sig) do
        if type(v) == "table" then
            parts[#parts + 1] = string.format("%s=%s", k, serialize_context_signature(v))
        else
            parts[#parts + 1] = string.format("%s=%s", k, tostring(v))
        end
    end
    table.sort(parts)
    return table.concat(parts, "|")
end

local function retriever_cfg()
    return ((config.settings or {}).experience or {}).retriever or {}
end

local function adaptive_cfg()
    return ((config.settings or {}).experience or {}).adaptive or {}
end

local function count_results_above_gate(candidates, gate)
    local count = 0
    for _, exp in ipairs(candidates or {}) do
        if (tonumber(exp.relevance_score) or 0.0) >= gate then
            count = count + 1
        end
    end
    return count
end

local function stage_name(idx)
    return STAGE_NAMES[idx] or ("stage_" .. tostring(idx))
end

local function detect_language(text, text_lower)
    text = tostring(text or "")
    text_lower = tostring(text_lower or text:lower())

    if text:find("C++", 1, true) or text_lower:find("c++", 1, true) then
        return "cpp"
    end

    for _, hint in ipairs(LANGUAGE_HINTS) do
        if text_lower:match(hint.pattern) then
            return hint.language
        end
    end

    return nil
end

local function merge_missing_fields(dst, src)
    for k, v in pairs(src or {}) do
        if dst[k] == nil then
            dst[k] = v
        end
    end
end

local function utility_confidence_count()
    local cfg = retriever_cfg()
    return math.max(1, math.floor(tonumber(cfg.utility_confidence_count) or 6))
end

-- ==================== 查询特征提取 ====================

function M.extract_query_features(query)
    local features = {}

    if type(query) == "string" then
        features.query_text = query
        merge_missing_fields(features, M.extract_features_from_text(query))
    elseif type(query) == "table" then
        features.type = query.type
        features.task_type = query.task_type
        features.context_signature = query.context_signature
        features.tool_name = query.tool_name or query.tool_used
        features.query_embedding = query.query_embedding
        features.domain = query.domain
        features.language = query.language
        features.output_strategy = query.output_strategy
        features.pattern_key = query.pattern_key
        features.error_type = query.error_type
        features.success_key = query.success_key
        features.query_text = query.text or query.content or query.description

        if features.query_text and features.query_text ~= "" then
            merge_missing_fields(features, M.extract_features_from_text(features.query_text))
        end
    end

    if features.context_signature then
        features.context_key = serialize_context_signature(features.context_signature)
        if not features.language then
            features.language = features.context_signature.language
        end
        if not features.domain then
            features.domain = features.context_signature.domain
        end
    end

    if (not features.query_embedding) and features.query_text and features.query_text ~= "" then
        features.query_embedding = tool.get_embedding_query(features.query_text)
    end

    return features
end

function M.extract_features_from_text(text)
    local features = {}
    text = tostring(text or "")
    local text_lower = text:lower()

    if text:match("失败") or text:match("错误") then
        features.type = "failure"
    elseif text:match("成功") or text:match("完成") then
        features.type = "success"
    end

    if text:match("代码") or text:match("编程") or text_lower:find("debug", 1, true) then
        features.task_type = "coding"
        features.domain = features.domain or "coding"
    elseif text:match("分析") then
        features.task_type = "analysis"
        features.domain = features.domain or "analysis"
    elseif text:match("对话") then
        features.task_type = "conversation"
    end

    local tool_match = text:match("使用%s*([%w_]+)%s*工具")
    if tool_match then
        features.tool_name = tool_match
    end

    features.language = detect_language(text, text_lower)
    return features
end

-- ==================== 统计信息 ====================

function M.get_stats()
    return {
        store_stats = store.get_stats(),
        config = retriever_cfg()
    }
end

-- ==================== 打分 ====================

function M.compute_relevance_score(experience, query_state)
    local semantic_sim = math.max(0.0, tonumber(experience.query_similarity or experience.vector_similarity) or 0.0)
    local context_sim = clamp(tonumber(experience.context_similarity) or 0.0, 0.0, 1.0)
    local index_signal = clamp(tonumber(experience.index_match_score) or 0.0, 0.0, 1.0)

    local route_signal = 0.0
    local route_weight = 0.0
    local ad_cfg = adaptive_cfg()
    local context_sig = query_state and query_state.context_signature or nil

    if ad_cfg.enabled == true then
        if experience.type then
            route_signal = route_signal + 0.20 * adaptive.get_retrieval_route_score("by_type", experience.type)
            route_weight = route_weight + 0.20
        end

        if experience.task_type then
            route_signal = route_signal + 0.30 * adaptive.get_retrieval_route_score("by_task", experience.task_type)
            route_weight = route_weight + 0.30
        end

        if context_sig then
            local ctx_key = type(context_sig) == "string" and context_sig or serialize_context_signature(context_sig)
            route_signal = route_signal + 0.30 * adaptive.get_retrieval_route_score("by_context", ctx_key)
            route_weight = route_weight + 0.30
        end

        if experience.tools_used then
            local tool_bonus = 0.0
            local tool_count = 0
            for tool_name in pairs(experience.tools_used) do
                tool_bonus = tool_bonus + adaptive.get_retrieval_route_score("by_tool", tool_name)
                tool_count = tool_count + 1
            end
            if tool_count > 0 then
                route_signal = route_signal + 0.20 * (tool_bonus / tool_count)
                route_weight = route_weight + 0.20
            end
        end
    end

    local route_prior = 0.5
    if route_weight > 0.0 then
        route_prior = clamp(0.5 + (route_signal / route_weight) * 0.25, 0.0, 1.0)
    end

    return clamp(
        0.50 * semantic_sim +
        0.20 * context_sim +
        0.20 * index_signal +
        0.10 * route_prior,
        0.0,
        1.0
    )
end

function M.compute_utility_score(experience)
    local base_prior = tonumber(experience.utility_prior)
    if base_prior == nil then
        base_prior = tonumber(experience.success_rate)
    end
    if base_prior == nil and experience.outcome and type(experience.outcome.success) == "boolean" then
        base_prior = experience.outcome.success and 0.70 or 0.35
    end
    base_prior = clamp(base_prior or 0.50, 0.0, 1.0)

    local ad_cfg = adaptive_cfg()
    if ad_cfg.enabled ~= true or not experience.id then
        experience.utility_observation_count = 0
        return base_prior
    end

    local learned = adaptive.get_experience_utility(experience.id)
    local seen = adaptive.get_experience_utility_count and adaptive.get_experience_utility_count(experience.id) or 0
    local confidence = clamp(seen / utility_confidence_count(), 0.0, 1.0)
    experience.utility_observation_count = seen

    return clamp(
        (1.0 - confidence) * base_prior + confidence * learned,
        0.0,
        1.0
    )
end

function M.compute_joint_score(experience)
    local relevance = clamp(tonumber(experience.relevance_score) or 0.0, 0.0, 1.0)
    local utility = clamp(tonumber(experience.utility_score) or 0.0, 0.0, 1.0)
    return clamp(relevance * utility, 0.0, 1.0)
end

-- ==================== 排序与回退 ====================

function M.rank_candidates(candidates, gate, k, options)
    options = options or {}
    local exploration_slots = math.max(0, math.floor(tonumber(options.exploration_slots) or 0))
    local confidence_den = utility_confidence_count()
    local ranked = {}

    for _, exp in ipairs(candidates or {}) do
        if (tonumber(exp.relevance_score) or 0.0) >= gate then
            ranked[#ranked + 1] = exp
        end
    end

    table.sort(ranked, function(a, b)
        local joint_a = tonumber(a.joint_score) or 0.0
        local joint_b = tonumber(b.joint_score) or 0.0
        if joint_a == joint_b then
            local rel_a = tonumber(a.relevance_score) or 0.0
            local rel_b = tonumber(b.relevance_score) or 0.0
            if rel_a == rel_b then
                return (tonumber(a.created_at) or 0) > (tonumber(b.created_at) or 0)
            end
            return rel_a > rel_b
        end
        return joint_a > joint_b
    end)

    local selected = {}
    local selected_ids = {}
    local main_limit = math.max(0, k - exploration_slots)

    for i = 1, math.min(main_limit, #ranked) do
        local item = ranked[i]
        selected[#selected + 1] = item
        selected_ids[item.id] = true
    end

    if exploration_slots > 0 and #selected < k then
        local exploration = {}
        for _, exp in ipairs(ranked) do
            if not selected_ids[exp.id] then
                local seen = tonumber(exp.utility_observation_count) or 0
                local confidence = clamp(seen / confidence_den, 0.0, 1.0)
                exp.exploration_score =
                    (tonumber(exp.relevance_score) or 0.0) *
                    (0.35 + 0.65 * (tonumber(exp.utility_score) or 0.0)) *
                    (1.0 - confidence)
                exploration[#exploration + 1] = exp
            end
        end

        table.sort(exploration, function(a, b)
            return (tonumber(a.exploration_score) or 0.0) > (tonumber(b.exploration_score) or 0.0)
        end)

        local missing = math.min(exploration_slots, k - #selected)
        for i = 1, missing do
            local item = exploration[i]
            if item then
                item.selected_as_exploration = true
                selected[#selected + 1] = item
                selected_ids[item.id] = true
            end
        end
    end

    return selected
end

local function build_store_query_options(query_features, options, fetch_limit)
    local cfg = retriever_cfg()
    local context_sig = options.context_sig or options.context_signature or query_features.context_signature

    return {
        limit = fetch_limit,
        type = options.type or query_features.type,
        task_type = options.task_type or query_features.task_type,
        context_signature = context_sig,
        tool_used = options.tool_used or options.tool_name or query_features.tool_name,
        domain = options.domain or query_features.domain,
        language = options.language or query_features.language,
        output_strategy = options.output_strategy or query_features.output_strategy,
        pattern_key = options.pattern_key or query_features.pattern_key,
        error_type = options.error_type or query_features.error_type,
        success_key = options.success_key or query_features.success_key,
        query_embedding = options.query_embedding or query_features.query_embedding,
        context_threshold = tonumber(options.context_threshold) or tonumber(cfg.context_threshold) or 0.30,
        semantic_threshold = tonumber(options.semantic_threshold) or tonumber(cfg.semantic_threshold) or 0.12,
        semantic_scan_limit = tonumber(options.semantic_scan_limit) or math.max(fetch_limit, tonumber(cfg.semantic_scan_limit) or fetch_limit),
    }
end

-- ==================== 核心检索 ====================

function M.retrieve_intersection_priority(query, options)
    options = options or {}
    local cfg = retriever_cfg()
    local k = math.max(1, math.floor(tonumber(options.limit) or 5))
    local fetch_limit = math.max(k * (tonumber(cfg.fetch_multiplier) or 8), 24)
    local base_gate = clamp(tonumber(options.relevance_gate) or tonumber(cfg.relevance_gate) or 0.32, 0.05, 0.95)
    local min_needed_ratio = clamp(tonumber(options.min_needed_ratio) or tonumber(cfg.min_needed_ratio) or 0.60, 0.0, 1.0)
    local min_needed = math.max(1, math.floor(k * min_needed_ratio + 0.5))
    local exploration_slots = math.max(0, math.floor(tonumber(options.exploration_slots) or tonumber(cfg.exploration_slots) or 1))

    local query_features = M.extract_query_features(query)
    local store_query = build_store_query_options(query_features, options, fetch_limit)
    local candidates = store.retrieve(store_query)

    if not candidates or #candidates == 0 then
        return {}, { strategy = "empty", reason = "no_candidates" }
    end

    local query_state = {
        context_signature = store_query.context_signature,
        context_key = query_features.context_key,
        query_embedding = store_query.query_embedding,
    }

    for _, exp in ipairs(candidates) do
        exp.context_key = exp.context_key
            or ((exp.context_signature and serialize_context_signature(exp.context_signature)) or nil)
        exp.relevance_score = M.compute_relevance_score(exp, query_state)
        exp.utility_score = M.compute_utility_score(exp)
        exp.joint_score = M.compute_joint_score(exp)
        exp.final_score = exp.joint_score
    end

    table.sort(candidates, function(a, b)
        return (tonumber(a.joint_score) or 0.0) > (tonumber(b.joint_score) or 0.0)
    end)

    local selected = {}
    local chosen_stage = "empty"
    local chosen_gate = base_gate

    for idx, scale in ipairs(STAGE_SCALES) do
        local gate = clamp(base_gate * scale, 0.0, 1.0)
        local results = M.rank_candidates(candidates, gate, k, {
            exploration_slots = exploration_slots,
        })

        if #results >= min_needed or idx == #STAGE_SCALES then
            selected = results
            chosen_stage = stage_name(idx)
            chosen_gate = gate
            break
        end
    end

    return selected, {
        strategy = chosen_stage,
        candidate_count = #candidates,
        result_count = #selected,
        applied_relevance_gate = chosen_gate,
        gate_hit_count = count_results_above_gate(candidates, chosen_gate),
        min_needed = min_needed,
        stats = {
            base_gate = base_gate,
            stage_scales = STAGE_SCALES,
            exploration_slots = exploration_slots,
        },
    }
end

-- ==================== 反馈 ====================

function M.record_utility_feedback(retrieved_items, effective_ids)
    local ad_cfg = adaptive_cfg()
    if ad_cfg.enabled ~= true then
        return
    end

    effective_ids = effective_ids or {}
    local retrieved_ids = {}
    local samples = {}
    local positive_ids = {}
    local negative_ids = {}

    for _, entry in ipairs(retrieved_items or {}) do
        local item = entry
        local exp_id = nil
        if type(item) == "table" then
            exp_id = item.id
        else
            exp_id = item
            item = nil
        end

        if exp_id then
            retrieved_ids[#retrieved_ids + 1] = exp_id
            if effective_ids[exp_id] == true then
                positive_ids[exp_id] = true
            else
                negative_ids[exp_id] = true
            end

            if item then
                samples[#samples + 1] = {
                    id = exp_id,
                    exp_type = item.type,
                    task_type = item.task_type,
                    tools_used = item.tools_used,
                    context_signature = item.context_signature,
                    context_key = item.context_key,
                }
            end
        end
    end

    adaptive.update_utility_from_feedback({
        retrieved_ids = retrieved_ids,
        effective_ids = effective_ids
    })

    if #samples > 0 then
        adaptive.update_after_retrieval({
            candidate_samples = samples,
            positive_ids = positive_ids,
            negative_ids = negative_ids,
        })
    end

    adaptive.record_retrieval_outcome(next(positive_ids) ~= nil)
end

function M.retrieve_with_feedback(query, options)
    local results, strategy_info = M.retrieve_intersection_priority(query, options)

    local feedback_items = {}
    for _, exp in ipairs(results) do
        feedback_items[#feedback_items + 1] = {
            id = exp.id,
            type = exp.type,
            task_type = exp.task_type,
            tools_used = exp.tools_used,
            context_signature = exp.context_signature,
            context_key = exp.context_key,
        }
    end

    local feedback_fn = function(effective_ids)
        M.record_utility_feedback(feedback_items, effective_ids or {})
    end

    return results, feedback_fn, strategy_info
end

return M
