
-- module/experience/retriever.lua
-- 经验检索器：基于双信号交集优先的智能检索

local M = {}

local store = require("module.experience.store")
local adaptive = require("module.experience.adaptive")
local intersection = require("module.experience.intersection")
local config = require("module.config")

-- ==================== 本地工具函数 ====================

-- 序列化上下文签名（用于路由评分索引）
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

-- ==================== 配置 ====================

local function retriever_cfg()
    return ((config.settings or {}).experience or {}).retriever or {}
end

local function adaptive_cfg()
    return ((config.settings or {}).experience or {}).adaptive or {}
end

-- ==================== 查询特征提取 ====================

function M.extract_query_features(query)
    local features = {}

    if type(query) == "string" then
        -- 从文本查询中提取特征
        features = M.extract_features_from_text(query)
    elseif type(query) == "table" then
        -- 从结构化查询中提取特征
        features.type = query.type
        features.task_type = query.task_type
        features.context_signature = query.context_signature
        features.tool_name = query.tool_name
        features.query_embedding = query.query_embedding

        -- 如果有文本内容，也提取文本特征
        if query.text or query.content then
            local text_features = M.extract_features_from_text(
                query.text or query.content
            )
            for k, v in pairs(text_features) do
                if not features[k] then
                    features[k] = v
                end
            end
        end
    end

    return features
end

function M.extract_features_from_text(text)
    local features = {}
    text = tostring(text or "")

    -- 提取经验类型
    if text:match("失败") or text:match("错误") or text:match("失败") then
        features.type = "failure"
    elseif text:match("成功") or text:match("完成") then
        features.type = "success"
    end

    -- 提取任务类型
    if text:match("代码") or text:match("编程") then
        features.task_type = "coding"
    elseif text:match("分析") then
        features.task_type = "analysis"
    elseif text:match("对话") then
        features.task_type = "conversation"
    end

    -- 提取工具名称
    local tool_match = text:match("使用[\s]*([%w_]+)[\s]*工具")
    if tool_match then
        features.tool_name = tool_match
    end

    return features
end

-- ==================== 统计信息 ====================

function M.get_stats()
    local store_stats = store.get_stats()
    local cfg = retriever_cfg()

    return {
        store_stats = store_stats,
        config = cfg
    }
end

-- ==================== 核心检索：双信号交集优先策略 ====================

--- 计算经验的相关性分数 (relevance_score)
-- 结合语义相似度和路由学习分数
-- @param experience 经验对象
-- @param context_sig 上下文签名
-- @return relevance_score
function M.compute_relevance_score(experience, context_sig)
    local score = 0.0
    local ad_cfg = adaptive_cfg()

    -- 基础语义相似度
    local semantic_sim = experience.context_similarity or experience.vector_similarity or 0.0
    score = 0.6 * semantic_sim

    -- 路由学习加成
    if ad_cfg.enabled == true then
        local route_bonus = 0.0

        -- 类型路由
        if experience.type then
            route_bonus = route_bonus + 0.1 * adaptive.get_retrieval_route_score("by_type", experience.type)
        end

        -- 任务类型路由
        if experience.task_type then
            route_bonus = route_bonus + 0.1 * adaptive.get_retrieval_route_score("by_task", experience.task_type)
        end

        -- 上下文路由
        if context_sig then
            local ctx_key = type(context_sig) == "string" and context_sig or serialize_context_signature(context_sig)
            route_bonus = route_bonus + 0.15 * adaptive.get_retrieval_route_score("by_context", ctx_key)
        end

        -- 工具路由
        if experience.tools_used then
            for tool_name in pairs(experience.tools_used) do
                route_bonus = route_bonus + 0.05 * adaptive.get_retrieval_route_score("by_tool", tool_name)
            end
        end

        score = score + 0.4 * math.max(0, math.min(1, route_bonus + 0.5))
    end

    return score
end

--- 计算经验的效用分数 (utility_score)
-- 结合语义相似度和效用学习分数
-- @param experience 经验对象
-- @return utility_score
function M.compute_utility_score(experience)
    local score = 0.0
    local ad_cfg = adaptive_cfg()

    -- 基础语义相似度
    local semantic_sim = experience.context_similarity or experience.vector_similarity or 0.0
    score = 0.5 * semantic_sim

    -- 效用学习加成
    if ad_cfg.enabled == true and experience.id then
        local utility = adaptive.get_experience_utility(experience.id)
        score = score + 0.5 * utility
    else
        -- 无学习时使用成功率作为效用代理
        local success_rate = experience.success_rate or 0.5
        score = score + 0.5 * success_rate
    end

    return score
end

--- 交集优先检索
-- 核心检索策略：先取两种信号的交集，不足则逐步放宽
-- @param query 查询条件（可为字符串或结构化表）
-- @param options 选项：limit, context_sig, task_type等
-- @return 结果列表, 检索策略信息
function M.retrieve_intersection_priority(query, options)
    options = options or {}
    local k = options.limit or 5
    local cfg = retriever_cfg()

    -- 1. 获取基础候选集（扩大范围以便排序）
    local fetch_limit = math.max(k * 4, 20)  -- 获取足够多的候选
    local candidates = store.retrieve({
        limit = fetch_limit,
        context_threshold = options.context_threshold or 0.5
    })

    if not candidates or #candidates == 0 then
        return {}, {strategy = "empty", reason = "no_candidates"}
    end

    -- 2. 提取查询特征
    local query_features = M.extract_query_features(query)
    local context_sig = options.context_sig or query_features.context_signature

    -- 3. 为每个候选计算两种信号分数
    for _, exp in ipairs(candidates) do
        exp.relevance_score = M.compute_relevance_score(exp, context_sig)
        exp.utility_score = M.compute_utility_score(exp)
    end

    -- 4. 按两种信号分别排序
    local relevance_sorted = intersection.sort_by_score(candidates, "relevance_score", true)
    local utility_sorted = intersection.sort_by_score(candidates, "utility_score", true)

    -- 5. 执行交集优先搜索
    local results, strategy = intersection.intersection_priority_search(
        relevance_sorted,
        utility_sorted,
        k,
        {
            min_needed_ratio = options.min_needed_ratio or cfg.intersection_min_ratio or 0.5,
            expand_factor = options.expand_factor or cfg.intersection_expand or 2.0,
        }
    )

    -- 6. 添加最终分数
    for _, exp in ipairs(results) do
        exp.final_score = (exp.relevance_score + exp.utility_score) / 2
    end

    return results, {
        strategy = strategy,
        candidate_count = #candidates,
        result_count = #results,
        stats = intersection.compute_stats(relevance_sorted, utility_sorted, k)
    }
end

--- 记录效用反馈（用于交集检索后的学习）
-- @param retrieved_ids 检索返回的经验ID列表
-- @param effective_ids 真正有效的经验ID集合 {id=true, ...}
function M.record_utility_feedback(retrieved_ids, effective_ids)
    local ad_cfg = adaptive_cfg()
    if ad_cfg.enabled ~= true then return end

    adaptive.update_utility_from_feedback({
        retrieved_ids = retrieved_ids,
        effective_ids = effective_ids
    })
end

--- 带反馈的检索（简化接口）
-- 返回结果和反馈函数
-- @param query 查询条件
-- @param options 选项
-- @return 结果列表, 反馈函数, 策略信息
function M.retrieve_with_feedback(query, options)
    local results, strategy_info = M.retrieve_intersection_priority(query, options)

    -- 构建反馈函数
    local retrieved_ids = {}
    for _, exp in ipairs(results) do
        retrieved_ids[#retrieved_ids + 1] = exp.id
    end

    local feedback_fn = function(effective_ids)
        M.record_utility_feedback(retrieved_ids, effective_ids or {})
    end

    return results, feedback_fn, strategy_info
end

return M
