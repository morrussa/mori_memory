
-- module/experience/retriever.lua
-- 经验检索器：智能检索和排序agent经验

local M = {}

local store = require("module.experience.store")
local adaptive = require("module.experience.adaptive")
local tool = require("module.tool")
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

-- ==================== 检索策略 ====================

-- 基础检索：按类型和任务类型检索
function M.retrieve_by_type_and_task(exp_type, task_type, options)
    options = options or {}

    local results = store.retrieve({
        type = exp_type,
        task_type = task_type,
        limit = options.limit or 10
    })

    return M.rerank_by_recency(results, options)
end

-- 上下文感知检索：考虑当前上下文
function M.retrieve_by_context(context_signature, options)
    options = options or {}

    local results = store.retrieve({
        context_signature = context_signature,
        context_threshold = options.context_threshold or 0.7,
        limit = options.limit or 10
    })

    -- 按上下文相似度排序
    table.sort(results, function(a, b)
        return (a.context_similarity or 0) > (b.context_similarity or 0)
    end)

    return results
end

-- 工具使用检索：查找相关工具使用经验
function M.retrieve_by_tool(tool_name, options)
    options = options or {}

    local results = store.retrieve({
        tool_used = tool_name,
        limit = options.limit or 10
    })

    return M.rerank_by_success_rate(results, options)
end

-- 混合检索：综合多种因素
function M.retrieve_hybrid(query, options)
    options = options or {}

    -- 提取查询特征
    local query_features = M.extract_query_features(query)

    -- 收集候选经验
    local candidates = {}
    local seen_ids = {}

    -- 1. 按类型检索
    if query_features.type then
        local by_type = store.retrieve({
            type = query_features.type,
            limit = options.limit or 5
        })
        for _, exp in ipairs(by_type) do
            if not seen_ids[exp.id] then
                candidates[#candidates + 1] = exp
                seen_ids[exp.id] = true
            end
        end
    end

    -- 2. 按任务类型检索
    if query_features.task_type then
        local by_task = store.retrieve({
            task_type = query_features.task_type,
            limit = options.limit or 5
        })
        for _, exp in ipairs(by_task) do
            if not seen_ids[exp.id] then
                candidates[#candidates + 1] = exp
                seen_ids[exp.id] = true
            end
        end
    end

    -- 3. 按上下文检索
    if query_features.context_signature then
        local by_context = store.retrieve({
            context_signature = query_features.context_signature,
            context_threshold = options.context_threshold or 0.6,
            limit = options.limit or 5
        })
        for _, exp in ipairs(by_context) do
            if not seen_ids[exp.id] then
                candidates[#candidates + 1] = exp
                seen_ids[exp.id] = true
            end
        end
    end

    -- 4. 按工具使用检索
    if query_features.tool_name then
        local by_tool = store.retrieve({
            tool_used = query_features.tool_name,
            limit = options.limit or 5
        })
        for _, exp in ipairs(by_tool) do
            if not seen_ids[exp.id] then
                candidates[#candidates + 1] = exp
                seen_ids[exp.id] = true
            end
        end
    end

    -- 如果没有指定任何特征，返回所有经验
    if #candidates == 0 then
        candidates = store.retrieve({limit = options.limit or 10})
    end

    -- 计算综合评分
    for _, exp in ipairs(candidates) do
        exp.composite_score = M.compute_composite_score(
            exp, 
            query_features,
            options
        )
    end

    -- 按综合评分排序
    table.sort(candidates, function(a, b)
        return (a.composite_score or 0) > (b.composite_score or 0)
    end)

    -- 限制返回数量
    local limit = options.limit or 5
    if #candidates > limit then
        local result = {}
        for i = 1, limit do
            result[i] = candidates[i]
        end
        candidates = result
    end

    return candidates
end

-- ==================== 评分和重排序 ====================

-- 计算综合评分（集成自适应权重，无时间衰减）
function M.compute_composite_score(experience, query_features, options)
    local cfg = retriever_cfg()
    local ad_cfg = adaptive_cfg()

    -- 使用自适应学习的权重（如果启用）
    local weights
    if ad_cfg.enabled == true then
        local adaptive_weights = adaptive.get_strategy_weights()
        weights = {
            context_similarity = adaptive_weights.context_weight or 0.40,
            success_rate = adaptive_weights.type_weight or 0.25,
            route_score = 0.35
        }
    else
        weights = options.weights or cfg.weights or {
            context_similarity = 0.40,
            success_rate = 0.25,
            route_score = 0.35
        }
    end

    local score = 0.0

    -- 上下文相似度
    if experience.context_similarity then
        score = score + weights.context_similarity * experience.context_similarity
    end

    -- 成功率
    if experience.success_rate then
        score = score + weights.success_rate * experience.success_rate
    end

    -- 检索路由评分加成（基于历史学习）
    if ad_cfg.enabled == true and experience.type then
        local route_score = adaptive.get_retrieval_route_score("by_type", experience.type)
        if route_score ~= 0 then
            score = score + weights.route_score * 0.25 * route_score
        end
    end

    if ad_cfg.enabled == true and experience.task_type then
        local route_score = adaptive.get_retrieval_route_score("by_task", experience.task_type)
        if route_score ~= 0 then
            score = score + weights.route_score * 0.25 * route_score
        end
    end

    -- 上下文路由评分
    if ad_cfg.enabled == true and experience.context_signature then
        local ctx_key = serialize_context_signature(experience.context_signature)
        local route_score = adaptive.get_retrieval_route_score("by_context", ctx_key)
        if route_score ~= 0 then
            score = score + weights.route_score * 0.25 * route_score
        end
    end

    -- 工具路由评分
    if ad_cfg.enabled == true and experience.tools_used then
        for tool_name in pairs(experience.tools_used) do
            local route_score = adaptive.get_retrieval_route_score("by_tool", tool_name)
            if route_score ~= 0 then
                score = score + weights.route_score * 0.25 * route_score
            end
        end
    end

    return score
end

-- 按时间重排序（无衰减，直接按时间戳排序）
function M.rerank_by_recency(experiences, options)
    options = options or {}
    local ascending = options.ascending or false

    for _, exp in ipairs(experiences) do
        if exp.created_at then
            exp.recency_score = exp.created_at
        else
            exp.recency_score = 0
        end
    end

    table.sort(experiences, function(a, b)
        if ascending then
            return (a.recency_score or 0) < (b.recency_score or 0)
        else
            return (a.recency_score or 0) > (b.recency_score or 0)
        end
    end)

    return experiences
end

-- 按成功率重排序
function M.rerank_by_success_rate(experiences, options)
    for _, exp in ipairs(experiences) do
        if exp.success_rate then
            exp.success_score = exp.success_rate
        elseif exp.outcome and exp.outcome.success then
            exp.success_score = 1.0
        else
            exp.success_score = 0.0
        end
    end

    table.sort(experiences, function(a, b)
        return (a.success_score or 0) > (b.success_score or 0)
    end)

    return experiences
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

-- ==================== 失败案例检索 ====================

-- 检查当前方案是否会导致类似失败
function M.check_potential_failure(proposed_solution, current_context, options)
    options = options or {}

    -- 提取当前上下文特征
    local current_sig = M.extract_context_signature(current_context)

    -- 检索相似上下文下的失败案例
    local failures = store.retrieve({
        type = "failure",
        context_signature = current_sig,
        context_threshold = options.context_threshold or 0.7,
        limit = options.limit or 10
    })

    -- 分析每个失败案例
    local risk_assessment = {}
    for _, failure in ipairs(failures) do
        local risk = M.assess_failure_risk(
            proposed_solution,
            failure,
            current_context
        )

        if risk.risk_level > 0 then
            risk_assessment[#risk_assessment + 1] = risk
        end
    end

    -- 按风险等级排序
    table.sort(risk_assessment, function(a, b)
        return a.risk_level > b.risk_level
    end)

    return risk_assessment
end

function M.extract_context_signature(context)
    local sig = {}

    if type(context) == "table" then
        -- 语言
        sig.language = context.language or "unknown"

        -- 领域
        sig.domain = context.domain or "general"

        -- 任务类型
        sig.task_category = context.task_category or "general"

        -- 环境
        if context.environment then
            sig.environment = context.environment
        end
    end

    return sig
end

function M.assess_failure_risk(proposed_solution, failure, current_context)
    local risk = {
        failure_id = failure.id,
        risk_level = 0.0,
        reasons = {},
        suggestions = {}
    }

    -- 1. 检查上下文相似度
    if failure.context_similarity then
        risk.risk_level = risk.risk_level + failure.context_similarity * 0.4
        table.insert(risk.reasons, string.format(
            "上下文相似度: %.2f", failure.context_similarity
        ))
    end

    -- 2. 检查解决方案相似度
    local solution_sim = M.compute_solution_similarity(
        proposed_solution,
        failure
    )
    if solution_sim > 0.5 then
        risk.risk_level = risk.risk_level + solution_sim * 0.6
        table.insert(risk.reasons, string.format(
            "解决方案相似度: %.2f", solution_sim
        ))
    end

    -- 3. 提取建议
    if failure.lessons then
        for _, lesson in ipairs(failure.lessons) do
            if lesson.type == "solution" and lesson.content then
                table.insert(risk.suggestions, lesson.content)
            end
        end
    end

    return risk
end

function M.compute_solution_similarity(proposed_solution, failure)
    -- 简单实现：比较工具使用
    local sim = 0.0

    if type(proposed_solution) == "table" and 
       type(failure) == "table" and
       failure.tools_used then

        local proposed_tools = proposed_solution.tools or {}
        local common_tools = 0
        local total_tools = 0

        for tool, _ in pairs(proposed_tools) do
            total_tools = total_tools + 1
            if failure.tools_used[tool] then
                common_tools = common_tools + 1
            end
        end

        if total_tools > 0 then
            sim = common_tools / total_tools
        end
    end

    return sim
end

-- ==================== 成功案例检索 ====================

-- 检索成功案例以指导当前任务
function M.retrieve_success_cases(task_description, current_context, options)
    options = options or {}

    -- 提取任务特征
    local task_features = M.extract_query_features({
        text = task_description,
        context = current_context
    })

    -- 检索成功案例
    local successes = store.retrieve({
        type = "success",
        task_type = task_features.task_type,
        context_signature = task_features.context_signature,
        context_threshold = options.context_threshold or 0.6,
        limit = options.limit or 5
    })

    -- 按成功率排序
    successes = M.rerank_by_success_rate(successes, options)

    -- 提取可复用的模式
    local reusable_patterns = {}
    for _, success in ipairs(successes) do
        if success.patterns then
            for _, pattern in ipairs(success.patterns) do
                if pattern.type == "tool_usage" and pattern.tool then
                    reusable_patterns[#reusable_patterns + 1] = {
                        tool = pattern.tool,
                        frequency = pattern.frequency,
                        source = success.id
                    }
                end
            end
        end
    end

    return {
        successes = successes,
        reusable_patterns = reusable_patterns
    }
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

-- ==================== [新增] 检索后学习反馈 ====================

-- 记录检索结果用于自适应学习
-- results: 检索结果列表
-- positive_ids: 成功匹配的经验ID集合 {id=true, ...}
-- negative_ids: 失败匹配的经验ID集合 {id=true, ...}
function M.record_retrieval_feedback(results, positive_ids, negative_ids)
    local ad_cfg = adaptive_cfg()
    if ad_cfg.enabled ~= true then return end

    if not results or #results == 0 then return end
    positive_ids = positive_ids or {}
    negative_ids = negative_ids or {}

    -- 构建候选样本
    local samples = {}
    for _, exp in ipairs(results) do
        if exp and exp.id then
            samples[#samples + 1] = {
                id = exp.id,
                exp_type = exp.type,
                task_type = exp.task_type,
                tools_used = exp.tools_used,
                sim = exp.context_similarity or exp.vector_similarity or 0
            }
        end
    end

    -- 更新检索路由评分
    adaptive.update_after_retrieval({
        candidate_samples = samples,
        positive_ids = positive_ids,
        negative_ids = negative_ids
    })

    -- 记录检索成功率
    local has_positive = false
    for _ in pairs(positive_ids) do
        has_positive = true
        break
    end
    adaptive.record_retrieval_outcome(has_positive)
end

-- 获取推荐输出策略
-- context_sig: 上下文签名
-- task_type: 任务类型
-- 返回: 推荐策略名称, 置信度
function M.get_recommended_output_strategy(context_sig, task_type)
    local ad_cfg = adaptive_cfg()
    if ad_cfg.enabled ~= true then
        return nil, 0.5
    end
    return adaptive.get_best_output_for_context(context_sig, task_type)
end

-- 记录输出结果
-- strategy: 输出策略名称
-- context_sig: 上下文签名
-- task_type: 任务类型
-- success: 是否成功
function M.record_output_result(strategy, context_sig, task_type, success)
    local ad_cfg = adaptive_cfg()
    if ad_cfg.enabled ~= true then return end
    adaptive.record_output_strategy(strategy, context_sig, task_type, success)
end

return M
