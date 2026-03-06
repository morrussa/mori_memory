-- module/experience/adaptive.lua
-- Experience模块的自适应系统
-- 参考memory/adaptive.lua的实现，为experience添加自组织能力

local M = {}

local config = require("module.config")
local persistence = require("module.persistence")
local tool = require("module.tool")

local VERSION = "EXPAD1"

M.state = nil
M.dirty = false

-- ==================== 工具函数 ====================

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function adaptive_cfg()
    return ((config.settings or {}).experience or {}).adaptive or {}
end

local function state_file()
    local storage_cfg = ((config.settings or {}).experience or {}).storage or {}
    local root = tostring(storage_cfg.root or "memory/experience_policy")
    if root == "" then
        root = "memory/experience_policy"
    end
    return root .. "/adaptive_state.txt"
end

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function make_default_state()
    return {
        -- 策略评分系统（检索策略权重）
        strategy_scores = {
            type_weight = 0.25,
            task_weight = 0.25,
            context_weight = 0.30,
            tool_weight = 0.20
        },

        -- 成功模式学习
        success_patterns = {},
        pattern_counts = {},

        -- 上下文聚类评分（类似memory的cluster_route_score）
        context_cluster_scores = {},
        context_cluster_seen = {},

        -- [新增] 输出策略学习
        -- 记录每种输出策略的成功率，让agent学习哪种输出方式更有效
        output_strategy_scores = {},
        output_strategy_counts = {},

        -- [新增] 上下文-输出亲和度
        -- 记录特定上下文签名下哪种输出策略成功率更高
        context_output_affinity = {},

        -- [新增] 任务类型-输出策略映射
        -- 记录每种任务类型下最佳输出策略
        task_output_best = {},

        -- [新增] 检索路由评分（类似memory的cluster_route_score）
        -- 记录从哪个经验源检索成功率更高
        retrieval_route_scores = {
            by_type = {},      -- 按类型检索的评分
            by_task = {},      -- 按任务类型检索的评分
            by_context = {},   -- 按上下文检索的评分
            by_tool = {}       -- 按工具检索的评分
        },

        -- [新增] 经验效用评分（独立于route_score）
        -- 用于记录特定经验ID的效用性（是否在实际使用中有效）
        experience_utility_scores = {},  -- id -> utility_score
        experience_utility_counts = {},  -- id -> 使用次数

        -- 学习统计
        learning_events = 0,
        successful_retrievals = 0,
        failed_retrievals = 0,

        -- [新增] 输出成功率统计
        output_success_total = 0,
        output_failure_total = 0,

        -- 相似度阈值（用于上下文匹配）
        learned_context_threshold = 0.7,
    }
end

-- ==================== 状态管理 ====================

function M.reset_defaults()
    M.state = make_default_state()
    M.dirty = false
end

function M.mark_dirty()
    M.dirty = true
end

function M.load()
    M.reset_defaults()

    local path = state_file()

    if not tool.file_exists(path) then
        print("[ExperienceAdaptive] adaptive_state.txt 不存在，使用默认状态")
        return
    end

    local f = io.open(path, "r")
    if not f then
        print("[ExperienceAdaptive] adaptive_state.txt 打开失败，使用默认状态")
        return
    end

    local header = f:read("*l")
    if header ~= VERSION then
        f:close()
        print("[ExperienceAdaptive] adaptive_state.txt 版本不匹配，已回退默认状态")
        return
    end

    for line in f:lines() do
        line = tostring(line or ""):gsub("^%s*(.-)%s*$", "%1")
        if line ~= "" then
            local k, v = line:match("^([%w_]+)%s*=%s*(.+)$")
            if k and v then
                -- 处理策略权重
                if k:match("^strategy_") then
                    local weight_name = k:match("^strategy_(.+)$")
                    if weight_name and M.state.strategy_scores[weight_name] ~= nil then
                        M.state.strategy_scores[weight_name] = tonumber(v) or M.state.strategy_scores[weight_name]
                    end
                -- 处理成功模式
                elseif k == "pattern" then
                    local pattern, score, count = v:match("^([^,]+),([%+%-]?[%d%.eE]+),([%+%-]?[%d%.eE]+)$")
                    if pattern and score then
                        M.state.success_patterns[pattern] = tonumber(score) or 0
                        M.state.pattern_counts[pattern] = tonumber(count) or 0
                    end
                -- 处理上下文聚类评分
                elseif k == "context_cluster" then
                    local cluster_id, score, seen = v:match("^([%w_]+),([%+%-]?[%d%.eE]+),([%+%-]?[%d%.eE]+)$")
                    if cluster_id and score then
                        M.state.context_cluster_scores[cluster_id] = tonumber(score) or 0
                        M.state.context_cluster_seen[cluster_id] = tonumber(seen) or 0
                    end
                -- [新增] 处理输出策略评分
                elseif k == "output_strategy" then
                    local strategy, score, count = v:match("^([^,]+),([%+%-]?[%d%.eE]+),([%+%-]?[%d%.eE]+)$")
                    if strategy and score then
                        M.state.output_strategy_scores[strategy] = tonumber(score) or 0.5
                        M.state.output_strategy_counts[strategy] = tonumber(count) or 0
                    end
                -- [新增] 处理上下文-输出亲和度
                elseif k == "ctx_output" then
                    local affinity_key, score = v:match("^([^,]+),([%+%-]?[%d%.eE]+)$")
                    if affinity_key and score then
                        M.state.context_output_affinity[affinity_key] = tonumber(score) or 0.5
                    end
                -- [新增] 处理任务类型-输出策略映射
                elseif k == "task_output" then
                    local task_key, score = v:match("^([^,]+),([%+%-]?[%d%.eE]+)$")
                    if task_key and score then
                        M.state.task_output_best[task_key] = tonumber(score) or 0.5
                    end
                -- [新增] 处理检索路由评分
                elseif k == "route" then
                    local route_type, route_key, score = v:match("^([^|]+)|([^,]+),([%+%-]?[%d%.eE]+)$")
                    if route_type and route_key and score then
                        local route_scores = M.state.retrieval_route_scores[route_type]
                        if route_scores then
                            route_scores[route_key] = tonumber(score) or 0.0
                        end
                    end
                -- [新增] 处理经验效用评分
                elseif k == "exp_utility" then
                    local exp_id, score, count = v:match("^([^,]+),([%+%-]?[%d%.eE]+),([%+%-]?[%d%.eE]+)$")
                    if exp_id and score then
                        M.state.experience_utility_scores[exp_id] = tonumber(score) or 0.5
                        M.state.experience_utility_counts[exp_id] = tonumber(count) or 0
                    end
                -- 处理标量值
                else
                    local n = tonumber(v)
                    if n ~= nil and M.state[k] ~= nil then
                        M.state[k] = n
                    end
                end
            end
        end
    end

    f:close()
    M.dirty = false

    local pattern_n = 0
    for _ in pairs(M.state.success_patterns) do
        pattern_n = pattern_n + 1
    end

    local strategy_n = 0
    for _ in pairs(M.state.output_strategy_scores) do
        strategy_n = strategy_n + 1
    end

    print(string.format("[ExperienceAdaptive] 状态加载完成: patterns=%d, output_strategies=%d", pattern_n, strategy_n))
end

function M.save_to_disk()
    if not M.dirty then return true end

    local path = state_file()
    local root = path:match("^(.*)/[^/]+$")
    if root and root ~= "" then
        ensure_dir(root)
    end

    local ok, err = persistence.write_atomic(path, "w", function(f)
        local w_ok, w_err = f:write(VERSION .. "\n")
        if not w_ok then
            return false, w_err
        end

        -- 写入策略权重
        for name, weight in pairs(M.state.strategy_scores) do
            local ok_line, err_line = f:write(string.format("strategy_%s=%.10f\n", name, weight))
            if not ok_line then
                return false, err_line
            end
        end

        -- 写入成功模式
        for pattern, score in pairs(M.state.success_patterns) do
            local count = tonumber(M.state.pattern_counts[pattern]) or 0
            if count > 0 then
                local ok_pattern, err_pattern = f:write(string.format("pattern=%s,%.10f,%.2f\n", pattern, score, count))
                if not ok_pattern then
                    return false, err_pattern
                end
            end
        end

        -- 写入上下文聚类评分
        for cluster_id, score in pairs(M.state.context_cluster_scores) do
            local seen = tonumber(M.state.context_cluster_seen[cluster_id]) or 0
            if math.abs(tonumber(score) or 0) > 1e-6 or seen > 0 then
                local ok_cluster, err_cluster = f:write(string.format("context_cluster=%s,%.10f,%.2f\n", cluster_id, score, seen))
                if not ok_cluster then
                    return false, err_cluster
                end
            end
        end

        -- [新增] 写入输出策略评分
        for strategy, score in pairs(M.state.output_strategy_scores) do
            local count = tonumber(M.state.output_strategy_counts[strategy]) or 0
            if count > 0 then
                local ok_out, err_out = f:write(string.format("output_strategy=%s,%.10f,%.2f\n", strategy, score, count))
                if not ok_out then
                    return false, err_out
                end
            end
        end

        -- [新增] 写入上下文-输出亲和度
        for affinity_key, score in pairs(M.state.context_output_affinity) do
            if math.abs(tonumber(score) or 0.5) > 0.51 then
                local ok_aff, err_aff = f:write(string.format("ctx_output=%s,%.10f\n", affinity_key, score))
                if not ok_aff then
                    return false, err_aff
                end
            end
        end

        -- [新增] 写入任务类型-输出策略映射
        for task_key, score in pairs(M.state.task_output_best) do
            if math.abs(tonumber(score) or 0.5) > 0.51 then
                local ok_task, err_task = f:write(string.format("task_output=%s,%.10f\n", task_key, score))
                if not ok_task then
                    return false, err_task
                end
            end
        end

        -- [新增] 写入检索路由评分
        for route_type, route_scores in pairs(M.state.retrieval_route_scores) do
            for route_key, score in pairs(route_scores) do
                if math.abs(tonumber(score) or 0) > 1e-6 then
                    local ok_route, err_route = f:write(string.format("route=%s|%s,%.10f\n", route_type, route_key, score))
                    if not ok_route then
                        return false, err_route
                    end
                end
            end
        end

        -- [新增] 写入经验效用评分
        for exp_id, score in pairs(M.state.experience_utility_scores) do
            local count = tonumber(M.state.experience_utility_counts[exp_id]) or 0
            if count > 0 then
                local ok_util, err_util = f:write(string.format("exp_utility=%s,%.10f,%.2f\n", exp_id, score, count))
                if not ok_util then
                    return false, err_util
                end
            end
        end

        -- 写入标量值
        local scalar_keys = {
            "learning_events", "successful_retrievals", "failed_retrievals",
            "learned_context_threshold",
            "output_success_total", "output_failure_total"
        }

        for _, k in ipairs(scalar_keys) do
            local val = tonumber(M.state[k]) or 0
            local ok_line, err_line = f:write(string.format("%s=%.10f\n", k, val))
            if not ok_line then
                return false, err_line
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

-- ==================== 策略权重学习 ====================

function M.get_strategy_weights()
    local cfg = adaptive_cfg()

    -- 如果启用了自适应学习，返回学习到的权重
    if cfg.enabled == true then
        return M.state.strategy_scores
    end

    -- 否则返回配置中的默认权重
    return cfg.default_weights or M.state.strategy_scores
end

function M.update_strategy_weights(outcome)
    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local lr = cfg.learning_rate or 0.05
    local weights = M.state.strategy_scores

    -- 根据结果调整权重
    if outcome.success then
        -- 成功时增强有效策略的权重
        if outcome.type_effective then
            weights.type_weight = clamp(weights.type_weight + lr, 0.05, 0.50)
        end
        if outcome.task_effective then
            weights.task_weight = clamp(weights.task_weight + lr, 0.05, 0.50)
        end
        if outcome.context_effective then
            weights.context_weight = clamp(weights.context_weight + lr, 0.05, 0.50)
        end
        if outcome.tool_effective then
            weights.tool_weight = clamp(weights.tool_weight + lr, 0.05, 0.50)
        end
    else
        -- 失败时降低无效策略的权重
        if not outcome.type_effective then
            weights.type_weight = clamp(weights.type_weight - lr * 0.5, 0.05, 0.50)
        end
        if not outcome.task_effective then
            weights.task_weight = clamp(weights.task_weight - lr * 0.5, 0.05, 0.50)
        end
        if not outcome.context_effective then
            weights.context_weight = clamp(weights.context_weight - lr * 0.5, 0.05, 0.50)
        end
        if not outcome.tool_effective then
            weights.tool_weight = clamp(weights.tool_weight - lr * 0.5, 0.05, 0.50)
        end
    end

    -- 归一化权重
    local total = weights.type_weight + weights.task_weight + weights.context_weight + weights.tool_weight
    if total > 0 then
        weights.type_weight = weights.type_weight / total
        weights.task_weight = weights.task_weight / total
        weights.context_weight = weights.context_weight / total
        weights.tool_weight = weights.tool_weight / total
    end

    M.state.learning_events = M.state.learning_events + 1
    M.mark_dirty()
end

-- ==================== 成功模式学习 ====================

-- 记录成功模式（无衰减，长期保留学习结果）
function M.record_success_pattern(pattern_key, effectiveness)
    if not pattern_key or type(pattern_key) ~= "string" then return end

    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local lr = cfg.pattern_learning_rate or 0.1

    -- 更新模式评分（只更新目标模式，不影响其他模式）
    local current_score = M.state.success_patterns[pattern_key] or 0.5
    local new_score = current_score + lr * (effectiveness - current_score)
    M.state.success_patterns[pattern_key] = clamp(new_score, 0.0, 1.0)

    -- 更新模式使用计数
    M.state.pattern_counts[pattern_key] = (M.state.pattern_counts[pattern_key] or 0) + 1

    M.mark_dirty()
end

function M.get_pattern_score(pattern_key)
    if not pattern_key then return 0.5 end
    return tonumber(M.state.success_patterns[pattern_key]) or 0.5
end

-- ==================== 上下文聚类学习 ====================

-- 更新上下文聚类评分（无衰减，长期保留学习结果）
function M.update_context_cluster(cluster_id, effectiveness)
    if not cluster_id then return end

    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local lr = cfg.cluster_learning_rate or 0.1

    -- 更新聚类评分（只更新目标聚类，不影响其他聚类）
    local current_score = M.state.context_cluster_scores[cluster_id] or 0.0
    local new_score = current_score + lr * (effectiveness - current_score)
    M.state.context_cluster_scores[cluster_id] = clamp(new_score, -2.0, 2.0)

    -- 更新聚类访问计数
    M.state.context_cluster_seen[cluster_id] = (M.state.context_cluster_seen[cluster_id] or 0) + 1

    M.mark_dirty()
end

function M.get_cluster_score(cluster_id)
    if not cluster_id then return 0.0 end
    return tonumber(M.state.context_cluster_scores[cluster_id]) or 0.0
end

-- ==================== 阈值学习 ====================

function M.update_thresholds(positive_sims, negative_sims)
    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    if #positive_sims == 0 and #negative_sims == 0 then return end

    local lr = cfg.threshold_learning_rate or 0.05

    -- 更新上下文相似度阈值
    if #positive_sims > 0 and #negative_sims > 0 then
        local pos_floor = M.quantile(positive_sims, 0.20)
        local neg_ceil = M.quantile(negative_sims, 0.80)
        if pos_floor and neg_ceil then
            local target = 0.5 * (pos_floor + neg_ceil)
            local current = M.state.learned_context_threshold
            M.state.learned_context_threshold = current + lr * (target - current)
        end
    elseif #negative_sims > 0 then
        M.state.learned_context_threshold = M.state.learned_context_threshold + lr * 0.02
    elseif #positive_sims > 0 then
        M.state.learned_context_threshold = M.state.learned_context_threshold - lr * 0.01
    end

    M.state.learned_context_threshold = clamp(
        M.state.learned_context_threshold,
        0.4, 0.9
    )

    M.mark_dirty()
end

function M.get_context_threshold(base_threshold)
    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then
        return tonumber(base_threshold) or 0.7
    end
    return tonumber(M.state.learned_context_threshold) or tonumber(base_threshold) or 0.7
end

-- ==================== [新增] 输出策略学习 ====================

-- 记录输出策略的结果（无衰减，长期保留学习结果）
-- strategy_key: 输出策略标识，如 "structured", "narrative", "code_first", "step_by_step" 等
-- context_sig: 上下文签名
-- task_type: 任务类型
-- success: 是否成功
function M.record_output_strategy(strategy_key, context_sig, task_type, success)
    if not strategy_key then return end

    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local lr = cfg.output_learning_rate or 0.1

    -- 1. 更新全局输出策略评分（只更新目标策略）
    local current_score = M.state.output_strategy_scores[strategy_key] or 0.5
    local delta = success and 1.0 or 0.0
    local new_score = current_score + lr * (delta - current_score)
    M.state.output_strategy_scores[strategy_key] = clamp(new_score, 0.0, 1.0)
    M.state.output_strategy_counts[strategy_key] = (M.state.output_strategy_counts[strategy_key] or 0) + 1

    -- 2. 更新上下文-输出亲和度
    if context_sig then
        local ctx_key = type(context_sig) == "table" and serialize_context_signature(context_sig) or tostring(context_sig)
        local affinity_key = ctx_key .. "|" .. strategy_key
        local current_affinity = M.state.context_output_affinity[affinity_key] or 0.5
        local new_affinity = current_affinity + lr * (delta - current_affinity)
        M.state.context_output_affinity[affinity_key] = clamp(new_affinity, 0.0, 1.0)
    end

    -- 3. 更新任务类型-输出策略映射
    if task_type then
        local task_key = task_type .. "|" .. strategy_key
        local current_task_score = M.state.task_output_best[task_key] or 0.5
        local new_task_score = current_task_score + lr * (delta - current_task_score)
        M.state.task_output_best[task_key] = clamp(new_task_score, 0.0, 1.0)
    end

    -- 4. 更新统计
    if success then
        M.state.output_success_total = M.state.output_success_total + 1
    else
        M.state.output_failure_total = M.state.output_failure_total + 1
    end

    M.mark_dirty()
end

-- 获取输出策略评分
function M.get_output_strategy_score(strategy_key)
    if not strategy_key then return 0.5 end
    return tonumber(M.state.output_strategy_scores[strategy_key]) or 0.5
end

-- 获取特定上下文下最佳输出策略
function M.get_best_output_for_context(context_sig, task_type)
    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then
        return nil, 0.5
    end

    local best_strategy = nil
    local best_score = 0.5

    -- 优先检查上下文-输出亲和度
    if context_sig then
        local ctx_key = type(context_sig) == "table" and serialize_context_signature(context_sig) or tostring(context_sig)
        for affinity_key, score in pairs(M.state.context_output_affinity) do
            local ctx_prefix = affinity_key:match("^([^|]+)|")
            if ctx_prefix == ctx_key and score > best_score then
                local strat = affinity_key:match("|([^|]+)$")
                if strat then
                    best_strategy = strat
                    best_score = score
                end
            end
        end
    end

    -- 其次检查任务类型-输出策略映射
    if not best_strategy and task_type then
        for task_key, score in pairs(M.state.task_output_best) do
            local task_prefix = task_key:match("^([^|]+)|")
            if task_prefix == task_type and score > best_score then
                local strat = task_key:match("|([^|]+)$")
                if strat then
                    best_strategy = strat
                    best_score = score
                end
            end
        end
    end

    -- 最后检查全局最佳策略
    if not best_strategy then
        for strat, score in pairs(M.state.output_strategy_scores) do
            if score > best_score then
                best_strategy = strat
                best_score = score
            end
        end
    end

    return best_strategy, best_score
end

-- ==================== [新增] 检索路由学习 ====================

-- 更新检索路由评分（无衰减，长期保留学习结果）
-- route_type: "by_type", "by_task", "by_context", "by_tool"
-- route_key: 具体的路由键值
-- effectiveness: 有效性评分 (-1.0 到 1.0)
function M.update_retrieval_route(route_type, route_key, effectiveness)
    if not route_type or not route_key then return end

    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local lr = cfg.route_learning_rate or 0.1

    local route_scores = M.state.retrieval_route_scores[route_type]
    if not route_scores then
        route_scores = {}
        M.state.retrieval_route_scores[route_type] = route_scores
    end

    -- 更新路由评分（只更新目标路由，不影响其他路由）
    local current_score = route_scores[route_key] or 0.0
    local new_score = current_score + lr * (effectiveness - current_score)
    route_scores[route_key] = clamp(new_score, -2.0, 2.0)

    M.mark_dirty()
end

-- 获取检索路由评分
function M.get_retrieval_route_score(route_type, route_key)
    if not route_type or not route_key then return 0.0 end
    local route_scores = M.state.retrieval_route_scores[route_type]
    if not route_scores then return 0.0 end
    return tonumber(route_scores[route_key]) or 0.0
end

-- 根据检索结果更新路由评分
function M.update_after_retrieval(event)
    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local samples = event and event.candidate_samples or {}
    local positive_ids = event and event.positive_ids or {}
    local negative_ids = event and event.negative_ids or {}

    if #samples == 0 then return end

    -- 统计各路由的正负样本
    local route_stats = {
        by_type = {},
        by_task = {},
        by_context = {},
        by_tool = {}
    }

    for _, sample in ipairs(samples) do
        local id = sample.id
        if id then
            local is_pos = positive_ids[id] == true
            local is_neg = negative_ids[id] == true

            if is_pos or is_neg then
                local weight = is_pos and 1.0 or -1.0

                -- 按类型统计
                if sample.exp_type then
                    local stats = route_stats.by_type[sample.exp_type] or {pos=0, neg=0}
                    if is_pos then stats.pos = stats.pos + 1 else stats.neg = stats.neg + 1 end
                    route_stats.by_type[sample.exp_type] = stats
                end

                -- 按任务类型统计
                if sample.task_type then
                    local stats = route_stats.by_task[sample.task_type] or {pos=0, neg=0}
                    if is_pos then stats.pos = stats.pos + 1 else stats.neg = stats.neg + 1 end
                    route_stats.by_task[sample.task_type] = stats
                end

                -- 按上下文统计
                local context_key = sample.context_key
                if (not context_key) and sample.context_signature then
                    context_key = type(sample.context_signature) == "table"
                        and serialize_context_signature(sample.context_signature)
                        or tostring(sample.context_signature)
                end
                if context_key and context_key ~= "" then
                    local stats = route_stats.by_context[context_key] or {pos=0, neg=0}
                    if is_pos then stats.pos = stats.pos + 1 else stats.neg = stats.neg + 1 end
                    route_stats.by_context[context_key] = stats
                end

                -- 按工具统计
                if sample.tools_used then
                    for tool_name in pairs(sample.tools_used) do
                        local stats = route_stats.by_tool[tool_name] or {pos=0, neg=0}
                        if is_pos then stats.pos = stats.pos + 1 else stats.neg = stats.neg + 1 end
                        route_stats.by_tool[tool_name] = stats
                    end
                end
            end
        end
    end

    -- 更新各路由评分
    for route_type, stats_dict in pairs(route_stats) do
        for key, stats in pairs(stats_dict) do
            local total = stats.pos + stats.neg
            if total > 0 then
                local effectiveness = (stats.pos - stats.neg) / total
                M.update_retrieval_route(route_type, key, effectiveness)
            end
        end
    end

    M.state.learning_events = M.state.learning_events + 1
    M.mark_dirty()
end

-- ==================== 辅助函数 ====================

function M.quantile(values, q)
    local n = #values
    if n <= 0 then return nil end

    local copy = {}
    for i, v in ipairs(values) do
        copy[i] = v
    end
    table.sort(copy)

    if n == 1 then return copy[1] end

    local qq = clamp(tonumber(q) or 0.5, 0.0, 1.0)
    local pos = 1 + (n - 1) * qq
    local lo = math.floor(pos)
    local hi = math.ceil(pos)

    if lo == hi then return copy[lo] end

    local t = pos - lo
    return copy[lo] * (1 - t) + copy[hi] * t
end

function M.record_retrieval_outcome(success)
    if success then
        M.state.successful_retrievals = M.state.successful_retrievals + 1
    else
        M.state.failed_retrievals = M.state.failed_retrievals + 1
    end
    M.mark_dirty()
end

function M.get_stats()
    local output_strategy_n = 0
    for _ in pairs(M.state.output_strategy_scores) do
        output_strategy_n = output_strategy_n + 1
    end

    local ctx_affinity_n = 0
    for _ in pairs(M.state.context_output_affinity) do
        ctx_affinity_n = ctx_affinity_n + 1
    end

    local route_n = 0
    for _, route_scores in pairs(M.state.retrieval_route_scores) do
        for _ in pairs(route_scores) do
            route_n = route_n + 1
        end
    end

    -- [新增] 效用评分统计
    local utility_n = 0
    for _ in pairs(M.state.experience_utility_scores) do
        utility_n = utility_n + 1
    end

    local output_success_rate = 0.0
    local total_outputs = M.state.output_success_total + M.state.output_failure_total
    if total_outputs > 0 then
        output_success_rate = M.state.output_success_total / total_outputs
    end

    return {
        strategy_scores = M.state.strategy_scores,
        pattern_count = M._count_keys(M.state.success_patterns),
        cluster_count = M._count_keys(M.state.context_cluster_scores),
        learning_events = M.state.learning_events,
        successful_retrievals = M.state.successful_retrievals,
        failed_retrievals = M.state.failed_retrievals,
        retrieval_success_rate = M._compute_retrieval_rate(),
        context_threshold = M.state.learned_context_threshold,
        -- [新增] 输出策略统计
        output_strategy_count = output_strategy_n,
        ctx_affinity_count = ctx_affinity_n,
        output_success_total = M.state.output_success_total,
        output_failure_total = M.state.output_failure_total,
        output_success_rate = output_success_rate,
        -- [新增] 检索路由统计
        route_score_count = route_n,
        -- [新增] 效用评分统计
        utility_score_count = utility_n,
    }
end

function M._count_keys(tbl)
    local count = 0
    for _ in pairs(tbl) do
        count = count + 1
    end
    return count
end

function M._compute_retrieval_rate()
    local total = M.state.successful_retrievals + M.state.failed_retrievals
    if total == 0 then return 0.0 end
    return M.state.successful_retrievals / total
end

-- ==================== [新增] 经验效用学习 ====================

--- 记录经验效用反馈
-- 用于学习哪些经验在实际使用中真正有效
-- @param experience_id 经验ID
-- @param utility 效用值 (0.0 无效 ~ 1.0 有效)
function M.record_experience_utility(experience_id, utility)
    if not experience_id then return end

    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local lr = cfg.utility_learning_rate or 0.1

    -- 更新效用评分（只更新目标经验）
    local current_score = M.state.experience_utility_scores[experience_id] or 0.5
    local new_score = current_score + lr * (utility - current_score)
    M.state.experience_utility_scores[experience_id] = clamp(new_score, 0.0, 1.0)
    M.state.experience_utility_counts[experience_id] = (M.state.experience_utility_counts[experience_id] or 0) + 1

    M.mark_dirty()
end

--- 获取经验效用评分
-- @param experience_id 经验ID
-- @return 效用评分 (默认 0.5)
function M.get_experience_utility(experience_id)
    if not experience_id then return 0.5 end
    return tonumber(M.state.experience_utility_scores[experience_id]) or 0.5
end

function M.get_experience_utility_count(experience_id)
    if not experience_id then return 0 end
    return tonumber(M.state.experience_utility_counts[experience_id]) or 0
end

--- 批量获取经验效用评分
-- @param experience_ids 经验ID列表
-- @return id -> utility_score 映射
function M.get_utility_scores_batch(experience_ids)
    local result = {}
    for _, id in ipairs(experience_ids or {}) do
        result[id] = M.get_experience_utility(id)
    end
    return result
end

--- 更新经验效用（基于检索后的使用反馈）
-- @param event 包含 retrieved_ids (检索返回的ID列表) 和 effective_ids (真正有效的ID集合)
function M.update_utility_from_feedback(event)
    local cfg = adaptive_cfg()
    if cfg.enabled ~= true then return end

    local retrieved_ids = event and event.retrieved_ids or {}
    local effective_ids = event and event.effective_ids or {}

    for _, exp_id in ipairs(retrieved_ids) do
        local is_effective = effective_ids[exp_id] == true
        M.record_experience_utility(exp_id, is_effective and 1.0 or 0.0)
    end

    M.mark_dirty()
end

-- 初始化
M.reset_defaults()

return M
