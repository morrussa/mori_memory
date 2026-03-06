-- module/experience/intersection.lua
-- 交集优先检索策略模块
-- 独立设计，可被experience和memory模块复用

local M = {}

-- ==================== 配置 ====================

-- 默认配置
local DEFAULT_CONFIG = {
    min_needed_ratio = 0.5,    -- 最少需要 k * ratio 个结果
    expand_factor = 2.0,       -- 扩展因子 (top_2k)
    max_expand_factor = 4.0,   -- 最大扩展因子
}

-- ==================== 核心算法 ====================

--- 构建ID到排名的映射
-- @param list 排序后的列表，每个元素必须有 id 字段
-- @param max_rank 只考虑前 max_rank 个元素
-- @return id -> rank 的映射表
local function build_rank_map(list, max_rank)
    local rank_map = {}
    local limit = math.min(#list, max_rank or #list)
    for i = 1, limit do
        local item = list[i]
        if item and item.id then
            rank_map[item.id] = i
        end
    end
    return rank_map
end

--- 计算两个列表的交集
-- @param rank_a 第一个列表的排名映射
-- @param rank_b 第二个列表的排名映射
-- @return 交集中所有ID的集合
local function compute_intersection(rank_a, rank_b)
    local intersection = {}
    for id, rank in pairs(rank_a) do
        if rank_b[id] then
            intersection[id] = {
                rank_a = rank,
                rank_b = rank_b[id]
            }
        end
    end
    return intersection
end

--- 计算两个列表的并集
-- @param rank_a 第一个列表的排名映射
-- @param rank_b 第二个列表的排名映射
-- @return 并集中所有ID的集合
local function compute_union(rank_a, rank_b)
    local union = {}
    -- 添加来自A的所有元素
    for id, rank in pairs(rank_a) do
        union[id] = {
            rank_a = rank,
            rank_b = rank_b[id]  -- 可能为nil
        }
    end
    -- 添加来自B但不在A中的元素
    for id, rank in pairs(rank_b) do
        if not union[id] then
            union[id] = {
                rank_a = nil,
                rank_b = rank
            }
        end
    end
    return union
end

--- 交集优先搜索
-- @param relevance_list 按relevance排序的候选列表（高分在前）
-- @param utility_list 按utility排序的候选列表（高分在前）
-- @param k 需要返回的数量
-- @param options 配置选项
-- @return 结果列表, 使用的策略 ("narrow", "wide", "union")
function M.intersection_priority_search(relevance_list, utility_list, k, options)
    options = options or {}
    local cfg = {
        min_needed_ratio = options.min_needed_ratio or DEFAULT_CONFIG.min_needed_ratio,
        expand_factor = options.expand_factor or DEFAULT_CONFIG.expand_factor,
        max_expand_factor = options.max_expand_factor or DEFAULT_CONFIG.max_expand_factor,
    }

    local min_needed = math.max(1, math.floor(k * cfg.min_needed_ratio))

    -- 构建排名映射（用最大范围构建一次）
    local max_range = math.floor(k * cfg.max_expand_factor)
    local rel_rank = build_rank_map(relevance_list, max_range)
    local util_rank = build_rank_map(utility_list, max_range)

    -- Phase 1: 窄交集 (top_k ∩ top_k)
    local rel_top_k = build_rank_map(relevance_list, k)
    local util_top_k = build_rank_map(utility_list, k)
    local narrow_intersection = compute_intersection(rel_top_k, util_top_k)

    local narrow_count = 0
    for _ in pairs(narrow_intersection) do
        narrow_count = narrow_count + 1
    end

    if narrow_count >= min_needed then
        -- 窄交集足够，直接返回
        return M._build_result(narrow_intersection, relevance_list, utility_list, k), "narrow"
    end

    -- Phase 2: 宽交集 (top_2k ∩ top_2k)
    local wide_k = math.floor(k * cfg.expand_factor)
    local rel_top_wide = build_rank_map(relevance_list, wide_k)
    local util_top_wide = build_rank_map(utility_list, wide_k)
    local wide_intersection = compute_intersection(rel_top_wide, util_top_wide)

    local wide_count = 0
    for _ in pairs(wide_intersection) do
        wide_count = wide_count + 1
    end

    if wide_count >= min_needed then
        -- 宽交集足够
        return M._build_result(wide_intersection, relevance_list, utility_list, k), "wide"
    end

    -- Phase 3: 并集降级 (top_k ∪ top_k)
    local union_set = compute_union(rel_top_k, util_top_k)
    return M._build_result(union_set, relevance_list, utility_list, k), "union"
end

--- 构建最终结果列表
-- @param id_set ID集合 {id -> {rank_a, rank_b}}
-- @param relevance_list relevance排序列表
-- @param utility_list utility排序列表
-- @param k 返回数量限制
-- @return 排序后的结果列表
function M._build_result(id_set, relevance_list, utility_list, k)
    -- 构建ID到原始数据的映射
    local rel_map = {}
    for _, item in ipairs(relevance_list) do
        if item and item.id then
            rel_map[item.id] = item
        end
    end

    local util_map = {}
    for _, item in ipairs(utility_list) do
        if item and item.id then
            util_map[item.id] = item
        end
    end

    -- 构建结果列表
    local results = {}
    for id, ranks in pairs(id_set) do
        -- 合并数据，优先使用relevance_list的数据作为基础
        local base_item = rel_map[id] or util_map[id] or {id = id}
        local item = {}
        for k, v in pairs(base_item) do
            item[k] = v
        end

        -- 添加排名信息
        item.relevance_rank = ranks.rank_a
        item.utility_rank = ranks.rank_b

        -- 计算综合排名分数（排名越小越好，转换为分数）
        local rel_score = ranks.rank_a and (1.0 / (ranks.rank_a + 1)) or 0
        local util_score = ranks.rank_b and (1.0 / (ranks.rank_b + 1)) or 0
        item.intersection_score = 0.5 * rel_score + 0.5 * util_score

        -- 标记来源
        item.in_intersection = ranks.rank_a and ranks.rank_b
        item.from_relevance = ranks.rank_a ~= nil
        item.from_utility = ranks.rank_b ~= nil

        results[#results + 1] = item
    end

    -- 排序：
    -- 1. 交集优先（双信号确认）
    -- 2. 按综合排名分数排序
    table.sort(results, function(a, b)
        -- 交集元素优先
        if a.in_intersection and not b.in_intersection then return true end
        if not a.in_intersection and b.in_intersection then return false end
        -- 同类型按综合分数排序
        return (a.intersection_score or 0) > (b.intersection_score or 0)
    end)

    -- 限制返回数量
    if #results > k then
        local trimmed = {}
        for i = 1, k do
            trimmed[i] = results[i]
        end
        return trimmed
    end

    return results
end

-- ==================== 辅助函数 ====================

--- 按指定分数字段排序列表
-- @param list 待排序列表
-- @param score_field 分数字段名
-- @param descending 是否降序（默认true，高分在前）
-- @return 排序后的新列表
function M.sort_by_score(list, score_field, descending)
    descending = descending ~= false  -- 默认降序

    local sorted = {}
    for i, item in ipairs(list) do
        sorted[i] = item
    end

    if descending then
        table.sort(sorted, function(a, b)
            return (a[score_field] or 0) > (b[score_field] or 0)
        end)
    else
        table.sort(sorted, function(a, b)
            return (a[score_field] or 0) < (b[score_field] or 0)
        end)
    end

    return sorted
end

--- 计算两个列表的统计信息
-- @param relevance_list relevance排序列表
-- @param utility_list utility排序列表
-- @param k 计算范围
-- @return 统计信息表
function M.compute_stats(relevance_list, utility_list, k)
    local rel_rank = build_rank_map(relevance_list, k)
    local util_rank = build_rank_map(utility_list, k)

    local intersection = compute_intersection(rel_rank, util_rank)

    local intersection_count = 0
    for _ in pairs(intersection) do
        intersection_count = intersection_count + 1
    end

    return {
        relevance_count = #relevance_list,
        utility_count = #utility_list,
        intersection_count = intersection_count,
        intersection_ratio = k > 0 and (intersection_count / k) or 0,
        k = k
    }
end

return M
