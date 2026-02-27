-- recall.lua
-- 自动记忆召回模块（已集成 Topic 过滤 - 你的核心需求）
-- 修改点：
-- 1. 新增 require("module.topic")
-- 2. 在 retrieve() 的打分环节加入 Topic 加权过滤（同话题强烈 boost，跨话题大幅 penalty）
-- 3. 使用 config.settings.ai_query 可配置参数（已兼容旧 config，未改 config.lua 也可直接跑）
-- 4. 加入安全防护：如果 topic.get_topic_for_turn 不存在则优雅降级

local M = {}

local tool = require("module.tool")
local memory = require("module.memory")
local history = require("module.history")
local config = require("module.config")
local cluster = require("module.cluster")
local topic = require("module.topic")   -- 【新增】用于 topic 过滤

-- 定义与 history.lua 一致的分隔符
-- 预定义的概念句子（用于计算语气向量）
local ANXIETY_SENTENCES = {
    "我很焦虑", "我急死了", "快点", "急急急", "我现在很着急"
}

local HELP_CRY_SENTENCES = {
    "救命", "帮帮我", "求助", "救救我", "help"
}

local PAST_TALK_SENTENCES = {
    "之前", "过去", "曾经", "上一次", "以前", "recall", "remember"
}

-- 向量缓存
local anxiety_vec = nil
local help_cry_vec = nil
local past_talk_vec = nil

-- 辅助函数：根据一组文本计算平均向量并归一化
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

-- 统一初始化所有语气向量
function M.init_all_sentiment_vectors()
    anxiety_vec = compute_average_vector(ANXIETY_SENTENCES)
    help_cry_vec = compute_average_vector(HELP_CRY_SENTENCES)
    past_talk_vec = compute_average_vector(PAST_TALK_SENTENCES)
    return anxiety_vec, help_cry_vec, past_talk_vec
end

-- 计算回忆分数（保留原有逻辑）
local function compute_recall_score(user_input, user_vec)
    local score = 0
    local cfg = config.settings.ai_query

    -- 1. 历史/过去关键词匹配
    local past_keywords = {"之前", "上次", "以前", "过去", "曾经", "recall", "remember"}
    for _, kw in ipairs(past_keywords) do
        if user_input:find(kw, 1, true) then
            score = score + cfg.history_search_bonus
            break
        end
    end

    -- 2. 专业术语关键词匹配
    local tech_keywords = {"代码", "函数", "API", "算法", "编程", "Python", "Lua", "配置", "参数"}
    for _, kw in ipairs(tech_keywords) do
        if user_input:find(kw, 1, true) then
            score = score + cfg.technical_term_bonus
            break
        end
    end

    -- 3. 长度加分
    if #user_input >= cfg.length_limit then
        score = score + cfg.length_bonus
    end

    -- 4. 语气检测
    if anxiety_vec and #anxiety_vec > 0 and user_vec and #user_vec > 0 then
        local sim = tool.cosine_similarity(user_vec, anxiety_vec)
        score = score + sim * cfg.anxiety_multi
    end

    if help_cry_vec and #help_cry_vec > 0 and user_vec and #user_vec > 0 then
        local sim = tool.cosine_similarity(user_vec, help_cry_vec)
        score = score + sim * cfg.help_cry_multi
    end

    if past_talk_vec and #past_talk_vec > 0 and user_vec and #user_vec > 0 then
        local sim = tool.cosine_similarity(user_vec, past_talk_vec)
        score = score + sim * cfg.past_talk_multi
    end

    return score
end

-- 判断是否需要触发记忆召回
local function need_recall(user_input, user_vec)
    user_vec = user_vec or tool.get_embedding(user_input)
    local score = compute_recall_score(user_input, user_vec)
    local cfg = config.settings.ai_query
    local threshold = cfg.recall_base
    print(string.format("[Recall] 回忆分数 = %.2f (阈值 %.2f)", score, threshold))
    return score >= threshold
end

-- 使用 LLM 生成搜索关键词（原子事实预测）
local function generate_search_keywords(user_input)
    if not py_pipeline then
        return {}
    end

    local prompt = string.format([[
用户提问：%s
请推测用户想要回忆起过去的哪些具体内容。输出3-5个最可能的或原子事实。
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
        max_tokens = 512,
        temperature = 0.3,
        seed = 42
    }

    local result_str = py_pipeline:generate_chat_sync(messages, params)
    result_str = tool.remove_cot(result_str)

    local keywords = {}
    local max_keywords = tonumber(((config.settings or {}).ai_query or {}).max_keywords) or 4
    if max_keywords < 1 then max_keywords = 4 end

    result_str = tostring(result_str or ""):gsub("%s+", " ")
    result_str = result_str:match("^%s*(.-)%s*$")

    local parsed, err = tool.parse_lua_string_array_strict(result_str, {
        max_items = max_keywords,
        max_item_chars = 64,
        must_full = true,
    })
    if parsed then
        local seen = {}
        for _, kw in ipairs(parsed) do
            local k = tostring(kw or ""):match("^%s*(.-)%s*$")
            if k ~= "" and k ~= "无" and not seen[k] then
                seen[k] = true
                table.insert(keywords, k)
            end
        end
    else
        print(string.format("[Recall] 关键词输出格式非法，已丢弃: %s", tostring(err)))
    end

    if #keywords > 0 then
        print("[Recall] LLM 生成的搜索关键词: " .. table.concat(keywords, ", "))
    else
        print("[Recall] LLM 未能生成关键词，将仅使用原文搜索")
    end

    return keywords
end

local function retrieve(user_input, user_vec)
    user_vec = user_vec or tool.get_embedding(user_input)

    -- 1. 查询向量集合（主查询权重更高）
    local query_vectors = {
        { vec = user_vec, weight = 1.00, is_primary = true }   -- 主查询
    }
    local keywords = generate_search_keywords(user_input)
    for _, kw in ipairs(keywords) do
        local kw_vec = tool.get_embedding(kw)
        table.insert(query_vectors, { vec = kw_vec, weight = 0.55, is_primary = false })
    end

    -- 2. 多路搜索 → Max-Pooling + 非线性压制
    local cfg = config.settings.ai_query
    local max_mem = cfg.max_memory or 5
    local MIN_SIM_GATE = cfg.min_sim_gate or 0.58
    local POWER = cfg.power_suppress or 1.80

    local turn_best = {}   -- turn → 最高有效分数

    for _, q in ipairs(query_vectors) do
        local q_vec = q.vec
        local weight = q.weight

        -- 簇内优先搜索（原有逻辑）
        local sim_results = {}
        local best_id, best_sim = cluster.find_best_cluster(q_vec)
        if best_id and best_sim >= (config.settings.cluster.cluster_sim or 0.75) then
            sim_results = cluster.find_sim_in_cluster(q_vec, best_id, { only_hot = true })
        else
            sim_results = memory.find_similar_all_fast(q_vec)  -- 热区全扫
        end

        for i, mem in ipairs(sim_results) do
            if i > max_mem then break end
            if mem.similarity < MIN_SIM_GATE then break end

            -- 非线性压制 + 权重
            local effective = (mem.similarity ^ POWER) * weight

            local mem_data = memory.memories[mem.index]
            if mem_data and mem_data.turns then
                for _, turn in ipairs(mem_data.turns) do
                    if not turn_best[turn] or effective > turn_best[turn] then
                        turn_best[turn] = effective
                    end
                end
            end
        end
    end

    -- 3. 转成候选列表（不做 topic 乘分，避免语义自激）
    local sorted_turns = {}
    for turn, score in pairs(turn_best) do
        table.insert(sorted_turns, {turn = turn, score = score})
    end
    table.sort(sorted_turns, function(a, b) return a.score > b.score end)

    -- 4. Topic 用作“检索范围约束”：same -> near -> cross
    local current_topic_info = topic.get_topic_for_turn and topic.get_topic_for_turn(history.get_turn() + 1) or nil
    local topic_gate = topic.get_topic_for_turn ~= nil and current_topic_info ~= nil
    local same_topic = {}
    local near_topic = {}
    local cross_topic = {}

    local sim_th = cfg.topic_sim_threshold or 0.7
    local cross_ratio = tonumber(cfg.topic_cross_quota_ratio) or 0.25
    if cross_ratio < 0 then cross_ratio = 0 end
    if cross_ratio > 0.5 then cross_ratio = 0.5 end

    local function push_bucket(item)
        if not topic_gate then
            table.insert(same_topic, item)
            return
        end

        local ti = topic.get_topic_for_turn(item.turn)
        if not ti then
            table.insert(cross_topic, item)
            return
        end

        if ti.is_active and current_topic_info.is_active then
            table.insert(same_topic, item)
            return
        end

        if (not ti.is_active) and (not current_topic_info.is_active)
            and ti.topic_idx and current_topic_info.topic_idx
            and ti.topic_idx == current_topic_info.topic_idx then
            table.insert(same_topic, item)
            return
        end

        if ti.centroid and current_topic_info.centroid then
            local ts = tool.cosine_similarity(ti.centroid, current_topic_info.centroid)
            if ts >= sim_th then
                table.insert(near_topic, item)
            else
                table.insert(cross_topic, item)
            end
            return
        end

        table.insert(cross_topic, item)
    end

    for _, item in ipairs(sorted_turns) do
        push_bucket(item)
    end

    local max_turns = cfg.max_turns or 10
    local reserved_cross = topic_gate and math.min(#cross_topic, math.floor(max_turns * cross_ratio)) or 0
    local selected = {}
    local used = {}

    local function append_until(src, limit)
        for _, item in ipairs(src) do
            if #selected >= limit then break end
            if not used[item.turn] then
                used[item.turn] = true
                table.insert(selected, item)
            end
        end
    end

    local in_topic_budget = max_turns - reserved_cross
    append_until(same_topic, in_topic_budget)
    append_until(near_topic, in_topic_budget)
    append_until(cross_topic, max_turns)

    -- 回填：若某个桶不足，允许其他桶补齐到 max_turns
    if #selected < max_turns then
        append_until(near_topic, max_turns)
        append_until(same_topic, max_turns)
        append_until(cross_topic, max_turns)
    end

    table.sort(selected, function(a, b) return a.score > b.score end)
    if #selected > max_turns then
        for i = #selected, max_turns + 1, -1 do
            table.remove(selected, i)
        end
    end
    print(string.format(
        "[Recall] 候选分桶: same=%d near=%d cross=%d | 选中=%d",
        #same_topic, #near_topic, #cross_topic, #selected
    ))

    -- 6. 拼接最终记忆文本（保持不变）
    local memory_text_lines = {}
    for _, item in ipairs(selected) do
        local entry = history.get_by_turn(item.turn)
        if entry then
            local user_part, ai_part = history.parse_entry(entry)
            if user_part then
                table.insert(memory_text_lines, string.format("第%d轮 用户：%s\n助手：%s", item.turn, user_part, ai_part))
            end
        end
    end

    return #memory_text_lines > 0 and "【相关记忆】\n" .. table.concat(memory_text_lines, "\n\n") or ""
end

-- 组合函数
function M.check_and_retrieve(user_input, user_vec)
    if need_recall(user_input, user_vec) then
        return retrieve(user_input, user_vec)
    end
    return ""
end

return M
