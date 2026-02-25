-- recall.lua
-- 自动记忆召回模块
-- 优化：引入 LLM 生成搜索关键词，进行多路向量检索，合并轮次

local M = {}

local tool = require("module.tool")
local memory = require("module.memory")
local history = require("module.history")
local config = require("module.config")

-- 定义与 history.lua 一致的分隔符
local FIELD_SEP = "\x1F"
local NEWLINE_REPLACE = "\x1E"

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

-- 【新增】使用 LLM 生成搜索关键词（原子事实预测）
-- @param user_input: 用户当前输入
-- @return table: 关键词字符串列表
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
        temperature = 0.3, -- 低温度保证输出格式稳定
        seed = 42
    }

    local result_str = py_pipeline:generate_chat_sync(messages, params)
    result_str = tool.remove_cot(result_str)
    
    -- 解析 Lua table
    local keywords = {}
    -- 提取 {...} 部分
    local table_str = result_str:match("{.*}")
    if table_str then
        local load_ok, load_func = pcall(load, "return " .. table_str)
        if load_ok and load_func then
            local call_ok, tbl = pcall(load_func)
            if call_ok and type(tbl) == "table" then
                keywords = tbl
            end
        end
    end

    if #keywords > 0 then
        print("[Recall] LLM 生成的搜索关键词: " .. table.concat(keywords, ", "))
    else
        print("[Recall] LLM 未能生成关键词，将仅使用原文搜索")
    end

    return keywords
end

-- 检索相关记忆并格式化返回（重写）
local function retrieve(user_input, user_vec)
    user_vec = user_vec or tool.get_embedding(user_input)

    -- ========== 1. 获取搜索向量集合 ==========
    -- 原始用户输入向量
    local query_vectors = { { vec = user_vec, weight = 1.0 } } 

    -- LLM 生成关键词 -> 获取向量
    local keywords = generate_search_keywords(user_input)
    for _, kw in ipairs(keywords) do
        local kw_vec = tool.get_embedding(kw)
        table.insert(query_vectors, { vec = kw_vec, weight = 0.8 }) -- 关键词权重可以略低于原文
    end

    -- ========== 2. 多路搜索 & 合并 Turn 分数 ==========
    local cfg = config.settings.ai_query
    local max_mem = cfg.max_memory or 5
    local turn_scores = {} -- key=turn, value=最高分

    for _, query_item in ipairs(query_vectors) do
        local q_vec = query_item.vec
        local weight = query_item.weight

        -- 搜索相似记忆行
        local sim_results = tool.find_sim_all_heat(q_vec)
        if sim_results then
            for i, mem in ipairs(sim_results) do
                if i > max_mem then break end -- 每路搜索只取前 N 个

                local mem_line = mem.index
                local mem_sim = mem.similarity * weight -- 应用权重
                local mem_data = memory.memories[mem_line]

                if mem_data and mem_data.turns then
                    -- 将该记忆行关联的轮次得分累加或更新
                    for _, turn in ipairs(mem_data.turns) do
                        local current_score = turn_scores[turn] or 0
                        -- 这里采用累加逻辑：多次命中视为更重要
                        turn_scores[turn] = current_score + mem_sim
                    end
                end
            end
        end
    end

    -- ========== 3. 排序与筛选 ==========
    local sorted_turns = {}
    for turn, score in pairs(turn_scores) do
        table.insert(sorted_turns, {turn = turn, score = score})
    end
    table.sort(sorted_turns, function(a, b) return a.score > b.score end)

    -- 限制最终轮次数
    local max_turns = cfg.max_turns or 10
    if #sorted_turns > max_turns then
        local limited = {}
        for i = 1, max_turns do
            limited[i] = sorted_turns[i]
        end
        sorted_turns = limited
    end

    if #sorted_turns == 0 then
        return ""
    end

    -- ========== 4. 获取对话内容并拼接 ==========
    local memory_text_lines = {}
    for _, item in ipairs(sorted_turns) do
        local turn = item.turn
        local entry = history.get_by_turn(turn)
        if entry then
            -- 解析逻辑（兼容新旧格式）
            local user_part, ai_part = entry:match("^(.-)" .. FIELD_SEP .. "(.*)$")
            if user_part and ai_part then
                ai_part = ai_part:gsub(NEWLINE_REPLACE, "\n")
                table.insert(memory_text_lines, string.format("第%d轮 用户：%s\n助手：%s", turn, user_part, ai_part))
            else
                local old_user, old_ai = entry:match("^user:(.+)ai:(.+)$")
                if old_user and old_ai then
                    old_ai = old_ai:gsub("\x1F", "\n")
                    table.insert(memory_text_lines, string.format("第%d轮 用户：%s\n助手：%s", turn, old_user, old_ai))
                else
                    table.insert(memory_text_lines, string.format("第%d轮 %s", turn, entry))
                end
            end
        end
    end

    if #memory_text_lines == 0 then
        return ""
    end

    local result = "【相关记忆】\n" .. table.concat(memory_text_lines, "\n\n")
    return result
end

-- 组合函数
function M.check_and_retrieve(user_input, user_vec)
    if need_recall(user_input, user_vec) then
        return retrieve(user_input, user_vec)
    end
    return ""
end

return M