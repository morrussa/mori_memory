local M = {}

local tool = require("module.tool")
local history = require("module.memory.history")
local config = require("module.config")
local topic = require("module.memory.topic")

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

local function trim(text)
    return tostring(text or ""):match("^%s*(.-)%s*$")
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

local function unique_sorted_memories(memories)
    local seen = {}
    local out = {}
    for _, mem_idx in ipairs(memories or {}) do
        local idx = tonumber(mem_idx)
        if idx and idx > 0 and not seen[idx] then
            seen[idx] = true
            out[#out + 1] = idx
        end
    end
    table.sort(out)
    return out
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
                sum[i] = (sum[i] or 0.0) + (tonumber(vec[i]) or 0.0)
            end
        end
    end
    local norm = 0.0
    for i = 1, #(sum or {}) do
        norm = norm + (tonumber(sum[i]) or 0.0) * (tonumber(sum[i]) or 0.0)
    end
    norm = math.sqrt(norm)
    if norm > 0.0 then
        for i = 1, #sum do
            sum[i] = (tonumber(sum[i]) or 0.0) / norm
        end
    end
    return sum or {}
end

function M.init_all_sentiment_vectors()
    anxiety_vec = compute_average_vector(ANXIETY_SENTENCES)
    help_cry_vec = compute_average_vector(HELP_CRY_SENTENCES)
    past_talk_vec = compute_average_vector(PAST_TALK_SENTENCES)
    return anxiety_vec, help_cry_vec, past_talk_vec
end

local function compute_recall_score(user_input, user_vec)
    local cfg = ai_cfg()
    local score = 0.0

    local past_keywords = {"之前", "上次", "以前", "过去", "曾经", "recall", "remember"}
    for _, kw in ipairs(past_keywords) do
        if user_input:find(kw, 1, true) then
            score = score + (tonumber(cfg.history_search_bonus) or 0.0)
            break
        end
    end

    local tech_keywords = {"代码", "函数", "API", "算法", "编程", "Python", "Lua", "配置", "参数"}
    for _, kw in ipairs(tech_keywords) do
        if user_input:find(kw, 1, true) then
            score = score + (tonumber(cfg.technical_term_bonus) or 0.0)
            break
        end
    end

    if #user_input >= (tonumber(cfg.length_limit) or 20) then
        score = score + (tonumber(cfg.length_bonus) or 0.0)
    end

    if anxiety_vec and #anxiety_vec > 0 and user_vec and #user_vec > 0 then
        score = score + (tonumber(tool.cosine_similarity(user_vec, anxiety_vec)) or 0.0) * (tonumber(cfg.anxiety_multi) or 0.0)
    end
    if help_cry_vec and #help_cry_vec > 0 and user_vec and #user_vec > 0 then
        score = score + (tonumber(tool.cosine_similarity(user_vec, help_cry_vec)) or 0.0) * (tonumber(cfg.help_cry_multi) or 0.0)
    end
    if past_talk_vec and #past_talk_vec > 0 and user_vec and #user_vec > 0 then
        score = score + (tonumber(tool.cosine_similarity(user_vec, past_talk_vec)) or 0.0) * (tonumber(cfg.past_talk_multi) or 0.0)
    end

    return score
end

local function need_recall(user_input, user_vec, _current_turn)
    local cfg = ai_cfg()
    local score = compute_recall_score(user_input, user_vec)
    local threshold = tonumber(cfg.recall_base) or 5.3

    local lower = user_input:lower()
    if lower:find("不要回忆", 1, true) or lower:find("不查记忆", 1, true) then
        score = score - (tonumber(cfg.suppress_recall_penalty) or 8.0)
    end

    if #user_input < (tonumber(cfg.short_query_penalty_len) or 8)
        and not (user_input:find("继续", 1, true) or user_input:find("刚才", 1, true)) then
        score = score - (tonumber(cfg.short_query_penalty) or 0.65)
    end

    return score >= threshold, score
end

local function empty_recall_result(current_anchor, score)
    return {
        context = "",
        score = score,
        topic_anchor = tostring(current_anchor or ""),
        predicted_topics = {},
        predicted_memories = {},
        predicted_nodes = {},
        selected_turns = {},
        selected_memories = {},
        fragments = {},
        adopted_memories = {},
    }
end

function M.check_and_retrieve(user_input, user_vec, _opts)
    local topic_graph = require("module.memory.topic_graph")
    user_input = trim(user_input)
    user_vec = user_vec or tool.get_embedding_query(user_input)

    local current_turn = (history.get_turn() or 0) + 1
    local current_anchor = topic.get_stable_anchor and topic.get_stable_anchor(current_turn) or nil
    if trim(current_anchor) == "" and topic.get_topic_anchor then
        current_anchor = topic.get_topic_anchor(current_turn)
    end
    current_anchor = trim(current_anchor)

    local should_recall, score = need_recall(user_input, user_vec, current_turn)
    if not should_recall then
        return empty_recall_result(current_anchor, score)
    end

    local result = topic_graph.retrieve(user_vec, current_anchor, current_turn, {
        user_input = user_input,
    })
    if type(result) ~= "table" then
        result = {}
    end
    result.score = score
    result.topic_anchor = trim(result.topic_anchor or current_anchor)
    result.predicted_topics = shallow_copy_array(result.predicted_topics or {})
    result.predicted_memories = shallow_copy_array(result.predicted_memories or {})
    result.predicted_nodes = shallow_copy_array(result.predicted_nodes or {})
    result.selected_turns = shallow_copy_array(result.selected_turns or {})
    result.selected_memories = shallow_copy_array(result.selected_memories or {})
    result.fragments = result.fragments or {}
    result.adopted_memories = shallow_copy_array(result.adopted_memories or {})
    result.context = tostring(result.context or "")
    return result
end

function M.infer_adopted_memories(final_text, recall_state)
    final_text = trim(final_text)
    recall_state = type(recall_state) == "table" and recall_state or {}

    local fragments = type(recall_state.fragments) == "table" and recall_state.fragments or {}
    if final_text == "" or #fragments <= 0 then
        return {}
    end

    local embed = tool.get_embedding_passage or tool.get_embedding
    local final_vec = embed and embed(final_text) or nil
    if type(final_vec) ~= "table" or #final_vec <= 0 then
        return {}
    end

    local cfg = ai_cfg()
    local min_sim = tonumber(cfg.topic_activation_feedback_min_sim) or 0.58
    local topn = math.max(1, math.floor(tonumber(cfg.topic_activation_feedback_topn) or 3))
    local margin = math.max(0.0, tonumber(cfg.topic_activation_feedback_margin) or 0.05)
    local ranked = {}

    for _, fragment in ipairs(fragments) do
        local mem_idx = tonumber((fragment or {}).mem_idx)
        local assistant_text = trim((fragment or {}).assistant)
        local full_text = trim((fragment or {}).text)
        local base_text = assistant_text ~= "" and assistant_text or full_text
        if mem_idx and base_text ~= "" then
            local base_vec = embed and embed(base_text) or nil
            if type(base_vec) == "table" and #base_vec > 0 then
                local sim = tonumber(tool.cosine_similarity(final_vec, base_vec)) or 0.0
                if full_text ~= "" and full_text ~= base_text then
                    local full_vec = embed(full_text)
                    if type(full_vec) == "table" and #full_vec > 0 then
                        sim = math.max(sim, (tonumber(tool.cosine_similarity(final_vec, full_vec)) or 0.0) * 0.97)
                    end
                end
                ranked[#ranked + 1] = {
                    mem_idx = mem_idx,
                    similarity = sim,
                }
            end
        end
    end

    table.sort(ranked, function(a, b)
        if (a.similarity or 0.0) ~= (b.similarity or 0.0) then
            return (a.similarity or 0.0) > (b.similarity or 0.0)
        end
        return (a.mem_idx or 0) < (b.mem_idx or 0)
    end)

    local out = {}
    local seen = {}
    local best = ranked[1] and (tonumber(ranked[1].similarity) or 0.0) or 0.0
    for i = 1, math.min(topn, #ranked) do
        local item = ranked[i]
        local sim = tonumber(item.similarity) or 0.0
        if sim >= min_sim or (best > 0 and sim >= math.max(min_sim - 0.04, best - margin)) then
            local mem_idx = tonumber(item.mem_idx)
            if mem_idx and not seen[mem_idx] then
                seen[mem_idx] = true
                out[#out + 1] = mem_idx
            end
        end
    end

    return unique_sorted_memories(out)
end

return M
