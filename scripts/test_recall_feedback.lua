package.path = "./?.lua;./?/init.lua;" .. package.path

package.loaded["module.config"] = {
    settings = {
        ai_query = {
            topic_activation_feedback_min_sim = 0.58,
            topic_activation_feedback_topn = 2,
            topic_activation_feedback_margin = 0.05,
        },
    },
}

local function cosine(a, b)
    local dot = 0.0
    local na = 0.0
    local nb = 0.0
    local dim = math.max(#(a or {}), #(b or {}))
    for i = 1, dim do
        local va = tonumber((a or {})[i]) or 0.0
        local vb = tonumber((b or {})[i]) or 0.0
        dot = dot + va * vb
        na = na + va * va
        nb = nb + vb * vb
    end
    if na <= 0.0 or nb <= 0.0 then
        return 0.0
    end
    return dot / math.sqrt(na * nb)
end

local function embed(text)
    text = tostring(text or "")
    if text:find("Lua", 1, true) then
        return { 1.0, 0.0 }
    end
    if text:find("Python", 1, true) then
        return { 0.0, 1.0 }
    end
    return { 0.6, 0.4 }
end

package.loaded["module.tool"] = {
    cosine_similarity = cosine,
    get_embedding_passage = embed,
    get_embedding_query = embed,
    get_embeddings_query = function() return {} end,
    remove_cot = function(text) return text end,
    parse_lua_string_array_strict = function() return {}, nil end,
}

package.loaded["module.memory.store"] = {
    get_total_lines = function() return 0 end,
    begin_turn = function() end,
}

package.loaded["module.memory.history"] = {
    get_turn = function() return 0 end,
}

package.loaded["module.memory.ghsom"] = {}
package.loaded["module.memory.topic"] = {}
package.loaded["module.memory.adaptive"] = {
    get_merge_limit = function(v) return v end,
    update_after_recall = function() end,
    add_counter = function() end,
}

package.loaded["module.memory.topic_predictor"] = {
    predict = function() return { lines = {}, ranked_nodes = {} } end,
}

local recall = require("module.memory.recall")

local adopted = recall.infer_adopted_memories("最终回答里明确提到了 Lua table 的处理方式", {
    fragments = {
        { mem_idx = 11, assistant = "之前你提过 Lua table 的结构", text = "Lua table" },
        { mem_idx = 12, assistant = "之前聊过 Python list 的遍历", text = "Python list" },
    },
})

assert(#adopted == 1, "should only adopt the best-aligned memory fragment")
assert(adopted[1] == 11, "Lua fragment should be selected as adopted memory")

print("test_recall_feedback: ok")
