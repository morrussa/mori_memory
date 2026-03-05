local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function default_graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function count_tokens(messages)
    local n = py_pipeline:count_chat_tokens(messages)
    return math.max(0, math.floor(tonumber(n) or 0))
end

local function extract_history_pairs(conversation_history)
    local out = {}
    if type(conversation_history) ~= "table" then
        return out
    end

    local idx = 2
    local turn = 0
    while idx <= #conversation_history do
        local user_msg = conversation_history[idx]
        local assistant_msg = conversation_history[idx + 1]
        if type(user_msg) == "table" and tostring(user_msg.role or "") == "user" then
            turn = turn + 1
            local pair = {
                turn = turn,
                user = tostring(user_msg.content or ""),
                assistant = "",
            }
            if type(assistant_msg) == "table" and tostring(assistant_msg.role or "") == "assistant" then
                pair.assistant = tostring(assistant_msg.content or "")
                idx = idx + 2
            else
                idx = idx + 1
            end
            out[#out + 1] = pair
        else
            idx = idx + 1
        end
    end

    return out
end

local function compose_system_prompt(base_system_prompt, context)
    local lines = { tostring(base_system_prompt or "") }
    if util.trim((context or {}).memory_context or "") ~= "" then
        lines[#lines + 1] = "[MemoryContext]"
        lines[#lines + 1] = tostring(context.memory_context)
    end
    if util.trim((context or {}).tool_context or "") ~= "" then
        lines[#lines + 1] = "[ToolContext]"
        lines[#lines + 1] = tostring(context.tool_context)
    end
    return table.concat(lines, "\n\n")
end

local function build_messages(system_prompt, user_input, history_pairs)
    local msgs = {
        { role = "system", content = tostring(system_prompt or "") },
    }
    for _, pair in ipairs(history_pairs or {}) do
        if util.trim(pair.user or "") ~= "" then
            msgs[#msgs + 1] = { role = "user", content = tostring(pair.user or "") }
        end
        if util.trim(pair.assistant or "") ~= "" then
            msgs[#msgs + 1] = { role = "assistant", content = tostring(pair.assistant or "") }
        end
    end
    msgs[#msgs + 1] = { role = "user", content = tostring(user_input or "") }
    return msgs
end

function M.build_chat_messages(state)
    local graph_cfg = default_graph_cfg()
    local token_budget = math.max(256, math.floor(tonumber(graph_cfg.input_token_budget) or 12000))

    local conversation_history = (((state or {}).messages or {}).conversation_history) or {}
    local base_system_prompt = (((state or {}).messages or {}).system_prompt) or ""
    if util.trim(base_system_prompt) == "" and type(conversation_history[1]) == "table" then
        if tostring(conversation_history[1].role or "") == "system" then
            base_system_prompt = tostring(conversation_history[1].content or "")
        end
    end

    local system_prompt = compose_system_prompt(base_system_prompt, (state or {}).context or {})
    local user_input = tostring((((state or {}).input or {}).message) or "")

    local pairs = extract_history_pairs(conversation_history)
    local kept = {}
    local dropped = 0

    local messages = build_messages(system_prompt, user_input, kept)
    local total_tokens = count_tokens(messages)

    for i = #pairs, 1, -1 do
        local candidate = { pairs[i] }
        for k = 1, #kept do
            candidate[#candidate + 1] = kept[k]
        end
        local candidate_messages = build_messages(system_prompt, user_input, candidate)
        local candidate_tokens = count_tokens(candidate_messages)
        if candidate_tokens <= token_budget then
            kept = candidate
            messages = candidate_messages
            total_tokens = candidate_tokens
        else
            dropped = dropped + 1
        end
    end

    if total_tokens > token_budget then
        error(string.format("[GraphContext] token budget exceeded total=%d budget=%d", total_tokens, token_budget))
    end

    return messages, {
        token_budget = token_budget,
        total_tokens = total_tokens,
        kept_pairs = #kept,
        dropped_pairs = dropped,
    }
end

return M
