local M = {}

local config = require("module.config")

local function trim(s)
    if s == nil then
        return ""
    end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function to_int(v, fallback, min_v)
    local n = tonumber(v)
    if not n then
        n = tonumber(fallback) or 0
    end
    n = math.floor(n)
    if min_v and n < min_v then
        n = min_v
    end
    return n
end

local function append_unique(arr, value)
    for _, v in ipairs(arr) do
        if v == value then
            return
        end
    end
    arr[#arr + 1] = value
end

local function extract_history_pairs(conversation_history)
    local out = {}
    if type(conversation_history) ~= "table" then
        return out
    end

    local idx = 2
    while idx <= #conversation_history do
        local user_msg = conversation_history[idx]
        local assistant_msg = conversation_history[idx + 1]

        local pair = nil
        if type(user_msg) == "table" and tostring(user_msg.role or "") == "user" then
            pair = {
                user = tostring(user_msg.content or ""),
                assistant = "",
            }
            if type(assistant_msg) == "table" and tostring(assistant_msg.role or "") == "assistant" then
                pair.assistant = tostring(assistant_msg.content or "")
                idx = idx + 2
            else
                idx = idx + 1
            end
        else
            idx = idx + 1
        end

        if pair then
            out[#out + 1] = pair
        end
    end
    return out
end

local function count_messages_tokens(messages)
    local n = py_pipeline:count_chat_tokens(messages)
    return to_int(n, 0, 0)
end

local function build_messages(system_prompt, user_input, history_pairs, blocks)
    local msgs = {
        { role = "system", content = system_prompt },
    }

    -- 注入顺序保持与旧链路一致：plan -> tool -> memory
    if blocks.plan_bom and blocks.plan_bom ~= "" then
        msgs[#msgs + 1] = { role = "system", content = blocks.plan_bom }
    end
    if blocks.tool_context and blocks.tool_context ~= "" then
        msgs[#msgs + 1] = { role = "system", content = blocks.tool_context }
    end
    if blocks.memory_context and blocks.memory_context ~= "" then
        msgs[#msgs + 1] = { role = "system", content = blocks.memory_context }
    end

    for _, pair in ipairs(history_pairs or {}) do
        if trim(pair.user) ~= "" then
            msgs[#msgs + 1] = { role = "user", content = tostring(pair.user) }
        end
        if trim(pair.assistant) ~= "" then
            msgs[#msgs + 1] = { role = "assistant", content = tostring(pair.assistant) }
        end
    end

    msgs[#msgs + 1] = { role = "user", content = user_input }
    return msgs
end

function M.build_messages(opts)
    opts = opts or {}
    local agent_cfg = (config.settings or {}).agent or {}

    local budget = to_int(
        opts.input_token_budget,
        agent_cfg.input_token_budget or 12000,
        128
    )

    local user_input = trim(opts.user_input)
    if user_input == "" then
        error("[ContextWindow] user_input is empty")
    end

    local system_prompt = trim(opts.system_prompt)
    local history_src = opts.conversation_history
    if system_prompt == "" and type(history_src) == "table" and type(history_src[1]) == "table" then
        if tostring(history_src[1].role or "") == "system" then
            system_prompt = tostring(history_src[1].content or "")
        end
    end
    if system_prompt == "" then
        error("[ContextWindow] system prompt missing")
    end

    local blocks = {
        plan_bom = trim(opts.plan_bom),
        tool_context = trim(opts.tool_context),
        memory_context = trim(opts.memory_context),
    }

    local dropped_blocks = {}
    local function drop_block(name)
        if blocks[name] and blocks[name] ~= "" then
            blocks[name] = ""
            append_unique(dropped_blocks, name)
            return true
        end
        return false
    end

    local base_messages = build_messages(system_prompt, user_input, {}, blocks)
    local total_tokens = count_messages_tokens(base_messages)

    if total_tokens > budget then
        -- 固定顺序：memory -> tool -> plan
        local order = { "memory_context", "tool_context", "plan_bom" }
        for _, name in ipairs(order) do
            if total_tokens <= budget then
                break
            end
            local changed = drop_block(name)
            if changed then
                base_messages = build_messages(system_prompt, user_input, {}, blocks)
                total_tokens = count_messages_tokens(base_messages)
            end
        end
    end

    if total_tokens > budget then
        error(string.format(
            "[ContextWindow] required context exceeds budget: total=%d budget=%d",
            total_tokens,
            budget
        ))
    end

    local all_pairs = extract_history_pairs(history_src)
    local kept_pairs = {}
    local kept_history_pairs = 0
    local dropped_history_pairs = 0
    local final_messages = base_messages

    for i = #all_pairs, 1, -1 do
        local pair = all_pairs[i]
        local candidate_pairs = { pair }
        for k = 1, #kept_pairs do
            candidate_pairs[#candidate_pairs + 1] = kept_pairs[k]
        end

        local candidate_messages = build_messages(system_prompt, user_input, candidate_pairs, blocks)
        local candidate_tokens = count_messages_tokens(candidate_messages)
        if candidate_tokens <= budget then
            kept_pairs = candidate_pairs
            kept_history_pairs = kept_history_pairs + 1
            final_messages = candidate_messages
            total_tokens = candidate_tokens
        else
            dropped_history_pairs = dropped_history_pairs + 1
        end
    end

    local meta = {
        total_tokens = total_tokens,
        kept_history_pairs = kept_history_pairs,
        dropped_history_pairs = dropped_history_pairs,
        dropped_blocks = dropped_blocks,
        budget = budget,
    }

    return final_messages, meta
end

return M
