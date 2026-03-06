local util = require("module.graph.util")
local config = require("module.config")
local context_manager = require("module.graph.context_manager")
local project_knowledge = require("module.graph.project_knowledge")

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

local function summarize_working_memory(state)
    local active_task = ((((state or {}).session or {}).active_task) or {})
    local memory = ((state or {}).working_memory) or {}
    local read_count = 0
    local written_count = 0
    for _, _ in pairs(memory.files_read_set or {}) do
        read_count = read_count + 1
    end
    for _, _ in pairs(memory.files_written_set or {}) do
        written_count = written_count + 1
    end

    local lines = {
        "[ActiveTask]",
        string.format("goal=%s", tostring(active_task.goal or "")),
        string.format("status=%s", tostring(active_task.status or "")),
        string.format("profile=%s", tostring(active_task.profile or "")),
    }
    if util.trim(active_task.carryover_summary or "") ~= "" then
        lines[#lines + 1] = string.format("carryover=%s", tostring(active_task.carryover_summary))
    end
    lines[#lines + 1] = "[WorkingMemory]"
    lines[#lines + 1] = string.format("current_plan=%s", tostring(memory.current_plan or ""))
    lines[#lines + 1] = string.format("plan_step_index=%s", tostring(memory.plan_step_index or 0))
    lines[#lines + 1] = string.format("files_read=%d", read_count)
    lines[#lines + 1] = string.format("files_written=%d", written_count)
    if util.trim(memory.last_tool_batch_summary or "") ~= "" then
        lines[#lines + 1] = "last_tool_batch:"
        lines[#lines + 1] = util.utf8_take(tostring(memory.last_tool_batch_summary), 800)
    end
    if util.trim(memory.last_repair_error or "") ~= "" then
        lines[#lines + 1] = string.format("last_repair_error=%s", tostring(memory.last_repair_error))
    end
    return table.concat(lines, "\n")
end

local function compose_system_prompt(base_system_prompt, state)
    local context = ((state or {}).context) or {}
    local lines = { tostring(base_system_prompt or "") }

    local pk_overview = project_knowledge.get_project_knowledge(state)
    if util.trim(pk_overview or "") ~= "" then
        lines[#lines + 1] = ""
        lines[#lines + 1] = pk_overview
    end

    lines[#lines + 1] = summarize_working_memory(state)

    if util.trim((context or {}).memory_context or "") ~= "" then
        lines[#lines + 1] = "[MemoryContext]"
        lines[#lines + 1] = tostring(context.memory_context)
    end
    local policy_context = util.trim((context or {}).policy_context or (context or {}).experience_context or "")
    if policy_context ~= "" then
        lines[#lines + 1] = "[PolicyHints]"
        lines[#lines + 1] = tostring(policy_context)
    end
    if util.trim((context or {}).tool_context or "") ~= "" then
        lines[#lines + 1] = "[ToolContext]"
        lines[#lines + 1] = tostring(context.tool_context)
    end
    if util.trim((context or {}).planner_context or "") ~= "" then
        lines[#lines + 1] = "[PlannerContext]"
        lines[#lines + 1] = tostring(context.planner_context)
    end

    -- 添加预算警告（如果有）
    if (context or {})._budget_warning then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "[System Note: " .. context._budget_warning .. "]"
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

-- 压缩历史对话对（简单版本：只保留关键信息）
local function compress_history_pair(pair, max_chars)
    max_chars = max_chars or 200
    local user_text = util.trim(pair.user or "")
    local assistant_text = util.trim(pair.assistant or "")

    -- 如果内容不长，直接返回
    if #user_text + #assistant_text <= max_chars then
        return pair
    end

    -- 压缩：保留用户问题的主要部分，压缩回答
    local compressed_user = util.utf8_take(user_text, math.floor(max_chars * 0.4))
    local compressed_assistant = util.utf8_take(assistant_text, math.floor(max_chars * 0.4))

    return {
        turn = pair.turn,
        user = compressed_user .. (#user_text > #compressed_user and "..." or ""),
        assistant = compressed_assistant .. (#assistant_text > #compressed_assistant and "..." or ""),
        _compressed = true,
    }
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

    local system_prompt = compose_system_prompt(base_system_prompt, state)
    local user_input = tostring((((state or {}).input or {}).message) or "")

    local pairs = extract_history_pairs(conversation_history)
    local kept = {}
    local dropped = {}
    local compressed = 0

    local messages = build_messages(system_prompt, user_input, kept)
    local total_tokens = count_tokens(messages)

    -- 从最近的对话开始保留（更相关）
    for i = #pairs, 1, -1 do
        local candidate_pair = pairs[i]

        -- 尝试加入候选对话
        local candidate = { candidate_pair }
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
            -- 尝试压缩后再加入
            local compressed_pair = compress_history_pair(candidate_pair, 150)
            local compressed_candidate = { compressed_pair }
            for k = 1, #kept do
                compressed_candidate[#compressed_candidate + 1] = kept[k]
            end
            local compressed_messages = build_messages(system_prompt, user_input, compressed_candidate)
            local compressed_tokens = count_tokens(compressed_messages)

            if compressed_tokens <= token_budget then
                kept = compressed_candidate
                messages = compressed_messages
                total_tokens = compressed_tokens
                compressed = compressed + 1
            else
                -- 完全丢弃
                dropped[#dropped + 1] = pairs[i]
            end
        end
    end

    -- 对过长的tool消息进行优化
    local optimized_messages, opt_stats = context_manager.optimize_runtime_messages(
        messages,
        4000 -- 单条tool消息最大4000字符
    )

    if total_tokens > token_budget then
        -- 最后的补救：直接截断过长的消息
        for i, msg in ipairs(optimized_messages) do
            if msg.role == "tool" and #(msg.content or "") > 2000 then
                optimized_messages[i].content = util.utf8_take(msg.content, 2000)
                    .. "\n[Truncated for context budget]"
            end
        end
        total_tokens = count_tokens(optimized_messages)
    end

    if total_tokens > token_budget then
        -- 记录警告但不报错，让系统继续运行
        print(string.format("[GraphContext] Warning: token budget exceeded total=%d budget=%d", total_tokens, token_budget))
    end

    return optimized_messages, {
        token_budget = token_budget,
        total_tokens = total_tokens,
        kept_pairs = #kept,
        dropped_pairs = #dropped,
        compressed_pairs = compressed,
        optimized_messages = opt_stats.truncated_count,
    }
end

return M
