local util = require("module.graph.util")
local tool_registry = require("module.graph.tool_registry_v2")
local config = require("module.config")
local context_manager = require("module.graph.context_manager")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local READ_EVIDENCE_TOOLS = {
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
}

-- 工具结果大小警告阈值
local TOOL_RESULT_WARN_CHARS = 3000
local TOOL_RESULT_HARD_MAX_CHARS = 8000

local function build_tool_message(row)
    local ok = row.ok == true
    local content = ""
    if ok then
        content = tostring(row.result or "")
    else
        local err = util.trim(row.error or "tool_exec_failed")
        content = string.format("Error: %s\n Please fix your mistakes.", err)
    end

    return {
        role = "tool",
        name = tostring(row.tool or ""),
        tool_call_id = tostring(row.call_id or ""),
        content = content,
        status = ok and "success" or "error",
    }
end

local function merge_tool_context(state, fragments)
    state.context = state.context or {}

    -- 使用context_manager进行智能合并
    local existing = util.trim((((state or {}).context or {}).tool_context) or "")
    local merged = context_manager.merge_tool_results(
        state.tool_exec.results or {},
        existing
    )

    -- 如果context_manager没有产生结果，使用原有逻辑
    if merged == "" then
        merged = table.concat(fragments or {}, "\n\n")
        if existing ~= "" and merged ~= "" then
            merged = existing .. "\n\n" .. merged
        elseif existing ~= "" then
            merged = existing
        end
    end

    local max_chars = math.max(120, math.floor(tonumber((((graph_cfg() or {}).tools or {}).file_context_max_chars) or 6000)))
    state.context.tool_context = util.trim(util.utf8_take(merged, max_chars))
end

function M.run(state, _ctx)
    state.agent_loop = state.agent_loop or {
        remaining_steps = 25,
        pending_tool_calls = {},
        stop_reason = "",
        iteration = 0,
    }
    state.messages = state.messages or {}
    state.messages.runtime_messages = state.messages.runtime_messages or {}
    state.tool_exec = state.tool_exec or {
        loop_count = 0,
        executed = 0,
        failed = 0,
        executed_total = 0,
        failed_total = 0,
        read_evidence_total = 0,
        results = {},
        all_results = {},
        context_fragments = {},
        truncated_count = 0,
        total_result_chars = 0,
        read_files = {},  -- 记录所有已读取的文件路径
    }

    local calls = state.agent_loop.pending_tool_calls or {}

    -- DEBUG
    print(string.format("[ToolsNode][DEBUG] executing %d tool calls", #calls))
    for i, call in ipairs(calls) do
        print(string.format("[ToolsNode][DEBUG] call[%d]: %s", i, tostring(call.tool or "unknown")))
    end

    local result = tool_registry.execute_calls(calls)
    local prior_stop_reason = util.trim(state.agent_loop.stop_reason)

    -- DEBUG
    print(string.format("[ToolsNode][DEBUG] result: executed=%d failed=%d control_action=%s",
        tonumber(result.executed) or 0,
        tonumber(result.failed) or 0,
        tostring(result.control_action or "nil")
    ))

    state.tool_exec.loop_count = (tonumber(state.tool_exec.loop_count) or 0) + 1
    state.tool_exec.executed = tonumber(result.executed) or 0
    state.tool_exec.failed = tonumber(result.failed) or 0
    state.tool_exec.executed_total = (tonumber(state.tool_exec.executed_total) or 0) + (tonumber(result.executed) or 0)
    state.tool_exec.failed_total = (tonumber(state.tool_exec.failed_total) or 0) + (tonumber(result.failed) or 0)
    state.tool_exec.read_evidence_total = tonumber(state.tool_exec.read_evidence_total) or 0
    state.tool_exec.truncated_count = 0
    state.tool_exec.total_result_chars = 0

    -- 处理控制信号
    local control_action = result.control_action
    local control_data = result.control_data
    
    -- 如果收到 finish_turn 信号，设置最终消息和停止原因
    if control_action == "finish" then
        state.agent_loop.stop_reason = "finish_turn_called"
        if result.final_message and result.final_message ~= "" then
            state.final_response = state.final_response or {}
            state.final_response.message = result.final_message
            state.final_response.from_control_tool = true
        end
    end
    
    -- 如果收到 continue_task 信号，确保循环继续
    -- (通过不清除 pending_tool_calls 的方式让 agent_node 在下一轮处理)
    -- continue_task 的理由会被添加到下一个消息中

    -- 处理工具结果，进行大小优化
    local processed_results = {}
    local processed_fragments = {}
    for _, row in ipairs(result.call_results or {}) do
        -- 跳过控制工具的结果（它们不产生文件上下文）
        if row.is_control then
            processed_results[#processed_results + 1] = {
                call_id = row.call_id,
                tool = row.tool,
                args = row.args,
                ok = row.ok,
                error = row.error,
                result = row.result,
                is_control = true,
            }
        else
            local processed_row = {
                call_id = row.call_id,
                tool = row.tool,
                args = row.args,
                ok = row.ok,
                error = row.error,
                result = row.result,
                original_chars = #(tostring(row.result or "")),
                was_truncated = false,
                from_cache = false,
            }

            if row.ok == true and row.result then
                -- 使用context_manager处理结果
                local processed_text, from_cache, original_len = context_manager.process_tool_result(
                    row.tool,
                    row.args,
                    row.result
                )
                processed_row.result = processed_text
                processed_row.from_cache = from_cache
                processed_row.original_chars = original_len
                processed_row.was_truncated = (original_len > #processed_text)

                if processed_row.was_truncated then
                    state.tool_exec.truncated_count = (state.tool_exec.truncated_count or 0) + 1
                end

                -- 更新context fragment
                local tool_name = tostring(row.tool or "")
                local is_read_evidence = READ_EVIDENCE_TOOLS[tool_name]
                local read_path = tostring((row.args or {}).path or "")
                
                -- DEBUG: 打印工具检查详情
                print(string.format("[ToolsNode][DEBUG] tool=%s is_read_evidence=%s path=%s",
                    tool_name, tostring(is_read_evidence), read_path))
                
                if is_read_evidence then
                    state.tool_exec.read_evidence_total = state.tool_exec.read_evidence_total + 1
                    processed_fragments[#processed_fragments + 1] = string.format(
                        "[Tool:%s]\n%s",
                        row.tool,
                        processed_text
                    )
                    -- 记录已读取的文件路径
                    if read_path ~= "" then
                        state.tool_exec.read_files = state.tool_exec.read_files or {}
                        -- 使用统一的路径标准化
                        read_path = util.normalize_tool_path(read_path)
                        state.tool_exec.read_files[read_path] = true
                        print(string.format("[ToolsNode][DEBUG] Recorded read file: %s", read_path))
                    end
                end
            end

            processed_results[#processed_results + 1] = processed_row
            state.tool_exec.total_result_chars = (state.tool_exec.total_result_chars or 0) + #(processed_row.result or "")
        end
    end

    state.tool_exec.results = processed_results
    state.tool_exec.all_results = state.tool_exec.all_results or {}
    for _, row in ipairs(processed_results) do
        state.tool_exec.all_results[#state.tool_exec.all_results + 1] = row
    end
    state.tool_exec.context_fragments = processed_fragments
    state.tool_exec.parallel_groups = tonumber(result.parallel_groups) or 0

    merge_tool_context(state, state.tool_exec.context_fragments)

    -- 构建工具消息时进行最终大小检查
    local max_tool_msg_chars = TOOL_RESULT_HARD_MAX_CHARS
    for _, row in ipairs(state.tool_exec.results) do
        -- 控制工具的消息也需要添加到 runtime_messages
        if row.is_control then
            state.messages.runtime_messages[#state.messages.runtime_messages + 1] = {
                role = "tool",
                name = tostring(row.tool or ""),
                tool_call_id = tostring(row.call_id or ""),
                content = row.ok and "Control signal processed." or ("Error: " .. tostring(row.error or "")),
                status = row.ok and "success" or "error",
            }
        else
            local msg = build_tool_message(row)
            -- 对过大的tool消息进行最终截断
            if #msg.content > max_tool_msg_chars then
                msg.content = util.utf8_take(msg.content, max_tool_msg_chars)
                    .. "\n[Output truncated to prevent context overflow]"
            end
            state.messages.runtime_messages[#state.messages.runtime_messages + 1] = msg
        end
    end

    -- 检查上下文预算
    local stats = context_manager.get_context_stats(state)
    local budget = math.max(256, math.floor(tonumber((graph_cfg().input_token_budget) or 12000)))
    local status, warning = context_manager.check_budget(stats.estimated_tokens, budget)
    if status == "warning" or status == "exceeded" then
        -- 标记状态，让后续节点可以采取措施
        state.context._budget_warning = warning
    end

    state.agent_loop.pending_tool_calls = {}
    
    -- 只有在没有 finish_turn 控制信号时才清除停止原因
    if control_action ~= "finish" and prior_stop_reason ~= "remaining_steps_exhausted" then
        -- 清除可能的停止原因，让循环可以继续
        if util.trim(state.agent_loop.stop_reason) ~= "" then
            state.agent_loop.stop_reason = ""
        end
    end
    
    -- 如果是 continue_task，添加一个系统提示让 agent 知道要继续
    if control_action == "continue" and control_data then
        local continue_hint = string.format(
            "[System] Agent requested to continue: %s",
            tostring(control_data.reason or "continuing task")
        )
        state.messages.runtime_messages[#state.messages.runtime_messages + 1] = {
            role = "user",
            content = continue_hint,
        }
    end

    return state
end

return M
