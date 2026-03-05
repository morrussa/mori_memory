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
        context_fragments = {},
        truncated_count = 0,
        total_result_chars = 0,
    }

    local calls = state.agent_loop.pending_tool_calls or {}
    local result = tool_registry.execute_calls(calls)

    state.tool_exec.loop_count = (tonumber(state.tool_exec.loop_count) or 0) + 1
    state.tool_exec.executed = tonumber(result.executed) or 0
    state.tool_exec.failed = tonumber(result.failed) or 0
    state.tool_exec.executed_total = (tonumber(state.tool_exec.executed_total) or 0) + (tonumber(result.executed) or 0)
    state.tool_exec.failed_total = (tonumber(state.tool_exec.failed_total) or 0) + (tonumber(result.failed) or 0)
    state.tool_exec.read_evidence_total = tonumber(state.tool_exec.read_evidence_total) or 0
    state.tool_exec.truncated_count = 0
    state.tool_exec.total_result_chars = 0

    -- 处理工具结果，进行大小优化
    local processed_results = {}
    local processed_fragments = {}
    for _, row in ipairs(result.call_results or {}) do
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
            if READ_EVIDENCE_TOOLS[tostring(row.tool or "")] then
                state.tool_exec.read_evidence_total = state.tool_exec.read_evidence_total + 1
                processed_fragments[#processed_fragments + 1] = string.format(
                    "[Tool:%s]\n%s",
                    row.tool,
                    processed_text
                )
            end
        end

        processed_results[#processed_results + 1] = processed_row
        state.tool_exec.total_result_chars = (state.tool_exec.total_result_chars or 0) + #(processed_row.result or "")
    end

    state.tool_exec.results = processed_results
    state.tool_exec.context_fragments = processed_fragments
    state.tool_exec.parallel_groups = tonumber(result.parallel_groups) or 0

    merge_tool_context(state, state.tool_exec.context_fragments)

    -- 构建工具消息时进行最终大小检查
    local max_tool_msg_chars = TOOL_RESULT_HARD_MAX_CHARS
    for _, row in ipairs(state.tool_exec.results) do
        local msg = build_tool_message(row)
        -- 对过大的tool消息进行最终截断
        if #msg.content > max_tool_msg_chars then
            msg.content = util.utf8_take(msg.content, max_tool_msg_chars)
                .. "\n[Output truncated to prevent context overflow]"
        end
        state.messages.runtime_messages[#state.messages.runtime_messages + 1] = msg
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
    if util.trim(state.agent_loop.stop_reason) == "" then
        state.agent_loop.stop_reason = ""
    end

    return state
end

return M
