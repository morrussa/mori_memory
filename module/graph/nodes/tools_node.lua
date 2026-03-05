local util = require("module.graph.util")
local tool_registry = require("module.graph.tool_registry_v2")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

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
    local merged = table.concat(fragments or {}, "\n\n")
    local prev = util.trim((((state or {}).context or {}).tool_context) or "")
    if prev ~= "" and merged ~= "" then
        merged = prev .. "\n\n" .. merged
    elseif prev ~= "" then
        merged = prev
    end

    local max_chars = math.max(120, math.floor(tonumber((((graph_cfg() or {}).tools or {}).file_context_max_chars) or 1600)))
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
        results = {},
        context_fragments = {},
    }

    local calls = state.agent_loop.pending_tool_calls or {}
    local result = tool_registry.execute_calls(calls)

    state.tool_exec.loop_count = (tonumber(state.tool_exec.loop_count) or 0) + 1
    state.tool_exec.executed = tonumber(result.executed) or 0
    state.tool_exec.failed = tonumber(result.failed) or 0
    state.tool_exec.executed_total = (tonumber(state.tool_exec.executed_total) or 0) + (tonumber(result.executed) or 0)
    state.tool_exec.failed_total = (tonumber(state.tool_exec.failed_total) or 0) + (tonumber(result.failed) or 0)
    state.tool_exec.results = result.call_results or {}
    state.tool_exec.context_fragments = result.context_fragments or {}
    state.tool_exec.parallel_groups = tonumber(result.parallel_groups) or 0

    merge_tool_context(state, state.tool_exec.context_fragments)

    for _, row in ipairs(state.tool_exec.results or {}) do
        state.messages.runtime_messages[#state.messages.runtime_messages + 1] = build_tool_message(row)
    end

    state.agent_loop.pending_tool_calls = {}
    if util.trim(state.agent_loop.stop_reason) == "" then
        state.agent_loop.stop_reason = ""
    end

    return state
end

return M
