local util = require("module.graph.util")
local tool_registry = require("module.graph.tool_registry_v2")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

function M.run(state, _ctx)
    state.tool_exec = state.tool_exec or {
        loop_count = 0,
        executed = 0,
        failed = 0,
        executed_total = 0,
        failed_total = 0,
        results = {},
        context_fragments = {},
    }

    local calls = (((state or {}).planner or {}).tool_calls) or {}
    local result = tool_registry.execute_calls(calls)

    state.tool_exec.loop_count = (tonumber(state.tool_exec.loop_count) or 0) + 1
    state.tool_exec.executed = tonumber(result.executed) or 0
    state.tool_exec.failed = tonumber(result.failed) or 0
    state.tool_exec.executed_total = (tonumber(state.tool_exec.executed_total) or 0) + (tonumber(result.executed) or 0)
    state.tool_exec.failed_total = (tonumber(state.tool_exec.failed_total) or 0) + (tonumber(result.failed) or 0)
    state.tool_exec.results = result.call_results or {}
    state.tool_exec.context_fragments = result.context_fragments or {}
    state.tool_exec.parallel_groups = tonumber(result.parallel_groups) or 0

    local merged = table.concat(state.tool_exec.context_fragments or {}, "\n\n")
    local prev = util.trim((((state or {}).context or {}).tool_context) or "")
    if prev ~= "" and merged ~= "" then
        merged = prev .. "\n\n" .. merged
    elseif prev ~= "" then
        merged = prev
    end

    local max_chars = math.max(120, math.floor(tonumber((((graph_cfg() or {}).tools or {}).file_context_max_chars) or 1600)))
    merged = util.utf8_take(merged, max_chars)

    state.context = state.context or {}
    state.context.tool_context = util.trim(merged)

    return state
end

return M
