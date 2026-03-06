local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

function M.run(state, _ctx)
    state.repair = state.repair or {}
    state.termination = state.termination or {}
    state.context = state.context or {}
    state.working_memory = state.working_memory or {}
    state.planner = state.planner or {}

    local pending = state.repair.pending == true or util.trim(state.repair.last_error or "") ~= ""
    if not pending then
        state.repair.retry_requested = false
        return state
    end

    local max_attempts = tonumber(state.repair.max_attempts)
    if not max_attempts then
        max_attempts = math.max(0, math.floor(tonumber((graph_cfg().repair or {}).max_attempts) or 2))
        state.repair.max_attempts = max_attempts
    end

    state.repair.attempts = (tonumber(state.repair.attempts) or 0) + 1
    local last_error = util.trim(state.repair.last_error or "repair_required")
    state.working_memory.last_repair_error = last_error

    if state.repair.attempts > max_attempts then
        state.termination.finish_requested = true
        state.termination.final_status = "failed"
        state.termination.stop_reason = "repair_exhausted"
        state.termination.final_message = "I could not complete the request after repeated tool/planner repair attempts."
        state.context.planner_context = "Repair exhausted: stop and explain the failure."
        state.repair.pending = false
        state.repair.retry_requested = false
        state.agent_loop = state.agent_loop or {}
        state.agent_loop.stop_reason = "repair_exhausted"
        state.planner.tool_calls = {}
        return state
    end

    state.context.planner_context = table.concat({
        "Repair the previous step.",
        "Last error: " .. last_error,
        "Do not repeat the same invalid tool batch.",
        "If the task is complete, call finish_turn.",
    }, "\n")
    state.repair.pending = false
    state.repair.retry_requested = true
    state.planner.tool_calls = {}
    return state
end

return M
