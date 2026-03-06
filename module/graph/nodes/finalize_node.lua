local util = require("module.graph.util")

local M = {}

function M.run(state, _ctx)
    state.final_response = state.final_response or {}
    state.termination = state.termination or {}
    state.agent_loop = state.agent_loop or {}
    state.router_decision = state.router_decision or { route = "respond", raw = "", reason = "" }

    local final_text = util.trim(state.final_response.message or state.termination.final_message or "")
    if final_text == "" then
        final_text = "Protocol error: planner did not produce finish_turn, so the turn cannot be finalized normally."
        state.termination.final_status = util.trim(state.termination.final_status or "failed")
        state.termination.stop_reason = util.trim(state.termination.stop_reason or "missing_finish_turn")
    end

    state.final_response.message = final_text
    if util.trim(state.termination.final_status or "") == "" then
        state.termination.final_status = util.trim(state.final_response.status or "completed")
    end
    if util.trim(state.termination.stop_reason or "") == "" then
        state.termination.stop_reason = util.trim(state.agent_loop.stop_reason or "completed")
    end

    local executed_total = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0
    state.router_decision.route = executed_total > 0 and "tool_loop" or "respond"
    state.agent_loop.stop_reason = tostring(state.termination.stop_reason or "")
    return state
end

return M
