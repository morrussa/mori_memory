local util = require("module.graph.util")

local M = {}

local NEED_MORE_STEPS_TEXT = "Sorry, need more steps to process this request."

local function last_assistant_message(runtime_messages)
    local rows = runtime_messages or {}
    for i = #rows, 1, -1 do
        local row = rows[i]
        if type(row) == "table" and tostring(row.role or "") == "assistant" then
            return row
        end
    end
    return nil
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

    local final_text = ""
    local stop_reason = util.trim(((state or {}).agent_loop or {}).stop_reason or "")

    if stop_reason == "remaining_steps_exhausted" or stop_reason == "tool_loop_max_exceeded" then
        final_text = NEED_MORE_STEPS_TEXT
    else
        local last_ai = last_assistant_message(state.messages.runtime_messages)
        if type(last_ai) == "table" then
            final_text = util.trim(last_ai.content or "")
        end
        if final_text == "" then
            final_text = "好的，已处理。"
        end
    end

    state.final_response = state.final_response or {}
    state.final_response.message = final_text

    state.router_decision = state.router_decision or { route = "respond", raw = "", reason = "" }
    if (tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0) > 0 then
        state.router_decision.route = "tool_loop"
    else
        state.router_decision.route = "respond"
    end

    return state
end

return M
