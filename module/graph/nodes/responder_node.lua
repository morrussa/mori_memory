local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function is_callable(obj)
    if type(obj) == "function" then
        return true
    end
    local ok, _ = pcall(function()
        return obj and obj.__call or (getmetatable(obj) or {}).__call
    end)
    if ok then
        return true
    end
    if obj ~= nil then
        local ok2, _ = pcall(function()
            obj({})
        end)
        return ok2
    end
    return false
end

local function emit_stream(state, event_name, payload)
    local sink = state.stream_sink
    if not is_callable(sink) then
        return
    end
    pcall(sink, {
        event = event_name,
        data = payload or {},
    })
end

local function emit_tokens(state, text, chunk_chars)
    local s = tostring(text or "")
    if s == "" then
        return
    end
    local n = math.max(1, math.floor(tonumber(chunk_chars) or 24))
    local buf = {}
    local count = 0
    for ch in s:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        buf[#buf + 1] = ch
        count = count + 1
        if count >= n then
            emit_stream(state, "token", { token = table.concat(buf) })
            buf = {}
            count = 0
        end
    end
    if #buf > 0 then
        emit_stream(state, "token", { token = table.concat(buf) })
    end
end

function M.run(state, _ctx)
    state.termination = state.termination or {}
    state.final_response = state.final_response or {}
    state.session = state.session or { active_task = {} }
    state.session.active_task = state.session.active_task or {}
    state.task = state.task or {}

    local final_text = util.trim(state.termination.final_message or "")
    if final_text == "" then
        final_text = "I do not have a valid finish_turn result for this turn."
    end

    state.final_response.message = final_text
    state.final_response.status = util.trim(state.termination.final_status or "completed")
    state.final_response.stop_reason = util.trim(state.termination.stop_reason or "")
    local task_decision = (state.task.decision or {})
    local decision_kind = util.trim(task_decision.kind or "")
    if decision_kind ~= "meta_turn" then
        state.session.active_task.carryover_summary = util.utf8_take(final_text, 600)
    end

    local status = state.final_response.status
    if decision_kind == "meta_turn" and util.trim(task_decision.previous_status or state.session.active_task.status or "") ~= "" then
        state.session.active_task.status = util.trim(task_decision.previous_status or state.session.active_task.status or "")
    elseif status == "completed" then
        state.session.active_task.status = "completed"
    elseif status == "need_more_info" then
        state.session.active_task.status = "waiting_user"
    elseif status == "partial" then
        state.session.active_task.status = "open"
    else
        state.session.active_task.status = "failed"
    end

    if not state._streaming_sent and final_text ~= "" then
        local stream_cfg = graph_cfg().streaming or {}
        emit_tokens(state, final_text, tonumber(stream_cfg.token_chunk_chars) or 24)
        state._streaming_sent = true
    end

    return state
end

return M
