local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local NEED_MORE_STEPS_TEXT = "Sorry, need more steps to process this request."

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

-- 检查对象是否可调用（支持 Lua function 和 Lupa userdata）
local function is_callable(obj)
    if type(obj) == "function" then
        return true
    end
    -- Lupa 传递的 Python 函数是 userdata 或 table，尝试检测
    local ok, _ = pcall(function()
        return obj and obj.__call or (getmetatable(obj) or {}).__call
    end)
    if ok then
        return true
    end
    -- 最后尝试直接调用检查（Lupa 的 Python 函数可以直接调用）
    if obj ~= nil then
        local ok2, _ = pcall(function()
            obj({})
        end)
        return ok2
    end
    return false
end

-- 发送流式事件
local function emit_stream(state, event_name, payload)
    local sink = state.stream_sink
    if not is_callable(sink) then
        return
    end
    local ok, err = pcall(sink, {
        event = event_name,
        data = payload or {},
    })
    if not ok then
        print(string.format("[FinalizeNode][WARN] stream emit failed: %s", tostring(err)))
    end
end

-- 分块发送文本作为 token 事件
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

    -- 检查是否从 finish_turn 控制工具获得了最终消息
    local from_control_tool = (((state or {}).final_response or {}).from_control_tool) == true
    
    if from_control_tool and ((state or {}).final_response or {}).message then
        -- 使用 finish_turn 提供的消息
        final_text = util.trim(state.final_response.message)
    elseif stop_reason == "remaining_steps_exhausted" or stop_reason == "tool_loop_max_exceeded" then
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

    -- 流式发送最终回复（如果之前没有发送过）
    if not state._streaming_sent and final_text ~= "" then
        local stream_cfg = graph_cfg().streaming or {}
        local chunk_chars = tonumber(stream_cfg.token_chunk_chars) or 24
        emit_tokens(state, final_text, chunk_chars)
        state._streaming_sent = true
    end

    state.router_decision = state.router_decision or { route = "respond", raw = "", reason = "" }
    if (tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0) > 0 then
        state.router_decision.route = "tool_loop"
    else
        state.router_decision.route = "respond"
    end

    return state
end

return M
