local util = require("module.graph.util")
local config = require("module.config")
local tool_registry = require("module.graph.tool_registry_v2")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function has_method(runtime, method_name)
    if runtime == nil then
        return false
    end
    local ok, attr = pcall(function()
        return runtime[method_name]
    end)
    return ok and attr ~= nil
end

local function get_field(obj, key)
    if obj == nil then
        return nil
    end
    if type(obj) == "table" then
        local v = obj[key]
        if v ~= nil then
            return v
        end
        return obj[tostring(key)]
    end

    local ok, v = pcall(function()
        return obj[key]
    end)
    if ok and v ~= nil then
        return v
    end
    ok, v = pcall(function()
        return obj[tostring(key)]
    end)
    if ok and v ~= nil then
        return v
    end
    return nil
end

local function get_seq_item(seq, index1)
    local ok, v = pcall(function()
        return seq[index1]
    end)
    if ok and v ~= nil then
        return v
    end
    ok, v = pcall(function()
        return seq[index1 - 1]
    end)
    if ok then
        return v
    end
    return nil
end

local function clone_seq_rows(src, limit)
    local out = {}
    if src == nil then
        return out
    end
    if type(src) == "table" then
        for i, item in ipairs(src) do
            out[#out + 1] = item
            if limit and #out >= limit then
                break
            end
        end
        return out
    end

    local n = 0
    local ok_len, len_or_err = pcall(function()
        return #src
    end)
    if ok_len then
        n = math.max(0, math.floor(tonumber(len_or_err) or 0))
    end

    if n > 0 then
        for i = 1, n do
            local item = get_seq_item(src, i)
            if item ~= nil then
                out[#out + 1] = item
                if limit and #out >= limit then
                    break
                end
            end
        end
        return out
    end

    local probe_limit = math.max(1, math.floor(tonumber(limit) or 128))
    for i = 1, probe_limit do
        local item = get_seq_item(src, i)
        if item == nil then
            break
        end
        out[#out + 1] = item
    end
    return out
end

local function normalize_tool_call(item, idx)
    local fn = get_field(item, "function")
    local name = util.trim(
        get_field(item, "name")
        or get_field(item, "tool")
        or (type(fn) == "table" and fn.name)
        or get_field(fn, "name")
    )
    if name == "" then
        return nil
    end

    local args = get_field(item, "args")
    if args == nil then
        args = get_field(item, "arguments")
    end
    if args == nil and fn ~= nil then
        args = get_field(fn, "arguments")
    end
    if type(args) ~= "table" then
        args = {}
    end

    local call_id = util.trim(
        get_field(item, "id")
        or get_field(item, "call_id")
    )
    if call_id == "" then
        call_id = string.format("tool_call_%d", tonumber(idx) or 0)
    end

    return {
        tool = name,
        args = args,
        call_id = call_id,
    }
end

local function normalize_tool_calls(raw_calls)
    local rows = clone_seq_rows(raw_calls, 256)
    local out = {}
    for i, item in ipairs(rows) do
        local call = normalize_tool_call(item, i)
        if call then
            out[#out + 1] = call
        end
    end
    return out
end

local function parse_model_output(raw)
    if type(raw) == "string" then
        local parsed = util.parse_lua_table_literal(raw)
        if parsed and type(parsed) == "table" then
            raw = parsed
        else
            return {
                content = util.trim(raw),
                tool_calls = {},
                raw = util.trim(raw),
            }
        end
    end

    if type(raw) ~= "table" and raw ~= nil then
        return {
            content = util.trim(tostring(raw)),
            tool_calls = {},
            raw = util.trim(tostring(raw)),
        }
    end

    local content = util.trim(
        get_field(raw, "content")
        or get_field(raw, "text")
    )
    local calls = normalize_tool_calls(get_field(raw, "tool_calls"))
    local raw_text = util.trim(get_field(raw, "raw"))
    return {
        content = content,
        tool_calls = calls,
        raw = raw_text,
    }
end

local function call_agent_with_tools(state)
    local runtime = _G.py_pipeline
    if runtime == nil then
        return nil, "python_runtime_unavailable"
    end
    if not has_method(runtime, "generate_chat_with_tools_sync") then
        return nil, "python_method_unavailable"
    end

    local cfg = (graph_cfg().agent or {})
    local params = {
        max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 1024)),
        temperature = tonumber(cfg.temperature) or 0.6,
        seed = tonumber(cfg.seed) or 42,
    }

    local tools = tool_registry.get_tool_schemas()
    local tool_choice = cfg.tool_choice
    if util.trim(tool_choice) == "" then
        tool_choice = "auto"
    end
    local parallel_tool_calls = util.to_bool(cfg.parallel_tool_calls, true)

    local ok, result_or_err = pcall(function()
        return runtime:generate_chat_with_tools_sync(
            (((state or {}).messages or {}).runtime_messages) or {},
            params,
            tools,
            tool_choice,
            parallel_tool_calls
        )
    end)
    if not ok then
        return nil, tostring(result_or_err or "agent_llm_call_failed")
    end

    local out = parse_model_output(result_or_err)
    return out, nil
end

local function fallback_without_tools(state)
    local cfg = (graph_cfg().agent or {})
    local ok, text_or_err = pcall(function()
        return py_pipeline:generate_chat_sync(
            (((state or {}).messages or {}).runtime_messages) or {},
            {
                max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 1024)),
                temperature = tonumber(cfg.temperature) or 0.6,
                seed = tonumber(cfg.seed) or 42,
            }
        )
    end)
    if not ok then
        return nil, tostring(text_or_err or "agent_llm_fallback_failed")
    end
    return {
        content = util.trim(tostring(text_or_err or "")),
        tool_calls = {},
        raw = "",
    }, nil
end

function M.run(state, _ctx)
    state.messages = state.messages or {}
    state.messages.runtime_messages = state.messages.runtime_messages or {}
    state.agent_loop = state.agent_loop or {
        remaining_steps = 25,
        pending_tool_calls = {},
        stop_reason = "",
        iteration = 0,
    }
    state.planner = state.planner or { tool_calls = {}, errors = {}, raw = "", force_reason = "" }
    state.router_decision = state.router_decision or { route = "respond", raw = "", reason = "" }

    local out, err = call_agent_with_tools(state)
    if not out then
        out, err = fallback_without_tools(state)
    end

    if not out then
        state.agent_loop.pending_tool_calls = {}
        state.agent_loop.stop_reason = "model_call_failed"
        state.planner.tool_calls = {}
        state.planner.raw = ""
        state.planner.errors = state.planner.errors or {}
        state.planner.errors[#state.planner.errors + 1] = tostring(err or "agent_call_failed")
        return state
    end

    local tool_calls = out.tool_calls or {}
    state.messages.runtime_messages[#state.messages.runtime_messages + 1] = {
        role = "assistant",
        content = tostring(out.content or ""),
        tool_calls = tool_calls,
    }

    state.agent_loop.iteration = (tonumber(state.agent_loop.iteration) or 0) + 1
    state.agent_loop.remaining_steps = math.max(0, (tonumber(state.agent_loop.remaining_steps) or 0) - 1)
    state.agent_loop.pending_tool_calls = tool_calls

    state.planner.tool_calls = tool_calls
    state.planner.raw = tostring(out.raw or "")

    if #tool_calls > 0 then
        state.router_decision.route = "tool_loop"
    else
        state.router_decision.route = "respond"
    end

    if #tool_calls > 0 and (tonumber(state.agent_loop.remaining_steps) or 0) <= 0 then
        state.agent_loop.pending_tool_calls = {}
        state.agent_loop.stop_reason = "remaining_steps_exhausted"
        state.planner.tool_calls = {}
    elseif util.trim(state.agent_loop.stop_reason) == "" then
        state.agent_loop.stop_reason = ""
    end

    return state
end

return M
