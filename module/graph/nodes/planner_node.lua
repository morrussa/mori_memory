local context_builder = require("module.graph.context_builder")
local tool_registry = require("module.graph.tool_registry_v2")
local control_tools = require("module.graph.control_tools")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local NEED_MORE_STEPS_TEXT = "Sorry, need more steps to process this request."

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
    local ok, value = pcall(function()
        return obj[key]
    end)
    if ok and value ~= nil then
        return value
    end
    ok, value = pcall(function()
        return obj[tostring(key)]
    end)
    if ok then
        return value
    end
    return nil
end

local function normalize_tool_calls(raw)
    local out = {}
    if type(raw) == "table" then
        for idx, item in ipairs(raw) do
            local name = util.trim(get_field(item, "name") or get_field(item, "tool"))
            if name ~= "" then
                out[#out + 1] = {
                    tool = name,
                    args = get_field(item, "args") or {},
                    call_id = util.trim(get_field(item, "id") or get_field(item, "call_id") or ("planner_call_" .. tostring(idx))),
                }
            end
        end
        return out
    end

    for i = 1, 32 do
        local item = get_field(raw, i) or get_field(raw, i - 1)
        if item == nil then
            break
        end
        local name = util.trim(get_field(item, "name") or get_field(item, "tool"))
        if name ~= "" then
            out[#out + 1] = {
                tool = name,
                args = get_field(item, "args") or {},
                call_id = util.trim(get_field(item, "id") or get_field(item, "call_id") or ("planner_call_" .. tostring(i))),
            }
        end
    end
    return out
end

local function parse_model_output(raw)
    local content = util.trim(get_field(raw, "content") or get_field(raw, "text") or "")
    local tool_calls = normalize_tool_calls(get_field(raw, "tool_calls"))
    local raw_text = util.trim(get_field(raw, "raw") or "")

    if content == "" and #tool_calls == 0 and raw ~= nil then
        raw_text = util.trim(tostring(raw))
    end

    return {
        content = content,
        tool_calls = tool_calls,
        raw = raw_text,
    }
end

local function ensure_contract_system_message(messages, state)
    local out = {}
    for i, row in ipairs(messages or {}) do
        out[i] = {
            role = tostring((row or {}).role or ""),
            content = tostring((row or {}).content or ""),
        }
    end
    local profile = util.trim((((state or {}).context or {}).task_profile) or "general")
    local contract = table.concat({
        "You are the planner for a single-session tool agent.",
        "Return tool calls when more work is needed.",
        "If the turn is complete, call finish_turn.",
        "Do not rely on plain assistant text to end the turn.",
        "Never mix finish_turn with any other tool call in the same batch.",
        "Available workspace root: " .. util.workspace_virtual_root(),
        "Current task profile: " .. profile,
    }, "\n")

    if #out > 0 and tostring((out[1] or {}).role or "") == "system" then
        out[1].content = table.concat({ tostring(out[1].content or ""), "", "[PlannerContract]", contract }, "\n")
    else
        table.insert(out, 1, { role = "system", content = contract })
    end
    return out
end

local function set_terminal_failure(state, status, stop_reason, message)
    state.termination = state.termination or {}
    state.termination.finish_requested = true
    state.termination.final_status = status
    state.termination.stop_reason = stop_reason
    state.termination.final_message = util.trim(message or "")
    state.agent_loop = state.agent_loop or {}
    state.agent_loop.stop_reason = stop_reason
    state.agent_loop.pending_tool_calls = {}
    state.planner = state.planner or {}
    state.planner.tool_calls = {}
end

function M.run(state, _ctx)
    state.agent_loop = state.agent_loop or {}
    state.planner = state.planner or {}
    state.repair = state.repair or {}
    state.termination = state.termination or {}
    state.working_memory = state.working_memory or {}
    state.messages = state.messages or {}
    state.messages.runtime_messages = state.messages.runtime_messages or {}

    state.planner.tool_calls = {}
    state.planner.errors = {}
    state.planner.raw = ""
    state.planner.force_reason = ""
    state.planner.missing_terminal_signal = false
    state.repair.pending = false
    state.repair.retry_requested = false

    if state.termination.finish_requested == true then
        return state
    end

    local tool_loop_max = math.max(1, math.floor(tonumber((graph_cfg() or {}).tool_loop_max) or 5))
    local loop_count = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0
    if loop_count >= tool_loop_max then
        set_terminal_failure(state, "failed", "tool_loop_max_exceeded", NEED_MORE_STEPS_TEXT)
        return state
    end

    local remaining_steps = tonumber(state.agent_loop.remaining_steps) or 0
    if remaining_steps <= 0 then
        set_terminal_failure(state, "failed", "remaining_steps_exhausted", NEED_MORE_STEPS_TEXT)
        return state
    end
    state.agent_loop.remaining_steps = remaining_steps - 1
    state.agent_loop.iteration = (tonumber(state.agent_loop.iteration) or 0) + 1

    local runtime = _G.py_pipeline
    if runtime == nil or not has_method(runtime, "generate_chat_with_tools_sync") then
        state.planner.errors[#state.planner.errors + 1] = "planner_runtime_unavailable"
        state.repair.pending = true
        state.repair.last_error = "planner_runtime_unavailable"
        return state
    end

    local cfg = graph_cfg().planner or {}
    local params = {
        max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 768)),
        temperature = tonumber(cfg.temperature) or 0.1,
        seed = tonumber(cfg.seed) or 11,
    }

    local messages = ensure_contract_system_message(context_builder.build_chat_messages(state), state)
    local tools = tool_registry.get_tool_schemas(state)

    local ok, result_or_err = pcall(function()
        return runtime:generate_chat_with_tools_sync(
            messages,
            params,
            tools,
            "auto",
            false
        )
    end)
    if not ok then
        state.planner.errors[#state.planner.errors + 1] = "planner_model_call_failed"
        state.repair.pending = true
        state.repair.last_error = tostring(result_or_err or "planner_model_call_failed")
        return state
    end

    local parsed = parse_model_output(result_or_err)
    state.planner.raw = parsed.raw ~= "" and parsed.raw or parsed.content

    local finish_call = nil
    local tool_calls = {}
    for _, call in ipairs(parsed.tool_calls or {}) do
        if util.trim(call.tool) == "finish_turn" then
            finish_call = call
        else
            tool_calls[#tool_calls + 1] = call
        end
    end

    if finish_call and #tool_calls > 0 then
        state.planner.errors[#state.planner.errors + 1] = "invalid_mixed_terminal_batch"
        state.repair.pending = true
        state.repair.last_error = "invalid_mixed_terminal_batch"
        state.working_memory.last_repair_error = "invalid_mixed_terminal_batch"
        return state
    end

    if finish_call then
        local ok_finish, _, control_info = control_tools.execute(finish_call)
        if not ok_finish or not control_info then
            state.planner.errors[#state.planner.errors + 1] = "finish_turn_invalid"
            state.repair.pending = true
            state.repair.last_error = "finish_turn_invalid"
            return state
        end

        state.termination.finish_requested = true
        state.termination.final_message = util.trim(control_info.message or parsed.content or "好的，已处理。")
        state.termination.final_status = util.trim(control_info.status or "completed")
        state.termination.stop_reason = "finish_turn_called"
        state.agent_loop.stop_reason = "finish_turn_called"
        state.agent_loop.pending_tool_calls = {}
        state.working_memory.current_plan = "finish_turn"
        return state
    end

    if #tool_calls == 0 then
        state.planner.missing_terminal_signal = true
        state.planner.errors[#state.planner.errors + 1] = "missing_terminal_signal"
        state.repair.pending = true
        state.repair.last_error = "missing_terminal_signal"
        state.working_memory.last_repair_error = "missing_terminal_signal"
        return state
    end

    state.planner.tool_calls = tool_calls
    state.agent_loop.pending_tool_calls = tool_calls

    local plan_names = {}
    for _, call in ipairs(tool_calls) do
        plan_names[#plan_names + 1] = call.tool
    end
    state.working_memory.current_plan = table.concat(plan_names, " -> ")
    state.working_memory.plan_step_index = (tonumber(state.working_memory.plan_step_index) or 0) + 1
    return state
end

return M
