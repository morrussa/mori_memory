local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function normalize_tool_call(item, idx)
    if type(item) ~= "table" then
        return nil
    end
    local name = util.trim(item.tool or item.name)
    if name == "" then
        return nil
    end
    local args = item.args
    if args == nil then
        args = {}
    end
    if type(args) ~= "table" then
        return nil
    end
    local call_id = util.trim(item.call_id)
    if call_id == "" then
        call_id = string.format("repair_call_%d", tonumber(idx) or 0)
    end
    return {
        tool = name,
        args = args,
        call_id = call_id,
    }
end

local function parse_output(raw)
    local parsed = util.parse_lua_table_literal(raw)
    if not parsed then
        return nil
    end
    local calls = parsed.tool_calls
    if type(calls) ~= "table" then
        return nil
    end
    local out = {}
    for i, item in ipairs(calls) do
        local norm = normalize_tool_call(item, i)
        if norm then
            out[#out + 1] = norm
        end
    end
    return out
end

local function build_prompt(state)
    local failed = {}
    for _, row in ipairs((((state or {}).tool_exec or {}).results) or {}) do
        if row.ok ~= true then
            failed[#failed + 1] = {
                tool = row.tool,
                call_id = row.call_id,
                error = row.error,
            }
        end
    end

    return table.concat({
        "You are a strict tool repair planner.",
        "Given failed tool calls, return corrected tool calls.",
        "Output exactly one Lua table: {tool_calls={...}}",
        "If no retry should be made, output {tool_calls={}}",
        "",
        "[UserInput]",
        tostring((((state or {}).input or {}).message) or ""),
        "",
        "[PreviousToolCalls]",
        util.encode_lua_value((((state or {}).planner or {}).tool_calls) or {}, 0),
        "",
        "[FailedResults]",
        util.encode_lua_value(failed, 0),
    }, "\n")
end

function M.run(state, _ctx)
    state.repair = state.repair or { attempts = 0, max_attempts = 2, last_error = "" }

    local max_attempts = tonumber(state.repair.max_attempts)
    if not max_attempts then
        max_attempts = math.max(0, math.floor(tonumber((graph_cfg().repair or {}).max_attempts) or 2))
        state.repair.max_attempts = max_attempts
    end

    if (tonumber((((state or {}).tool_exec or {}).failed)) or 0) <= 0 then
        return state
    end

    local attempts = tonumber(state.repair.attempts) or 0
    if attempts >= max_attempts then
        return state
    end

    local cfg = graph_cfg().repair or {}
    local raw = py_pipeline:generate_chat_sync(
        { { role = "user", content = build_prompt(state) } },
        {
            max_tokens = math.max(32, math.floor(tonumber(cfg.max_tokens) or 256)),
            temperature = tonumber(cfg.temperature) or 0,
            seed = tonumber(cfg.seed) or 29,
        }
    )

    local repaired_calls = parse_output(raw or "")
    state.repair.attempts = attempts + 1
    state.repair.last_error = ""

    if type(repaired_calls) == "table" and #repaired_calls > 0 then
        state.planner = state.planner or {}
        state.planner.tool_calls = repaired_calls
    else
        state.repair.last_error = "repair_parse_failed_or_empty"
        state.planner = state.planner or {}
        state.planner.tool_calls = {}
    end

    return state
end

return M
