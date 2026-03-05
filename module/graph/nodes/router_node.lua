local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local ROUTER_PROMPT = [[
You are a strict routing classifier.
Output exactly one Lua table on a single line.
Allowed outputs:
{route="tool_loop"}
{route="respond"}
Rules:
- route="tool_loop" when file reading/searching or external tool usage is needed before final answer.
- route="respond" when direct final response is possible without tool calls.
Do not output anything else.
]]

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function has_uploads(state)
    local rows = ((state or {}).uploads) or {}
    return type(rows) == "table" and #rows > 0
end

local function should_force_tool_loop(state)
    if not has_uploads(state) then
        return false
    end
    local executed_total = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0
    return executed_total <= 0
end

local function parse_router_output(raw)
    local parsed, err = util.parse_lua_table_literal(raw)
    if not parsed then
        return nil, err
    end
    local route = util.trim(parsed.route)
    if route ~= "tool_loop" and route ~= "respond" then
        return nil, "invalid_route"
    end
    return {
        route = route,
        raw = util.trim(raw),
        reason = util.trim(parsed.reason),
    }
end

function M.run(state, _ctx)
    local cfg = graph_cfg().router or {}
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local tool_context = tostring((((state or {}).context or {}).tool_context) or "")
    local memory_context = tostring((((state or {}).context or {}).memory_context) or "")
    local force_upload_route = should_force_tool_loop(state)

    local prompt = table.concat({
        ROUTER_PROMPT,
        "",
        "[UserInput]",
        user_input,
        "",
        "[ToolContext]",
        tool_context,
        "",
        "[MemoryContext]",
        memory_context,
    }, "\n")

    local raw = py_pipeline:generate_chat_sync(
        { { role = "user", content = prompt } },
        {
            max_tokens = math.max(16, math.floor(tonumber(cfg.max_tokens) or 48)),
            temperature = tonumber(cfg.temperature) or 0,
            seed = tonumber(cfg.seed) or 7,
        }
    )

    local decision, err = parse_router_output(raw)
    if decision and force_upload_route and decision.route ~= "tool_loop" then
        decision.route = "tool_loop"
        decision.reason = "forced_upload_tool_loop"
    end

    if not decision then
        local fallback_route = force_upload_route and "tool_loop" or "respond"
        decision = {
            route = fallback_route,
            raw = util.trim(raw),
            reason = "fallback:" .. tostring(err or "router_parse_failed"),
        }
    end

    state.router_decision = decision
    return state
end

return M
