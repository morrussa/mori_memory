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

local function lower_text(v)
    return tostring(v or ""):lower()
end

local function has_uploads(state)
    local rows = ((state or {}).uploads) or {}
    return type(rows) == "table" and #rows > 0
end

local function collect_history_text(state, max_chars)
    local rows = ((((state or {}).messages or {}).conversation_history) or {})
    if type(rows) ~= "table" or #rows <= 0 then
        return ""
    end
    local out = {}
    for i = #rows, 1, -1 do
        local row = rows[i]
        if type(row) == "table" then
            local role = tostring(row.role or "")
            local content = util.trim(row.content or "")
            if content ~= "" then
                out[#out + 1] = string.format("[%s] %s", role, content)
            end
            if #out >= 6 then
                break
            end
        end
    end
    local merged = table.concat(out, "\n")
    return util.utf8_take(merged, math.max(160, math.floor(tonumber(max_chars) or 1800)))
end

local function has_history_tool_path(state)
    local text = collect_history_text(state, 2400)
    if text == "" then
        return false
    end
    if text:find("tool_path=download/", 1, true) then
        return true
    end
    if text:find("download/", 1, true) then
        return true
    end
    return false
end

local function has_file_intent(user_input)
    local q = lower_text(user_input)
    if q == "" then
        return false
    end
    local keywords = {
        "读取", "读一下", "读一读", "查看", "前几行", "几行", "第几行",
        "文件", "代码", "模型设置", "配置", "search", "grep", "read", "line", "lines",
        "list_files", "read_file", "read_lines", "search_file", "search_files",
        "download/", ".py", ".lua", ".txt",
    }
    for _, key in ipairs(keywords) do
        if q:find(key, 1, true) then
            return true
        end
    end
    return false
end

local function is_followup_execute_intent(state, user_input)
    local q = lower_text(user_input)
    if q == "" then
        return false
    end
    local asks_execute = q:find("执行", 1, true) ~= nil
    local asks_read = (
        q:find("读取", 1, true) ~= nil
        or q:find("读", 1, true) ~= nil
        or q:find("前几行", 1, true) ~= nil
        or q:find("几行", 1, true) ~= nil
    )
    local request_exec = (
        asks_execute
        or asks_read
        or q:find("先做", 1, true) ~= nil
    )
    if not request_exec then
        return false
    end
    local history_text = collect_history_text(state, 1200)
    if history_text == "" then
        return false
    end
    if history_text:find("当前回合没有任何工具执行结果可用", 1, true) then
        return true
    end
    if asks_execute and history_text:find("tool_path=download/", 1, true) then
        return true
    end
    return false
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
    local force_followup_route = has_history_tool_path(state) and has_file_intent(user_input)
    local force_retry_route = is_followup_execute_intent(state, user_input)
    local force_tool_route = force_upload_route or force_followup_route or force_retry_route

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
        "",
        "[ConversationTail]",
        collect_history_text(state, 1600),
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
    if decision and force_tool_route and decision.route ~= "tool_loop" then
        decision.route = "tool_loop"
        decision.reason = "forced_tool_loop"
    end

    if not decision then
        local fallback_route = force_tool_route and "tool_loop" or "respond"
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
