local util = require("module.graph.util")
local config = require("module.config")
local tool_registry = require("module.graph.tool_registry_v2")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function has_uploads(state)
    local rows = ((state or {}).uploads) or {}
    return type(rows) == "table" and #rows > 0
end

local function should_force_upload_toolchain(state)
    if not has_uploads(state) then
        return false
    end
    local executed_total = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0
    return executed_total <= 0
end

local function build_supported_tool_list()
    local supported = tool_registry.get_supported_tools()
    local names = {}
    for name, enabled in pairs(supported or {}) do
        if enabled then
            names[#names + 1] = name
        end
    end
    table.sort(names)
    return names
end

local function build_prompt(state)
    local names = build_supported_tool_list()
    local route = tostring((((state or {}).router_decision or {}).route) or "respond")
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local memory_context = tostring((((state or {}).context or {}).memory_context) or "")
    local tool_context = tostring((((state or {}).context or {}).tool_context) or "")

    return table.concat({
        "You are a strict tool planner.",
        "Output exactly one Lua table on a single line.",
        "Schema:",
        "{tool_calls={",
        "  {tool=\"list_files\", args={...}, call_id=\"optional\"},",
        "  ...",
        "}}",
        "If no tool is needed, output {tool_calls={}}.",
        "Do not output explanation.",
        "",
        "Allowed tools:",
        table.concat(names, ", "),
        "",
        "[Route] " .. route,
        "[UserInput]",
        user_input,
        "",
        "[MemoryContext]",
        memory_context,
        "",
        "[ToolContext]",
        tool_context,
    }, "\n")
end

local function normalize_upload_tool_path(item)
    local raw = util.trim((item or {}).tool_path or (item or {}).path)
    if raw == "" then
        return ""
    end
    local path = raw:gsub("\\", "/")

    while path:sub(1, 2) == "./" do
        path = path:sub(3)
    end
    if path:sub(1, 10) == "workspace/" then
        path = path:sub(11)
    end

    local idx = path:find("/workspace/", 1, true)
    if idx then
        path = path:sub(idx + 11)
    end

    path = path:gsub("^/+", "")
    return util.trim(path)
end

local function build_forced_upload_calls(state, max_reads, read_max_chars)
    local calls = {
        {
            tool = "list_files",
            args = { prefix = "download" },
            call_id = "forced_upload_list_1",
        },
    }

    local upload_rows = ((state or {}).uploads) or {}
    local seen = {}
    local read_count = 0

    for _, upload in ipairs(upload_rows) do
        if read_count >= max_reads then
            break
        end
        local tool_path = normalize_upload_tool_path(upload)
        if tool_path ~= "" and not seen[tool_path] then
            seen[tool_path] = true
            read_count = read_count + 1
            calls[#calls + 1] = {
                tool = "read_file",
                args = {
                    path = tool_path,
                    max_chars = read_max_chars,
                },
                call_id = string.format("forced_upload_read_%d", read_count),
            }
        end
    end

    return calls
end

local function normalize_tool_call(item, idx)
    if type(item) ~= "table" then
        return nil, "tool_call_not_table"
    end
    local name = util.trim(item.tool or item.name)
    if name == "" then
        return nil, "missing_tool"
    end
    local args = item.args
    if args == nil then
        args = {}
    end
    if type(args) ~= "table" then
        return nil, "args_not_table"
    end
    local call_id = util.trim(item.call_id)
    if call_id == "" then
        call_id = string.format("planner_call_%d", tonumber(idx) or 0)
    end
    return {
        tool = name,
        args = args,
        call_id = call_id,
    }
end

local function parse_output(raw)
    local parsed, err = util.parse_lua_table_literal(raw)
    if not parsed then
        return nil, err
    end

    local calls = parsed.tool_calls
    if calls == nil then
        return nil, "missing_tool_calls"
    end
    if type(calls) ~= "table" then
        return nil, "tool_calls_not_table"
    end

    local out = {}
    for i, item in ipairs(calls) do
        local norm, nerr = normalize_tool_call(item, i)
        if not norm then
            return nil, nerr
        end
        out[#out + 1] = norm
    end

    return out
end

function M.run(state, _ctx)
    local route = tostring((((state or {}).router_decision or {}).route) or "respond")
    state.planner = state.planner or { tool_calls = {}, errors = {}, raw = "", force_reason = "" }

    if route ~= "tool_loop" then
        state.planner.tool_calls = {}
        state.planner.raw = ""
        state.planner.force_reason = ""
        return state
    end

    local cfg = graph_cfg().planner or {}
    local force_upload_toolchain = util.to_bool(cfg.force_upload_toolchain, true)
    local force_upload = force_upload_toolchain and should_force_upload_toolchain(state)
    local max_upload_reads = math.max(1, math.floor(tonumber(cfg.force_upload_read_limit) or 2))
    local upload_read_max_chars = math.max(256, math.floor(tonumber(cfg.force_upload_read_max_chars) or 2400))
    local max_calls = math.max(1, math.floor(tonumber(cfg.max_calls_per_loop) or 6))

    local raw = py_pipeline:generate_chat_sync(
        { { role = "user", content = build_prompt(state) } },
        {
            max_tokens = math.max(32, math.floor(tonumber(cfg.max_tokens) or 256)),
            temperature = tonumber(cfg.temperature) or 0.1,
            seed = tonumber(cfg.seed) or 11,
        }
    )

    local calls, err = parse_output(raw)
    state.planner.raw = util.trim(raw)
    state.planner.errors = state.planner.errors or {}
    state.planner.force_reason = ""

    if not calls then
        state.planner.errors[#state.planner.errors + 1] = tostring(err or "planner_parse_failed")
        if force_upload then
            calls = build_forced_upload_calls(state, max_upload_reads, upload_read_max_chars)
            state.planner.force_reason = "upload_toolchain_parse_fallback"
        else
            state.planner.tool_calls = {}
            return state
        end
    end

    if #calls <= 0 and force_upload then
        calls = build_forced_upload_calls(state, max_upload_reads, upload_read_max_chars)
        state.planner.force_reason = "upload_toolchain_empty_fallback"
    end

    if #calls > max_calls then
        for i = #calls, max_calls + 1, -1 do
            table.remove(calls, i)
        end
    end

    if #calls <= 0 then
        state.planner.tool_calls = {}
        return state
    end

    state.planner.tool_calls = calls
    return state
end

return M
