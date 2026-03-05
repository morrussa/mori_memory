local util = require("module.graph.util")
local config = require("module.config")
local tool_registry = require("module.graph.tool_registry_v2")

local M = {}

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

local function should_force_upload_toolchain(state)
    if not has_uploads(state) then
        return false
    end
    local executed_total = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0
    return executed_total <= 0
end

local function collect_history_rows(state)
    local rows = ((((state or {}).messages or {}).conversation_history) or {})
    if type(rows) ~= "table" then
        return {}
    end
    return rows
end

local function has_history_tool_path(state)
    local rows = collect_history_rows(state)
    for i = #rows, 1, -1 do
        local row = rows[i]
        if type(row) == "table" then
            local content = tostring(row.content or "")
            if content:find("tool_path=download/", 1, true) or content:find("download/", 1, true) then
                return true
            end
        end
    end
    return false
end

local function should_force_history_toolchain(state, user_input)
    local q = lower_text(user_input)
    if q == "" then
        return false
    end
    local likely_tool_intent = (
        q:find("读取", 1, true) ~= nil
        or q:find("查看", 1, true) ~= nil
        or q:find("前几行", 1, true) ~= nil
        or q:find("几行", 1, true) ~= nil
        or q:find("执行", 1, true) ~= nil
        or q:find("模型设置", 1, true) ~= nil
        or q:find("配置", 1, true) ~= nil
        or q:find("read", 1, true) ~= nil
        or q:find("line", 1, true) ~= nil
        or q:find("search", 1, true) ~= nil
        or q:find("download/", 1, true) ~= nil
    )
    if not likely_tool_intent then
        return false
    end

    local executed_total = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0
    if executed_total > 0 then
        return false
    end
    return has_history_tool_path(state)
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
        "",
        "[ConversationTail]",
        util.utf8_take(util.trim(table.concat((function()
            local lines = {}
            local rows = collect_history_rows(state)
            for i = #rows, 1, -1 do
                local row = rows[i]
                if type(row) == "table" then
                    local role = tostring(row.role or "")
                    local content = util.trim(row.content or "")
                    if content ~= "" then
                        lines[#lines + 1] = string.format("[%s] %s", role, content)
                    end
                    if #lines >= 6 then
                        break
                    end
                end
            end
            return lines
        end)(), "\n")), 2200),
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

local function extract_tool_paths_from_text(text, out, seen)
    local s = tostring(text or "")
    if s == "" then
        return
    end

    for path in s:gmatch("tool_path=([^,%s%)]+)") do
        local p = normalize_upload_tool_path({ tool_path = path })
        if p ~= "" and not seen[p] then
            seen[p] = true
            out[#out + 1] = p
        end
    end

    for path in s:gmatch("download/[0-9A-Za-z%._%-%/]+") do
        local p = normalize_upload_tool_path({ tool_path = path })
        if p ~= "" and not seen[p] then
            seen[p] = true
            out[#out + 1] = p
        end
    end
end

local function collect_known_tool_paths(state)
    local out = {}
    local seen = {}

    local uploads = ((state or {}).uploads) or {}
    for _, item in ipairs(uploads) do
        local p = normalize_upload_tool_path(item)
        if p ~= "" and not seen[p] then
            seen[p] = true
            out[#out + 1] = p
        end
    end

    local rows = collect_history_rows(state)
    for i = #rows, 1, -1 do
        local row = rows[i]
        if type(row) == "table" then
            extract_tool_paths_from_text(row.content or "", out, seen)
        end
    end

    local user_input = tostring((((state or {}).input or {}).message) or "")
    extract_tool_paths_from_text(user_input, out, seen)
    return out
end

local function build_forced_history_calls(state, known_paths, read_max_chars)
    local q = lower_text((((state or {}).input or {}).message) or "")
    local path = tostring((known_paths or {})[1] or "")
    if path == "" then
        return {}
    end

    local calls = {
        {
            tool = "list_files",
            args = { prefix = "download" },
            call_id = "forced_history_list_1",
        },
    }

    local wants_lines = (
        q:find("前几行", 1, true) ~= nil
        or q:find("几行", 1, true) ~= nil
        or q:find("line", 1, true) ~= nil
        or q:find("lines", 1, true) ~= nil
    )

    local mentions_config = (
        q:find("模型设置", 1, true) ~= nil
        or q:find("配置", 1, true) ~= nil
        or q:find("model", 1, true) ~= nil
        or q:find("config", 1, true) ~= nil
    )

    if wants_lines or mentions_config then
        calls[#calls + 1] = {
            tool = "read_lines",
            args = {
                path = path,
                start_line = 1,
                max_lines = mentions_config and 220 or 40,
            },
            call_id = "forced_history_read_lines_1",
        }
    else
        calls[#calls + 1] = {
            tool = "read_file",
            args = {
                path = path,
                max_chars = read_max_chars,
            },
            call_id = "forced_history_read_file_1",
        }
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
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local force_history_toolchain = util.to_bool(cfg.force_history_toolchain, true)
    local force_history = force_history_toolchain and should_force_history_toolchain(state, user_input)
    local max_upload_reads = math.max(1, math.floor(tonumber(cfg.force_upload_read_limit) or 2))
    local upload_read_max_chars = math.max(256, math.floor(tonumber(cfg.force_upload_read_max_chars) or 2400))
    local history_read_max_chars = math.max(256, math.floor(tonumber(cfg.force_history_read_max_chars) or upload_read_max_chars))
    local max_calls = math.max(1, math.floor(tonumber(cfg.max_calls_per_loop) or 6))
    local known_tool_paths = collect_known_tool_paths(state)

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
        elseif force_history then
            calls = build_forced_history_calls(state, known_tool_paths, history_read_max_chars)
            state.planner.force_reason = "history_toolchain_parse_fallback"
        else
            state.planner.tool_calls = {}
            return state
        end
    end

    if #calls <= 0 and force_upload then
        calls = build_forced_upload_calls(state, max_upload_reads, upload_read_max_chars)
        state.planner.force_reason = "upload_toolchain_empty_fallback"
    elseif #calls <= 0 and force_history then
        calls = build_forced_history_calls(state, known_tool_paths, history_read_max_chars)
        state.planner.force_reason = "history_toolchain_empty_fallback"
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
