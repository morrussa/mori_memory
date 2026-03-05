local M = {}

local tool = require("module.tool")

local DEFAULT_SUPPORTED_ACTS = {
    upsert_record = true,
    query_record = true,
    delete_record = true,
    list_agent_files = true,
    read_agent_file = true,
    read_agent_file_lines = true,
    search_agent_file = true,
    search_agent_files = true,
}

local ACT_ALIAS = {
    upsert = "upsert_record",
    upsert_record = "upsert_record",
    insert_record = "upsert_record",
    save_record = "upsert_record",

    query = "query_record",
    search = "query_record",
    search_record = "query_record",
    query_record = "query_record",

    delete = "delete_record",
    remove = "delete_record",
    remove_record = "delete_record",
    delete_record = "delete_record",

    list_files = "list_agent_files",
    list_file = "list_agent_files",
    list_uploaded_files = "list_agent_files",
    list_agent_files = "list_agent_files",

    read_file = "read_agent_file",
    open_file = "read_agent_file",
    read_uploaded_file = "read_agent_file",
    read_agent_file = "read_agent_file",

    read_file_lines = "read_agent_file_lines",
    open_file_lines = "read_agent_file_lines",
    read_agent_file_lines = "read_agent_file_lines",

    find_in_file = "search_agent_file",
    search_file = "search_agent_file",
    grep_file = "search_agent_file",
    search_agent_file = "search_agent_file",

    find_in_files = "search_agent_files",
    search_files = "search_agent_files",
    grep_files = "search_agent_files",
    search_agent_files = "search_agent_files",
}

local CALL_FIELDS = {
    "tool_call_id",
    "arguments",
    "string",
    "query",
    "path",
    "file",
    "prefix",
    "pattern",
    "type",
    "types",
    "entity",
    "evidence",
    "confidence",
    "start_char",
    "offset_char",
    "max_chars",
    "start_line",
    "end_line",
    "max_lines",
    "max_hits",
    "max_files",
    "per_file_hits",
    "context_lines",
    "regex",
    "case_sensitive",
    "limit",
    "namespace",
    "key",
    "value",
}

local function trim(s)
    if not s then return "" end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function normalize_act(name, supported_acts)
    local raw = trim(name):lower()
    if raw == "" then return nil end
    raw = raw:gsub("%-", "_")
    raw = raw:gsub("%s+", "_")
    local mapped = ACT_ALIAS[raw] or raw
    local allow = supported_acts or DEFAULT_SUPPORTED_ACTS
    if allow[mapped] then
        return mapped
    end
    return nil
end

local extract_balanced = nil

local function parse_lua_field(tbl_line, field)
    tbl_line = tostring(tbl_line or "")
    field = tostring(field or "")
    if field == "" or tbl_line == "" then
        return nil
    end

    local safe_field = field:gsub("(%W)", "%%%1")
    local p1, p2 = tbl_line:find("%f[%w_]" .. safe_field .. "%f[^%w_]%s*=%s*")
    if not p1 then
        return nil
    end
    local i = (p2 or 0) + 1
    while i <= #tbl_line do
        local ch = tbl_line:sub(i, i)
        if ch == " " or ch == "\t" or ch == "\r" or ch == "\n" then
            i = i + 1
        else
            break
        end
    end
    if i > #tbl_line then
        return nil
    end

    local first = tbl_line:sub(i, i)
    if first == '"' or first == "'" then
        local quote = first
        i = i + 1
        local out = {}
        while i <= #tbl_line do
            local ch = tbl_line:sub(i, i)
            if ch == "\\" then
                local nxt = tbl_line:sub(i + 1, i + 1)
                if nxt ~= "" then
                    out[#out + 1] = nxt
                    i = i + 2
                else
                    i = i + 1
                end
            elseif ch == quote then
                return table.concat(out)
            else
                out[#out + 1] = ch
                i = i + 1
            end
        end
        return table.concat(out)
    end

    if first == "{" then
        if extract_balanced then
            local seg = extract_balanced(tbl_line, i, "{", "}")
            if seg then
                return trim(seg)
            end
        end
    end

    local raw = tbl_line:match("%f[%w_]" .. safe_field .. "%f[^%w_]%s*=%s*([^,%}]+)")
    if raw then return trim(raw) end
    return nil
end

extract_balanced = function(text, start_pos, open_char, close_char)
    local n = #text
    local i = tonumber(start_pos) or 1
    if i < 1 or i > n then
        return nil, nil, nil
    end
    if text:sub(i, i) ~= open_char then
        return nil, nil, nil
    end

    local depth = 0
    local quote = nil
    local start_idx = i

    while i <= n do
        local ch = text:sub(i, i)
        if quote then
            if ch == "\\" then
                i = i + 2
            elseif ch == quote then
                quote = nil
                i = i + 1
            else
                i = i + 1
            end
        else
            if ch == '"' or ch == "'" then
                quote = ch
                i = i + 1
            elseif ch == open_char then
                depth = depth + 1
                i = i + 1
            elseif ch == close_char then
                depth = depth - 1
                i = i + 1
                if depth == 0 then
                    local end_idx = i - 1
                    return text:sub(start_idx, end_idx), start_idx, end_idx
                end
            else
                i = i + 1
            end
        end
    end

    return nil, nil, nil
end

local function parse_mixed_field(blob, field)
    blob = tostring(blob or "")
    if blob == "" then return nil end

    local lua_v = parse_lua_field(blob, field)
    if lua_v ~= nil and lua_v ~= "" then
        return trim(lua_v)
    end

    return nil
end

local function pick_blob_field(primary_blob, fallback_blob, field)
    local v = parse_mixed_field(primary_blob, field)
    if v == nil or v == "" then
        v = parse_mixed_field(fallback_blob, field)
    end
    return v
end

local function build_call(act, raw, args_blob, fallback_blob)
    local call = {
        raw = trim(raw or ""),
        act = act,
    }

    local from_args = tostring(args_blob or "")
    local from_fallback = tostring(fallback_blob or "")

    for _, field in ipairs(CALL_FIELDS) do
        call[field] = pick_blob_field(from_args, from_fallback, field)
    end

    return call
end

local function parse_lua_call_block(raw, supported_acts)
    local s = trim(raw)
    if s == "" then return nil end
    if not s:match("^%b{}$") then
        local first = (tool.extract_first_lua_table and tool.extract_first_lua_table(s)) or s:match("%b{}")
        if not first then return nil end
        s = trim(first)
    end

    local act_raw = parse_lua_field(s, "act")
    if not act_raw then return nil end
    act_raw = act_raw:gsub('^["\'](.-)["\']$', "%1")
    local act = normalize_act(act_raw, supported_acts)
    if not act then return nil end

    return build_call(act, s, s, s)
end

local function append_unique_call(calls, seen, call)
    if not call or type(call) ~= "table" then return end
    local sig = table.concat({
        tostring(call.act or ""),
        tostring(call.string or ""),
        tostring(call.query or ""),
        tostring(call.type or ""),
        tostring(call.types or ""),
        tostring(call.entity or ""),
        tostring(call.value or ""),
        tostring(call.confidence or ""),
    }, "\x1F")
    if seen[sig] then return end
    seen[sig] = true
    calls[#calls + 1] = call
end

local function add_span(spans, s, e)
    s = tonumber(s) or 0
    e = tonumber(e) or 0
    if s <= 0 or e <= 0 or e < s then return end
    spans[#spans + 1] = { s = s, e = e }
end

local function parse_lua_lines(text, calls, spans, seen, supported_acts)
    local cursor = 1
    for line in (text .. "\n"):gmatch("(.-)\n") do
        local line_start = cursor
        local line_end = cursor + #line - 1
        cursor = cursor + #line + 1
        local call = parse_lua_call_block(line, supported_acts)
        if call then
            append_unique_call(calls, seen, call)
            add_span(spans, line_start, math.max(line_start, line_end))
        end
    end
end

local function is_lua_table_literal(text)
    local s = trim(text)
    if s == "" then return false end
    if not s:match("^%b{}$") then
        return false
    end
    local chunk = load("return " .. s, "tool_args", "t", {})
    if not chunk then
        return false
    end
    local ok, value = pcall(chunk)
    return ok and type(value) == "table"
end

local function parse_qwen_symbol_calls(text, calls, spans, seen, supported_acts)
    local fn_tag = "✿FUNCTION✿"
    local args_tag = "✿ARGS✿"
    local pos = 1
    local n = #text

    while pos <= n do
        local s_fn = text:find(fn_tag, pos, true)
        if not s_fn then break end

        local colon_fn = text:find(":", s_fn + #fn_tag, true)
        if not colon_fn then
            pos = s_fn + #fn_tag
            goto continue
        end
        local fn_line_end = text:find("\n", colon_fn + 1, true) or (n + 1)
        local fn_name = trim(text:sub(colon_fn + 1, fn_line_end - 1))
        local act = normalize_act(fn_name, supported_acts)

        local s_args = text:find(args_tag, fn_line_end, true)
        if not s_args then
            if act then
                append_unique_call(calls, seen, build_call(act, fn_name, "", ""))
                add_span(spans, s_fn, fn_line_end - 1)
            end
            pos = fn_line_end + 1
            goto continue
        end

        local colon_args = text:find(":", s_args + #args_tag, true)
        if not colon_args then
            pos = s_args + #args_tag
            goto continue
        end

        local next_fn = text:find(fn_tag, colon_args + 1, true)
        local end_pos = next_fn and (next_fn - 1) or n
        local args_blob = trim(text:sub(colon_args + 1, end_pos))

        if act then
            local args_clean = trim(args_blob)
            if args_clean == "" or is_lua_table_literal(args_clean) then
                local call = build_call(act, text:sub(s_fn, end_pos), args_clean, args_clean)
                if args_clean ~= "" then
                    call.arguments = args_clean
                end
                append_unique_call(calls, seen, call)
                add_span(spans, s_fn, end_pos)
            end
        end
        pos = (next_fn or (n + 1))
        ::continue::
    end
end

local function is_react_boundary_line(line)
    local low = trim(line):lower()
    if low == "" then return false end
    if low:match("^action%s*:") then return true end
    if low:match("^observation%s*:") then return true end
    if low:match("^final%s+answer%s*:") then return true end
    if low:match("^thought%s*:") then return true end
    if low:match("^question%s*:") then return true end
    return false
end

local function parse_react_action_calls(text, calls, spans, seen, supported_acts)
    local lines = {}
    local cursor = 1
    for line in (text .. "\n"):gmatch("(.-)\n") do
        local line_start = cursor
        local line_end = cursor + #line - 1
        if line_end < line_start then
            line_end = line_start
        end
        lines[#lines + 1] = {
            text = line,
            s = line_start,
            e = line_end,
        }
        cursor = cursor + #line + 1
    end

    local idx = 1
    while idx <= #lines do
        local line = lines[idx].text
        local action_name = line:match("^%s*[Aa]ction%s*:%s*(.-)%s*$")
        if action_name and action_name ~= "" then
            local act = normalize_act(action_name, supported_acts)
            local raw_start = lines[idx].s
            local raw_end = lines[idx].e
            local args_blob = ""
            local next_idx = idx + 1

            if next_idx <= #lines then
                local first_input = lines[next_idx].text:match("^%s*[Aa]ction%s*[Ii]nput%s*:%s*(.*)$")
                if first_input ~= nil then
                    args_blob = tostring(first_input or "")
                    raw_end = lines[next_idx].e
                    next_idx = next_idx + 1
                    while next_idx <= #lines do
                        local next_line = lines[next_idx].text
                        if is_react_boundary_line(next_line) then
                            break
                        end
                        args_blob = args_blob .. "\n" .. tostring(next_line or "")
                        raw_end = lines[next_idx].e
                        next_idx = next_idx + 1
                    end
                end
            end

            if act then
                local args_clean = trim(args_blob)
                if args_clean == "" or is_lua_table_literal(args_clean) then
                    local raw_block = trim(text:sub(raw_start, raw_end))
                    local call = build_call(act, raw_block, args_clean, args_clean)
                    if args_clean ~= "" then
                        call.arguments = args_clean
                    end
                    append_unique_call(calls, seen, call)
                    add_span(spans, raw_start, raw_end)
                end
            end

            if next_idx > idx then
                idx = next_idx
            else
                idx = idx + 1
            end
        else
            idx = idx + 1
        end
    end
end

local function merge_ranges(ranges)
    if type(ranges) ~= "table" or #ranges == 0 then
        return {}
    end
    table.sort(ranges, function(a, b)
        if a.s == b.s then
            return a.e < b.e
        end
        return a.s < b.s
    end)

    local merged = {}
    for _, r in ipairs(ranges) do
        if #merged == 0 then
            merged[#merged + 1] = { s = r.s, e = r.e }
        else
            local last = merged[#merged]
            if r.s <= (last.e + 1) then
                if r.e > last.e then
                    last.e = r.e
                end
            else
                merged[#merged + 1] = { s = r.s, e = r.e }
            end
        end
    end
    return merged
end

local function strip_ranges(text, ranges)
    if #ranges == 0 then
        return trim(text)
    end
    local merged = merge_ranges(ranges)
    local out = {}
    local pos = 1
    local n = #text
    for _, r in ipairs(merged) do
        local s = math.max(1, tonumber(r.s) or 1)
        local e = math.min(n, tonumber(r.e) or 0)
        if s > pos then
            out[#out + 1] = text:sub(pos, s - 1)
        end
        pos = math.max(pos, e + 1)
    end
    if pos <= n then
        out[#out + 1] = text:sub(pos)
    end
    local visible = table.concat(out)
    return trim(visible)
end

local function parse_calls_with_spans(text, opts)
    local supported_acts = (opts or {}).supported_acts or DEFAULT_SUPPORTED_ACTS
    text = tostring(text or "")
    local calls, spans = {}, {}
    local seen = {}

    -- 协议收敛：移除 XML/JSON 作为工具调用入口，仅保留 Lua table + Qwen/ReAct。
    parse_qwen_symbol_calls(text, calls, spans, seen, supported_acts)
    parse_react_action_calls(text, calls, spans, seen, supported_acts)
    parse_lua_lines(text, calls, spans, seen, supported_acts)

    return calls, spans
end

local function collect_first_call_with_parser(text, supported_acts, parser_fn)
    local calls = {}
    local spans = {}
    local seen = {}
    parser_fn(text, calls, spans, seen, supported_acts)
    if #calls > 0 then
        return calls[1]
    end
    return nil
end

local function copy_supported_acts(source)
    local out = {}
    for act, enabled in pairs(source or {}) do
        if enabled then
            out[act] = true
        end
    end
    return out
end

function M.collect_tool_calls_only(text, opts)
    local calls = parse_calls_with_spans(text, opts)
    return calls
end

function M.split_tool_calls_and_text(text, opts)
    text = tostring(text or "")
    local calls, spans = parse_calls_with_spans(text, opts)
    local visible = strip_ranges(text, spans)
    return calls, visible
end

function M.parse_tool_call_line(line, opts)
    local supported_acts = (opts or {}).supported_acts or DEFAULT_SUPPORTED_ACTS
    local text = tostring(line or "")

    local call = parse_lua_call_block(text, supported_acts)
    if call then return call end

    call = collect_first_call_with_parser(text, supported_acts, parse_qwen_symbol_calls)
    if call then return call end

    call = collect_first_call_with_parser(text, supported_acts, parse_react_action_calls)
    if call then return call end

    return nil
end

function M.clone_supported_acts(overrides)
    local acts = copy_supported_acts(DEFAULT_SUPPORTED_ACTS)
    if type(overrides) == "table" then
        for act, enabled in pairs(overrides) do
            if enabled then
                acts[act] = true
            else
                acts[act] = nil
            end
        end
    end
    return acts
end

function M.normalize_function_choice(raw_choice, opts)
    local supported_acts = ((opts or {}).supported_acts) or DEFAULT_SUPPORTED_ACTS
    local raw = trim(raw_choice):lower()
    if raw == "" or raw == "auto" then
        return "auto"
    end
    if raw == "none" then
        return "none"
    end
    raw = raw:gsub("%-", "_")
    raw = raw:gsub("%s+", "_")
    if supported_acts[raw] then
        return raw
    end
    return "auto"
end

return M
