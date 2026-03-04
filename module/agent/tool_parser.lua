local M = {}

local tool = require("module.tool")

local DEFAULT_SUPPORTED_ACTS = {
    upsert_record = true,
    query_record = true,
    delete_record = true,
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
}

local CALL_FIELDS = {
    "string",
    "query",
    "type",
    "types",
    "entity",
    "evidence",
    "confidence",
    "namespace",
    "key",
    "value",
}

local function trim(s)
    if not s then return "" end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function is_space(ch)
    return ch == " " or ch == "\t" or ch == "\r" or ch == "\n"
end

local function skip_spaces(text, pos)
    local i = pos or 1
    local n = #text
    while i <= n and is_space(text:sub(i, i)) do
        i = i + 1
    end
    return i
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

local function parse_lua_field(tbl_line, field)
    local dq = tbl_line:match(field .. '%s*=%s*"([^"]*)"')
    if dq then return dq end
    local sq = tbl_line:match(field .. "%s*=%s*'([^']*)'")
    if sq then return sq end
    local raw = tbl_line:match(field .. "%s*=%s*([^,%}]+)")
    if raw then return trim(raw) end
    return nil
end

local function extract_balanced(text, start_pos, open_char, close_char)
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

local function parse_json_string(text, start_pos)
    local pos = tonumber(start_pos) or 1
    if text:sub(pos, pos) ~= '"' then
        return nil, pos
    end
    pos = pos + 1
    local out = {}
    local n = #text
    while pos <= n do
        local ch = text:sub(pos, pos)
        if ch == '"' then
            return table.concat(out), pos + 1
        end
        if ch == "\\" then
            local nxt = text:sub(pos + 1, pos + 1)
            if nxt == "" then
                return table.concat(out), n + 1
            end
            if nxt == "n" then
                out[#out + 1] = "\n"
            elseif nxt == "r" then
                out[#out + 1] = "\r"
            elseif nxt == "t" then
                out[#out + 1] = "\t"
            elseif nxt == "b" then
                out[#out + 1] = "\b"
            elseif nxt == "f" then
                out[#out + 1] = "\f"
            elseif nxt == '"' then
                out[#out + 1] = '"'
            elseif nxt == "\\" then
                out[#out + 1] = "\\"
            elseif nxt == "/" then
                out[#out + 1] = "/"
            elseif nxt == "u" then
                local hx = text:sub(pos + 2, pos + 5)
                if #hx == 4 and hx:match("^[0-9a-fA-F]+$") then
                    out[#out + 1] = "\\u" .. hx
                    pos = pos + 6
                    goto continue
                else
                    out[#out + 1] = "u"
                end
            else
                out[#out + 1] = nxt
            end
            pos = pos + 2
        else
            out[#out + 1] = ch
            pos = pos + 1
        end
        ::continue::
    end
    return table.concat(out), pos
end

local function parse_json_value(text, start_pos)
    local pos = skip_spaces(text, start_pos)
    local n = #text
    if pos > n then
        return nil, "nil", pos
    end

    local ch = text:sub(pos, pos)
    if ch == '"' then
        local val, next_pos = parse_json_string(text, pos)
        return val, "string", next_pos
    end
    if ch == "{" then
        local seg, _, e = extract_balanced(text, pos, "{", "}")
        if seg then
            return seg, "object", e + 1
        end
        return nil, "object", pos + 1
    end
    if ch == "[" then
        local seg, _, e = extract_balanced(text, pos, "[", "]")
        if seg then
            return seg, "array", e + 1
        end
        return nil, "array", pos + 1
    end

    local i = pos
    while i <= n do
        local c = text:sub(i, i)
        if c == "," or c == "}" or c == "]" then
            break
        end
        i = i + 1
    end
    local token = trim(text:sub(pos, i - 1))
    return token, "literal", i
end

local function find_json_key_value(text, key, from_pos)
    local pos = tonumber(from_pos) or 1
    local needle = '"' .. tostring(key) .. '"'
    while true do
        local i = text:find(needle, pos, true)
        if not i then
            return nil, nil, nil, nil
        end
        local j = i + #needle
        j = skip_spaces(text, j)
        if text:sub(j, j) ~= ":" then
            pos = j + 1
        else
            local value_start = skip_spaces(text, j + 1)
            local value, kind, next_pos = parse_json_value(text, value_start)
            return value, kind, value_start, next_pos
        end
    end
end

local function parse_mixed_field(blob, field)
    blob = tostring(blob or "")
    if blob == "" then return nil end

    local value, kind = find_json_key_value(blob, field, 1)
    if value ~= nil then
        if kind == "string" then
            return trim(value)
        end
        return trim(tostring(value))
    end

    local dq = blob:match('"' .. field .. '"%s*:%s*"([^"]*)"')
    if dq then return trim(dq) end
    local sq = blob:match("'" .. field .. "'%s*:%s*'([^']*)'")
    if sq then return trim(sq) end
    local raw_json = blob:match('"' .. field .. '"%s*:%s*([^,%}%]]+)')
    if raw_json then return trim(raw_json) end

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

local function first_non_empty_field(blob, names)
    for _, name in ipairs(names or {}) do
        local v = parse_mixed_field(blob, name)
        if v ~= nil and v ~= "" then
            return v
        end
    end
    return nil
end

local function json_value_to_blob(value, kind)
    if value == nil then
        return nil
    end
    if kind == "string" then
        return tostring(value)
    end
    return trim(tostring(value))
end

local function extract_json_args_blob(obj_text)
    local args_val, args_kind = find_json_key_value(obj_text, "arguments", 1)
    local args_blob = json_value_to_blob(args_val, args_kind)
    if args_blob == nil then
        local input_val, input_kind = find_json_key_value(obj_text, "input", 1)
        args_blob = json_value_to_blob(input_val, input_kind)
    end
    if trim(args_blob or "") == "" then
        return obj_text
    end
    return args_blob
end

local function parse_json_function_object(raw, supported_acts)
    local s = trim(raw)
    if s == "" then return nil end
    if s:sub(1, 1) ~= "{" then return nil end

    local name = first_non_empty_field(s, { "name", "act" })
    if not name then return nil end

    local act = normalize_act(name, supported_acts)
    if not act then return nil end

    local args_blob = extract_json_args_blob(s)
    return build_call(act, s, args_blob, s)
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
            append_unique_call(calls, seen, build_call(act, text:sub(s_fn, end_pos), args_blob, args_blob))
            add_span(spans, s_fn, end_pos)
        end
        pos = (next_fn or (n + 1))
        ::continue::
    end
end

local function parse_tool_call_xml(text, calls, spans, seen, supported_acts)
    local lower = text:lower()
    local open_tag = "<tool_call>"
    local close_tag = "</tool_call>"
    local pos = 1

    while true do
        local s_tag, e_tag = lower:find(open_tag, pos, true)
        if not s_tag then break end
        local s_close, e_close = lower:find(close_tag, e_tag + 1, true)
        if not s_close then break end

        local inner = text:sub(e_tag + 1, s_close - 1)
        local found = false
        local scan = 1
        while scan <= #inner do
            local obj_start = inner:find("{", scan, true)
            if not obj_start then break end
            local obj, _, obj_end = extract_balanced(inner, obj_start, "{", "}")
            if not obj then
                scan = obj_start + 1
            else
                local call = parse_json_function_object(obj, supported_acts)
                if call then
                    append_unique_call(calls, seen, call)
                    found = true
                end
                scan = obj_end + 1
            end
        end

        if found then
            add_span(spans, s_tag, e_close)
        end

        pos = e_close + 1
    end
end

local function parse_json_objects(text, calls, spans, seen, supported_acts)
    local pos = 1
    local n = #text
    while pos <= n do
        local obj_start = text:find("{", pos, true)
        if not obj_start then break end
        local obj, _, obj_end = extract_balanced(text, obj_start, "{", "}")
        if not obj then
            pos = obj_start + 1
        else
            local call = parse_json_function_object(obj, supported_acts)
            if call then
                append_unique_call(calls, seen, call)
                add_span(spans, obj_start, obj_end)
            end
            pos = obj_end + 1
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

    parse_tool_call_xml(text, calls, spans, seen, supported_acts)
    parse_qwen_symbol_calls(text, calls, spans, seen, supported_acts)
    parse_lua_lines(text, calls, spans, seen, supported_acts)
    parse_json_objects(text, calls, spans, seen, supported_acts)

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

    call = parse_json_function_object(text, supported_acts)
    if call then return call end

    call = collect_first_call_with_parser(text, supported_acts, parse_qwen_symbol_calls)
    if call then return call end

    call = collect_first_call_with_parser(text, supported_acts, parse_tool_call_xml)
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
