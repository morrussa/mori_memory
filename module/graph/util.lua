local M = {}

local function trim(s)
    if s == nil then
        return ""
    end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function to_bool(v, fallback)
    if type(v) == "boolean" then
        return v
    end
    if type(v) == "number" then
        return v ~= 0
    end
    if type(v) == "string" then
        local s = v:lower()
        if s == "1" or s == "true" or s == "yes" or s == "on" then
            return true
        end
        if s == "0" or s == "false" or s == "no" or s == "off" then
            return false
        end
    end
    return fallback == true
end

local function cfg_number(v, fallback, min_v, max_v)
    local n = tonumber(v)
    if not n then
        n = tonumber(fallback) or 0
    end
    if min_v and n < min_v then
        n = min_v
    end
    if max_v and n > max_v then
        n = max_v
    end
    return n
end

local function utf8_take(s, max_chars)
    s = tostring(s or "")
    max_chars = tonumber(max_chars) or 0
    if max_chars <= 0 then
        return s
    end

    local out = {}
    local count = 0
    for ch in s:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        count = count + 1
        if count > max_chars then
            break
        end
        out[count] = ch
    end
    return table.concat(out)
end

local function shallow_copy(tbl)
    local out = {}
    for k, v in pairs(tbl or {}) do
        out[k] = v
    end
    return out
end

local function is_array_like_table(tbl)
    if type(tbl) ~= "table" then
        return false, 0
    end
    local count = 0
    local max_idx = 0
    for k, _ in pairs(tbl) do
        if type(k) ~= "number" or k < 1 or k % 1 ~= 0 then
            return false, 0
        end
        count = count + 1
        if k > max_idx then
            max_idx = k
        end
    end
    if max_idx ~= count then
        return false, 0
    end
    return true, max_idx
end

local function lua_escape_str(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\")
    s = s:gsub('"', '\\"')
    s = s:gsub("\r", "\\r")
    s = s:gsub("\n", "\\n")
    return s
end

local function encode_lua_value(v, depth)
    depth = tonumber(depth) or 0
    if depth > 32 then
        return "nil"
    end

    local vt = type(v)
    if vt == "table" then
        local is_arr, arr_len = is_array_like_table(v)
        if is_arr then
            local parts = {}
            for i = 1, arr_len do
                parts[#parts + 1] = encode_lua_value(v[i], depth + 1)
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end

        local entries = {}
        for k, value in pairs(v) do
            entries[#entries + 1] = { key = k, value = value, key_text = tostring(k) }
        end
        table.sort(entries, function(a, b)
            return a.key_text < b.key_text
        end)

        local parts = {}
        for _, item in ipairs(entries) do
            local k = item.key
            local key_expr = ""
            if type(k) == "string" and k:match("^[A-Za-z_][A-Za-z0-9_]*$") then
                key_expr = k
            elseif type(k) == "number" then
                key_expr = "[" .. tostring(k) .. "]"
            else
                key_expr = '["' .. lua_escape_str(tostring(k)) .. '"]'
            end
            parts[#parts + 1] = key_expr .. "=" .. encode_lua_value(item.value, depth + 1)
        end
        return "{" .. table.concat(parts, ",") .. "}"
    end

    if vt == "string" then
        return '"' .. lua_escape_str(v) .. '"'
    end
    if vt == "number" then
        if v ~= v or v == math.huge or v == -math.huge then
            return "0"
        end
        return tostring(v)
    end
    if vt == "boolean" then
        return v and "true" or "false"
    end
    if v == nil then
        return "nil"
    end
    return '"' .. lua_escape_str(tostring(v)) .. '"'
end

local function parse_lua_table_literal(raw)
    local text = trim(raw)
    if text == "" or (not text:match("^%b{}$")) then
        return nil, "not_lua_table"
    end
    local chunk, load_err = load("return " .. text, "graph_literal", "t", {})
    if not chunk then
        return nil, tostring(load_err or "load_failed")
    end
    local ok, parsed = pcall(chunk)
    if not ok then
        return nil, tostring(parsed or "eval_failed")
    end
    if type(parsed) ~= "table" then
        return nil, "not_table"
    end
    return parsed
end

local function json_escape(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\")
    s = s:gsub('"', '\\"')
    s = s:gsub("\b", "\\b")
    s = s:gsub("\f", "\\f")
    s = s:gsub("\n", "\\n")
    s = s:gsub("\r", "\\r")
    s = s:gsub("\t", "\\t")
    return s
end

local function json_encode(v, depth)
    depth = tonumber(depth) or 0
    if depth > 32 then
        return "null"
    end

    local vt = type(v)
    if vt == "nil" then
        return "null"
    end
    if vt == "boolean" then
        return v and "true" or "false"
    end
    if vt == "number" then
        if v ~= v or v == math.huge or v == -math.huge then
            return "null"
        end
        return tostring(v)
    end
    if vt == "string" then
        return '"' .. json_escape(v) .. '"'
    end
    if vt ~= "table" then
        return '"' .. json_escape(tostring(v)) .. '"'
    end

    local is_arr, arr_len = is_array_like_table(v)
    if is_arr then
        local parts = {}
        for i = 1, arr_len do
            parts[#parts + 1] = json_encode(v[i], depth + 1)
        end
        return "[" .. table.concat(parts, ",") .. "]"
    end

    local entries = {}
    for k, value in pairs(v) do
        entries[#entries + 1] = { key = tostring(k), value = value }
    end
    table.sort(entries, function(a, b)
        return a.key < b.key
    end)

    local parts = {}
    for _, item in ipairs(entries) do
        parts[#parts + 1] = '"' .. json_escape(item.key) .. '":' .. json_encode(item.value, depth + 1)
    end
    return "{" .. table.concat(parts, ",") .. "}"
end

local function ensure_dir(path)
    local p = tostring(path or "")
    if p == "" then
        return
    end
    os.execute(string.format('mkdir -p "%s"', p:gsub('"', '\\"')))
end

local function random_hex(n)
    local len = math.max(1, math.floor(tonumber(n) or 16))
    local out = {}
    for i = 1, len do
        out[i] = string.format("%x", math.random(0, 15))
    end
    return table.concat(out)
end

local function new_run_id()
    local ts = os.time()
    return string.format("run_%d_%s", ts, random_hex(16))
end

local function now_ms()
    return math.floor((os.time() or 0) * 1000)
end

M.trim = trim
M.to_bool = to_bool
M.cfg_number = cfg_number
M.utf8_take = utf8_take
M.shallow_copy = shallow_copy
M.is_array_like_table = is_array_like_table
M.encode_lua_value = encode_lua_value
M.parse_lua_table_literal = parse_lua_table_literal
M.json_encode = json_encode
M.ensure_dir = ensure_dir
M.new_run_id = new_run_id
M.now_ms = now_ms

return M
