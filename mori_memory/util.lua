local M = {}

local function trim(s)
    if s == nil then
        return ""
    end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
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
    if text == "" then
        return nil, "not_lua_table"
    end

    if not text:match("^%b{}$") then
        local candidate = text:match("(%b{})")
        if not candidate then
            return nil, "not_lua_table"
        end
        text = candidate
    end

    local chunk, load_err = load("return " .. text, "mori_memory_literal", "t", {})
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

M.trim = trim
M.utf8_take = utf8_take
M.encode_lua_value = encode_lua_value
M.parse_lua_table_literal = parse_lua_table_literal

return M

