local ok_ffi, ffi = pcall(require, "ffi")
if not ok_ffi then
    ffi = nil
end

local M = {}
local WORKSPACE_VIRTUAL_ROOT = "/mori/workspace"
local _clock_gettime_cdef_ready = false

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

local function utf8_len(s)
    local count = 0
    for _ in tostring(s or ""):gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        count = count + 1
    end
    return count
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
    -- 使用 LuaJIT 的 FFI 获取高精度时间戳
    if type(jit) == "table" and ffi then
        if not _clock_gettime_cdef_ready then
            local ok = pcall(function()
                ffi.cdef[[
                    typedef long time_t;
                    typedef struct timespec {
                        time_t tv_sec;
                        long tv_nsec;
                    } timespec;
                    int clock_gettime(int clk_id, struct timespec *tp);
                ]]
            end)
            _clock_gettime_cdef_ready = ok
        end
        local CLOCK_REALTIME = 0
        local ts = ffi.new("struct timespec")
        if ffi.C.clock_gettime(CLOCK_REALTIME, ts) == 0 then
            return tonumber(ts.tv_sec) * 1000 + math.floor(tonumber(ts.tv_nsec) / 1000000)
        end
    end
    -- 回退到秒级精度
    return math.floor((os.time() or 0) * 1000)
end

local function workspace_virtual_root()
    return WORKSPACE_VIRTUAL_ROOT
end

local function build_workspace_virtual_path(rel_path)
    local rel = trim(rel_path)
    if rel == "" then
        return WORKSPACE_VIRTUAL_ROOT
    end
    rel = tostring(rel):gsub("\\", "/")
    while rel:sub(1, 2) == "./" do
        rel = rel:sub(3)
    end
    rel = rel:gsub("^/+", "")
    if rel == "" then
        return WORKSPACE_VIRTUAL_ROOT
    end
    return WORKSPACE_VIRTUAL_ROOT .. "/" .. rel
end

local function normalize_tool_path(raw)
    local path = trim(raw)
    if path == "" then
        return ""
    end
    path = tostring(path):gsub("\\", "/")

    while path:sub(1, 2) == "./" do
        path = path:sub(3)
    end

    if path == WORKSPACE_VIRTUAL_ROOT then
        return WORKSPACE_VIRTUAL_ROOT
    end
    if path:sub(1, #WORKSPACE_VIRTUAL_ROOT + 1) == WORKSPACE_VIRTUAL_ROOT .. "/" then
        local rel = path:sub(#WORKSPACE_VIRTUAL_ROOT + 2)
        rel = rel:gsub("^/+", "")
        if rel == "" then
            return WORKSPACE_VIRTUAL_ROOT
        end
        local norm = rel
        local segments = {}
        for seg in norm:gmatch("[^/]+") do
            if seg == ".." then
                return ""
            end
            if seg ~= "." and seg ~= "" then
                segments[#segments + 1] = seg
            end
        end
        return build_workspace_virtual_path(table.concat(segments, "/"))
    end

    if path:sub(1, 1) == "/" then
        return ""
    end

    if path:sub(1, 10) == "workspace/" or path:sub(1, 12) == "agent_files/" then
        return ""
    end
    local legacy_idx = path:find("/workspace/", 1, true)
    if legacy_idx then
        return ""
    end

    path = path:gsub("^/+", "")
    if path == "" then
        return WORKSPACE_VIRTUAL_ROOT
    end

    local segments = {}
    for seg in path:gmatch("[^/]+") do
        if seg == ".." then
            return ""
        end
        if seg ~= "." and seg ~= "" then
            segments[#segments + 1] = seg
        end
    end
    return build_workspace_virtual_path(table.concat(segments, "/"))
end

local function relative_workspace_path(raw)
    local path = normalize_tool_path(raw)
    if path == "" then
        return ""
    end
    if path == WORKSPACE_VIRTUAL_ROOT then
        return ""
    end
    return path:sub(#WORKSPACE_VIRTUAL_ROOT + 2)
end

local EXPLICIT_CONTINUE_PREFIXES = {
    "继续",
    "接着",
    "接下来",
    "继续处理",
    "继续做",
    "继续改",
    "继续修",
    "继续这个",
    "继续上次",
    "继续刚才",
}

local EXPLICIT_CONTINUE_PREFIXES_EN = {
    "continue",
    "resume",
    "go on",
    "carry on",
    "keep going",
    "pick up where we left off",
    "continue with",
    "resume the task",
}

local FOLLOWUP_PREFIXES = {
    "那就",
    "那把",
    "然后",
    "然后把",
    "接着把",
    "接下来",
    "顺手",
    "顺便把",
    "再把",
    "再跑",
    "再试",
    "再看",
}

local FOLLOWUP_PREFIXES_EN = {
    "then ",
    "next ",
    "so ",
    "and then ",
}

local SHORT_FOLLOWUP_ACTIONS = {
    "跑一下测试",
    "跑下测试",
    "测一下",
    "试一下",
    "看看结果",
    "看下结果",
    "看看报错",
    "看下报错",
    "修一下",
    "补一下",
    "改一下",
    "提交一下",
    "检查一下",
    "再检查一下",
    "再跑一下",
    "再试一下",
}

local SHORT_FOLLOWUP_ACTIONS_EN = {
    "run tests",
    "run the tests",
    "check the result",
    "try again",
    "fix it",
    "patch it",
}

local DEICTIC_CUES = {
    "这个",
    "这个任务",
    "这个文件",
    "这个改动",
    "它",
    "它们",
    "上次",
    "刚才",
    "前面",
    "上述",
    "剩下",
    "余下",
    "后面",
    "同一个",
    "这一步",
    "那一步",
    "这里",
    "那里",
}

local DEICTIC_CUES_EN = {
    "same task",
    "same file",
    "previous",
    "earlier",
    "above",
    "remaining",
    "rest of it",
    "that file",
    "that task",
}

local ACTION_CUES = {
    "改",
    "修",
    "补",
    "跑",
    "测",
    "试",
    "看",
    "查",
    "完成",
    "处理",
    "提交",
    "优化",
}

local ACTION_CUES_EN = {
    "fix",
    "patch",
    "run",
    "test",
    "check",
    "finish",
    "complete",
    "apply",
    "update",
    "review",
}

local NEW_TOPIC_PREFIXES = {
    "换个问题",
    "换个任务",
    "新问题",
    "新任务",
    "另开一个",
    "另外一个问题",
    "题外话",
}

local NEW_TOPIC_PREFIXES_EN = {
    "btw",
    "by the way",
    "unrelated",
    "separate question",
    "different task",
}

local FOLLOWUP_STOPWORDS = {
    ["resume"] = true,
    ["original"] = true,
    ["task"] = true,
    ["tests"] = true,
    ["test"] = true,
    ["file"] = true,
    ["workspace"] = true,
    ["patch"] = true,
    ["apply"] = true,
    ["run"] = true,
    ["current"] = true,
    ["plan"] = true,
    ["goal"] = true,
    ["open"] = true,
}

local function starts_with_any(text, patterns)
    local s = tostring(text or "")
    for _, pattern in ipairs(patterns or {}) do
        local candidate = tostring(pattern or "")
        if candidate ~= "" and s:sub(1, #candidate) == candidate then
            return true, candidate
        end
    end
    return false, ""
end

local function contains_any(text, patterns)
    local s = tostring(text or "")
    for _, pattern in ipairs(patterns or {}) do
        local candidate = tostring(pattern or "")
        if candidate ~= "" and s:find(candidate, 1, true) ~= nil then
            return true, candidate
        end
    end
    return false, ""
end

local function path_basename(raw)
    local path = tostring(raw or "")
    if path == "" then
        return ""
    end
    return path:match("([^/]+)$") or path
end

local function add_followup_ref(dst, raw)
    local token = trim(raw):lower()
    if token == "" or FOLLOWUP_STOPWORDS[token] then
        return
    end
    if not token:find("/", 1, true)
        and not token:find(".", 1, true)
        and not token:find("_", 1, true)
        and not token:find("%d")
        and #token < 6 then
        return
    end
    dst[token] = true
end

local function collect_followup_refs(active_task, working_memory)
    local refs = {}

    local function add_from_text(text)
        local lower = tostring(text or ""):lower()
        for token in lower:gmatch("[%w_%.%-/]+") do
            add_followup_ref(refs, token)
            add_followup_ref(refs, path_basename(token))
        end
    end

    add_from_text((active_task or {}).goal)
    add_from_text((active_task or {}).carryover_summary)
    add_from_text((working_memory or {}).current_plan)

    for path, enabled in pairs((working_memory or {}).files_read_set or {}) do
        if enabled then
            add_followup_ref(refs, path)
            add_followup_ref(refs, path_basename(path))
        end
    end
    for path, enabled in pairs((working_memory or {}).files_written_set or {}) do
        if enabled then
            add_followup_ref(refs, path)
            add_followup_ref(refs, path_basename(path))
        end
    end

    return refs
end

local function is_strong_followup_ref(ref)
    local token = tostring(ref or "")
    if token == "" then
        return false
    end
    if token:find("/", 1, true) ~= nil then
        return true
    end
    if token:find(".", 1, true) ~= nil then
        return true
    end
    return false
end

local function count_followup_ref_hits(text, refs)
    local lower = tostring(text or ""):lower()
    local strong_hits = 0
    local generic_hits = 0
    for ref, _ in pairs(refs or {}) do
        if lower:find(ref, 1, true) ~= nil then
            if is_strong_followup_ref(ref) then
                strong_hits = strong_hits + 1
            else
                generic_hits = generic_hits + 1
            end
        end
    end
    return strong_hits, generic_hits
end

local function is_continue_request(raw)
    local text = trim(raw)
    if text == "" then
        return false
    end
    local lower = text:lower()
    local cn_match = starts_with_any(text, EXPLICIT_CONTINUE_PREFIXES)
    local en_match = starts_with_any(lower, EXPLICIT_CONTINUE_PREFIXES_EN)
    return cn_match == true or en_match == true
end

local function should_continue_task(raw, active_task, working_memory, opts)
    local text = trim(raw)
    if text == "" then
        return false
    end
    if is_continue_request(text) then
        return true
    end

    active_task = type(active_task) == "table" and active_task or {}
    working_memory = type(working_memory) == "table" and working_memory or {}
    opts = type(opts) == "table" and opts or {}

    local has_goal = trim(active_task.goal or "") ~= ""
    local has_resumable_run = opts.has_resumable_run == true
    if not has_goal and not has_resumable_run then
        return false
    end

    local status = trim(active_task.status or ""):lower()
    local task_openish = status == ""
        or status == "open"
        or status == "waiting_user"
        or status == "partial"
        or status == "in_progress"
        or status == "blocked"

    local lower = text:lower()
    local score = 0
    if task_openish or trim(working_memory.current_plan or "") ~= "" then
        score = score + 1
    end
    if has_resumable_run then
        score = score + 1
    end

    local new_topic = starts_with_any(text, NEW_TOPIC_PREFIXES) or starts_with_any(lower, NEW_TOPIC_PREFIXES_EN)
    if new_topic then
        score = score - 3
    end

    local has_followup_prefix = starts_with_any(text, FOLLOWUP_PREFIXES) or starts_with_any(lower, FOLLOWUP_PREFIXES_EN)
    if has_followup_prefix then
        score = score + 2
    end

    local has_short_followup_action = starts_with_any(text, SHORT_FOLLOWUP_ACTIONS) or starts_with_any(lower, SHORT_FOLLOWUP_ACTIONS_EN)
    if has_short_followup_action then
        score = score + 2
    end

    local has_deictic = contains_any(text, DEICTIC_CUES) or contains_any(lower, DEICTIC_CUES_EN)
    if has_deictic then
        score = score + 2
    end

    local strong_ref_hits, generic_ref_hits = count_followup_ref_hits(
        lower,
        collect_followup_refs(active_task, working_memory)
    )
    if strong_ref_hits > 0 then
        score = score + math.min(2, strong_ref_hits * 2)
    end

    local has_action = contains_any(text, ACTION_CUES) or contains_any(lower, ACTION_CUES_EN)
    local has_anchor = has_deictic
        or has_followup_prefix
        or has_short_followup_action
        or has_resumable_run
    if generic_ref_hits > 0 and (has_anchor or has_action) then
        score = score + 1
    end

    if has_action and (has_anchor or strong_ref_hits > 0 or generic_ref_hits > 0) then
        score = score + 1
    end

    if utf8_len(text) <= 36 and (has_anchor or (has_action and strong_ref_hits > 0)) then
        score = score + 1
    end

    return score >= 3 and (has_anchor or has_action or strong_ref_hits > 0)
end

M.trim = trim
M.to_bool = to_bool
M.cfg_number = cfg_number
M.utf8_take = utf8_take
M.utf8_len = utf8_len
M.shallow_copy = shallow_copy
M.is_array_like_table = is_array_like_table
M.encode_lua_value = encode_lua_value
M.parse_lua_table_literal = parse_lua_table_literal
M.json_encode = json_encode
M.ensure_dir = ensure_dir
M.new_run_id = new_run_id
M.now_ms = now_ms
M.workspace_virtual_root = workspace_virtual_root
M.build_workspace_virtual_path = build_workspace_virtual_path
M.normalize_tool_path = normalize_tool_path
M.relative_workspace_path = relative_workspace_path
M.is_continue_request = is_continue_request
M.should_continue_task = should_continue_task

return M
