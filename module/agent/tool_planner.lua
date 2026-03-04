local M = {}

local tool = require("module.tool")
local config_mem = require("module.config")

local TOOL_CFG = ((config_mem.settings or {}).keyring or {}).tool_calling or {}

local function trim(s)
    if not s then return "" end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function to_bool(v, fallback)
    if type(v) == "boolean" then return v end
    if type(v) == "number" then return v ~= 0 end
    if type(v) == "string" then
        local s = v:lower()
        if s == "true" or s == "1" or s == "yes" then return true end
        if s == "false" or s == "0" or s == "no" then return false end
    end
    return fallback == true
end

local function cfg_number(v, fallback, min_v, max_v)
    local n = tonumber(v)
    if not n then n = tonumber(fallback) or 0 end
    if min_v and n < min_v then n = min_v end
    if max_v and n > max_v then n = max_v end
    return n
end

local function strip_cot_safe(text)
    text = tostring(text or "")
    local cleaned = tool.remove_cot(text)
    if cleaned == "" and text ~= "" and not text:find("</think>", 1, true) then
        cleaned = text
    end
    return cleaned
end

local function parse_field(tbl_line, field)
    local dq = tbl_line:match(field .. '%s*=%s*"([^"]*)"')
    if dq then return dq end
    local sq = tbl_line:match(field .. "%s*=%s*'([^']*)'")
    if sq then return sq end
    local raw = tbl_line:match(field .. "%s*=%s*([^,%}]+)")
    if raw then return trim(raw) end
    return nil
end

local function parse_tool_call_line(line)
    local s = trim(line)
    if s == "" then return nil end
    if not s:match("^%b{}$") then
        local first = tool.extract_first_lua_table and tool.extract_first_lua_table(s) or s:match("%b{}")
        if not first then return nil end
        s = trim(first)
    end
    if not s:find("act%s*=") then return nil end

    local act_raw = parse_field(s, "act")
    if not act_raw then return nil end
    local act = string.lower(trim((act_raw:gsub('^["\'](.-)["\']$', "%1"))))
    if act == "" then return nil end

    return {
        raw = s,
        act = act,
        string = parse_field(s, "string"),
        query = parse_field(s, "query"),
        type = parse_field(s, "type"),
        types = parse_field(s, "types"),
        entity = parse_field(s, "entity"),
        evidence = parse_field(s, "evidence"),
        confidence = parse_field(s, "confidence"),
        namespace = parse_field(s, "namespace"),
        key = parse_field(s, "key"),
        value = parse_field(s, "value"),
    }
end

local function collect_tool_calls_only(text)
    local calls = {}
    text = tostring(text or "")
    for line in (text .. "\n"):gmatch("(.-)\n") do
        local call = parse_tool_call_line(line)
        if call then
            table.insert(calls, call)
        end
    end
    if #calls > 0 then return calls end

    for block in text:gmatch("%b{}") do
        local call = parse_tool_call_line(block)
        if call then
            table.insert(calls, call)
        end
    end
    return calls
end

local function build_tool_pass_prompt(user_input, assistant_text, policy)
    local delete_flag = policy.delete_enabled and "允许" or "禁用"
    return string.format([[
你是 keyring 工具调用规划器，只负责输出工具调用，不负责回复用户。

可用调用（每条必须单独一行，严格 Lua table）：
{act="upsert_record", type="preference|constraint|identity|credential_hint|long_term_plan", entity="对象", value="内容", evidence="原话片段", confidence=0.0}
{act="query_record", query="检索内容", types="可选逗号分隔type列表"}
{act="delete_record", type="...", entity="对象", evidence="删除依据"}

硬约束：
1. 只输出工具调用行，不要解释、不要 markdown、不要代码块。
2. 如果不需要调用，输出空字符串。
3. upsert_record 只在“明确且长期稳定事实”时调用，confidence 必须 >= %.2f。
4. 本轮最多 upsert_record %d 条，最多 query_record %d 条。
5. delete_record 当前%s。
6. query_record 可适当发散（同义词、上位词）提高命中，但必须与当前问题相关。

输入上下文：
用户原话：
%s

助手回复：
%s
]], policy.upsert_min_confidence, policy.upsert_max_per_turn, policy.query_max_per_turn, delete_flag, user_input, assistant_text)
end

local function get_default_policy()
    return {
        upsert_min_confidence = cfg_number(TOOL_CFG.upsert_min_confidence, 0.82, 0, 1),
        upsert_max_per_turn = cfg_number(TOOL_CFG.upsert_max_per_turn, 1, 0),
        query_max_per_turn = cfg_number(TOOL_CFG.query_max_per_turn, 2, 0),
        delete_enabled = to_bool(TOOL_CFG.delete_enabled, false),
        query_max_types = cfg_number(TOOL_CFG.query_max_types, 3, 1),
        query_fetch_limit = cfg_number(TOOL_CFG.query_fetch_limit, 18, 1),
        query_inject_top = cfg_number(TOOL_CFG.query_inject_top, 3, 1),
        query_inject_max_chars = cfg_number(TOOL_CFG.query_inject_max_chars, 800, 200),
        tool_pass_temperature = cfg_number(TOOL_CFG.tool_pass_temperature, 0.15, 0, 1),
        tool_pass_max_tokens = cfg_number(TOOL_CFG.tool_pass_max_tokens, 128, 32),
        tool_pass_seed = cfg_number(TOOL_CFG.tool_pass_seed, 42),
    }
end

function M.get_policy()
    return get_default_policy()
end

function M.plan_calls(user_input, assistant_text, policy)
    local p = policy or get_default_policy()
    local safe_user = tostring(user_input or "")
    local safe_assistant = tostring(assistant_text or "")
    if tool.utf8_sanitize_lossy then
        safe_user = tool.utf8_sanitize_lossy(safe_user)
        safe_assistant = tool.utf8_sanitize_lossy(safe_assistant)
    end
    local prompt = build_tool_pass_prompt(safe_user, safe_assistant, p)
    local tool_messages = {
        { role = "user", content = prompt }
    }
    local tool_params = {
        max_tokens = p.tool_pass_max_tokens,
        temperature = p.tool_pass_temperature,
        seed = p.tool_pass_seed,
    }
    local raw = py_pipeline:generate_chat_sync(tool_messages, tool_params)
    raw = strip_cot_safe(raw or "")
    local calls = collect_tool_calls_only(raw)
    if #calls > 0 then
        print(string.format("[ToolPlanner] two_step 产出 %d 条调用", #calls))
    end
    return calls
end

return M
