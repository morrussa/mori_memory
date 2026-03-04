local M = {}

local tool = require("module.tool")
local history = require("module.memory.history")
local topic = require("module.memory.topic")
local memory = require("module.memory.store")
local heat = require("module.memory.heat")
local config_mem = require("module.config")
local notebook = require("module.agent.notebook")

M._pending_system_context = ""
M._pending_topic_anchor = nil
M._pending_created_turn = 0

local function clear_pending_system_context()
    M._pending_system_context = ""
    M._pending_topic_anchor = nil
    M._pending_created_turn = 0
end

local TOOL_CFG = ((config_mem.settings or {}).keyring or {}).tool_calling or {}
local FACT_CFG = ((config_mem.settings or {}).keyring or {}).fact_extractor or {}
local trim

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

local function cfg_string(v, fallback)
    if type(v) == "string" then
        local s = tostring(v):gsub("^%s*(.-)%s*$", "%1")
        if s ~= "" then
            return s
        end
    end
    return tostring(fallback or "")
end

local function get_tool_policy()
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

local function get_fact_policy()
    local style = cfg_string(FACT_CFG.prompt_style, "high_recall_v1")
    local default_extract_tokens = (style == "high_recall_v1") and 320 or 256
    return {
        prompt_style = style,
        verify_pass = to_bool(FACT_CFG.verify_pass, true),
        max_facts = cfg_number(FACT_CFG.max_facts, 8, 1, 16),
        max_parse_items = cfg_number(FACT_CFG.max_parse_items, 12, 1, 24),
        max_item_chars = cfg_number(FACT_CFG.max_item_chars, 64, 8, 256),
        extract_max_tokens = cfg_number(FACT_CFG.extract_max_tokens, default_extract_tokens, 64, 1024),
        extract_temperature = cfg_number(FACT_CFG.extract_temperature, 0.15, 0, 1),
        extract_seed = cfg_number(FACT_CFG.extract_seed, 42),
        repair_max_tokens = cfg_number(FACT_CFG.repair_max_tokens, 192, 64, 512),
        repair_temperature = cfg_number(FACT_CFG.repair_temperature, 0.0, 0, 1),
        repair_seed = cfg_number(FACT_CFG.repair_seed, 43),
        verify_max_tokens = cfg_number(FACT_CFG.verify_max_tokens, 192, 64, 512),
        verify_temperature = cfg_number(FACT_CFG.verify_temperature, 0.0, 0, 1),
        verify_seed = cfg_number(FACT_CFG.verify_seed, 46),
    }
end

trim = function(s)
    if not s then return "" end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
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

local function split_csv(s)
    local out = {}
    s = trim(s)
    if s == "" then return out end
    for part in s:gmatch("[^,]+") do
        local p = trim(part)
        if p ~= "" then table.insert(out, p) end
    end
    return out
end

local function clamp01(v, fallback)
    local n = tonumber(v)
    if not n then return fallback or 0.7 end
    if n < 0 then return 0 end
    if n > 1 then return 1 end
    return n
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

local function split_tool_calls_and_text(text)
    local calls = {}
    local kept = {}

    text = tostring(text or "")
    for line in (text .. "\n"):gmatch("(.-)\n") do
        local call = parse_tool_call_line(line)
        if call then
            table.insert(calls, call)
        else
            table.insert(kept, line)
        end
    end

    local visible = table.concat(kept, "\n")
    visible = trim(visible)
    return calls, visible
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

local function utf8_take(s, max_chars)
    s = tostring(s or "")
    max_chars = tonumber(max_chars) or 0
    if max_chars <= 0 then return s end

    local out = {}
    local count = 0
    for ch in s:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        count = count + 1
        if count > max_chars then break end
        out[count] = ch
    end
    return table.concat(out)
end

local function dedupe_and_clip_types(types, max_types)
    local out = {}
    local seen = {}
    max_types = tonumber(max_types) or 3
    for _, raw in ipairs(types or {}) do
        local t = trim(raw)
        if t ~= "" then
            t = t:gsub("%-", "_")
            t = t:lower()
            if not seen[t] then
                seen[t] = true
                table.insert(out, t)
                if #out >= max_types then break end
            end
        end
    end
    return out
end

local function deterministic_rerank(results, query)
    local ranked = {}
    local now = os.time()
    local q = trim(query):lower()

    for _, r in ipairs(results or {}) do
        local base = tonumber(r.score) or 0
        local conf = clamp01(r.confidence, 0.7)
        local updated_at = tonumber(r.updated_at) or 0
        local recency = 0
        if updated_at > 0 then
            local age_hours = math.max(0, (now - updated_at) / 3600)
            recency = 1 / (1 + age_hours / 96)
        end

        local lexical = 0
        if q ~= "" then
            local entity_l = tostring(r.entity or ""):lower()
            local value_l = tostring(r.value or ""):lower()
            if entity_l:find(q, 1, true) then lexical = lexical + 0.45 end
            if value_l:find(q, 1, true) then lexical = lexical + 0.70 end
        end

        local final = base + conf * 0.40 + recency * 0.25 + lexical
        table.insert(ranked, {
            rec = r,
            final = final,
            updated_at = updated_at,
            id = tonumber(r.id) or 0,
        })
    end

    table.sort(ranked, function(a, b)
        if a.final == b.final then
            if a.updated_at == b.updated_at then
                return a.id < b.id
            end
            return a.updated_at > b.updated_at
        end
        return a.final > b.final
    end)

    local out = {}
    for _, item in ipairs(ranked) do
        table.insert(out, item.rec)
    end
    return out
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

local function generate_two_step_tool_calls(user_input, assistant_text, policy)
    local prompt = build_tool_pass_prompt(user_input, assistant_text, policy)
    local tool_messages = {
        -- jinja chat template 需要 user 消息作为查询入口
        { role = "user", content = prompt }
    }
    local tool_params = {
        max_tokens = policy.tool_pass_max_tokens,
        temperature = policy.tool_pass_temperature,
        seed = policy.tool_pass_seed,
    }
    local raw = py_pipeline:generate_chat_sync(tool_messages, tool_params)
    raw = strip_cot_safe(raw or "")
    local calls = collect_tool_calls_only(raw)
    if #calls > 0 then
        print(string.format("[ToolCalling] two_step 产出 %d 条调用", #calls))
    end
    return calls
end

local function apply_upsert_record(call, current_turn, policy, state)
    if state.upsert_count >= policy.upsert_max_per_turn then
        return false, string.format("upsert_record 超出预算（max=%d）", policy.upsert_max_per_turn)
    end

    local rec_type = trim(call.type)
    local entity = trim(call.entity)
    local value = trim(call.value)
    if value == "" then value = trim(call.string) end
    if rec_type == "" then return false, "upsert_record 缺少 type" end
    if entity == "" then return false, "upsert_record 缺少 entity" end
    if value == "" then return false, "upsert_record 缺少 value" end

    local confidence = clamp01(call.confidence, 0.75)
    if confidence < policy.upsert_min_confidence then
        return false, string.format(
            "upsert_record 置信度 %.2f 低于阈值 %.2f",
            confidence,
            policy.upsert_min_confidence
        )
    end

    local id, op = notebook.upsert_record(rec_type, entity, value, {
        turn = current_turn,
        confidence = confidence,
        evidence = trim(call.evidence),
        source = "tool_call_upsert_record",
    })
    if not id then
        return false, "upsert_record 写入失败: " .. tostring(op)
    end
    state.upsert_count = state.upsert_count + 1
    return true, string.format("upsert_record %s (id=%d)", tostring(op), id)
end

local function apply_delete_record(call, current_turn, policy)
    if not policy.delete_enabled then
        return false, "delete_record 已禁用"
    end

    local rec_type = trim(call.type)
    local entity = trim(call.entity)
    if rec_type == "" then return false, "delete_record 缺少 type" end
    if entity == "" then return false, "delete_record 缺少 entity" end

    local id, op = notebook.delete_record(rec_type, entity, {
        turn = current_turn,
        evidence = trim(call.evidence),
        source = "tool_call_delete_record",
    })
    if not id then
        return false, "delete_record 失败: " .. tostring(op)
    end
    return true, string.format("delete_record %s (id=%d)", tostring(op), id)
end

local function apply_query_record(call, policy, state)
    if state.query_count >= policy.query_max_per_turn then
        return false, string.format("query_record 超出预算（max=%d）", policy.query_max_per_turn)
    end

    local query = trim(call.query)
    if query == "" then query = trim(call.string) end
    if query == "" then query = trim(call.value) end
    if query == "" then return false, "query_record 缺少 query/string/value" end

    local types = split_csv(call.types or "")
    if #types == 0 and trim(call.type) ~= "" then
        table.insert(types, trim(call.type))
    end
    types = dedupe_and_clip_types(types, policy.query_max_types)

    local results = notebook.query_records(query, {
        types = types,
        limit = policy.query_fetch_limit,
        mark_hit = true,
    })
    results = deterministic_rerank(results, query)
    if #results > policy.query_inject_top then
        for i = #results, policy.query_inject_top + 1, -1 do
            table.remove(results, i)
        end
    end

    state.query_count = state.query_count + 1
    print(string.format("[ToolCalling] query_record 命中 %d 条", #results))
    if #results > 0 then
        print(notebook.render_results(results))
        local block = {}
        table.insert(block, "【Tool:query_record 上一轮检索结果】")
        table.insert(block, "query: " .. query)
        table.insert(block, "请把以下记录当作可引用事实，若不相关可忽略：")
        for i, r in ipairs(results) do
            table.insert(block, string.format(
                "%d) [%s] entity=%s | value=%s | conf=%.2f",
                i,
                r.type or "identity",
                r.entity or "",
                r.value or "",
                r.confidence or 0
            ))
        end
        local payload_raw = table.concat(block, "\n")
        local payload = utf8_take(payload_raw, policy.query_inject_max_chars)
        if payload ~= payload_raw then
            payload = payload .. "\n...(truncated)"
        end
        local anchor = topic.get_topic_anchor and topic.get_topic_anchor(call._turn_for_anchor) or nil
        if M._pending_system_context ~= "" and M._pending_topic_anchor == anchor then
            M._pending_system_context = M._pending_system_context .. "\n\n" .. payload
        else
            M._pending_system_context = payload
            M._pending_topic_anchor = anchor
            M._pending_created_turn = tonumber(call._turn_for_anchor) or 0
        end
    end
    return true, string.format("query_record ok (%d hits)", #results)
end

local function execute_tool_calls(calls, current_turn, policy)
    if #calls == 0 then return end
    local state = {
        upsert_count = 0,
        query_count = 0,
    }

    for _, call in ipairs(calls) do
        local ok, msg
        if call.act == "save_key" then
            ok, msg = false, "two_step 模式禁用 save_key"
        elseif call.act == "upsert_record" then
            ok, msg = apply_upsert_record(call, current_turn, policy, state)
        elseif call.act == "delete_record" then
            ok, msg = apply_delete_record(call, current_turn, policy)
        elseif call.act == "query_record" then
            call._turn_for_anchor = current_turn
            ok, msg = apply_query_record(call, policy, state)
        else
            ok, msg = false, "跳过未知 act: " .. tostring(call.act)
        end

        print(string.format("[ToolCalling] %s | %s", ok and "OK" or "FAIL", msg))
    end
end

local function resolve_calls_for_turn(user_input, clean_result, policy)
    return generate_two_step_tool_calls(user_input, clean_result, policy)
end

local function build_fact_prompt(user_input, assistant_clean, style)
    style = trim(style or "high_recall_v1")
    if style == "baseline" then
        return string.format([[
You are an atomic fact extractor for long-term memory.
Task: extract reusable long-term facts from one dialogue turn.

Output format (exactly one Lua string array):
1) {"fact1","fact2"}
2) {"none"}

Hard rules:
- Output only one Lua table. No prefix, no explanation, no markdown.
- Each fact must be an independent statement.
- Prefer: preference, constraint, identity, long-term plan, persistent need.
- Ignore small talk and one-off context.
- No meta phrasing like 'user said' or 'assistant said'.

User: %s
Assistant: %s
]], user_input, assistant_clean)
    end
    if style == "balanced_en_v1" then
        return string.format([[
You are an atomic memory fact extractor.
Task: extract reusable facts from this turn.

Output format (strict, choose one):
1) {"fact1","fact2"}
2) {"none"}

Rules:
- Output exactly one Lua string array, and nothing else.
- Each fact must be one atomic claim (single proposition).
- Fact must be directly supported by the current turn.
- Prefer reusable preferences, constraints, goals, commitments,
  capability limits, and durable needs.
- Medium-term reusable facts are allowed (not only permanent traits).
- Ignore small talk, one-off filler, and meta wording.
- Do not include phrases like 'user said' or 'assistant said'.

User: %s
Assistant: %s
]], user_input, assistant_clean)
    end
    if style == "high_recall_v1" then
        return string.format([[
You are an atomic memory fact extractor.
Goal: maximize recall while keeping controllable noise.
Extract as many valid reusable facts as possible from this turn.

Output format (strict, choose one):
1) {"fact1","fact2","fact3"}
2) {"none"}

Rules:
- Output exactly one Lua string array and nothing else.
- Each fact must be one atomic claim (single proposition).
- Fact must be directly supported by current turn text.
- Allowed reusable scope: long-term and medium-term facts,
  including preferences, constraints, goals, commitments,
  stable abilities/limitations, planned actions.
- Prefer concise, factual statements; avoid meta wording.
- Use {"none"} only when no concrete reusable fact exists.

User: %s
Assistant: %s
]], user_input, assistant_clean)
    end
    if style == "balanced_v3" then
        return string.format([[
你是“原子事实提取器”，只输出 Lua 字符串数组。
任务：从本轮对话提取可复用事实（优先未来多轮可用）。

输出格式（只能二选一）：
1) {"事实1","事实2"}
2) {"无"}

规则：
- 只能输出一个 Lua table，本体外任何字符都禁止。
- 每条事实必须单原子断言，不要并列多结论。
- 事实必须由当前对话直接支持；不确定就不要写。
- 优先：偏好、约束、目标、承诺、能力边界、可持续需求。
- 可接受“短中期可复用”事实，不必强制永久事实。
- 忽略纯寒暄、无信息重复、情绪口头语。
- 禁止“用户说/助手说/本轮对话”等元话术。

用户：%s
助手：%s
]], user_input, assistant_clean)
    end
    return string.format([[
你是“长期记忆原子事实提取器”，只输出 Lua 字符串数组。
任务：从本轮对话抽取可长期复用、可直接验证的原子事实。

输出格式（只能二选一）：
1) {"事实1","事实2"}
2) {"无"}

硬规则：
- 只能输出一个 Lua table，本体之外任何字符都禁止。
- 每条事实只允许一个主断言（单原子），不得出现并列多断言。
- 必须可由当前对话直接支持，禁止猜测、补全背景、引入外部信息。
- 优先提取：稳定偏好、长期约束、身份信息、长期计划、持续需求。
- 忽略一次性寒暄、短期动作、情绪感叹。
- 禁止“用户说/助手说/本轮对话”等元话术。

用户：%s
助手：%s
]], user_input, assistant_clean)
end

local function build_fact_repair_prompt(raw_output, style)
    style = trim(style or "high_recall_v1")
    if style == "baseline" then
        return string.format([[
Repair the text into exactly one Lua string array.
Allowed:
{"fact1","fact2"} or {"none"}
Forbidden: any extra characters.
Raw text:
%s
]], tostring(raw_output or ""))
    end
    return string.format([[
把下面文本修复为且仅为一个 Lua 字符串数组。
允许输出：{"事实1","事实2"} 或 {"无"}
禁止任何解释、前后缀、代码块标记。
原始文本：
%s
]], tostring(raw_output or ""))
end

local function build_fact_verify_prompt(user_input, assistant_clean, candidates, style)
    style = trim(style or "high_recall_v1")
    local packed = {}
    for _, c in ipairs(candidates or {}) do
        local s = trim(c)
        if s ~= "" then
            packed[#packed + 1] = s
        end
    end
    local joined = table.concat(packed, " | ")
    if style == "high_recall_v1" then
        return string.format([[
You are an atomic fact quality gate for high recall mode.
Filter and lightly rewrite candidates.
Keep facts that satisfy:
1) directly supported by the turn,
2) one atomic claim,
3) reusable in future turns (long-term or medium-term).
Remove only clearly noisy/meta/unsupported items.
Keep top 1-4 facts.

Output exactly one Lua string array:
{"fact1","fact2"} or {"none"}
No explanation.

User: %s
Assistant: %s
Candidates: %s
]], user_input, assistant_clean, joined)
    end
    if style == "balanced_en_v1" or style == "baseline" then
        return string.format([[
You are an atomic fact quality checker.
Filter and lightly rewrite candidate facts.
Keep only facts that satisfy all conditions:
1) directly supported by the turn,
2) one atomic claim,
3) reusable for future turns.

Output exactly one Lua string array:
{"fact1","fact2"} or {"none"}
No explanation.

User: %s
Assistant: %s
Candidates: %s
]], user_input, assistant_clean, joined)
    end
    return string.format([[
你是“原子事实质检器”。
请对候选事实做筛选与轻量改写，只保留同时满足以下条件的事实：
1) 能被当前对话直接支持；2) 单原子断言；3) 具中长期复用价值。

输出格式只能是 Lua 字符串数组：{"事实1","事实2"} 或 {"无"}
禁止任何解释。

用户：%s
助手：%s
候选事实：%s
]], user_input, assistant_clean, joined)
end

local function normalize_fact(fact)
    if tool.utf8_sanitize_lossy then
        fact = tool.utf8_sanitize_lossy(fact)
    end
    fact = trim(tostring(fact or ""))
    fact = fact:gsub("[%c]+", " ")
    fact = fact:gsub("%s+", " ")
    fact = trim(fact)
    fact = fact:gsub("^[Uu][Ss][Ee][Rr]%s*[:：]%s*", "")
    fact = fact:gsub("^[Aa][Ss][Ss][Ii][Ss][Tt][Aa][Nn][Tt]%s*[:：]%s*", "")
    fact = fact:gsub("^用户[:：]?", "")
    fact = fact:gsub("^助手[:：]?", "")
    fact = trim(fact)
    return fact
end

local function is_bad_fact(fact)
    local n = #fact
    if n < 6 or n > 64 then return true end
    local low = fact:lower()
    if fact == "无" or low == "none" or low == "null" or low == "n/a" then return true end
    if fact:find("[{}]") then return true end
    if low:find("lua table", 1, true) then return true end
    if low:find("analysis", 1, true) then return true end
    if low:find("assistant", 1, true) or low:find("user", 1, true) then return true end
    if low:find("response", 1, true) or low:find("statement", 1, true) then return true end
    if low:find("dialogue", 1, true) or low:find("conversation", 1, true) then return true end
    if low:find("user said", 1, true) or low:find("assistant said", 1, true) then return true end
    if fact:find("用户说", 1, true) or fact:find("助手说", 1, true) then return true end
    if fact:find("?", 1, true) or fact:find("？", 1, true) then return true end
    return false
end

local function sanitize_facts(candidates, max_items)
    local out = {}
    local seen = {}
    max_items = tonumber(max_items) or 8
    for _, item in ipairs(candidates or {}) do
        local fact = normalize_fact(item)
        local key = fact:lower()
        if fact ~= "" and (not is_bad_fact(fact)) and (not seen[key]) then
            seen[key] = true
            out[#out + 1] = fact
            if #out >= max_items then break end
        end
    end
    return out
end

local function parse_quoted_candidates(text, max_items)
    local out = {}
    max_items = tonumber(max_items) or 12
    text = tostring(text or "")
    for q in text:gmatch('"(.-)"') do
        out[#out + 1] = q
        if #out >= max_items then return out end
    end
    for q in text:gmatch("'(.-)'") do
        out[#out + 1] = q
        if #out >= max_items then return out end
    end
    return out
end

local function run_fact_chat_once(prompt, max_tokens, temperature, seed)
    local messages = {
        -- jinja chat template 需要 user 消息作为查询入口
        { role = "user", content = prompt }
    }
    local params = {
        max_tokens = max_tokens,
        temperature = temperature,
        seed = seed,
    }
    return py_pipeline:generate_chat_sync(messages, params)
end

local function parse_facts_from_llm(raw_facts_str, fact_policy, stage_name)
    local facts_str = strip_cot_safe(raw_facts_str or "")
    facts_str = trim(facts_str)
    stage_name = trim(stage_name or "extract")

    local parsed, err = tool.parse_lua_string_array_strict(facts_str, {
        max_items = fact_policy.max_parse_items,
        max_item_chars = fact_policy.max_item_chars,
        must_full = true,
        extract_first_on_fail = true,
    })
    if not parsed then
        local quoted = parse_quoted_candidates(facts_str, fact_policy.max_parse_items)
        local recovered = sanitize_facts(quoted, fact_policy.max_facts)
        if #recovered > 0 then
            print(string.format("[Lua Fact Extract][%s] strict 解析失败(%s)，已从引号内容恢复 %d 条", stage_name, tostring(err), #recovered))
            return recovered
        end
        print(string.format("[Lua Fact Extract][%s] LLM 输出格式非法，已丢弃: %s", stage_name, tostring(err)))
        return {}
    end

    local facts = sanitize_facts(parsed, fact_policy.max_facts)
    if #facts > 0 then
        print(string.format("[Lua Fact Extract][%s] 成功提取 %d 条原子事实", stage_name, #facts))
    end
    return facts
end

local function verify_facts(user_input, assistant_clean, candidates, fact_policy)
    if #candidates <= 0 then return {} end
    local verify_prompt = build_fact_verify_prompt(
        user_input,
        assistant_clean,
        candidates,
        fact_policy.prompt_style
    )
    local raw = run_fact_chat_once(
        verify_prompt,
        fact_policy.verify_max_tokens,
        fact_policy.verify_temperature,
        fact_policy.verify_seed
    )
    return parse_facts_from_llm(raw, fact_policy, "verify")
end

function M.extract_atomic_facts(user_input, assistant_text)
    local fact_policy = get_fact_policy()
    local safe_user = tostring(user_input or "")
    local safe_assistant = tostring(assistant_text or "")
    if tool.utf8_sanitize_lossy then
        safe_user = tool.utf8_sanitize_lossy(safe_user)
        safe_assistant = tool.utf8_sanitize_lossy(safe_assistant)
    end
    local assistant_clean = tool.replace(safe_assistant, "\n", " ")
    local fact_prompt = build_fact_prompt(safe_user, assistant_clean, fact_policy.prompt_style)
    local facts_str = run_fact_chat_once(
        fact_prompt,
        fact_policy.extract_max_tokens,
        fact_policy.extract_temperature,
        fact_policy.extract_seed
    )
    local facts = parse_facts_from_llm(facts_str, fact_policy, "extract")

    if #facts > 0 and fact_policy.verify_pass then
        local checked = verify_facts(safe_user, assistant_clean, facts, fact_policy)
        if #checked > 0 then
            facts = checked
            print(string.format("[Lua Fact Extract] verify 通过，保留 %d 条", #facts))
        end
    end

    if #facts == 0 then
        local repair_prompt = build_fact_repair_prompt(facts_str, fact_policy.prompt_style)
        local repaired = run_fact_chat_once(
            repair_prompt,
            fact_policy.repair_max_tokens,
            fact_policy.repair_temperature,
            fact_policy.repair_seed
        )
        facts = parse_facts_from_llm(repaired, fact_policy, "repair")

        if #facts > 0 and fact_policy.verify_pass then
            local checked2 = verify_facts(safe_user, assistant_clean, facts, fact_policy)
            if #checked2 > 0 then
                facts = checked2
                print(string.format("[Lua Fact Extract] repair+verify 通过，保留 %d 条", #facts))
            end
        end

        if #facts > 0 then
            print(string.format("[Lua Fact Extract] repair 模式恢复成功：%d 条", #facts))
        end
    end

    if #facts == 0 then
        print("[Lua Fact Extract] 未提取到有效原子事实，本轮不写入 memory")
    end
    return facts
end

function M.save_turn_memory(facts, mem_turn)
    if history.get_turn() ~= mem_turn then
        print(string.format("[WARN] history turn mismatch: history=%d, current=%d", history.get_turn(), mem_turn))
    end

    local saved = 0
    for _, fact in ipairs(facts) do
        local fact_vec = tool.get_embedding_passage(fact)
        local affected_line, add_err = memory.add_memory(fact_vec, mem_turn)
        if affected_line then
            heat.neighbors_add_heat(fact_vec, mem_turn, affected_line)
            print(string.format("   → 原子事实存入记忆行 %d: %s", affected_line, fact:sub(1, 60)))
            saved = saved + 1
        else
            print(string.format("[Memory][WARN] 原子事实写入失败(%s): %s", tostring(add_err), fact:sub(1, 60)))
        end
    end
    if saved == 0 then
        print("[ToolCalling] 本轮无原子事实写入 memory")
    end

    if mem_turn % config_mem.settings.time.maintenance_task == 0 then
        heat.perform_cold_exchange()
    end
end

function M.handle_chat_result(ctx, result)
    local user_input = ctx.user_input
    local current_turn = ctx.current_turn
    local read_only = ctx.read_only == true
    local add_to_history = ctx.add_to_history

    local policy = get_tool_policy()
    local cot_clean = strip_cot_safe(result or "")
    local parsed_calls, visible_text = split_tool_calls_and_text(cot_clean)
    local clean_result = visible_text
    if clean_result == "" and #parsed_calls == 0 then
        clean_result = trim(cot_clean)
    end
    if clean_result == "" then
        clean_result = "好的，已记录。"
    end

    if not read_only then
        local calls_to_run = resolve_calls_for_turn(user_input, clean_result, policy)
        execute_tool_calls(calls_to_run, current_turn, policy)
    else
        print("[ToolCalling] read_only 模式：跳过工具写入")
    end
    print("\n[Assistant]: " .. clean_result)

    if not read_only then
        history.add_history(user_input, clean_result)
        topic.update_assistant(current_turn, clean_result)
        if add_to_history then
            add_to_history(user_input, clean_result)
        end
    else
        print("[ToolCalling] read_only 模式：跳过 history/topic 写入")
    end

    local facts = {}

    if not read_only then
        facts = M.extract_atomic_facts(user_input, clean_result)

        local cur_summary = topic.get_summary(current_turn)
        if cur_summary and cur_summary ~= "" then
            print("[当前话题摘要] " .. cur_summary)
        end

        M.save_turn_memory(facts, current_turn)
    else
        print("[ToolCalling] read_only 模式：跳过 memory 写入")
    end

    return clean_result, facts
end

function M.consume_pending_system_context(current_turn)
    local ctx = M.get_pending_system_context and M.get_pending_system_context(current_turn) or M._pending_system_context
    if ctx == "" then return "" end
    clear_pending_system_context()
    return ctx
end

function M.get_pending_system_context(current_turn)
    if M._pending_system_context == "" then return "" end
    local cur_anchor = topic.get_topic_anchor and topic.get_topic_anchor(current_turn) or nil
    if M._pending_topic_anchor and cur_anchor and M._pending_topic_anchor ~= cur_anchor then
        clear_pending_system_context()
        return ""
    end
    return M._pending_system_context
end

function M.consume_pending_system_context_for_turn(current_turn)
    local ctx = M.get_pending_system_context(current_turn)
    if ctx == "" then return "" end
    clear_pending_system_context()
    return ctx
end

function M.get_long_term_plan_bom()
    return notebook.build_long_term_plan_bom and notebook.build_long_term_plan_bom() or ""
end

return M
