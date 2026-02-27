local M = {}

local tool = require("module.tool")
local history = require("module.history")
local topic = require("module.topic")
local memory = require("module.memory")
local heat = require("module.heat")
local config_mem = require("module.config")
local notebook = require("module.notebook")

M._pending_system_context = ""
M._pending_topic_anchor = nil
M._pending_created_turn = 0

local function clear_pending_system_context()
    M._pending_system_context = ""
    M._pending_topic_anchor = nil
    M._pending_created_turn = 0
end

local KEYRING_CFG = ((config_mem.settings or {}).keyring or {})
local TOOL_CFG = KEYRING_CFG.tool_calling or {}
local FACT_CFG = KEYRING_CFG.fact_extraction or {}
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
    return {
        max_items = cfg_number(FACT_CFG.max_items, 16, 4, 48),
        max_item_chars = cfg_number(FACT_CFG.max_item_chars, 96, 16, 240),
        primary_max_tokens = cfg_number(FACT_CFG.primary_max_tokens, 420, 64, 2048),
        primary_temperature = cfg_number(FACT_CFG.primary_temperature, 0.18, 0, 1),
        primary_seed = cfg_number(FACT_CFG.primary_seed, 42),
        audit_rounds = cfg_number(FACT_CFG.audit_rounds, 2, 0, 4),
        audit_max_tokens = cfg_number(FACT_CFG.audit_max_tokens, 300, 64, 2048),
        audit_temperature = cfg_number(FACT_CFG.audit_temperature, 0.08, 0, 1),
        audit_seed = cfg_number(FACT_CFG.audit_seed, 43),
        min_facts = cfg_number(FACT_CFG.min_facts, 2, 0, 48),
        min_facts_per_sentence = cfg_number(FACT_CFG.min_facts_per_sentence, 0.65, 0, 2),
        quality_retry_max = cfg_number(FACT_CFG.quality_retry_max, 1, 0, 3),
        fallback_sentence_take = cfg_number(FACT_CFG.fallback_sentence_take, 6, 1, 24),
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
    if not s:match("^%b{}$") then return nil end
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
        { role = "system", content = prompt }
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

local function utf8_len(s)
    local n = 0
    for _ in tostring(s or ""):gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        n = n + 1
    end
    return n
end

local function normalize_fact_key(s)
    local v = trim(s):lower()
    v = v:gsub("%s+", " ")
    v = v:gsub("^[,，。;；:：]+", "")
    v = v:gsub("[,，。;；:：]+$", "")
    return v
end

local function escape_lua_string(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\")
    s = s:gsub("\"", "\\\"")
    s = s:gsub("\n", " ")
    s = s:gsub("\r", " ")
    return s
end

local function render_lua_string_table(items)
    local parts = {}
    for _, item in ipairs(items or {}) do
        parts[#parts + 1] = string.format("\"%s\"", escape_lua_string(item))
    end
    return "{" .. table.concat(parts, ", ") .. "}"
end

local function merge_fact_lists(base, incoming, max_items, max_item_chars)
    local out = {}
    local seen = {}
    max_items = tonumber(max_items) or 16
    max_item_chars = tonumber(max_item_chars) or 96

    local function add_one(raw)
        if #out >= max_items then return end
        local fact = trim(raw)
        if fact == "" or fact == "无" then return end
        fact = fact:gsub("%s+", " ")
        if utf8_len(fact) > max_item_chars then
            fact = utf8_take(fact, max_item_chars)
        end
        if utf8_len(fact) < 2 then return end
        local key = normalize_fact_key(fact)
        if key == "" or seen[key] then return end
        seen[key] = true
        out[#out + 1] = fact
    end

    for _, item in ipairs(base or {}) do add_one(item) end
    for _, item in ipairs(incoming or {}) do add_one(item) end
    return out
end

local function split_into_sentences(text)
    local out = {}
    text = tostring(text or "")
    text = text:gsub("\r", "\n")
    for seg in text:gmatch("[^。\n！？!?；;%.]+") do
        local s = trim(seg)
        if s ~= "" then
            out[#out + 1] = s
        end
    end
    return out
end

local function count_dialogue_sentences(user_input, assistant_clean)
    local count = 0
    local user_s = split_into_sentences(user_input)
    local ai_s = split_into_sentences(assistant_clean)
    count = #user_s + #ai_s
    if count <= 0 then count = 1 end
    return count
end

local function build_fallback_sentence_facts(user_input, assistant_clean, policy)
    local candidate = {}
    local seen = {}
    local max_take = tonumber(policy.fallback_sentence_take) or 6
    local max_chars = tonumber(policy.max_item_chars) or 96

    local function feed(list)
        for _, s in ipairs(list or {}) do
            local fact = trim(s)
            if fact ~= "" then
                if utf8_len(fact) > max_chars then
                    fact = utf8_take(fact, max_chars)
                end
                local key = normalize_fact_key(fact)
                if key ~= "" and not seen[key] then
                    seen[key] = true
                    candidate[#candidate + 1] = fact
                    if #candidate >= max_take then
                        return
                    end
                end
            end
        end
    end

    feed(split_into_sentences(user_input))
    if #candidate < max_take then
        feed(split_into_sentences(assistant_clean))
    end
    if #candidate == 0 then
        local raw = trim(user_input)
        if raw == "" then raw = trim(assistant_clean) end
        if raw ~= "" then
            candidate[1] = utf8_take(raw, max_chars)
        end
    end
    return candidate
end

local function build_fact_primary_prompt(user_input, assistant_clean, policy, strict_cover_mode)
    local strict_line = ""
    if strict_cover_mode then
        strict_line = "8. 先逐句核查“用户+助手”每一句是否被覆盖，再输出。"
    end

    return string.format([[
你是高召回原子事实抽取器，不是摘要器。

目标：覆盖率优先，宁可冗余也不要遗漏。

硬约束：
1. 只抽取对话里“明确说出”的事实，不推测。
2. 禁止压缩、禁止合并、禁止抽象改写。
3. 一条事实只表达一个谓词（一个动作/状态）。
4. 并列信息必须拆条，时间/数字/否定词尽量保留。
5. 不要主动去重，允许近似重复，后处理会去重。
6. 只输出 Lua 字符串数组，例如 {"事实1","事实2"}。
7. 无可用事实时输出 {"无"}。
%s

格式要求：
- 每条事实长度 <= %d 字。
- 输出必须以 { 开头，以 } 结尾，不要解释文本。

对话：
用户：%s
助手：%s

现在只输出 Lua table：
]], strict_line, policy.max_item_chars, user_input, assistant_clean)
end

local function build_fact_audit_prompt(user_input, assistant_clean, existing_facts, policy)
    local existing = render_lua_string_table(existing_facts or {})
    if existing == "{}" then
        existing = "{\"无\"}"
    end

    return string.format([[
你在做“漏抽审计”，不是摘要改写。

已提取事实（禁止重复）：
%s

任务：重新检查对话，只补充“原文明确出现但上面未覆盖”的事实。

硬约束：
1. 禁止抽象总结，禁止把多条信息合并成一条。
2. 只输出新增事实；若无新增，输出 {"无"}。
3. 输出格式必须是 Lua 字符串数组 {"...","..."}。
4. 每条事实长度 <= %d 字。

对话：
用户：%s
助手：%s

现在只输出 Lua table：
]], existing, policy.max_item_chars, user_input, assistant_clean)
end

local function parse_facts_from_llm(raw_facts_str, policy)
    local facts_str = strip_cot_safe(raw_facts_str or "")
    facts_str = facts_str:gsub("%s+", " ")
    facts_str = trim(facts_str)

    local parsed, err = tool.parse_lua_string_array_strict(facts_str, {
        max_items = policy.max_items,
        max_item_chars = policy.max_item_chars,
        must_full = true,
    })
    if not parsed then
        parsed, err = tool.parse_lua_string_array_strict(facts_str, {
            max_items = policy.max_items,
            max_item_chars = policy.max_item_chars,
            must_full = false,
        })
    end
    if not parsed then
        print(string.format("[Lua Fact Extract] LLM 输出格式非法，已丢弃: %s", tostring(err)))
        return {}
    end

    local facts = merge_fact_lists({}, parsed, policy.max_items, policy.max_item_chars)
    print(string.format("[Lua Fact Extract] 成功解析 %d 条事实", #facts))
    return facts
end

local function run_fact_extract_pass(user_input, assistant_clean, policy, strict_cover_mode, seed_offset)
    local prompt = build_fact_primary_prompt(user_input, assistant_clean, policy, strict_cover_mode == true)
    local messages = {
        { role = "system", content = prompt }
    }
    local params = {
        max_tokens = policy.primary_max_tokens,
        temperature = policy.primary_temperature,
        seed = policy.primary_seed + (tonumber(seed_offset) or 0),
    }

    local raw = py_pipeline:generate_chat_sync(messages, params)
    return parse_facts_from_llm(raw, policy)
end

local function run_fact_audit_rounds(user_input, assistant_clean, base_facts, policy)
    local merged = merge_fact_lists({}, base_facts, policy.max_items, policy.max_item_chars)
    if policy.audit_rounds <= 0 then
        return merged
    end

    for round = 1, policy.audit_rounds do
        local prompt = build_fact_audit_prompt(user_input, assistant_clean, merged, policy)
        local messages = {
            { role = "system", content = prompt }
        }
        local params = {
            max_tokens = policy.audit_max_tokens,
            temperature = policy.audit_temperature,
            seed = policy.audit_seed + round,
        }
        local raw = py_pipeline:generate_chat_sync(messages, params)
        local delta = parse_facts_from_llm(raw, policy)
        local before = #merged
        merged = merge_fact_lists(merged, delta, policy.max_items, policy.max_item_chars)
        local added = #merged - before
        print(string.format("[Lua Fact Extract] 漏抽审计 round=%d 新增 %d 条", round, math.max(0, added)))
        if added <= 0 then
            break
        end
    end
    return merged
end

local function evaluate_fact_quality(facts, user_input, assistant_clean, policy)
    local sentence_count = count_dialogue_sentences(user_input, assistant_clean)
    local effective_sentence_count = math.min(sentence_count, policy.max_items)
    if effective_sentence_count <= 0 then effective_sentence_count = 1 end
    local ratio = #facts / effective_sentence_count
    local pass = (#facts >= policy.min_facts) and (ratio >= policy.min_facts_per_sentence)
    return pass, ratio, sentence_count
end

function M.extract_atomic_facts(user_input, assistant_text)
    local policy = get_fact_policy()
    local assistant_clean = tool.replace(assistant_text or "", "\n", " ")

    local facts = run_fact_extract_pass(user_input, assistant_clean, policy, false, 0)
    facts = run_fact_audit_rounds(user_input, assistant_clean, facts, policy)

    local pass, ratio, sentence_count = evaluate_fact_quality(facts, user_input, assistant_clean, policy)
    print(string.format(
        "[Lua Fact Extract] 质量检查: facts=%d, sentence=%d, ratio=%.2f, pass=%s",
        #facts,
        sentence_count,
        ratio,
        pass and "yes" or "no"
    ))

    if (not pass) and policy.quality_retry_max > 0 then
        for retry = 1, policy.quality_retry_max do
            print(string.format("[Lua Fact Extract] 覆盖率不足，触发强制重抽 retry=%d", retry))
            local retry_facts = run_fact_extract_pass(user_input, assistant_clean, policy, true, retry * 11)
            retry_facts = merge_fact_lists(facts, retry_facts, policy.max_items, policy.max_item_chars)
            retry_facts = run_fact_audit_rounds(user_input, assistant_clean, retry_facts, policy)
            facts = retry_facts

            pass, ratio, sentence_count = evaluate_fact_quality(facts, user_input, assistant_clean, policy)
            print(string.format(
                "[Lua Fact Extract] 重抽后质量: facts=%d, sentence=%d, ratio=%.2f, pass=%s",
                #facts,
                sentence_count,
                ratio,
                pass and "yes" or "no"
            ))
            if pass then break end
        end
    end

    if #facts == 0 then
        print("[Lua Fact Extract] LLM 未产出可用事实，启用句子兜底")
        facts = build_fallback_sentence_facts(user_input, assistant_clean, policy)
    end

    return facts
end

function M.save_turn_memory(facts, mem_turn)
    if history.get_turn() ~= mem_turn then
        print(string.format("[WARN] history turn mismatch: history=%d, current=%d", history.get_turn(), mem_turn))
    end

    for _, fact in ipairs(facts) do
        local fact_text = ""
        if type(fact) == "table" then
            fact_text = trim(fact.fact or fact.value or fact.text)
        else
            fact_text = trim(fact)
        end
        if fact_text ~= "" then
            local fact_vec = tool.get_embedding_passage(fact_text)
            local affected_line, add_err = memory.add_memory(fact_vec, mem_turn)
            if affected_line then
                heat.neighbors_add_heat(fact_vec, mem_turn, affected_line)
                print(string.format("   → 原子事实存入记忆行 %d: %s", affected_line, fact_text:sub(1, 60)))
            else
                print(string.format("[Memory][WARN] 原子事实写入失败(%s): %s", tostring(add_err), fact_text:sub(1, 60)))
            end
        end
    end

    if mem_turn % config_mem.settings.time.maintenance_task == 0 then
        heat.perform_cold_exchange()
    end
end

function M.handle_chat_result(ctx, result)
    local user_input = ctx.user_input
    local current_turn = ctx.current_turn
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

    local calls_to_run = resolve_calls_for_turn(user_input, clean_result, policy)
    execute_tool_calls(calls_to_run, current_turn, policy)
    print("\n[Assistant]: " .. clean_result)

    history.add_history(user_input, clean_result)
    topic.update_assistant(current_turn, clean_result)
    if add_to_history then
        add_to_history(user_input, clean_result)
    end

    local facts = M.extract_atomic_facts(user_input, clean_result)

    local cur_summary = topic.get_summary(current_turn)
    if cur_summary and cur_summary ~= "" then
        print("[当前话题摘要] " .. cur_summary)
    end

    M.save_turn_memory(facts, current_turn)
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
