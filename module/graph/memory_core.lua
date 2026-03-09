local tool = require("module.tool")
local memory = require("module.memory.store")
local ghsom = require("module.memory.ghsom")
local history = require("module.memory.history")

local M = {}

local function trim(s)
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

local function build_fact_prompt(user_input, assistant_clean)
    return string.format([[
你是记忆系统的“原子事实提取器”。
任务：从下面一轮对话提取可长期存储的事实，输出 Lua 字符串数组。

硬性输出格式（只能二选一）：
1) {"事实1","事实2"}
2) {"无"}

硬性规则：
- 只能输出一个 Lua table，禁止任何解释、前后缀、代码块标记、标签。
- 每条事实 6~28 字，必须是可独立复用的陈述句。
- 优先提取：偏好、约束、身份、长期计划、持续需求；忽略寒暄/一次性上下文。
- 不要输出“用户说/助手说/这轮对话”等元话术。
- 没有可存事实时只能输出 {"无"}。

输入：
用户：%s
助手：%s

请使用 /no_think 模式推理；但输出中只能包含 Lua table 本体
]], user_input, assistant_clean)
end

local function build_fact_repair_prompt(raw_output)
    return string.format([[
把下面文本修复为**仅一个** Lua 字符串数组：
允许输出：
{"事实1","事实2"} 或 {"无"}
禁止任何其他字符。
原始文本：
%s

请使用 /no_think 模式推理；但输出中只能包含修复后的 Lua table 本体
]], tostring(raw_output or ""))
end

local function normalize_fact(fact)
    fact = trim(tostring(fact or ""))
    fact = fact:gsub("[%c]+", " ")
    fact = fact:gsub("%s+", " ")
    fact = trim(fact)
    fact = fact:gsub("^用户[:：]?", "")
    fact = fact:gsub("^助手[:：]?", "")
    fact = trim(fact)
    return fact
end

local function is_bad_fact(fact)
    local n = #fact
    if n < 6 or n > 64 then return true end
    if fact == "无" then return true end
    if fact:find("[{}]") then return true end
    local low = fact:lower()
    if low:find("lua table", 1, true) then return true end
    if low:find("analysis", 1, true) then return true end
    if fact:find("用户说", 1, true) or fact:find("助手说", 1, true) then return true end
    return false
end

local function sanitize_facts(candidates, max_items)
    local out = {}
    local seen = {}
    max_items = tonumber(max_items) or 8
    for _, item in ipairs(candidates or {}) do
        local fact = normalize_fact(item)
        if fact ~= "" and (not is_bad_fact(fact)) and (not seen[fact]) then
            seen[fact] = true
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

local function parse_facts_from_llm(raw_facts_str)
    local facts_str = strip_cot_safe(raw_facts_str or "")
    facts_str = trim(facts_str)

    local parsed, err = tool.parse_lua_string_array_strict(facts_str, {
        max_items = 12,
        max_item_chars = 64,
        must_full = true,
        extract_first_on_fail = true,
    })
    if not parsed then
        local quoted = parse_quoted_candidates(facts_str, 12)
        local recovered = sanitize_facts(quoted, 8)
        if #recovered > 0 then
            print(string.format("[Lua Fact Extract] strict 解析失败(%s)，已从引号内容恢复 %d 条", tostring(err), #recovered))
            return recovered
        end
        print(string.format("[Lua Fact Extract] LLM 输出格式非法，已丢弃: %s", tostring(err)))
        return {}
    end

    local facts = sanitize_facts(parsed, 8)
    print(string.format("[Lua Fact Extract] 成功提取 %d 条原子事实", #facts))
    return facts
end

function M.extract_atomic_facts(user_input, assistant_text)
    local assistant_clean = tool.replace(assistant_text or "", "\n", " ")
    local fact_prompt = build_fact_prompt(user_input, assistant_clean)

    local fact_messages = {
        { role = "system", content = fact_prompt }
    }
    local fact_params = {
        max_tokens = 256,
        temperature = 0.15,
        seed = 42,
    }

    local facts_str = py_pipeline:generate_chat_sync(fact_messages, fact_params)
    local facts = parse_facts_from_llm(facts_str)

    if #facts == 0 then
        local repair_prompt = build_fact_repair_prompt(facts_str)
        local repair_messages = {
            { role = "system", content = repair_prompt }
        }
        local repair_params = {
            max_tokens = 192,
            temperature = 0.0,
            seed = 43,
        }
        local repaired = py_pipeline:generate_chat_sync(repair_messages, repair_params)
        facts = parse_facts_from_llm(repaired)
        if #facts > 0 then
            print(string.format("[Lua Fact Extract] repair 模式恢复成功：%d 条", #facts))
        end
    end

    if #facts == 0 then
        print("[Lua Fact Extract] 未提取到事实，使用原始用户输入兜底")
        facts = sanitize_facts({ user_input }, 1)
        if #facts == 0 then
            facts = { "用户本轮提出需求" }
        end
    end
    return facts
end

function M.save_ingest_items(facts, mem_turn)
    if history.get_turn() ~= mem_turn then
        print(string.format("[GraphMemory][WARN] history turn mismatch: history=%d current=%d", history.get_turn(), mem_turn))
    end

    local saved = 0
    for _, fact in ipairs(facts or {}) do
        local fact_vec = tool.get_embedding_passage(fact)
        local line, err = memory.add_memory(fact_vec, mem_turn)
        if line then
            saved = saved + 1
        else
            print(string.format("[GraphMemory][WARN] add memory failed (%s): %s", tostring(err), tostring(fact):sub(1, 60)))
        end
    end

    return saved
end

M.extract_atomic_items = M.extract_atomic_facts
M.save_turn_memory = M.save_ingest_items

return M
