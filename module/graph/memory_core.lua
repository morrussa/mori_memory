local tool = require("module.tool")
local memory = require("module.memory.store")
local heat = require("module.memory.heat")
local history = require("module.memory.history")
local config = require("module.config")
local util = require("module.graph.util")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function fact_cfg()
    return (graph_cfg().fact_extractor or {})
end

local function sanitize_text(s)
    local text = tostring(s or "")
    if tool.utf8_sanitize_lossy then
        text = tool.utf8_sanitize_lossy(text)
    end
    return text
end

local function run_chat(prompt, max_tokens, temperature, seed)
    local messages = {
        { role = "user", content = prompt },
    }
    local params = {
        max_tokens = math.max(64, math.floor(tonumber(max_tokens) or 256)),
        temperature = tonumber(temperature) or 0,
        seed = tonumber(seed) or 42,
    }
    return tostring(py_pipeline:generate_chat_sync(messages, params) or "")
end

local function normalize_fact_item(item, max_chars)
    local fact = ""
    local type_name = memory.get_default_type_name()

    if type(item) == "string" then
        fact = item
    elseif type(item) == "table" then
        fact = item.fact or item.text or item.value or item.content or ""
        type_name = memory.normalize_type_name(item.type or item.kind or item.statement_type)
    else
        return nil
    end

    fact = sanitize_text(fact):gsub("%s+", " ")
    fact = util.trim(fact)
    if fact == "" or fact:lower() == "none" then
        return nil
    end
    if #fact > max_chars then
        fact = sanitize_text(fact:sub(1, max_chars))
        fact = util.trim(fact)
    end
    if fact == "" then
        return nil
    end

    return {
        fact = fact,
        type = type_name,
    }
end

local function parse_fact_array(raw, max_items, max_chars)
    local limit_items = math.max(1, math.floor(tonumber(max_items) or 8))
    local limit_chars = math.max(8, math.floor(tonumber(max_chars) or 96))
    local out = {}
    local seen = {}

    local function push_items(parsed)
        if type(parsed) ~= "table" then
            return
        end
        for i = 1, math.min(#parsed, limit_items) do
            local normalized = normalize_fact_item(parsed[i], limit_chars)
            if normalized then
                local key = normalized.type .. "\0" .. normalized.fact
                if not seen[key] then
                    seen[key] = true
                    out[#out + 1] = normalized
                    if #out >= limit_items then
                        break
                    end
                end
            end
        end
    end

    local parsed = select(1, util.parse_lua_table_literal(raw or ""))
    push_items(parsed)

    if #out <= 0 then
        local legacy, _err = tool.parse_lua_string_array_strict(raw or "", {
            max_items = limit_items,
            max_item_chars = limit_chars,
            must_full = true,
            extract_first_on_fail = true,
        })
        push_items(legacy)
    end

    if #out > limit_items then
        for i = #out, limit_items + 1, -1 do
            table.remove(out, i)
        end
    end
    return out
end

local function allowed_type_list_literal()
    return util.encode_lua_value(memory.get_allowed_type_names(), 0)
end

local function build_extract_prompt(user_input, assistant_text)
    return string.format([[
你是长期记忆原子事实提取器。
请从下面一轮对话提取“可复用且长期稳定”的事实。
输出格式必须且只能是 Lua 数组，优先使用表项：
{{fact="...",type="Constraint"},{fact="...",type="Preference"}} 或 {"none"}。
`type` 只能从以下候选中选择：%s
禁止输出解释、markdown、代码块。

用户输入：%s
助手回复：%s
]], allowed_type_list_literal(), user_input, assistant_text)
end

local function build_verify_prompt(user_input, assistant_text, facts)
    return string.format([[
请校验下列原子事实是否得到当前轮对话支持，删除不可靠项。
输出格式必须且只能是 Lua 数组，保留原有 `fact` 和 `type` 字段。
`type` 只能从以下候选中选择：%s

用户输入：%s
助手回复：%s
候选事实：%s
]], allowed_type_list_literal(), user_input, assistant_text, util.encode_lua_value(facts, 0))
end

local function build_repair_prompt(raw)
    return string.format([[
把下面文本修复成合法 Lua 数组，仅输出数组本体。
优先输出 {fact="...",type="..."} 的数组，`type` 只能从以下候选中选择：%s。
文本：%s
]], allowed_type_list_literal(), tostring(raw or ""))
end

function M.extract_atomic_items(user_input, assistant_text)
    local cfg = fact_cfg()
    local max_facts = math.max(1, math.floor(tonumber(cfg.max_facts) or 8))
    local max_parse_items = math.max(max_facts, math.floor(tonumber(cfg.max_parse_items) or 12))
    local max_item_chars = math.max(16, math.floor(tonumber(cfg.max_item_chars) or 96))

    local user = sanitize_text(user_input)
    local assistant = sanitize_text(assistant_text):gsub("\n", " ")

    local raw = run_chat(
        build_extract_prompt(user, assistant),
        cfg.extract_max_tokens or 320,
        cfg.extract_temperature or 0.15,
        cfg.extract_seed or 42
    )

    local facts = parse_fact_array(raw, max_parse_items, max_item_chars)

    if #facts > 0 and util.to_bool(cfg.verify_pass, true) then
        local checked_raw = run_chat(
            build_verify_prompt(user, assistant, facts),
            cfg.verify_max_tokens or 192,
            cfg.verify_temperature or 0,
            cfg.verify_seed or 46
        )
        local checked = parse_fact_array(checked_raw, max_parse_items, max_item_chars)
        if #checked > 0 then
            facts = checked
        end
    end

    if #facts == 0 then
        local repaired_raw = run_chat(
            build_repair_prompt(raw),
            cfg.repair_max_tokens or 192,
            cfg.repair_temperature or 0,
            cfg.repair_seed or 43
        )
        facts = parse_fact_array(repaired_raw, max_parse_items, max_item_chars)
    end

    if #facts > max_facts then
        for i = #facts, max_facts + 1, -1 do
            table.remove(facts, i)
        end
    end

    return facts
end

function M.save_ingest_items(items, mem_turn)
    if history.get_turn() ~= mem_turn then
        print(string.format("[GraphMemory][WARN] history turn mismatch: history=%d current=%d", history.get_turn(), mem_turn))
    end

    local saved = 0
    for _, item in ipairs(items or {}) do
        local normalized = normalize_fact_item(item, math.huge)
        if normalized then
            local fact_vec = tool.get_embedding_passage(normalized.fact)
            local line, err = memory.add_memory(fact_vec, mem_turn, {
                type_name = normalized.type,
            })
            if line then
                heat.neighbors_add_heat(fact_vec, mem_turn, line)
                saved = saved + 1
            else
                print(string.format("[GraphMemory][WARN] add memory failed (%s)", tostring(err)))
            end
        end
    end

    local maintenance_every = tonumber(((config.settings or {}).time or {}).maintenance_task) or 75
    if maintenance_every > 0 and mem_turn % maintenance_every == 0 then
        heat.perform_cold_exchange()
    end

    return saved
end

M.extract_atomic_facts = M.extract_atomic_items
M.save_turn_memory = M.save_ingest_items

return M
