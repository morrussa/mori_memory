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

local function parse_fact_array(raw, max_items, max_chars)
    local parsed, _err = tool.parse_lua_string_array_strict(raw or "", {
        max_items = math.max(1, math.floor(tonumber(max_items) or 8)),
        max_item_chars = math.max(8, math.floor(tonumber(max_chars) or 96)),
        must_full = true,
        extract_first_on_fail = true,
    })
    if not parsed then
        return {}
    end

    local out = {}
    local seen = {}
    for _, item in ipairs(parsed) do
        local fact = util.trim(item)
        if fact ~= "" and fact:lower() ~= "none" and (not seen[fact]) then
            seen[fact] = true
            out[#out + 1] = fact
        end
    end
    return out
end

local function build_extract_prompt(user_input, assistant_text)
    return string.format([[
你是长期记忆原子事实提取器。
请从下面一轮对话提取“可复用且长期稳定”的事实。
输出格式必须且只能是 Lua 字符串数组：{"fact1","fact2"} 或 {"none"}。
禁止输出解释、markdown、代码块。

用户输入：%s
助手回复：%s
]], user_input, assistant_text)
end

local function build_verify_prompt(user_input, assistant_text, facts)
    return string.format([[
请校验下列原子事实是否得到当前轮对话支持，删除不可靠项。
输出格式必须且只能是 Lua 字符串数组。

用户输入：%s
助手回复：%s
候选事实：%s
]], user_input, assistant_text, util.encode_lua_value(facts, 0))
end

local function build_repair_prompt(raw)
    return string.format([[
把下面文本修复成合法 Lua 字符串数组，仅输出数组本体。
文本：%s
]], tostring(raw or ""))
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
    for _, fact in ipairs(items or {}) do
        local fact_vec = tool.get_embedding_passage(fact)
        local line, err = memory.add_memory(fact_vec, mem_turn)
        if line then
            heat.neighbors_add_heat(fact_vec, mem_turn, line)
            saved = saved + 1
        else
            print(string.format("[GraphMemory][WARN] add memory failed (%s)", tostring(err)))
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
