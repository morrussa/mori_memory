local tool = require("module.tool")
local memory = require("module.memory.store")
local heat = require("module.memory.heat")
local history = require("module.memory.history")
local memtypes = require("module.memory.types")
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

local function normalize_item_row(item, max_chars)
    local text = ""
    local kind = nil

    if type(item) == "string" then
        text = util.trim(item)
    elseif type(item) == "table" then
        text = util.trim(item.text or item.content or item.fact or "")
        kind = item.type or item.kind or item.label
    end

    if text == "" then
        return nil
    end

    text = util.utf8_take(text, math.max(16, math.floor(tonumber(max_chars) or 96)))
    if text == "" or text:lower() == "none" then
        return nil
    end

    kind = memtypes.normalize(kind, memtypes.infer_text_type(text, "passage"))
    return {
        text = text,
        type = kind,
    }
end

local function parse_typed_item_array(raw, max_items, max_chars)
    local parsed, _err = util.parse_lua_table_literal(raw or "")
    if type(parsed) ~= "table" then
        local fallback, _fallback_err = tool.parse_lua_string_array_strict(raw or "", {
            max_items = math.max(1, math.floor(tonumber(max_items) or 8)),
            max_item_chars = math.max(8, math.floor(tonumber(max_chars) or 96)),
            must_full = true,
            extract_first_on_fail = true,
        })
        parsed = fallback or {}
    end

    local out = {}
    local seen = {}
    for _, item in ipairs(parsed or {}) do
        local row = normalize_item_row(item, max_chars)
        if row then
            local dedup_key = row.type .. "\31" .. row.text
            if not seen[dedup_key] then
                seen[dedup_key] = true
                out[#out + 1] = row
                if #out >= math.max(1, math.floor(tonumber(max_items) or 8)) then
                    break
                end
            end
        end
    end
    return out
end

local function build_extract_prompt(user_input, assistant_text)
    return string.format([[
你是长期记忆条目提取器。
请从下面一轮对话提取“未来轮次仍可复用”的记忆条目，并为每条标注唯一类型。
类型只能是：
- identity: 用户本人、他人身份、角色、稳定属性
- preference: 偏好、风格、习惯、常用选择
- state: 当前进展、阶段、上下文、临时约束、正在做的事
- decision: 已确定方案、冻结结论、采用或不采用什么
- fact: 客观结果、指标、数值、观察结论
- concept: 方法、机制、术语、抽象知识

输出格式必须且只能是 Lua table 数组：
{{text="...",type="state"},{text="...",type="decision"}}
如果没有可写入内容，输出：
{{text="none",type="none"}}
禁止输出解释、markdown、代码块。

用户输入：%s
助手回复：%s
]], user_input, assistant_text)
end

local function build_verify_prompt(user_input, assistant_text, items)
    return string.format([[
请校验下列记忆条目及其类型是否得到当前轮对话支持，删除不可靠项，必要时修正类型。
输出格式必须且只能是 Lua table 数组，字段只允许 text/type。

用户输入：%s
助手回复：%s
候选条目：%s
]], user_input, assistant_text, util.encode_lua_value(items, 0))
end

local function build_repair_prompt(raw)
    return string.format([[
把下面文本修复成合法 Lua table 数组，仅输出数组本体。
每个元素必须是 {text="...",type="identity|preference|state|decision|fact|concept"}。
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

    local items = parse_typed_item_array(raw, max_parse_items, max_item_chars)

    if #items > 0 and util.to_bool(cfg.verify_pass, true) then
        local checked_raw = run_chat(
            build_verify_prompt(user, assistant, items),
            cfg.verify_max_tokens or 192,
            cfg.verify_temperature or 0,
            cfg.verify_seed or 46
        )
        local checked = parse_typed_item_array(checked_raw, max_parse_items, max_item_chars)
        if #checked > 0 then
            items = checked
        end
    end

    if #items == 0 then
        local repaired_raw = run_chat(
            build_repair_prompt(raw),
            cfg.repair_max_tokens or 192,
            cfg.repair_temperature or 0,
            cfg.repair_seed or 43
        )
        items = parse_typed_item_array(repaired_raw, max_parse_items, max_item_chars)
    end

    if #items > max_facts then
        for i = #items, max_facts + 1, -1 do
            table.remove(items, i)
        end
    end

    return items
end

function M.save_ingest_items(items, mem_turn)
    if history.get_turn() ~= mem_turn then
        print(string.format("[GraphMemory][WARN] history turn mismatch: history=%d current=%d", history.get_turn(), mem_turn))
    end

    local saved = 0
    for _, item in ipairs(items or {}) do
        local row = normalize_item_row(item, 160)
        if row then
            local fact_vec = tool.get_embedding_passage(row.text)
            local line, err = memory.add_memory(fact_vec, mem_turn, {
                type = row.type,
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
