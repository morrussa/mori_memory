local M = {}

local tool = require("module.tool")
local history = require("module.memory.history")
local memory = require("module.memory.store")
local heat = require("module.memory.heat")
local config_mem = require("module.config")

local function get_keyring_cfg()
    return (config_mem.settings or {}).keyring or {}
end

local function get_keyring_defaults()
    local defaults = (config_mem.defaults or {}).keyring
    if type(defaults) ~= "table" then
        defaults = get_keyring_cfg()
    end
    return defaults or {}
end

local function get_fact_cfg()
    return (get_keyring_cfg().fact_extractor or {})
end

local function get_fact_defaults()
    return (get_keyring_defaults().fact_extractor or {})
end

local function get_memory_input_cfg()
    return (get_keyring_cfg().memory_input or {})
end

local function get_memory_input_defaults()
    return (get_keyring_defaults().memory_input or {})
end

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

local function cfg_string(v, fallback)
    if type(v) == "string" then
        local s = trim(v)
        if s ~= "" then
            return s
        end
    end
    return tostring(fallback or "")
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

local function strip_cot_safe(text)
    text = tostring(text or "")
    local cleaned = tool.remove_cot(text)
    if cleaned == "" and text ~= "" and not text:find("</think>", 1, true) then
        cleaned = text
    end
    return cleaned
end

local function resolve_file_payload_mode(mode, fallback_mode)
    local raw = trim(mode)
    if raw == "" then
        raw = trim(fallback_mode)
    end
    if raw == "" then
        raw = "ignore"
    end

    local low = raw:lower()
    if low == "filename" then
        low = "filename_only"
    end
    if low ~= "ignore" and low ~= "filename_only" and low ~= "keep" then
        low = "ignore"
    end
    return low
end

local function get_memory_input_policy()
    local memory_cfg = get_memory_input_cfg()
    local memory_defaults = get_memory_input_defaults()
    local fact_cfg = get_fact_cfg()
    local fact_defaults = get_fact_defaults()
    local default_mode = resolve_file_payload_mode(
        memory_cfg.file_payload_mode or memory_defaults.file_payload_mode,
        fact_cfg.file_payload_mode or fact_defaults.file_payload_mode
    )
    return {
        recall_mode = resolve_file_payload_mode(memory_cfg.recall_file_payload_mode, memory_defaults.recall_file_payload_mode or default_mode),
        fact_mode = resolve_file_payload_mode(memory_cfg.fact_file_payload_mode, memory_defaults.fact_file_payload_mode or default_mode),
        max_chars = cfg_number(memory_cfg.max_chars, memory_defaults.max_chars, 256, 32768),
        manifest_max_items = cfg_number(memory_cfg.manifest_max_items, memory_defaults.manifest_max_items, 1, 64),
        manifest_name_max_chars = cfg_number(memory_cfg.manifest_name_max_chars, memory_defaults.manifest_name_max_chars, 8, 256),
    }
end

function M.get_memory_input_policy()
    return get_memory_input_policy()
end

local function get_fact_policy()
    local fact_cfg = get_fact_cfg()
    local fact_defaults = get_fact_defaults()
    local style = cfg_string(fact_cfg.prompt_style, fact_defaults.prompt_style)
    local default_extract_tokens = tonumber(fact_defaults.extract_max_tokens)
        or ((style == "high_recall_v1") and 320 or 256)
    local input_policy = get_memory_input_policy()
    return {
        prompt_style = style,
        file_payload_mode = input_policy.fact_mode,
        memory_input_max_chars = input_policy.max_chars,
        memory_manifest_max_items = input_policy.manifest_max_items,
        memory_manifest_name_max_chars = input_policy.manifest_name_max_chars,
        verify_pass = to_bool(fact_cfg.verify_pass, fact_defaults.verify_pass),
        max_facts = cfg_number(fact_cfg.max_facts, fact_defaults.max_facts, 1, 16),
        max_parse_items = cfg_number(fact_cfg.max_parse_items, fact_defaults.max_parse_items, 1, 24),
        max_item_chars = cfg_number(fact_cfg.max_item_chars, fact_defaults.max_item_chars, 8, 256),
        extract_max_tokens = cfg_number(fact_cfg.extract_max_tokens, default_extract_tokens, 64, 1024),
        extract_temperature = cfg_number(fact_cfg.extract_temperature, fact_defaults.extract_temperature, 0, 1),
        extract_seed = cfg_number(fact_cfg.extract_seed, fact_defaults.extract_seed),
        repair_max_tokens = cfg_number(fact_cfg.repair_max_tokens, fact_defaults.repair_max_tokens, 64, 512),
        repair_temperature = cfg_number(fact_cfg.repair_temperature, fact_defaults.repair_temperature, 0, 1),
        repair_seed = cfg_number(fact_cfg.repair_seed, fact_defaults.repair_seed),
        verify_max_tokens = cfg_number(fact_cfg.verify_max_tokens, fact_defaults.verify_max_tokens, 64, 512),
        verify_temperature = cfg_number(fact_cfg.verify_temperature, fact_defaults.verify_temperature, 0, 1),
        verify_seed = cfg_number(fact_cfg.verify_seed, fact_defaults.verify_seed),
    }
end

local function is_file_attachment_label(label)
    local raw = trim(label)
    if raw == "" then return false end
    local low = raw:lower()
    if low:find("file", 1, true) then return true end
    if low:find("pdf", 1, true) then return true end
    if raw:find("文件", 1, true) then return true end
    if raw:find("附件", 1, true) then return true end
    return false
end

local function strip_file_payload_for_memory(text, opts)
    text = tostring(text or "")
    opts = opts or {}
    local mode = resolve_file_payload_mode(opts.mode, "ignore")
    local input_policy = get_memory_input_policy()
    local manifest_max_items = cfg_number(opts.manifest_max_items, input_policy.manifest_max_items, 1, 64)
    local manifest_name_max_chars = cfg_number(opts.manifest_name_max_chars, input_policy.manifest_name_max_chars, 8, 256)
    if text == "" or mode == "keep" then
        return text, false, 0
    end
    if not text:find("%-%-%-", 1, true) then
        return text, false, 0
    end

    local kept = {}
    local in_file_block = false
    local redacted_blocks = 0
    local file_names = {}

    for line in (text .. "\n"):gmatch("(.-)\n") do
        local tline = trim(line)
        local label, name = tline:match("^%-%-%-%s*([^:]+)%s*:%s*(.-)%s*%-%-%-%s*$")
        if label and is_file_attachment_label(label) then
            redacted_blocks = redacted_blocks + 1
            in_file_block = true
            local fname = trim(name)
            if fname == "" then fname = "unknown" end
            fname = fname:gsub("[%c]+", " ")
            if #fname > manifest_name_max_chars then
                fname = fname:sub(1, manifest_name_max_chars) .. "..."
            end
            file_names[#file_names + 1] = fname
        elseif label then
            in_file_block = false
            kept[#kept + 1] = line
        else
            if not in_file_block then
                kept[#kept + 1] = line
            end
        end
    end

    if redacted_blocks <= 0 then
        return text, false, 0
    end

    if mode == "filename_only" and #file_names > 0 then
        local listed = 0
        kept[#kept + 1] = ""
        kept[#kept + 1] = "[附件清单(仅文件名)]"
        for _, fname in ipairs(file_names) do
            listed = listed + 1
            if listed > manifest_max_items then
                break
            end
            kept[#kept + 1] = string.format("- %s", fname)
        end
        local rest = #file_names - listed
        if rest > 0 then
            kept[#kept + 1] = string.format("- ... (%d more files)", rest)
        end
    end

    local out = trim(table.concat(kept, "\n"))
    if out == "" then
        if mode == "filename_only" and #file_names > 0 then
            out = "[用户上传了文件附件: " .. table.concat(file_names, ", ") .. "]"
        else
            out = "[用户上传了文件附件，正文已忽略]"
        end
    end
    return out, true, redacted_blocks
end

function M.sanitize_memory_input(user_input, opts)
    local safe = tostring(user_input or "")
    if tool.utf8_sanitize_lossy then
        safe = tool.utf8_sanitize_lossy(safe)
    end
    local input_policy = get_memory_input_policy()
    local mode = ""
    local max_chars = input_policy.max_chars
    local manifest_max_items = input_policy.manifest_max_items
    local manifest_name_max_chars = input_policy.manifest_name_max_chars

    if type(opts) == "string" then
        mode = trim(opts)
    elseif type(opts) == "table" then
        mode = trim(opts.mode or opts.file_payload_mode or "")
        if opts.max_chars ~= nil then
            max_chars = cfg_number(opts.max_chars, max_chars, 256, 32768)
        end
        if opts.manifest_max_items ~= nil then
            manifest_max_items = cfg_number(opts.manifest_max_items, manifest_max_items, 1, 64)
        end
        if opts.manifest_name_max_chars ~= nil then
            manifest_name_max_chars = cfg_number(opts.manifest_name_max_chars, manifest_name_max_chars, 8, 256)
        end
    end

    local resolved_mode = resolve_file_payload_mode(mode, input_policy.recall_mode)
    local normalized, changed, blocks = strip_file_payload_for_memory(safe, {
        mode = resolved_mode,
        manifest_max_items = manifest_max_items,
        manifest_name_max_chars = manifest_name_max_chars,
    })

    local truncated = false
    if max_chars > 0 then
        local clipped = utf8_take(normalized, max_chars)
        if clipped ~= normalized then
            truncated = true
            normalized = trim(clipped) .. "\n\n[memory input truncated]"
        end
    end

    return normalized, (changed or truncated), blocks, resolved_mode, truncated
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
    local fact_policy = get_fact_policy()
    max_items = tonumber(max_items) or tonumber(fact_policy.max_facts) or 8
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
    local fact_policy = get_fact_policy()
    max_items = tonumber(max_items) or tonumber(fact_policy.max_parse_items) or 12
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
            print(string.format(
                "[Lua Fact Extract][%s] strict 解析失败(%s)，已从引号内容恢复 %d 条",
                stage_name,
                tostring(err),
                #recovered
            ))
            return recovered
        end
        print(string.format(
            "[Lua Fact Extract][%s] LLM 输出格式非法，已丢弃: %s",
            stage_name,
            tostring(err)
        ))
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

    local memory_user, changed, blocks, mode, truncated = M.sanitize_memory_input(safe_user, {
        mode = fact_policy.file_payload_mode,
        max_chars = fact_policy.memory_input_max_chars,
        manifest_max_items = fact_policy.memory_manifest_max_items,
        manifest_name_max_chars = fact_policy.memory_manifest_name_max_chars,
    })
    if changed then
        print(string.format(
            "[Lua Fact Extract] memory 输入净化（mode=%s, blocks=%d, truncated=%s）",
            tostring(mode),
            tonumber(blocks) or 0,
            truncated and "yes" or "no"
        ))
    end

    local assistant_clean = tool.replace(safe_assistant, "\n", " ")
    local fact_prompt = build_fact_prompt(memory_user, assistant_clean, fact_policy.prompt_style)
    local facts_str = run_fact_chat_once(
        fact_prompt,
        fact_policy.extract_max_tokens,
        fact_policy.extract_temperature,
        fact_policy.extract_seed
    )
    local facts = parse_facts_from_llm(facts_str, fact_policy, "extract")

    if #facts > 0 and fact_policy.verify_pass then
        local checked = verify_facts(memory_user, assistant_clean, facts, fact_policy)
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
            local checked2 = verify_facts(memory_user, assistant_clean, facts, fact_policy)
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
        print(string.format(
            "[WARN] history turn mismatch: history=%d, current=%d",
            history.get_turn(),
            mem_turn
        ))
    end

    local saved = 0
    for _, fact in ipairs(facts or {}) do
        local fact_vec = tool.get_embedding_passage(fact)
        local affected_line, add_err = memory.add_memory(fact_vec, mem_turn)
        if affected_line then
            heat.neighbors_add_heat(fact_vec, mem_turn, affected_line)
            print(string.format("   → 原子事实存入记忆行 %d: %s", affected_line, fact:sub(1, 60)))
            saved = saved + 1
        else
            print(string.format(
                "[Memory][WARN] 原子事实写入失败(%s): %s",
                tostring(add_err),
                fact:sub(1, 60)
            ))
        end
    end
    if saved == 0 then
        print("[ToolCalling] 本轮无原子事实写入 memory")
    end

    if mem_turn % config_mem.settings.time.maintenance_task == 0 then
        heat.perform_cold_exchange()
    end
end

return M
