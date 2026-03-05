local M = {}

local tool = require("module.tool")
local tool_parser = require("module.agent.tool_parser")
local tool_registry = require("module.agent.tool_registry")
local config_mem = require("module.config")

local function get_tool_cfg()
    return ((config_mem.settings or {}).keyring or {}).tool_calling or {}
end

local function get_tool_defaults()
    local defaults = ((config_mem.defaults or {}).keyring or {}).tool_calling
    if type(defaults) ~= "table" then
        defaults = get_tool_cfg()
    end
    return defaults or {}
end

local function get_agent_cfg()
    return (config_mem.settings or {}).agent or {}
end

local function get_agent_defaults()
    local defaults = (config_mem.defaults or {}).agent
    if type(defaults) ~= "table" then
        defaults = get_agent_cfg()
    end
    return defaults or {}
end

local function get_supported_tool_acts()
    local acts = tool_parser.clone_supported_acts()
    if tool_registry.get_supported_acts then
        local ok, merged = pcall(tool_registry.get_supported_acts, acts)
        if ok and type(merged) == "table" then
            acts = merged
        end
    end
    return acts
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

local function normalize_function_choice(raw_choice, supported_acts)
    return tool_parser.normalize_function_choice(raw_choice, {
        supported_acts = supported_acts or tool_parser.clone_supported_acts(),
    })
end

local function get_default_parallel_function_calls()
    local agent_cfg = get_agent_cfg()
    local agent_defaults = get_agent_defaults()
    return to_bool(
        agent_cfg.parallel_function_calls,
        to_bool(agent_defaults.parallel_function_calls, true)
    )
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

local function build_dynamic_tool_call_examples(policy)
    local lines = {
        '{act="upsert_record", type="preference|constraint|identity|credential_hint|long_term_plan", entity="对象", value="内容", evidence="原话片段", confidence=0.0}',
        '{act="query_record", query="检索内容", types="可选逗号分隔type列表"}',
        '{act="delete_record", type="...", entity="对象", evidence="删除依据"}',
    }
    local schemas = {}
    if tool_registry.get_openai_tools then
        local ok, out = pcall(tool_registry.get_openai_tools, policy)
        if ok and type(out) == "table" then
            schemas = out
        end
    end
    local known = {
        upsert_record = true,
        query_record = true,
        delete_record = true,
    }
    for _, item in ipairs(schemas) do
        local fn = type(item) == "table" and item["function"] or nil
        local name = trim((fn or {}).name or "")
        if name ~= "" and (not known[name]) then
            known[name] = true
            lines[#lines + 1] = string.format(
                '{act="%s", arguments={key="value"}}',
                name
            )
        end
    end
    return table.concat(lines, "\n")
end

local function build_tool_pass_prompt(user_input, assistant_text, policy, planner_ctx, supported_acts)
    planner_ctx = planner_ctx or {}
    supported_acts = supported_acts or tool_parser.clone_supported_acts()
    local delete_flag = policy.delete_enabled and "允许" or "禁用"
    local step_idx = math.max(1, math.floor(tonumber(planner_ctx.step_idx) or 1))
    local function_choice = normalize_function_choice(planner_ctx.function_choice, supported_acts)
    local parallel_function_calls = to_bool(
        planner_ctx.parallel_function_calls,
        get_default_parallel_function_calls()
    )
    local substep_name = trim(planner_ctx.substep_name or "general-purpose")
    local substep_label = trim(planner_ctx.substep_label or "")
    local substep_description = trim(planner_ctx.substep_description or "")
    local last_observation = trim(planner_ctx.last_observation or "")
    local tool_trace = trim(planner_ctx.tool_trace or "")
    local tool_examples = build_dynamic_tool_call_examples(policy)

    if tool.utf8_sanitize_lossy then
        last_observation = tool.utf8_sanitize_lossy(last_observation)
        tool_trace = tool.utf8_sanitize_lossy(tool_trace)
    end
    last_observation = utf8_take(last_observation, 700)
    tool_trace = utf8_take(tool_trace, 900)

    if last_observation == "" then
        last_observation = "（无）"
    end
    if tool_trace == "" then
        tool_trace = "（无）"
    end
    if substep_label == "" then
        substep_label = substep_name
    end
    if substep_description == "" then
        substep_description = "（无）"
    end

    local function_choice_rule = "function_choice=auto，可按需调用任意可用工具。"
    if function_choice == "none" then
        function_choice_rule = "function_choice=none，本轮禁止调用任何工具，必须输出空字符串。"
    elseif function_choice ~= "auto" then
        function_choice_rule = string.format(
            "function_choice=%s，本轮只允许输出 act=\"%s\"。",
            function_choice,
            function_choice
        )
    end
    local parallel_rule = parallel_function_calls
        and "parallel_function_calls=true：若有多个互补检索，可同轮输出多条调用。"
        or "parallel_function_calls=false：本轮最多输出 1 条工具调用。"

    return string.format([[
你是 keyring 工具调用规划器，只负责输出工具调用，不负责回复用户。

可用调用（每条必须单独一行，严格 Lua table）：
%s

硬约束：
1. 只输出工具调用行，不要解释、不要 markdown、不要代码块。
2. 如果不需要调用，输出空字符串。
3. upsert_record 只在“明确且长期稳定事实”时调用，confidence 必须 >= %.2f。
4. 本轮最多 upsert_record %d 条，最多 query_record %d 条。
5. delete_record 当前%s。
6. query_record 可适当发散（同义词、上位词）提高命中，但必须与当前问题相关。
7. 若上一轮 observation 已明确“无命中/无增量/重复”，不要重复同一 query_record。
8. 优先利用 observation 与 trace 纠错；仅在确有新增信息时再继续调用。
9. 若需要多个互补检索，可同轮输出多条独立 query_record（系统会并行批处理）。
10. %s
11. %s

输入上下文：
当前 step：
%d

当前子步骤：
%s (%s)
子步骤任务描述：
%s

用户原话：
%s

助手回复：
%s

上一轮 observation：
%s

当前轮 trace 摘要：
%s
]], tool_examples, policy.upsert_min_confidence, policy.upsert_max_per_turn, policy.query_max_per_turn, delete_flag, function_choice_rule, parallel_rule, step_idx, substep_name, substep_label, substep_description, user_input, assistant_text, last_observation, tool_trace)
end

local function filter_planner_calls(calls, planner_ctx, supported_acts)
    planner_ctx = planner_ctx or {}
    supported_acts = supported_acts or tool_parser.clone_supported_acts()
    local function_choice = normalize_function_choice(planner_ctx.function_choice, supported_acts)
    local parallel_function_calls = to_bool(
        planner_ctx.parallel_function_calls,
        get_default_parallel_function_calls()
    )

    local out = {}
    if function_choice == "none" then
        return out
    end

    for _, c in ipairs(calls or {}) do
        local act = trim((c or {}).act):lower()
        if function_choice == "auto" or act == function_choice then
            out[#out + 1] = c
        end
    end
    if (not parallel_function_calls) and #out > 1 then
        out = { out[1] }
    end
    return out
end

local function get_default_policy()
    local cfg = get_tool_cfg()
    local defaults = get_tool_defaults()
    return {
        upsert_min_confidence = cfg_number(cfg.upsert_min_confidence, defaults.upsert_min_confidence, 0, 1),
        upsert_max_per_turn = cfg_number(cfg.upsert_max_per_turn, defaults.upsert_max_per_turn, 0),
        query_max_per_turn = cfg_number(cfg.query_max_per_turn, defaults.query_max_per_turn, 0),
        delete_enabled = to_bool(cfg.delete_enabled, defaults.delete_enabled),
        query_max_types = cfg_number(cfg.query_max_types, defaults.query_max_types, 1),
        query_fetch_limit = cfg_number(cfg.query_fetch_limit, defaults.query_fetch_limit, 1),
        query_inject_top = cfg_number(cfg.query_inject_top, defaults.query_inject_top, 1),
        query_inject_max_chars = cfg_number(cfg.query_inject_max_chars, defaults.query_inject_max_chars, 200),
        tool_pass_temperature = cfg_number(cfg.tool_pass_temperature, defaults.tool_pass_temperature, 0, 1),
        tool_pass_max_tokens = cfg_number(cfg.tool_pass_max_tokens, defaults.tool_pass_max_tokens, 32),
        tool_pass_seed = cfg_number(cfg.tool_pass_seed, defaults.tool_pass_seed),
    }
end

function M.get_policy()
    return get_default_policy()
end

function M.plan_calls(user_input, assistant_text, policy, planner_ctx)
    local p = policy or get_default_policy()
    planner_ctx = planner_ctx or {}
    local supported_acts = get_supported_tool_acts()
    local safe_user = tostring(user_input or "")
    local safe_assistant = tostring(assistant_text or "")
    if tool.utf8_sanitize_lossy then
        safe_user = tool.utf8_sanitize_lossy(safe_user)
        safe_assistant = tool.utf8_sanitize_lossy(safe_assistant)
    end
    local prompt = build_tool_pass_prompt(safe_user, safe_assistant, p, planner_ctx, supported_acts)
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
    local calls = tool_parser.collect_tool_calls_only(raw, {
        supported_acts = supported_acts,
    })
    calls = filter_planner_calls(calls, planner_ctx, supported_acts)
    if #calls > 0 then
        print(string.format(
            "[ToolPlanner] two_step 产出 %d 条调用 (step=%d)",
            #calls,
            math.max(1, math.floor(tonumber(planner_ctx.step_idx) or 1))
        ))
    end
    return calls
end

return M
