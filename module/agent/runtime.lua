local M = {}

local tool = require("module.tool")
local history = require("module.memory.history")
local topic = require("module.memory.topic")
local recall = require("module.memory.recall")
local tool_calling = require("module.agent.tool_calling")
local tool_planner = require("module.agent.tool_planner")
local tool_registry = require("module.agent.tool_registry")
local context_window = require("module.agent.context_window")
local tool_parser = require("module.agent.tool_parser")
local substep = require("module.agent.substep")
local config = require("module.config")

local PLAN_SIGNAL_PROMPT = [[
【计划信号】
仅在回复最后一行输出一个 Lua table 计划信号：{act="plan"} 或 {act="no_plan"}。
- 需要后台二阶段 planner 调用工具：{act="plan"}
- 不需要工具：{act="no_plan"}
禁止输出其它工具调用格式（包括 query_record/upsert_record/delete_record 等）。
不要输出解释或其它计划信号。
]]

local PLAN_SIGNAL_SUPPORTED_ACTS = {
    plan = true,
    no_plan = true,
}

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

local function strip_cot_safe(text)
    text = tostring(text or "")
    local cleaned = tool.remove_cot(text)
    if cleaned == "" and text ~= "" and not text:find("</think>", 1, true) then
        cleaned = text
    end
    return cleaned
end

local function normalize_result_text(result)
    local visible = trim(strip_cot_safe(result or ""))
    if visible == "" then
        visible = "好的，已记录。"
    end
    if tool.utf8_sanitize_lossy then
        visible = tool.utf8_sanitize_lossy(visible)
    end
    return visible
end

local function parse_plan_signal_line(line)
    local raw = trim(line)
    if raw == "" or not raw:match("^%b{}$") then
        return nil
    end

    local call = tool_parser.parse_tool_call_line(line, {
        supported_acts = PLAN_SIGNAL_SUPPORTED_ACTS,
    })
    if type(call) ~= "table" then
        return nil
    end

    local act = trim(call.act):lower()
    if act == "plan" then
        return true
    end
    if act == "no_plan" then
        return false
    end
    return nil
end

local function extract_plan_signal(text)
    local raw = tostring(text or "")
    local signal = nil
    local signal_line_idx = nil
    local lines = {}

    for line in (raw .. "\n"):gmatch("(.-)\n") do
        lines[#lines + 1] = line
    end

    for i = #lines, 1, -1 do
        if trim(lines[i]) ~= "" then
            local parsed = parse_plan_signal_line(lines[i])
            if parsed ~= nil then
                signal = parsed
                signal_line_idx = i
            end
            break
        end
    end

    if signal_line_idx then
        table.remove(lines, signal_line_idx)
        raw = table.concat(lines, "\n")
    end

    raw = trim(raw)
    return raw, signal
end

local function resolve_base_system_prompt(conversation_history)
    if type(conversation_history) ~= "table" then
        return ""
    end
    local first = conversation_history[1]
    if type(first) ~= "table" then
        return ""
    end
    if tostring(first.role or "") ~= "system" then
        return ""
    end
    return tostring(first.content or "")
end

local function build_step_system_prompt(base_system_prompt, inject_plan_signal_prompt)
    local base = trim(base_system_prompt)
    if not inject_plan_signal_prompt then
        return base
    end
    if base == "" then
        return PLAN_SIGNAL_PROMPT
    end
    return base .. "\n\n" .. PLAN_SIGNAL_PROMPT
end

local function normalize_planner_gate_mode(raw_mode)
    local mode = trim(raw_mode):lower()
    if mode == "always" or mode == "assistant_signal" then
        return mode
    end
    return "assistant_signal"
end

local function resolve_planner_gate(function_choice, gate_mode, explicit_signal, default_when_missing)
    if function_choice == "none" then
        return false, "function_choice_none"
    end
    if gate_mode == "always" then
        return true, "gate_always"
    end
    if explicit_signal ~= nil then
        if explicit_signal then
            return true, "assistant_signal_yes"
        end
        return false, "assistant_signal_no"
    end
    if default_when_missing then
        return true, "assistant_signal_default_yes"
    end
    return false, "assistant_signal_default_no"
end

local function join_dropped_blocks(blocks)
    if type(blocks) ~= "table" or #blocks == 0 then
        return "none"
    end
    return table.concat(blocks, ",")
end

local function get_agent_cfg()
    return (config.settings or {}).agent or {}
end

local function get_agent_defaults()
    local defaults = (config.defaults or {}).agent
    if type(defaults) ~= "table" then
        defaults = get_agent_cfg()
    end
    return defaults or {}
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

local function sleep_seconds(sec)
    sec = tonumber(sec) or 0
    if sec <= 0 then return end

    local ok_socket, socket = pcall(require, "socket")
    if ok_socket and socket and type(socket.sleep) == "function" then
        pcall(socket.sleep, sec)
        return
    end

    local ok_sleep = pcall(os.execute, string.format("sleep %.3f", sec))
    if not ok_sleep then
        -- no-op fallback
    end
end

local function run_with_retry(label, max_retries, base_backoff_sec, fn)
    local retries = math.max(0, math.floor(tonumber(max_retries) or 0))
    local attempt = 0

    while true do
        attempt = attempt + 1
        local ok, r1, r2, r3 = pcall(fn)
        if ok then
            return true, r1, r2, r3, attempt - 1
        end

        if attempt > (retries + 1) then
            return false, r1, nil, nil, attempt - 1
        end

        local wait = math.max(0, tonumber(base_backoff_sec) or 0) * (2 ^ math.max(0, attempt - 1))
        print(string.format(
            "[AgentRuntime][WARN] %s 失败 (attempt=%d/%d): %s",
            tostring(label or "unknown_call"),
            attempt,
            retries + 1,
            tostring(r1)
        ))
        if wait > 0 then
            print(string.format(
                "[AgentRuntime][WARN] %s 将在 %.2fs 后重试",
                tostring(label or "unknown_call"),
                wait
            ))
            sleep_seconds(wait)
        end
    end
end

local function normalize_function_choice(raw_choice, supported_acts)
    return tool_parser.normalize_function_choice(raw_choice, {
        supported_acts = supported_acts or tool_parser.clone_supported_acts(),
    })
end

local function filter_calls_by_choice(calls, function_choice, parallel_enabled)
    local in_calls = {}
    if type(calls) == "table" then
        in_calls = calls
    end

    if function_choice == "none" then
        return {}, #in_calls, 0
    end

    local filtered = {}
    for _, call in ipairs(in_calls) do
        local act = trim((call or {}).act):lower()
        if function_choice == "auto" or act == function_choice then
            filtered[#filtered + 1] = call
        end
    end

    local pre_parallel_count = #filtered
    if (not parallel_enabled) and #filtered > 1 then
        filtered = { filtered[1] }
    end

    return filtered, #in_calls, pre_parallel_count
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

local function format_call_brief(call)
    call = call or {}
    local act = trim(call.act or "")
    if act == "" then
        return "{act=\"unknown\"}"
    end
    if act == "query_record" then
        local q = trim(call.query or call.string or call.value)
        local types = trim(call.types or call.type)
        local payload = string.format('{act="%s"', act)
        if q ~= "" then
            payload = payload .. string.format(', query="%s"', utf8_take(q, 80))
        end
        if types ~= "" then
            payload = payload .. string.format(', types="%s"', utf8_take(types, 48))
        end
        return payload .. "}"
    end
    if act == "upsert_record" then
        local rec_type = trim(call.type)
        local entity = trim(call.entity)
        local value = trim(call.value or call.string)
        return string.format(
            '{act="%s", type="%s", entity="%s", value="%s"}',
            act,
            utf8_take(rec_type, 28),
            utf8_take(entity, 48),
            utf8_take(value, 64)
        )
    end
    if act == "delete_record" then
        return string.format(
            '{act="%s", type="%s", entity="%s"}',
            act,
            utf8_take(trim(call.type), 28),
            utf8_take(trim(call.entity), 48)
        )
    end
    return string.format('{act="%s"}', act)
end

local function build_observation_entry(step_idx, call_source, calls, exec_result)
    local executed = tonumber((exec_result or {}).executed) or 0
    local failed = tonumber((exec_result or {}).failed) or 0
    local skipped = tonumber((exec_result or {}).skipped) or 0
    local count = #(calls or {})
    if count <= 0 and executed <= 0 and failed <= 0 and skipped <= 0 then
        return ""
    end

    local lines = {
        string.format(
            "step=%d source=%s calls=%d executed=%d failed=%d skipped=%d",
            tonumber(step_idx) or 0,
            tostring(call_source or "unknown"),
            count,
            executed,
            failed,
            skipped
        ),
    }
    if count > 0 then
        lines[#lines + 1] = "actions:"
        for i, call in ipairs(calls or {}) do
            if i > 4 then break end
            lines[#lines + 1] = string.format("%d) %s", i, format_call_brief(call))
        end
    end

    local logs = (exec_result and exec_result.logs) or {}
    if type(logs) == "table" and #logs > 0 then
        lines[#lines + 1] = "observations:"
        for i, msg in ipairs(logs) do
            if i > 4 then break end
            lines[#lines + 1] = string.format("%d) %s", i, utf8_take(msg, 160))
        end
    end

    return table.concat(lines, "\n")
end

local function build_tool_observation_trace(entries, max_steps, max_chars)
    if type(entries) ~= "table" or #entries <= 0 then
        return ""
    end
    local agent_defaults = get_agent_defaults()
    max_steps = math.max(1, math.floor(
        tonumber(max_steps) or tonumber(agent_defaults.tool_trace_max_steps) or 4
    ))
    max_chars = math.max(120, math.floor(
        tonumber(max_chars) or tonumber(agent_defaults.tool_trace_max_chars) or 1200
    ))
    local start_idx = math.max(1, #entries - max_steps + 1)
    local lines = {
        "【Tool Observation Trace】",
        "以下是本轮已发生的 Action/Observation 轨迹（按时间顺序）：",
    }
    local idx = 0
    for i = start_idx, #entries do
        idx = idx + 1
        lines[#lines + 1] = string.format("trace#%d", idx)
        lines[#lines + 1] = entries[i]
    end
    local text = table.concat(lines, "\n")
    local clipped = utf8_take(text, max_chars)
    if clipped ~= text then
        return trim(clipped) .. "\n...(trace truncated)"
    end
    return text
end

local function build_step_feedback(step_idx, clean_result, calls, exec_result, reason, has_pending_ctx, call_source)
    local lines = {
        "【Agent多步反馈】",
        string.format("step=%d reason=%s", tonumber(step_idx) or 0, tostring(reason or "unknown")),
        string.format("call_source=%s", tostring(call_source or "planner_pass")),
        string.format(
            "calls=%d executed=%d failed=%d skipped=%d pending_context=%s",
            #(calls or {}),
            tonumber((exec_result or {}).executed) or 0,
            tonumber((exec_result or {}).failed) or 0,
            tonumber((exec_result or {}).skipped) or 0,
            has_pending_ctx and "yes" or "no"
        ),
    }

    local draft = trim(clean_result)
    if draft ~= "" then
        lines[#lines + 1] = "上一版回答（可修正）："
        lines[#lines + 1] = utf8_take(draft, 220)
    end

    local logs = (exec_result and exec_result.logs) or {}
    if type(logs) == "table" and #logs > 0 then
        lines[#lines + 1] = "工具执行日志："
        for i, msg in ipairs(logs) do
            if i > 4 then break end
            lines[#lines + 1] = string.format("%d) %s", i, utf8_take(msg, 160))
        end
    end

    lines[#lines + 1] = "请基于最新工具反馈修正答案；若工具失败，请避免引用失败结果。"
    return table.concat(lines, "\n")
end

function M.run_turn(args)
    args = args or {}
    local user_input = trim(args.user_input or "")
    if user_input == "" then
        return ""
    end

    local read_only = args.read_only == true
    local stream_sink = args.stream_sink
    local conversation_history = args.conversation_history or {}
    local add_to_history = args.add_to_history
    local base_system_prompt = resolve_base_system_prompt(conversation_history)

    local agent_cfg = get_agent_cfg()
    local agent_defaults = get_agent_defaults()
    local supported_tool_acts = get_supported_tool_acts()
    local max_steps = math.max(
        1,
        math.floor(tonumber(agent_cfg.max_steps) or tonumber(agent_defaults.max_steps) or 4)
    )
    local completion_reserve_tokens = math.max(
        64,
        math.floor(
            tonumber(agent_cfg.completion_reserve_tokens)
            or tonumber(agent_defaults.completion_reserve_tokens)
            or 1024
        )
    )
    local token_budget = math.max(
        128,
        math.floor(
            tonumber(agent_cfg.input_token_budget)
            or tonumber(agent_defaults.input_token_budget)
            or 12000
        )
    )
    local continue_on_tool_context = to_bool(
        agent_cfg.continue_on_tool_context,
        agent_defaults.continue_on_tool_context
    )
    local continue_on_tool_failure = to_bool(
        agent_cfg.continue_on_tool_failure,
        agent_defaults.continue_on_tool_failure
    )
    local max_context_refine_steps = math.max(
        0,
        math.floor(
            tonumber(agent_cfg.max_context_refine_steps)
            or tonumber(agent_defaults.max_context_refine_steps)
            or 2
        )
    )
    local max_failure_refine_steps = math.max(
        0,
        math.floor(
            tonumber(agent_cfg.max_failure_refine_steps)
            or tonumber(agent_defaults.max_failure_refine_steps)
            or 2
        )
    )
    local planner_gate_mode = normalize_planner_gate_mode(
        agent_cfg.planner_gate_mode or agent_defaults.planner_gate_mode
    )
    local planner_default_when_missing = to_bool(
        agent_cfg.planner_default_when_missing,
        agent_defaults.planner_default_when_missing
    )
    local include_tool_observation_trace = to_bool(
        agent_cfg.include_tool_observation_trace,
        agent_defaults.include_tool_observation_trace
    )
    local function_choice = normalize_function_choice(agent_cfg.function_choice, supported_tool_acts)
    local parallel_function_calls = to_bool(
        agent_cfg.parallel_function_calls,
        agent_defaults.parallel_function_calls
    )
    local tool_trace_max_steps = math.max(
        1,
        math.floor(
            tonumber(agent_cfg.tool_trace_max_steps)
            or tonumber(agent_defaults.tool_trace_max_steps)
            or 4
        )
    )
    local tool_trace_max_chars = math.max(
        240,
        math.floor(
            tonumber(agent_cfg.tool_trace_max_chars)
            or tonumber(agent_defaults.tool_trace_max_chars)
            or 1200
        )
    )
    local llm_retry_max = math.max(
        0,
        math.floor(
            tonumber(agent_cfg.llm_retry_max)
            or tonumber(agent_defaults.llm_retry_max)
            or 2
        )
    )
    local llm_retry_backoff_sec = cfg_number(
        agent_cfg.llm_retry_backoff_sec,
        agent_defaults.llm_retry_backoff_sec,
        0,
        10
    )
    local llm_temperature = cfg_number(
        agent_cfg.llm_temperature,
        agent_defaults.llm_temperature,
        0,
        2
    )
    local llm_seed_min = math.floor(cfg_number(
        agent_cfg.llm_seed_min,
        agent_defaults.llm_seed_min,
        1
    ))
    local llm_seed_max = math.floor(cfg_number(
        agent_cfg.llm_seed_max,
        agent_defaults.llm_seed_max,
        llm_seed_min
    ))
    if llm_seed_max < llm_seed_min then
        llm_seed_min, llm_seed_max = llm_seed_max, llm_seed_min
    end
    local planner_retry_max = math.max(
        0,
        math.floor(
            tonumber(agent_cfg.planner_retry_max)
            or tonumber(agent_defaults.planner_retry_max)
            or 1
        )
    )
    local planner_retry_backoff_sec = cfg_number(
        agent_cfg.planner_retry_backoff_sec,
        agent_defaults.planner_retry_backoff_sec,
        0,
        10
    )
    local substep_route_cfg = agent_cfg.substep_route or {}
    local substep_route_defaults = agent_defaults.substep_route or {}
    local substep_registry = substep.resolve_registry(
        agent_cfg.substeps,
        agent_defaults.substeps
    )
    local substep_default = substep.resolve_default_name(agent_cfg, agent_defaults, substep_registry)
    local substep_auto_route = to_bool(
        substep_route_cfg.auto_route,
        to_bool(
            agent_cfg.substep_auto_route,
            to_bool(substep_route_defaults.auto_route, agent_defaults.substep_auto_route)
        )
    )
    local substep_profile, substep_source = substep.resolve_turn_substep({
        requested = args.substep,
        user_input = user_input,
        registry = substep_registry,
        default_name = substep_default,
        auto_route = substep_auto_route,
        plan_keywords = substep_route_cfg.plan_keywords
            or agent_cfg.substep_plan_keywords
            or substep_route_defaults.plan_keywords
            or agent_defaults.substep_plan_keywords,
        explore_keywords = substep_route_cfg.explore_keywords
            or agent_cfg.substep_explore_keywords
            or substep_route_defaults.explore_keywords
            or agent_defaults.substep_explore_keywords,
    })
    print(string.format(
        "[AgentRuntime] turn_substep=%s source=%s auto_route=%s",
        tostring((substep_profile or {}).name or "general-purpose"),
        tostring(substep_source or "default"),
        substep_auto_route and "true" or "false"
    ))
    local substep_planner_cfg = type((substep_profile or {}).planner) == "table"
        and substep_profile.planner or {}
    planner_gate_mode = normalize_planner_gate_mode(
        substep_planner_cfg.planner_gate_mode or planner_gate_mode
    )
    planner_default_when_missing = to_bool(
        substep_planner_cfg.planner_default_when_missing,
        planner_default_when_missing
    )
    function_choice = normalize_function_choice(
        substep_planner_cfg.function_choice or function_choice,
        supported_tool_acts
    )
    parallel_function_calls = to_bool(
        substep_planner_cfg.parallel_function_calls,
        parallel_function_calls
    )
    include_tool_observation_trace = to_bool(
        substep_planner_cfg.include_tool_observation_trace,
        include_tool_observation_trace
    )
    print(string.format(
        "[AgentRuntime] substep_planner gate=%s default_when_missing=%s function_choice=%s parallel=%s trace=%s",
        tostring(planner_gate_mode),
        planner_default_when_missing and "true" or "false",
        tostring(function_choice),
        parallel_function_calls and "true" or "false",
        include_tool_observation_trace and "true" or "false"
    ))
    local memory_input_policy = tool_calling.get_memory_input_policy()
    local memory_user_input, memory_input_sanitized, redacted_blocks, file_mode, memory_input_truncated =
        tool_calling.sanitize_memory_input(user_input, {
            mode = memory_input_policy.recall_mode,
            max_chars = memory_input_policy.max_chars,
            manifest_max_items = memory_input_policy.manifest_max_items,
            manifest_name_max_chars = memory_input_policy.manifest_name_max_chars,
        })
    if memory_input_sanitized then
        print(string.format(
            "[AgentRuntime] memory 输入净化（mode=%s, blocks=%d, truncated=%s）",
            tostring(file_mode),
            tonumber(redacted_blocks) or 0,
            memory_input_truncated and "yes" or "no"
        ))
    end

    local attempts = 0
    local done = false
    local final_result = ""
    local state_name = "INIT"

    state_name = "PREPARE"
    local current_turn = history.get_turn() + 1
    local user_vec_q = tool.get_embedding_query(memory_user_input)
    local user_vec_p = tool.get_embedding_passage(memory_user_input)
    if not read_only then
        topic.add_turn(current_turn, memory_user_input, user_vec_p)
    end

    state_name = "BUILD_CONTEXT"
    local memory_context = recall.check_and_retrieve(memory_user_input, user_vec_q, {
        read_only = read_only,
    })
    local plan_bom = tool_registry.get_long_term_plan_bom()
    local tool_policy = tool_registry.get_policy()
    local step_feedback = ""
    local context_refine_used = 0
    local failure_refine_used = 0
    local last_call_signature = ""
    local tool_observation_entries = {}
    local latest_observation = ""

    while (not done) and attempts < max_steps do
        attempts = attempts + 1

        state_name = "BUILD_CONTEXT"
        local consumed_tool_context = ""
        if not read_only then
            consumed_tool_context = tool_registry.consume_pending_system_context_for_turn(current_turn)
        end
        local merged_tool_context = trim(consumed_tool_context)
        if step_feedback ~= "" then
            if merged_tool_context ~= "" then
                merged_tool_context = merged_tool_context .. "\n\n" .. step_feedback
            else
                merged_tool_context = step_feedback
            end
            step_feedback = ""
        end
        if include_tool_observation_trace then
            local trace_context = build_tool_observation_trace(
                tool_observation_entries,
                tool_trace_max_steps,
                tool_trace_max_chars
            )
            if trace_context ~= "" then
                if merged_tool_context ~= "" then
                    merged_tool_context = merged_tool_context .. "\n\n" .. trace_context
                else
                    merged_tool_context = trace_context
                end
            end
        end

        local messages_for_llm, ctx_meta = context_window.build_messages({
            conversation_history = conversation_history,
            system_prompt = build_step_system_prompt(
                base_system_prompt,
                (not read_only) and planner_gate_mode == "assistant_signal" and attempts == 1
            ),
            user_input = user_input,
            plan_bom = plan_bom,
            tool_context = merged_tool_context,
            memory_context = memory_context,
            input_token_budget = token_budget,
            token_count_mode = agent_cfg.token_count_mode,
            context_drop_order = agent_cfg.context_drop_order,
        })
        print(string.format(
            "[AgentRuntime][step=%d] substep=%s context_tokens=%d kept_pairs=%d dropped_pairs=%d compressed_pairs=%d history_summary=%s dropped_blocks=%s",
            attempts,
            tostring((substep_profile or {}).name or "general-purpose"),
            tonumber(ctx_meta.total_tokens) or 0,
            tonumber(ctx_meta.kept_history_pairs) or 0,
            tonumber(ctx_meta.dropped_history_pairs) or 0,
            tonumber(ctx_meta.compressed_history_pairs) or 0,
            (ctx_meta.history_summary_used == true) and "Y" or "N",
            join_dropped_blocks(ctx_meta.dropped_blocks)
        ))

        state_name = "GENERATE"
        local params = {
            max_tokens = completion_reserve_tokens,
            temperature = llm_temperature,
            seed = math.random(llm_seed_min, llm_seed_max),
        }
        -- 多步模式下避免流式泄露中间草稿；最终文本由上层统一输出。
        if stream_sink and max_steps <= 1 then
            params.stream = true
        end

        local generated_text = ""
        local ok_chat, chat_err = run_with_retry(
            "chat generation",
            llm_retry_max,
            llm_retry_backoff_sec,
            function()
                local function chat_callback(result)
                    generated_text = tostring(result or "")
                end
                py_pipeline:generate_chat(messages_for_llm, params, chat_callback, params.stream and stream_sink or nil)
            end
        )
        if not ok_chat then
            error(string.format(
                "[AgentRuntime] chat generation failed after retries: %s",
                tostring(chat_err)
            ))
        end

        local clean_result = normalize_result_text(generated_text)
        local planner_signal = nil
        clean_result, planner_signal = extract_plan_signal(clean_result)
        if clean_result == "" then
            clean_result = "好的，已记录。"
        end
        final_result = clean_result

        state_name = "PLAN_TOOLS"
        local calls = {}
        local call_source = "none"
        if not read_only then
            local should_plan, gate_source = resolve_planner_gate(
                function_choice,
                planner_gate_mode,
                planner_signal,
                planner_default_when_missing
            )
            print(string.format(
                "[AgentRuntime][step=%d] plan_gate decision=%s source=%s",
                attempts,
                should_plan and "plan" or "skip",
                tostring(gate_source)
            ))

            local tool_trace = ""
            if include_tool_observation_trace then
                tool_trace = build_tool_observation_trace(
                    tool_observation_entries,
                    tool_trace_max_steps,
                    tool_trace_max_chars
                )
            end

            if should_plan then
                local ok_plan, plan_calls_or_err = run_with_retry(
                    "tool planner",
                    planner_retry_max,
                    planner_retry_backoff_sec,
                    function()
                        return tool_planner.plan_calls(user_input, clean_result, tool_policy, {
                            step_idx = attempts,
                            last_observation = latest_observation,
                            tool_trace = tool_trace,
                            function_choice = function_choice,
                            parallel_function_calls = parallel_function_calls,
                            substep_name = (substep_profile or {}).name,
                            substep_label = (substep_profile or {}).label,
                            substep_description = (substep_profile or {}).description,
                        })
                    end
                )
                if ok_plan then
                    calls = plan_calls_or_err
                    call_source = "planner_pass"
                else
                    calls = {}
                    call_source = "planner_error"
                    print(string.format(
                        "[AgentRuntime][step=%d][WARN] planner 失败（重试后）: %s",
                        attempts,
                        tostring(plan_calls_or_err)
                    ))
                end
            else
                calls = {}
                call_source = "planner_skipped"
            end
        else
            print("[AgentRuntime] read_only 模式：跳过工具规划")
        end

        local raw_call_count = #(calls or {})
        local calls_after_choice, _, pre_parallel_count = filter_calls_by_choice(
            calls,
            function_choice,
            parallel_function_calls
        )
        calls = calls_after_choice
        if raw_call_count ~= #calls then
            print(string.format(
                "[AgentRuntime][step=%d] tool calls filtered by choice=%s parallel=%s raw=%d kept=%d",
                attempts,
                tostring(function_choice),
                parallel_function_calls and "true" or "false",
                raw_call_count,
                #calls
            ))
            if function_choice == "none" then
                call_source = "none"
            end
        elseif pre_parallel_count ~= #calls then
            print(string.format(
                "[AgentRuntime][step=%d] parallel_function_calls=false, keep first call only (from %d to %d)",
                attempts,
                pre_parallel_count,
                #calls
            ))
        end

        local call_count = #(calls or {})
        print(string.format(
            "[AgentRuntime][step=%d] tool_calls_count=%d source=%s",
            attempts,
            call_count,
            call_source
        ))

        local call_signature = ""
        if call_count > 0 then
            local sig_parts = {}
            for _, c in ipairs(calls) do
                local raw = trim(c.raw or "")
                if raw == "" then raw = trim(c.act or "") end
                if raw ~= "" then
                    sig_parts[#sig_parts + 1] = raw
                end
            end
            table.sort(sig_parts)
            call_signature = table.concat(sig_parts, "||")
        end
        local repeated_calls = (call_signature ~= "" and call_signature == last_call_signature)
        if call_signature ~= "" then
            last_call_signature = call_signature
        end

        state_name = "EXECUTE_TOOLS"
        local exec_result = tool_registry.execute_calls(calls, {
            current_turn = current_turn,
            read_only = read_only,
            policy = tool_policy,
        })
        print(string.format(
            "[AgentRuntime][step=%d] tool_exec executed=%d failed=%d skipped=%d parallel_batches=%d retries=%d",
            attempts,
            tonumber(exec_result.executed) or 0,
            tonumber(exec_result.failed) or 0,
            tonumber(exec_result.skipped) or 0,
            tonumber(exec_result.parallel_batches) or 0,
            tonumber(exec_result.retry_total) or 0
        ))
        if not read_only then
            local observation_entry = build_observation_entry(
                attempts,
                call_source,
                calls,
                exec_result
            )
            if observation_entry ~= "" then
                latest_observation = observation_entry
                tool_observation_entries[#tool_observation_entries + 1] = observation_entry
                if #tool_observation_entries > tool_trace_max_steps then
                    table.remove(tool_observation_entries, 1)
                end
            end
        end

        local has_pending_context = false
        if not read_only then
            has_pending_context = trim(tool_registry.get_pending_system_context(current_turn)) ~= ""
        end
        local context_updated = (exec_result.context_updated == true)
        local context_novel = exec_result.context_novel
        if (not context_updated) and has_pending_context then
            context_updated = true
        end
        if context_novel == nil then
            context_novel = context_updated
        else
            context_novel = (context_novel == true)
        end

        local continue_reason = nil
        if attempts < max_steps then
            if continue_on_tool_context and context_updated and context_refine_used < max_context_refine_steps then
                continue_reason = "tool_context_updated"
                context_refine_used = context_refine_used + 1
            elseif continue_on_tool_failure and (tonumber(exec_result.failed) or 0) > 0 and failure_refine_used < max_failure_refine_steps then
                continue_reason = "tool_failed"
                failure_refine_used = failure_refine_used + 1
            end
        end

        if continue_reason == "tool_context_updated" and (not context_novel) then
            print(string.format(
                "[AgentRuntime][step=%d] 工具上下文无增量，提前收敛",
                attempts
            ))
            continue_reason = nil
        end

        if continue_reason and repeated_calls and continue_reason ~= "tool_context_updated" then
            print(string.format(
                "[AgentRuntime][step=%d] 检测到重复工具调用签名，提前收敛",
                attempts
            ))
            continue_reason = nil
        end

        if continue_reason then
            step_feedback = build_step_feedback(
                attempts,
                clean_result,
                calls,
                exec_result,
                continue_reason,
                context_updated,
                call_source
            )
            print(string.format("[AgentRuntime][step=%d] continue reason=%s", attempts, continue_reason))
        else
            done = true
        end
    end

    if not done then
        if final_result == "" then
            error(string.format("[AgentRuntime] max_steps exceeded: %d", max_steps))
        end
        print(string.format(
            "[AgentRuntime][WARN] max_steps reached (%d)，返回最后一版结果",
            max_steps
        ))
    end

    state_name = "PERSIST"
    print("\n[Assistant]: " .. final_result)
    if not read_only then
        history.add_history(user_input, final_result)
        topic.update_assistant(current_turn, final_result)
        if add_to_history then
            add_to_history(user_input, final_result)
        end

        local facts = tool_calling.extract_atomic_facts(memory_user_input, final_result)
        local cur_summary = topic.get_summary(current_turn)
        if cur_summary and cur_summary ~= "" then
            print("[当前话题摘要] " .. cur_summary)
        end
        tool_calling.save_turn_memory(facts, current_turn)
    else
        print("[AgentRuntime] read_only 模式：跳过 history/topic/memory 写入")
    end

    state_name = "DONE"
    print("[AgentRuntime] agent_state_end=" .. state_name)
    return final_result
end

return M
