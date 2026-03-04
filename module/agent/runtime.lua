local M = {}

local tool = require("module.tool")
local history = require("module.memory.history")
local topic = require("module.memory.topic")
local recall = require("module.memory.recall")
local tool_calling = require("module.agent.tool_calling")
local tool_planner = require("module.agent.tool_planner")
local tool_registry = require("module.agent.tool_registry")
local context_window = require("module.agent.context_window")
local config = require("module.config")

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
    return { act = act, raw = s }
end

local function remove_tool_call_lines(text)
    local kept = {}
    text = tostring(text or "")
    for line in (text .. "\n"):gmatch("(.-)\n") do
        local call = parse_tool_call_line(line)
        if not call then
            kept[#kept + 1] = line
        end
    end
    local visible = table.concat(kept, "\n")
    return trim(visible)
end

local function normalize_result_text(result)
    local cot_clean = strip_cot_safe(result or "")
    local visible = remove_tool_call_lines(cot_clean)
    if visible == "" then
        visible = trim(cot_clean)
    end
    if visible == "" then
        visible = "好的，已记录。"
    end
    if tool.utf8_sanitize_lossy then
        visible = tool.utf8_sanitize_lossy(visible)
    end
    return visible
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

local function build_step_feedback(step_idx, clean_result, calls, exec_result, reason, has_pending_ctx)
    local lines = {
        "【Agent多步反馈】",
        string.format("step=%d reason=%s", tonumber(step_idx) or 0, tostring(reason or "unknown")),
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

    local agent_cfg = get_agent_cfg()
    local max_steps = math.max(1, math.floor(tonumber(agent_cfg.max_steps) or 4))
    local completion_reserve_tokens = math.max(
        64,
        math.floor(tonumber(agent_cfg.completion_reserve_tokens) or 1024)
    )
    local token_budget = math.max(
        128,
        math.floor(tonumber(agent_cfg.input_token_budget) or 12000)
    )
    local continue_on_tool_context = to_bool(agent_cfg.continue_on_tool_context, true)
    local continue_on_tool_failure = to_bool(agent_cfg.continue_on_tool_failure, true)
    local max_context_refine_steps = math.max(
        0,
        math.floor(tonumber(agent_cfg.max_context_refine_steps) or 2)
    )
    local max_failure_refine_steps = math.max(
        0,
        math.floor(tonumber(agent_cfg.max_failure_refine_steps) or 2)
    )
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

        local messages_for_llm, ctx_meta = context_window.build_messages({
            conversation_history = conversation_history,
            user_input = user_input,
            plan_bom = plan_bom,
            tool_context = merged_tool_context,
            memory_context = memory_context,
            input_token_budget = token_budget,
        })
        print(string.format(
            "[AgentRuntime][step=%d] context_tokens=%d kept_pairs=%d dropped_pairs=%d compressed_pairs=%d history_summary=%s dropped_blocks=%s",
            attempts,
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
            temperature = 0.75,
            seed = math.random(1, 2147483647),
        }
        -- 多步模式下避免流式泄露中间草稿；最终文本由上层统一输出。
        if stream_sink and max_steps <= 1 then
            params.stream = true
        end

        local generated_text = ""
        local function chat_callback(result)
            generated_text = tostring(result or "")
        end
        py_pipeline:generate_chat(messages_for_llm, params, chat_callback, params.stream and stream_sink or nil)
        local clean_result = normalize_result_text(generated_text)
        final_result = clean_result

        state_name = "PLAN_TOOLS"
        local calls = {}
        if not read_only then
            calls = tool_planner.plan_calls(user_input, clean_result, tool_policy)
        else
            print("[AgentRuntime] read_only 模式：跳过工具规划")
        end
        local call_count = #(calls or {})
        print(string.format("[AgentRuntime][step=%d] tool_calls_count=%d", attempts, call_count))

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
            "[AgentRuntime][step=%d] tool_exec executed=%d failed=%d skipped=%d",
            attempts,
            tonumber(exec_result.executed) or 0,
            tonumber(exec_result.failed) or 0,
            tonumber(exec_result.skipped) or 0
        ))

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
                context_updated
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
