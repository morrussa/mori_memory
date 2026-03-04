local M = {}

local tool = require("module.tool")
local history = require("module.history")
local topic = require("module.topic")
local recall = require("module.recall")
local tool_calling = require("module.tool_calling")
local tool_planner = require("module.tool_planner")
local tool_registry = require("module.tool_registry")
local context_window = require("module.context_window")
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

    local attempts = 0
    local done = false
    local final_result = ""
    local state_name = "INIT"

    while (not done) and attempts < max_steps do
        attempts = attempts + 1

        state_name = "PREPARE"
        local current_turn = history.get_turn() + 1
        local user_vec_q = tool.get_embedding_query(user_input)
        local user_vec_p = tool.get_embedding_passage(user_input)
        if not read_only then
            topic.add_turn(current_turn, user_input, user_vec_p)
        end

        state_name = "BUILD_CONTEXT"
        local memory_context = recall.check_and_retrieve(user_input, user_vec_q)
        local plan_bom = tool_registry.get_long_term_plan_bom()
        local tool_context = ""
        if not read_only then
            tool_context = tool_registry.consume_pending_system_context_for_turn(current_turn)
        end

        local messages_for_llm, ctx_meta = context_window.build_messages({
            conversation_history = conversation_history,
            user_input = user_input,
            plan_bom = plan_bom,
            tool_context = tool_context,
            memory_context = memory_context,
            input_token_budget = token_budget,
        })
        print(string.format(
            "[AgentRuntime] context_tokens=%d kept_pairs=%d dropped_blocks=%s",
            tonumber(ctx_meta.total_tokens) or 0,
            tonumber(ctx_meta.kept_history_pairs) or 0,
            join_dropped_blocks(ctx_meta.dropped_blocks)
        ))

        state_name = "GENERATE"
        local params = {
            max_tokens = completion_reserve_tokens,
            temperature = 0.75,
            seed = math.random(1, 2147483647),
        }
        if stream_sink then
            params.stream = true
        end

        local generated_text = ""
        local function chat_callback(result)
            generated_text = tostring(result or "")
        end
        py_pipeline:generate_chat(messages_for_llm, params, chat_callback, stream_sink)
        local clean_result = normalize_result_text(generated_text)

        state_name = "PLAN_TOOLS"
        local tool_policy = tool_registry.get_policy()
        local calls = {}
        if not read_only then
            calls = tool_planner.plan_calls(user_input, clean_result, tool_policy)
        else
            print("[AgentRuntime] read_only 模式：跳过工具规划")
        end
        print(string.format("[AgentRuntime] tool_calls_count=%d", #(calls or {})))

        state_name = "EXECUTE_TOOLS"
        local exec_result = tool_registry.execute_calls(calls, {
            current_turn = current_turn,
            read_only = read_only,
            policy = tool_policy,
        })
        print(string.format(
            "[AgentRuntime] tool_exec executed=%d failed=%d skipped=%d",
            tonumber(exec_result.executed) or 0,
            tonumber(exec_result.failed) or 0,
            tonumber(exec_result.skipped) or 0
        ))

        state_name = "PERSIST"
        print("\n[Assistant]: " .. clean_result)
        if not read_only then
            history.add_history(user_input, clean_result)
            topic.update_assistant(current_turn, clean_result)
            if add_to_history then
                add_to_history(user_input, clean_result)
            end

            local facts = tool_calling.extract_atomic_facts(user_input, clean_result)
            local cur_summary = topic.get_summary(current_turn)
            if cur_summary and cur_summary ~= "" then
                print("[当前话题摘要] " .. cur_summary)
            end
            tool_calling.save_turn_memory(facts, current_turn)
        else
            print("[AgentRuntime] read_only 模式：跳过 history/topic/memory 写入")
        end

        final_result = clean_result
        done = true
    end

    if not done then
        error(string.format("[AgentRuntime] max_steps exceeded: %d", max_steps))
    end

    state_name = "DONE"
    print("[AgentRuntime] agent_state_end=" .. state_name)
    return final_result
end

return M
