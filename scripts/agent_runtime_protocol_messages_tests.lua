local function clear_mods()
    local mods = {
        "module.config",
        "module.tool",
        "module.memory.history",
        "module.memory.topic",
        "module.memory.recall",
        "module.agent.tool_calling",
        "module.agent.tool_planner",
        "module.agent.tool_registry",
        "module.agent.context_window",
        "module.agent.tool_parser",
        "module.agent.runtime",
    }
    for _, m in ipairs(mods) do
        package.loaded[m] = nil
        package.preload[m] = nil
    end
end

clear_mods()

local state = {
    build_ctx_calls = 0,
    tail_messages_by_step = {},
    generate_calls = 0,
    planner_calls = 0,
    exec_calls = 0,
}

package.preload["module.config"] = function()
    return {
        settings = {
            agent = {
                max_steps = 2,
                input_token_budget = 12000,
                completion_reserve_tokens = 256,
                continue_on_tool_context = true,
                continue_on_tool_failure = true,
                max_context_refine_steps = 1,
                max_failure_refine_steps = 1,
                function_choice = "auto",
                parallel_function_calls = true,
                function_protocol_enabled = true,
                function_protocol_result_max_chars = 600,
            },
        },
    }
end

package.preload["module.tool"] = function()
    local M = {}
    function M.remove_cot(text) return tostring(text or "") end
    function M.extract_first_lua_table(s)
        return tostring(s or ""):match("%b{}")
    end
    function M.get_embedding_query(_) return { 0.1, 0.2 } end
    function M.get_embedding_passage(_) return { 0.2, 0.3 } end
    return M
end

package.preload["module.memory.history"] = function()
    local M = {}
    local turn = 0
    function M.get_turn() return turn end
    function M.add_history() turn = turn + 1 end
    return M
end

package.preload["module.memory.topic"] = function()
    local M = {}
    function M.add_turn() end
    function M.update_assistant() end
    function M.get_summary() return "" end
    function M.get_topic_anchor(turn) return "anchor-" .. tostring(turn or 0) end
    return M
end

package.preload["module.memory.recall"] = function()
    local M = {}
    function M.check_and_retrieve() return "" end
    return M
end

package.preload["module.agent.tool_calling"] = function()
    local M = {}
    function M.get_memory_input_policy()
        return {
            recall_mode = "ignore",
            max_chars = 2048,
            manifest_max_items = 8,
            manifest_name_max_chars = 96,
        }
    end
    function M.sanitize_memory_input(user_input)
        return tostring(user_input or ""), false, 0, "ignore", false
    end
    function M.extract_atomic_facts() return {} end
    function M.save_turn_memory() end
    return M
end

package.preload["module.agent.tool_planner"] = function()
    local M = {}
    function M.plan_calls()
        state.planner_calls = state.planner_calls + 1
        if state.planner_calls == 1 then
            return {
                {
                    act = "query_record",
                    query = "协议测试",
                    raw = '{act="query_record",query="协议测试"}',
                },
            }
        end
        return {}
    end
    return M
end

package.preload["module.agent.tool_registry"] = function()
    local M = {}
    local pending = ""
    function M.get_policy()
        return {}
    end
    function M.get_long_term_plan_bom()
        return ""
    end
    function M.get_supported_acts(base)
        local acts = {}
        for k, v in pairs(base or {}) do
            if v then acts[k] = true end
        end
        acts.query_record = true
        return acts
    end
    function M.get_openai_tools()
        return {}
    end
    function M.consume_pending_system_context_for_turn()
        local x = pending
        pending = ""
        return x
    end
    function M.get_pending_system_context()
        return pending
    end
    function M.execute_calls(calls)
        state.exec_calls = state.exec_calls + 1
        if type(calls) == "table" and #calls > 0 then
            pending = "【Tool:query_record 上一轮检索结果】\nquery: 协议测试"
            return {
                executed = 1,
                failed = 0,
                skipped = 0,
                logs = { "query_record ok (1 hits)" },
                context_updated = true,
                context_novel = true,
                call_results = {
                    {
                        act = "query_record",
                        tool_call_id = "tc_protocol_1",
                        arguments_json = '{"query":"协议测试"}',
                        ok = true,
                        skipped = false,
                        message = "query_record ok (1 hits)",
                        result = "hits=1",
                    }
                },
            }
        end
        return {
            executed = 0,
            failed = 0,
            skipped = 0,
            logs = {},
            context_updated = false,
            context_novel = false,
            call_results = {},
        }
    end
    return M
end

package.preload["module.agent.context_window"] = function()
    local M = {}
    function M.build_messages(opts)
        state.build_ctx_calls = state.build_ctx_calls + 1
        local tails = {}
        for _, m in ipairs(opts.tail_messages or {}) do
            tails[#tails + 1] = m
        end
        state.tail_messages_by_step[state.build_ctx_calls] = tails

        return {
            { role = "system", content = "sys" },
            { role = "user", content = tostring(opts.user_input or "") },
        }, {
            total_tokens = 42,
            kept_history_pairs = 0,
            dropped_history_pairs = 0,
            dropped_blocks = {},
            history_summary_used = false,
            compressed_history_pairs = 0,
        }
    end
    return M
end

_G.py_pipeline = {
    generate_chat = function(_, _, _, cb, _)
        state.generate_calls = state.generate_calls + 1
        if state.generate_calls == 1 then
            cb("第一步草稿")
        else
            cb("第二步终稿")
        end
    end,
}

local runtime = require("module.agent.runtime")
local final = runtime.run_turn({
    user_input = "请触发工具并修正回答",
    read_only = false,
    conversation_history = {
        { role = "system", content = "你是测试助手" },
    },
})

assert(final == "第二步终稿", "final output should converge on step2")
assert(state.build_ctx_calls == 2, "should build context twice")

local step1_tail = state.tail_messages_by_step[1] or {}
local step2_tail = state.tail_messages_by_step[2] or {}
assert(#step1_tail == 0, "step1 should not contain protocol tail messages")
assert(#step2_tail >= 2, "step2 should contain assistant/tool protocol tail")
assert(tostring((step2_tail[1] or {}).role) == "assistant", "tail[1] should be assistant tool_call message")
assert(type((step2_tail[1] or {}).tool_calls) == "table", "assistant tail should contain tool_calls")
assert(tostring((step2_tail[2] or {}).role) == "tool", "tail[2] should be tool result message")
assert(tostring((step2_tail[2] or {}).tool_call_id) == "tc_protocol_1", "tool_call_id should be preserved")

print("AGENT_RUNTIME_PROTOCOL_MESSAGES_TESTS_PASS")
