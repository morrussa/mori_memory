local function contains(s, part)
    return tostring(s or ""):find(part, 1, true) ~= nil
end

local STUB_MODULES = {
    "module.config",
    "module.tool",
    "module.memory.history",
    "module.memory.topic",
    "module.memory.recall",
    "module.agent.tool_calling",
    "module.agent.tool_planner",
    "module.agent.tool_registry",
    "module.agent.tool_parser",
    "module.agent.context_window",
    "module.agent.substep",
    "module.agent.runtime",
}

local function clear_modules()
    for _, name in ipairs(STUB_MODULES) do
        package.loaded[name] = nil
        package.preload[name] = nil
    end
end

local function install_stubs(state, scenario)
    package.preload["module.config"] = function()
        return {
            settings = {
                agent = {
                    max_steps = scenario.max_steps or 4,
                    input_token_budget = 12000,
                    completion_reserve_tokens = 256,
                    continue_on_tool_context = true,
                    continue_on_tool_failure = true,
                    max_failure_refine_steps = scenario.max_failure_refine_steps or 3,
                    planner_gate_mode = scenario.planner_gate_mode or "assistant_signal",
                    planner_default_when_missing = (scenario.planner_default_when_missing == true),
                    function_choice = scenario.function_choice or "auto",
                    parallel_function_calls = (scenario.parallel_function_calls ~= false),
                    substep_default = scenario.substep_default or "general-purpose",
                    substep_auto_route = (scenario.substep_auto_route ~= false),
                    substep_route = scenario.substep_route or {
                        auto_route = (scenario.substep_auto_route ~= false),
                    },
                }
            }
        }
    end

    package.preload["module.tool"] = function()
        local M = {}
        function M.remove_cot(text) return tostring(text or "") end
        function M.extract_first_lua_table(s)
            local first = tostring(s or ""):match("%b{}")
            return first
        end
        function M.get_embedding_query(_) return { 0.1, 0.2 } end
        function M.get_embedding_passage(_) return { 0.2, 0.3 } end
        return M
    end

    package.preload["module.memory.history"] = function()
        local M = {}
        local turn = 0
        function M.get_turn() return turn end
        function M.add_history(user_text, assistant_text)
            state.history_adds = state.history_adds + 1
            state.last_history = { user = user_text, assistant = assistant_text }
            turn = turn + 1
        end
        return M
    end

    package.preload["module.memory.topic"] = function()
        local M = {}
        function M.add_turn(current_turn, user_text, _)
            state.topic_add_turns = state.topic_add_turns + 1
            state.last_topic_turn = current_turn
            state.last_topic_user = user_text
        end
        function M.update_assistant(current_turn, assistant_text)
            state.topic_updates = state.topic_updates + 1
            state.last_topic_update = { turn = current_turn, assistant = assistant_text }
        end
        function M.get_summary(_) return "" end
        function M.get_topic_anchor(turn)
            return "anchor-" .. tostring(turn or 0)
        end
        return M
    end

    package.preload["module.memory.recall"] = function()
        local M = {}
        function M.check_and_retrieve(user_input, _, opts)
            state.recall_calls = state.recall_calls + 1
            state.last_recall_user = user_input
            state.recall_opts[state.recall_calls] = opts or {}
            return "【相关记忆】\n记忆A"
        end
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
        function M.sanitize_memory_input(user_input, _)
            return tostring(user_input or ""), false, 0, "ignore", false
        end
        function M.extract_atomic_facts(_, assistant_text)
            state.extract_calls = state.extract_calls + 1
            state.last_extract_input = assistant_text
            return { "事实1" }
        end
        function M.save_turn_memory(facts, current_turn)
            state.memory_save_calls = state.memory_save_calls + 1
            state.last_saved = { facts = facts, turn = current_turn }
        end
        return M
    end

    package.preload["module.agent.tool_planner"] = function()
        local M = {}
        function M.plan_calls(_, assistant_text, _, planner_ctx)
            state.plan_calls = state.plan_calls + 1
            state.plan_inputs[state.plan_calls] = assistant_text
            state.plan_ctxs[state.plan_calls] = planner_ctx or {}
            local scripted = scenario.plan_outputs[state.plan_calls]
            if scripted == nil then
                scripted = scenario.plan_outputs[#scenario.plan_outputs] or {}
            end
            return scripted
        end
        return M
    end

    package.preload["module.agent.tool_registry"] = function()
        local M = {}
        local pending_ctx = ""

        function M.get_long_term_plan_bom()
            return "【LongTermPlan BOM】"
        end

        function M.get_policy()
            return {}
        end

        function M.get_openai_tools(_)
            return {
                {
                    type = "function",
                    ["function"] = {
                        name = "query_record",
                        description = "query memory records",
                        parameters = {
                            type = "object",
                            properties = {
                                query = { type = "string" },
                                types = { type = "string" },
                            },
                            required = { "query" },
                        },
                    },
                },
            }
        end

        function M.consume_pending_system_context_for_turn(_)
            local out = pending_ctx
            pending_ctx = ""
            return out
        end

        function M.get_pending_system_context(_)
            return pending_ctx
        end

        function M.execute_calls(calls, _)
            state.exec_calls = state.exec_calls + 1
            local result = {
                executed = 0,
                failed = 0,
                skipped = 0,
                logs = {},
            }
            for _, c in ipairs(calls or {}) do
                if c.act == "query_record" then
                    result.executed = result.executed + 1
                    pending_ctx = "【Tool:query_record 上一轮检索结果】\nquery: " .. tostring(c.query or "")
                    result.logs[#result.logs + 1] = "query_record ok (1 hits)"
                elseif c.act == "fail_call" then
                    result.failed = result.failed + 1
                    result.logs[#result.logs + 1] = "fail_call simulated error"
                else
                    result.executed = result.executed + 1
                    result.logs[#result.logs + 1] = "ok"
                end
            end
            state.exec_results[state.exec_calls] = result
            return result
        end

        return M
    end

    package.preload["module.agent.context_window"] = function()
        local M = {}
        function M.build_messages(opts)
            state.build_context_calls = state.build_context_calls + 1
            state.tool_contexts[state.build_context_calls] = tostring(opts.tool_context or "")
            local messages = {
                { role = "system", content = "sys" },
                { role = "user", content = tostring(opts.user_input or "") },
            }
            local meta = {
                total_tokens = 100,
                kept_history_pairs = 0,
                dropped_history_pairs = 0,
                dropped_blocks = {},
                budget = tonumber(opts.input_token_budget) or 0,
            }
            return messages, meta
        end
        return M
    end

    _G.py_pipeline = {
        generate_chat = function(_, _, _, cb, _)
            state.generate_calls = state.generate_calls + 1
            local out = scenario.generate_outputs[state.generate_calls]
            if out == nil then
                out = scenario.generate_outputs[#scenario.generate_outputs]
            end
            cb(out)
        end,
    }
end

local function run_scenario(name, scenario)
    clear_modules()
    local state = {
        generate_calls = 0,
        build_context_calls = 0,
        plan_calls = 0,
        exec_calls = 0,
        extract_calls = 0,
        memory_save_calls = 0,
        history_adds = 0,
        topic_add_turns = 0,
        topic_updates = 0,
        recall_calls = 0,
        recall_opts = {},
        tool_contexts = {},
        plan_inputs = {},
        plan_ctxs = {},
        exec_results = {},
    }

    install_stubs(state, scenario)

    local runtime = require("module.agent.runtime")
    local final = runtime.run_turn({
        user_input = scenario.user_input or "请结合我的信息回答",
        substep = scenario.substep,
        read_only = (scenario.read_only == true),
        conversation_history = {
            { role = "system", content = "你是测试助手" },
        },
        add_to_history = function(_, _)
            state.add_to_history_calls = (state.add_to_history_calls or 0) + 1
        end,
    })

    scenario.assertions(state, final)
    print(string.format("[PASS] %s | generate=%d plan=%d exec=%d", name, state.generate_calls, state.plan_calls, state.exec_calls))
end

local scenarios = {
    {
        name = "tool_hit_reinject",
        max_steps = 4,
        plan_outputs = {
            {
                { act = "query_record", query = "用户偏好", raw = '{act="query_record",query="用户偏好"}' },
            },
            {},
        },
        generate_outputs = {
            "初版回答：我先给你一个大概结论。\n{act=\"plan\"}",
            "最终答案（已结合工具检索）\n{act=\"no_plan\"}",
        },
        assertions = function(state, final)
            assert(state.generate_calls == 2, "tool_hit 应该触发第二步生成")
            assert(state.plan_calls == 1, "tool_hit 在 {act=\"no_plan\"} 收敛后不应继续规划")
            assert(state.history_adds == 1, "最终持久化应只写一次 history")
            assert(state.memory_save_calls == 1, "最终持久化应只写一次 memory")
            assert(contains(state.tool_contexts[2], "Tool:query_record"), "第二步上下文应包含工具检索结果")
            assert(final == "最终答案（已结合工具检索）", "最终答案应为第二步收敛结果")
        end,
    },
    {
        name = "function_choice_none_blocks_planner_calls",
        max_steps = 4,
        function_choice = "none",
        plan_outputs = {
            {
                { act = "query_record", query = "用户偏好", raw = '{act="query_record",query="用户偏好"}' },
            },
        },
        generate_outputs = {
            "先给你结论。\n{act=\"plan\"}",
        },
        assertions = function(state, final)
            assert(state.generate_calls == 1, "function_choice=none 不应触发下一步")
            assert(state.plan_calls == 0, "function_choice=none 时不应进入 planner")
            assert(state.exec_results[1] and state.exec_results[1].executed == 0, "function_choice=none 不应执行工具")
            assert(final == "先给你结论。", "应返回可见文本答案")
        end,
    },
    {
        name = "parallel_function_calls_false_keep_first",
        max_steps = 4,
        parallel_function_calls = false,
        plan_outputs = {
            {
                { act = "query_record", query = "用户偏好", raw = '{act="query_record",query="用户偏好"}' },
                { act = "query_record", query = "长期目标", raw = '{act="query_record",query="长期目标"}' },
            },
            {},
        },
        generate_outputs = {
            "我先查。\n{act=\"plan\"}",
            "最终答案：已按单调用约束收敛。\n{act=\"no_plan\"}",
        },
        assertions = function(state, final)
            assert(state.generate_calls == 2, "首轮命中工具后应触发下一步生成")
            assert(state.plan_calls == 1, "收到 {act=\"no_plan\"} 后第二步不应继续规划")
            assert(state.exec_results[1] and state.exec_results[1].executed == 1, "parallel_function_calls=false 应仅执行首条调用")
            assert(final == "最终答案：已按单调用约束收敛。", "最终答案应来自第二步收敛结果")
        end,
    },
    {
        name = "tool_fail_refine",
        max_steps = 4,
        max_failure_refine_steps = 2,
        plan_outputs = {
            {
                { act = "fail_call", raw = '{act="fail_call"}' },
            },
            {},
        },
        generate_outputs = {
            "初版回答：尝试执行工具。\n{act=\"plan\"}",
            "修正版回答：工具失败，给出无工具依赖方案。\n{act=\"no_plan\"}",
        },
        assertions = function(state, final)
            assert(state.generate_calls == 2, "tool_fail 应该触发失败后重修")
            assert(contains(state.tool_contexts[2], "reason=tool_failed"), "第二步应包含失败反馈")
            assert(state.exec_results[1] and state.exec_results[1].failed == 1, "第一步应有1次工具失败")
            assert(final == "修正版回答：工具失败，给出无工具依赖方案。", "最终答案应是失败后修正版")
        end,
    },
    {
        name = "repeat_call_converge",
        max_steps = 5,
        max_failure_refine_steps = 4,
        plan_outputs = {
            {
                { act = "fail_call", raw = '{act="fail_call",type="x"}' },
            },
            {
                { act = "fail_call", raw = '{act="fail_call",type="x"}' },
            },
            {
                { act = "fail_call", raw = '{act="fail_call",type="x"}' },
            },
        },
        generate_outputs = {
            "草稿1\n{act=\"plan\"}",
            "草稿2\n{act=\"plan\"}",
            "草稿3\n{act=\"no_plan\"}",
        },
        assertions = function(state, final)
            assert(state.generate_calls == 2, "重复同签名调用应在第2步收敛，避免死循环")
            assert(state.exec_calls == 2, "重复同签名调用应只执行两轮")
            assert(final == "草稿2", "重复签名收敛时应返回最后一版答案")
        end,
    },
    {
        name = "read_only_zero_side_effect",
        read_only = true,
        max_steps = 4,
        plan_outputs = {
            {},
        },
        generate_outputs = {
            "只读答案（不写状态）",
        },
        assertions = function(state, final)
            assert(state.recall_calls == 1, "read_only 仍应执行 recall 检索")
            assert(state.recall_opts[1] and state.recall_opts[1].read_only == true, "read_only 标记应透传 recall")
            assert(state.plan_calls == 0, "read_only 不应进入工具规划")
            assert(state.history_adds == 0, "read_only 不应写 history")
            assert(state.topic_add_turns == 0, "read_only 不应写 topic.add_turn")
            assert(state.topic_updates == 0, "read_only 不应写 topic.update_assistant")
            assert(state.memory_save_calls == 0, "read_only 不应写 memory")
            assert((state.add_to_history_calls or 0) == 0, "read_only 不应写 conversation_history")
            assert(final == "只读答案（不写状态）", "read_only 应返回生成结果")
        end,
    },
    {
        name = "auto_route_explore_substep",
        user_input = "先帮我搜索代码库里和会话相关的关键词",
        max_steps = 3,
        plan_outputs = {
            {},
        },
        generate_outputs = {
            "先做快速探索。\n{act=\"plan\"}",
        },
        assertions = function(state, final)
            local planner_ctx = state.plan_ctxs[1] or {}
            assert(planner_ctx.substep_name == "explore", "应自动路由到 explore 子步骤")
            assert(planner_ctx.substep_label == "Explore", "explore 子步骤应传递 label")
            assert(final == "先做快速探索。", "最终答案应返回清洗后的文本")
        end,
    },
    {
        name = "explore_substep_planner_gate_always",
        user_input = "帮我查找项目里与会话路由相关的代码位置",
        max_steps = 3,
        plan_outputs = {
            {},
        },
        generate_outputs = {
            "我先快速扫一遍。\n{act=\"no_plan\"}",
        },
        assertions = function(state, final)
            local planner_ctx = state.plan_ctxs[1] or {}
            assert(planner_ctx.substep_name == "explore", "关键词路由应命中 explore 子步骤")
            assert(state.plan_calls == 1, "explore 子步骤应使用 planner_gate_mode=always")
            assert(final == "我先快速扫一遍。", "最终答案应返回清洗后的文本")
        end,
    },
    {
        name = "requested_plan_substep",
        substep = "plan",
        user_input = "请给我一个完整实现方案",
        max_steps = 3,
        plan_outputs = {
            {},
        },
        generate_outputs = {
            "我先给出架构方案。\n{act=\"plan\"}",
        },
        assertions = function(state, final)
            local planner_ctx = state.plan_ctxs[1] or {}
            assert(planner_ctx.substep_name == "plan", "显式 substep=plan 应透传到 planner")
            assert(planner_ctx.substep_label == "Plan", "plan 子步骤应传递 label")
            assert(final == "我先给出架构方案。", "最终答案应返回清洗后的文本")
        end,
    },
}

for _, s in ipairs(scenarios) do
    run_scenario(s.name, s)
end

print("ALL_SCENARIOS_PASS")
