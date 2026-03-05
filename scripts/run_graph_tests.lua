local function printf(fmt, ...)
    io.write(string.format(fmt, ...) .. "\n")
end

local function fail(msg)
    error(msg, 0)
end

local function encode_lua_value(v)
    local tv = type(v)
    if tv == "string" then
        local s = v:gsub("\\", "\\\\"):gsub('"', '\\"'):gsub("\n", "\\n")
        return '"' .. s .. '"'
    end
    if tv == "number" or tv == "boolean" then
        return tostring(v)
    end
    if tv == "table" then
        local is_array = true
        local max_i = 0
        local count = 0
        for k, _ in pairs(v) do
            if type(k) ~= "number" or k < 1 or k % 1 ~= 0 then
                is_array = false
                break
            end
            if k > max_i then
                max_i = k
            end
            count = count + 1
        end
        if is_array and max_i == count then
            local parts = {}
            for i = 1, max_i do
                parts[#parts + 1] = encode_lua_value(v[i])
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end
        local parts = {}
        for k, val in pairs(v) do
            local key
            if type(k) == "string" and k:match("^[A-Za-z_][A-Za-z0-9_]*$") then
                key = k
            else
                key = "[" .. encode_lua_value(tostring(k)) .. "]"
            end
            parts[#parts + 1] = key .. "=" .. encode_lua_value(val)
        end
        table.sort(parts)
        return "{" .. table.concat(parts, ",") .. "}"
    end
    if v == nil then
        return "nil"
    end
    return encode_lua_value(tostring(v))
end

local function planner_literal(calls)
    return "{tool_calls=" .. encode_lua_value(calls or {}) .. "}"
end

local function mk_tool_call(name, args, call_id)
    return {
        tool = name,
        args = args or {},
        call_id = call_id or ("call_" .. tostring(name)),
    }
end

local function parse_lua_args(text)
    local src = tostring(text or "")
    local chunk = load("return " .. src, "tool_args", "t", {})
    if not chunk then
        return {}
    end
    local ok, value = pcall(chunk)
    if not ok or type(value) ~= "table" then
        return {}
    end
    return value
end

local CURRENT = nil

local function setup_stubs(case)
    CURRENT = {
        id = case.id,
        router = {},
        planner = {},
        repair = {},
        planner_default = case.planner_default,
        repair_default = case.repair_default,
        external_calls = 0,
        file_calls = {},
    }
    for _, v in ipairs(case.router_steps or {}) do
        CURRENT.router[#CURRENT.router + 1] = v
    end
    for _, v in ipairs(case.planner_steps or {}) do
        CURRENT.planner[#CURRENT.planner + 1] = v
    end
    for _, v in ipairs(case.repair_steps or {}) do
        CURRENT.repair[#CURRENT.repair + 1] = v
    end

    for k, _ in pairs(package.loaded) do
        if k:match("^module%.graph") or k:match("^module%.memory") or k == "module.tool" or k == "module.config" then
            package.loaded[k] = nil
        end
    end

    package.preload["module.config"] = function()
        return {
            settings = {
                graph = {
                    input_token_budget = 12000,
                    tool_loop_max = 5,
                    max_nodes_per_run = 128,
                    router = { max_tokens = 48, temperature = 0.0, seed = 7 },
                    planner = { max_tokens = 256, temperature = 0.1, seed = 11, max_calls_per_loop = 6 },
                    repair = { max_attempts = 2, max_tokens = 256, temperature = 0.0, seed = 29 },
                    responder = { max_tokens = 256, temperature = 0.0, seed = 42 },
                    recall = { enable_on_respond = false },
                    streaming = { token_chunk_chars = 16 },
                    tools = { file_context_max_chars = 4000 },
                    file_tools = {
                        list_default_limit = 12,
                        list_hard_limit = 64,
                        read_default_max_chars = 3000,
                        read_hard_max_chars = 12000,
                        read_lines_default_max_lines = 220,
                        read_lines_hard_max_lines = 1200,
                        search_default_max_hits = 20,
                        search_hard_max_hits = 200,
                        search_files_default_max_hits = 30,
                        search_files_hard_max_hits = 400,
                        search_files_default_max_files = 24,
                        search_files_hard_max_files = 200,
                        search_files_default_per_file_hits = 5,
                        search_files_hard_per_file_hits = 20,
                    },
                    providers = {
                        external = {
                            enabled = case.external_enabled == true,
                            provider = "qwen",
                            allowlist = case.external_allowlist or {},
                            context_inject = true,
                            context_max_chars = 800,
                        },
                    },
                    fact_extractor = {
                        verify_pass = false,
                        max_facts = 8,
                    },
                    cli = { debug_trace = false },
                },
                time = { maintenance_task = 9999999 },
            },
        }
    end

    package.preload["module.tool"] = function()
        return {
            get_embedding_query = function(_text)
                return { 0.1, 0.2, 0.3 }
            end,
            get_embedding_passage = function(_text)
                return { 0.3, 0.2, 0.1 }
            end,
        }
    end

    package.preload["module.memory.recall"] = function()
        return {
            check_and_retrieve = function(user_input, _vec, _opts)
                local s = tostring(user_input or "")
                if s:find("MEMORY_HIT", 1, true) then
                    return "memory_context_for_" .. s
                end
                return ""
            end,
        }
    end

    package.preload["module.memory.history"] = function()
        local turn = 0
        return {
            get_turn = function()
                return turn
            end,
            add_history = function(_user, _assistant)
                turn = turn + 1
            end,
        }
    end

    package.preload["module.memory.topic"] = function()
        return {
            add_turn = function(_turn, _user, _vec)
            end,
            update_assistant = function(_turn, _assistant)
            end,
            get_summary = function(turn)
                return "topic_summary_turn_" .. tostring(turn or 0)
            end,
        }
    end

    package.preload["module.graph.memory_core"] = function()
        return {
            extract_atomic_facts = function(_user, assistant)
                local text = tostring(assistant or "")
                if text == "" then
                    return {}
                end
                return { "fact_for_" .. tostring(case.id) }
            end,
            save_turn_memory = function(facts, _turn)
                return #(facts or {})
            end,
        }
    end

    _G.py_pipeline = {
        generate_chat_sync = function(_, messages, _params)
            local prompt = ""
            if type(messages) == "table" then
                local last = messages[#messages]
                if type(last) == "table" then
                    prompt = tostring(last.content or "")
                end
            end

            if prompt:find("strict routing classifier", 1, true) then
                if #CURRENT.router > 0 then
                    return table.remove(CURRENT.router, 1)
                end
                return '{route="respond"}'
            end

            if prompt:find("strict tool planner", 1, true) then
                if #CURRENT.planner > 0 then
                    return table.remove(CURRENT.planner, 1)
                end
                if type(CURRENT.planner_default) == "string" then
                    return CURRENT.planner_default
                end
                return "{tool_calls={}}"
            end

            if prompt:find("strict tool repair planner", 1, true) then
                if #CURRENT.repair > 0 then
                    return table.remove(CURRENT.repair, 1)
                end
                if type(CURRENT.repair_default) == "string" then
                    return CURRENT.repair_default
                end
                return "{tool_calls={}}"
            end

            return "FINAL_" .. tostring(case.id)
        end,

        count_chat_tokens = function(_, messages)
            local chars = 0
            for _, row in ipairs(messages or {}) do
                chars = chars + #(tostring((row or {}).content or ""))
            end
            return math.floor(chars / 4) + 1
        end,

        list_files = function(_, args_lua, _default_limit, _hard_limit)
            local args = parse_lua_args(args_lua)
            local prefix = tostring(args.prefix or "")
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "list_files"
            return "[list_files] prefix=" .. prefix
        end,

        read_file = function(_, args_lua, _default_chars, _hard_chars)
            local args = parse_lua_args(args_lua)
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "read_file"
            if tostring(args.path or "") == "" then
                return "read_file error: missing `path`"
            end
            return "[read_file] path=" .. tostring(args.path)
        end,

        read_lines = function(_, args_lua, _default_lines, _hard_lines)
            local args = parse_lua_args(args_lua)
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "read_lines"
            if tostring(args.path or "") == "" then
                return "read_lines error: missing `path`"
            end
            return "[read_lines] path=" .. tostring(args.path)
        end,

        search_file = function(_, args_lua, _default_hits, _hard_hits)
            local args = parse_lua_args(args_lua)
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "search_file"
            if tostring(args.path or "") == "" or tostring(args.pattern or "") == "" then
                return "search_file error: missing args"
            end
            return "[search_file] path=" .. tostring(args.path) .. " pattern=" .. tostring(args.pattern)
        end,

        search_files = function(_, args_lua, _a, _b, _c, _d, _e, _f)
            local args = parse_lua_args(args_lua)
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "search_files"
            if tostring(args.pattern or "") == "" then
                return "search_files error: missing `pattern`"
            end
            return "[search_files] pattern=" .. tostring(args.pattern)
        end,

        get_qwen_tool_schemas = function(_, allowlist)
            local out = {}
            local names = {}
            if type(allowlist) == "table" and #allowlist > 0 then
                for _, name in ipairs(allowlist) do
                    names[#names + 1] = tostring(name)
                end
            else
                names = { "web_search" }
            end
            for _, name in ipairs(names) do
                out[#out + 1] = {
                    type = "function",
                    ["function"] = {
                        name = name,
                        description = "stub tool",
                        parameters = { type = "object", properties = {} },
                    },
                }
            end
            return out
        end,

        call_qwen_tool = function(_, name, _args_lua)
            CURRENT.external_calls = CURRENT.external_calls + 1
            return "external_result_" .. tostring(name or "")
        end,
    }
end

local function cleanup_graph_dirs()
    os.execute('mkdir -p "memory/v3/graph"')
    os.execute('rm -rf "memory/v3/graph/checkpoints"')
    os.execute('rm -rf "memory/v3/graph/traces"')
end

local function ensure(condition, msg)
    if not condition then
        fail(msg)
    end
end

local function assert_event_order(events, case_id)
    ensure(#events > 0, case_id .. ": empty stream events")
    ensure((events[1] or {}).event == "run_start", case_id .. ": first event must be run_start")
    ensure((events[#events] or {}).event == "done", case_id .. ": last event must be done")

    local done_count = 0
    local node_start = 0
    local node_end = 0
    for _, evt in ipairs(events) do
        if evt.event == "done" then
            done_count = done_count + 1
        elseif evt.event == "node_start" then
            node_start = node_start + 1
        elseif evt.event == "node_end" then
            node_end = node_end + 1
        end
    end
    ensure(done_count == 1, case_id .. ": done must appear exactly once")
    ensure(node_start > 0 and node_end > 0 and node_start == node_end, case_id .. ": node events mismatch")
end

local function run_case(case)
    setup_stubs(case)
    local graph_runtime = require("module.graph.graph_runtime")

    local events = {}
    local conversation_history = case.conversation_history
    if type(conversation_history) ~= "table" or #conversation_history == 0 then
        conversation_history = {
            { role = "system", content = "SYSTEM_PROMPT" },
            { role = "user", content = "history_u1" },
            { role = "assistant", content = "history_a1" },
        }
    end

    local output = graph_runtime.run_turn({
        user_input = case.input,
        read_only = false,
        uploads = case.uploads or {},
        conversation_history = conversation_history,
        stream_sink = function(evt)
            events[#events + 1] = evt
        end,
    })

    local snapshot = _G.mori_last_state_snapshot or {}
    local trace = _G.mori_last_trace_summary or {}

    assert_event_order(events, case.id)
    ensure(type(output) == "string" and output ~= "", case.id .. ": empty output")
    ensure(type(trace.run_id) == "string" and trace.run_id ~= "", case.id .. ": missing run_id")

    if case.expect_route then
        ensure(tostring(snapshot.route or "") == case.expect_route, case.id .. ": unexpected route")
    end
    if case.expect_recall ~= nil then
        ensure((snapshot.recall_triggered == true) == (case.expect_recall == true), case.id .. ": recall mismatch")
    end
    if case.min_tool_executed then
        ensure((tonumber(snapshot.tool_executed) or 0) >= case.min_tool_executed, case.id .. ": tool_executed too small")
    end
    if case.expect_tool_executed ~= nil then
        ensure((tonumber(snapshot.tool_executed) or 0) == case.expect_tool_executed, case.id .. ": tool_executed mismatch")
    end
    if case.expect_tool_failed ~= nil then
        ensure((tonumber(snapshot.tool_failed) or 0) == case.expect_tool_failed, case.id .. ": tool_failed mismatch")
    end
    if case.min_tool_failed ~= nil then
        ensure((tonumber(snapshot.tool_failed) or 0) >= case.min_tool_failed, case.id .. ": tool_failed too small")
    end
    if case.expect_loops ~= nil then
        ensure((tonumber(trace.tool_loops) or 0) == case.expect_loops, case.id .. ": tool loop mismatch")
    end
    if case.expect_repair_attempts ~= nil then
        ensure((tonumber(trace.repair_attempts) or 0) == case.expect_repair_attempts, case.id .. ": repair attempts mismatch")
    end
    if case.expect_uploads_count ~= nil then
        ensure((tonumber(snapshot.uploads_count) or 0) == case.expect_uploads_count, case.id .. ": uploads count mismatch")
    end
    if case.expect_external_calls ~= nil then
        ensure((tonumber(CURRENT.external_calls) or 0) == case.expect_external_calls, case.id .. ": external call count mismatch")
    end
    if case.expect_writeback_saved ~= nil then
        ensure((tonumber(snapshot.writeback_saved) or 0) == case.expect_writeback_saved, case.id .. ": writeback_saved mismatch")
    end

    return {
        run_id = trace.run_id,
        trace = trace,
    }
end

local function build_cases()
    local cases = {}

    -- Memory 20
    for i = 1, 20 do
        cases[#cases + 1] = {
            id = string.format("memory_%02d", i),
            category = "memory",
            input = string.format("MEMORY_HIT case %d", i),
            router_steps = { '{route="tool_loop"}' },
            planner_steps = { "{tool_calls={}}" },
            expect_route = "tool_loop",
            expect_recall = true,
            expect_tool_executed = 0,
            expect_writeback_saved = 1,
            expect_uploads_count = 0,
        }
    end

    local file_tools = {
        mk_tool_call("list_files", { prefix = "download" }, "file_c1"),
        mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "file_c2"),
        mk_tool_call("read_lines", { path = "download/a.txt", start_line = 1, max_lines = 20 }, "file_c3"),
        mk_tool_call("search_file", { path = "download/a.txt", pattern = "todo" }, "file_c4"),
        mk_tool_call("search_files", { prefix = "download", pattern = "todo", max_hits = 4 }, "file_c5"),
    }

    -- File 20
    for i = 1, 20 do
        local tool_call = file_tools[((i - 1) % #file_tools) + 1]
        local planner_steps = {
            planner_literal({ tool_call }),
            "{tool_calls={}}",
        }
        local planner_default = nil
        local expect_loops = nil
        local input = string.format("file tool case %d", i)
        local router_steps = { '{route="tool_loop"}' }
        local conversation_history = nil
        if i == 1 then
            -- loop cap test: planner keeps emitting calls; runtime must stop at tool_loop_max=5
            planner_steps = {}
            planner_default = planner_literal({ tool_call })
            expect_loops = 5
        end
        if i == 20 then
            -- follow-up file reading test: router/planner must recover from history tool_path even without fresh uploads.
            input = "继续读取这个文件前几行，看看模型设置"
            router_steps = { '{route="respond"}' }
            planner_steps = { "{tool_calls={}}" }
            conversation_history = {
                { role = "system", content = "SYSTEM_PROMPT" },
                { role = "user", content = "我上传了文件" },
                {
                    role = "assistant",
                    content = "[上传文件已保存到 ./workspace/download]\n- main.py: ./workspace/download/mori_x_main.py (tool_path=download/mori_x_main.py, bytes=38505)",
                },
            }
        end
        cases[#cases + 1] = {
            id = string.format("file_%02d", i),
            category = "file",
            input = input,
            router_steps = router_steps,
            planner_steps = planner_steps,
            planner_default = planner_default,
            conversation_history = conversation_history,
            expect_route = "tool_loop",
            min_tool_executed = 1,
            expect_tool_failed = 0,
            expect_loops = expect_loops,
            expect_writeback_saved = 1,
            expect_uploads_count = 0,
        }
    end

    -- No-tool 10
    for i = 1, 10 do
        local has_uploads = (i == 1)
        local expect_route = "respond"
        local expect_tool_executed = 0
        local min_tool_executed = nil
        local expect_tool_failed = 0
        if has_uploads then
            expect_route = "tool_loop"
            expect_tool_executed = nil
            min_tool_executed = 1
            expect_tool_failed = nil
        end

        cases[#cases + 1] = {
            id = string.format("no_tool_%02d", i),
            category = "no_tool",
            input = string.format("plain response %d", i),
            router_steps = { '{route="respond"}' },
            expect_route = expect_route,
            expect_recall = false,
            expect_tool_executed = expect_tool_executed,
            min_tool_executed = min_tool_executed,
            expect_tool_failed = expect_tool_failed,
            expect_writeback_saved = 1,
            expect_uploads_count = has_uploads and 1 or 0,
            uploads = has_uploads and {
                { name = "a.txt", path = "./workspace/download/a.txt", bytes = 5 },
            } or {},
        }
    end

    -- External 10
    for i = 1, 10 do
        local external_enabled = true
        local external_allowlist = { "web_search" }
        local planner_steps = {
            planner_literal({
                mk_tool_call("web_search", { query = "q" .. tostring(i) }, "ext_c1"),
                mk_tool_call("web_search", { query = "q" .. tostring(i) .. "_2" }, "ext_c2"),
            }),
            "{tool_calls={}}",
        }
        local expect_external_calls = 2
        local expect_tool_failed = 0
        local min_tool_failed = nil
        local expect_repair_attempts = nil
        local expect_loops = nil
        local repair_steps = {}
        local min_tool_executed = 2

        if i == 2 then
            -- repair cap test: disallowed tool -> fail -> repair repeats -> max 2 repairs
            planner_steps = {
                planner_literal({ mk_tool_call("not_allowlisted", { query = "x" }, "ext_bad1") }),
            }
            repair_steps = {
                planner_literal({ mk_tool_call("not_allowlisted", { query = "x2" }, "ext_bad2") }),
                planner_literal({ mk_tool_call("not_allowlisted", { query = "x3" }, "ext_bad3") }),
            }
            expect_external_calls = 0
            expect_tool_failed = nil
            min_tool_failed = 1
            expect_repair_attempts = 2
            expect_loops = 3
            min_tool_executed = nil
        end

        cases[#cases + 1] = {
            id = string.format("external_%02d", i),
            category = "external",
            input = string.format("external tool case %d", i),
            router_steps = { '{route="tool_loop"}' },
            planner_steps = planner_steps,
            repair_steps = repair_steps,
            external_enabled = external_enabled,
            external_allowlist = external_allowlist,
            expect_route = "tool_loop",
            min_tool_executed = min_tool_executed,
            expect_tool_failed = expect_tool_failed,
            min_tool_failed = min_tool_failed,
            expect_external_calls = expect_external_calls,
            expect_repair_attempts = expect_repair_attempts,
            expect_loops = expect_loops,
            expect_writeback_saved = 1,
            expect_uploads_count = 0,
        }
    end

    return cases
end

local function run_consistency_check(sample_case)
    setup_stubs(sample_case)
    local graph_runtime = require("module.graph.graph_runtime")
    local out_a = graph_runtime.run_turn({
        user_input = sample_case.input,
        read_only = false,
        uploads = {},
        conversation_history = {
            { role = "system", content = "SYSTEM_PROMPT" },
        },
        stream_sink = nil,
    })

    setup_stubs(sample_case)
    graph_runtime = require("module.graph.graph_runtime")
    local events = {}
    local out_b = graph_runtime.run_turn({
        user_input = sample_case.input,
        read_only = false,
        uploads = {},
        conversation_history = {
            { role = "system", content = "SYSTEM_PROMPT" },
        },
        stream_sink = function(evt)
            events[#events + 1] = evt
        end,
    })
    ensure(out_a == out_b, "cli/webui consistency mismatch")
    assert_event_order(events, "consistency_sample")
end

local function run_artifact_check(run_id)
    local trace_path = string.format("memory/v3/graph/traces/%s.jsonl", tostring(run_id))
    local f = io.open(trace_path, "r")
    ensure(f ~= nil, "trace file missing: " .. trace_path)
    local text = f:read("*a") or ""
    f:close()
    ensure(text:find('"event":"done"', 1, true) ~= nil, "trace file missing done event")

    local checkpoint_store = require("module.graph.checkpoint_store")
    local loaded, err = checkpoint_store.load_last_checkpoint(run_id)
    ensure(loaded ~= nil, "checkpoint replay failed: " .. tostring(err))
    ensure(type((loaded or {}).snapshot) == "table", "checkpoint snapshot missing")
    ensure(tostring(((loaded or {}).snapshot or {}).run_id or "") == tostring(run_id), "checkpoint run_id mismatch")
end

local function main()
    cleanup_graph_dirs()
    math.randomseed(20260305)

    local cases = build_cases()
    ensure(#cases == 60, "expected exactly 60 cases")

    local expected_dist = {
        memory = 20,
        file = 20,
        no_tool = 10,
        external = 10,
    }
    local dist = {
        memory = 0,
        file = 0,
        no_tool = 0,
        external = 0,
    }
    for _, c in ipairs(cases) do
        dist[c.category] = (dist[c.category] or 0) + 1
    end
    for k, expected in pairs(expected_dist) do
        ensure((dist[k] or 0) == expected, string.format("case distribution mismatch: %s", k))
    end

    local passed = 0
    local first_run_id = nil

    for idx, case in ipairs(cases) do
        local ok, result_or_err = pcall(run_case, case)
        if ok then
            passed = passed + 1
            if (not first_run_id) and type(result_or_err) == "table" then
                first_run_id = result_or_err.run_id
            end
            printf("[PASS] (%d/%d) %s", idx, #cases, case.id)
        else
            printf("[FAIL] (%d/%d) %s -> %s", idx, #cases, case.id, tostring(result_or_err))
        end
    end

    ensure(first_run_id ~= nil, "no successful run to validate artifacts")
    run_artifact_check(first_run_id)

    run_consistency_check({
        id = "consistency",
        input = "MEMORY_HIT consistency check",
        router_steps = { '{route="tool_loop"}' },
        planner_steps = { "{tool_calls={}}" },
        external_enabled = false,
        external_allowlist = {},
    })

    local hit_rate = passed / #cases
    printf("\nGraph Tests Summary")
    printf("Total: %d", #cases)
    printf("Passed: %d", passed)
    printf("HitRate: %.2f%%", hit_rate * 100)
    printf("Distribution: memory=%d file=%d no_tool=%d external=%d",
        dist.memory, dist.file, dist.no_tool, dist.external)

    ensure(hit_rate >= 0.90, string.format("hit rate below threshold: %.4f", hit_rate))
    printf("All checks passed.")
end

main()
