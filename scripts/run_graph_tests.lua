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

local function mk_tool_call(name, args, id)
    return {
        name = name,
        args = args or {},
        id = id or ("call_" .. tostring(name)),
        type = "tool_call",
    }
end

local function mk_agent_step(content, tool_calls)
    return {
        content = tostring(content or ""),
        tool_calls = tool_calls or {},
    }
end

local CURRENT = nil

local function setup_stubs(case)
    CURRENT = {
        id = case.id,
        external_calls = 0,
        file_calls = {},
        agent_steps = {},
        agent_default_step = case.agent_default_step,
    }

    for _, step in ipairs(case.agent_steps or {}) do
        CURRENT.agent_steps[#CURRENT.agent_steps + 1] = {
            content = tostring(step.content or ""),
            tool_calls = step.tool_calls or {},
        }
    end

    for k, _ in pairs(package.loaded) do
        if k:match("^module%.graph")
            or k:match("^module%.memory")
            or k:match("^module%.experience")
            or k == "module.tool"
            or k == "module.config" then
            package.loaded[k] = nil
        end
    end

    package.preload["module.config"] = function()
        local settings = {
            graph = {
                input_token_budget = 12000,
                tool_loop_max = math.max(1, math.floor(tonumber(case.tool_loop_max) or 5)),
                max_nodes_per_run = 128,
                router = { max_tokens = 48, temperature = 0.0, seed = 7 },
                agent = {
                    max_tokens = 512,
                    temperature = 0.0,
                    seed = 42,
                    remaining_steps = math.max(1, math.floor(tonumber(case.remaining_steps) or 25)),
                    tool_choice = "auto",
                    parallel_tool_calls = true,
                },
                planner = { max_tokens = 256, temperature = 0.1, seed = 11, max_calls_per_loop = 6 },
                repair = { max_attempts = 2, max_tokens = 256, temperature = 0.0, seed = 29 },
                responder = { max_tokens = 256, temperature = 0.0, seed = 42 },
                recall = { enable_on_respond = true },
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
            experience = {
                storage = {
                    root = tostring(case.experience_root or ("memory/experience_policy_test/" .. tostring(case.id))),
                },
                retriever = {
                    fetch_multiplier = 8,
                    relevance_gate = 0.20,
                    context_threshold = 0.20,
                    semantic_threshold = 0.0,
                    semantic_scan_limit = 48,
                    min_needed_ratio = 0.34,
                    exploration_slots = 0,
                    utility_confidence_count = 2,
                },
                adaptive = {
                    enabled = true,
                    route_learning_rate = 0.10,
                    utility_learning_rate = 0.10,
                },
            },
        }

        local function get_by_path(root, path)
            local cur = root
            if type(path) ~= "string" or path == "" then
                return cur
            end
            for seg in path:gmatch("[^%.]+") do
                if type(cur) ~= "table" then
                    return nil
                end
                cur = cur[seg]
                if cur == nil then
                    return nil
                end
            end
            return cur
        end

        return {
            settings = settings,
            get = function(path, fallback)
                local value = get_by_path(settings, path)
                if value == nil then
                    return fallback
                end
                return value
            end,
        }
    end

    package.preload["module.tool"] = function()
        local ffi = require("ffi")

        local function file_exists(path)
            local f = io.open(path, "rb")
            if f then
                f:close()
                return true
            end
            return false
        end

        local function cosine_similarity(a, b)
            local dot = 0
            local norm_a = 0
            local norm_b = 0
            local len = math.max(#(a or {}), #(b or {}))
            for i = 1, len do
                local va = tonumber((a or {})[i]) or 0
                local vb = tonumber((b or {})[i]) or 0
                dot = dot + va * vb
                norm_a = norm_a + va * va
                norm_b = norm_b + vb * vb
            end
            if norm_a <= 0 or norm_b <= 0 then
                return 0.0
            end
            return dot / math.sqrt(norm_a * norm_b)
        end

        local function vector_to_bin(vec)
            if type(vec) ~= "table" then
                return ""
            end

            local n = #vec
            local elem_size = ffi.sizeof("float")
            local buf_size = 4 + n * elem_size
            local buf = ffi.new("uint8_t[?]", buf_size)

            ffi.cast("uint32_t*", buf)[0] = n

            local flt_arr = ffi.cast("float*", buf + 4)
            for i = 1, n do
                flt_arr[i - 1] = tonumber(vec[i]) or 0
            end

            return ffi.string(buf, buf_size)
        end

        local function bin_to_vector(bin, offset)
            offset = tonumber(offset) or 0
            if type(bin) ~= "string" or #bin < offset + 4 then
                return nil, 0
            end

            local p = ffi.cast("const uint8_t*", bin) + offset
            local n = ffi.cast("const uint32_t*", p)[0]
            local elem_size = ffi.sizeof("float")
            local expected_len = 4 + n * elem_size
            if #bin < offset + expected_len then
                return nil, 0
            end

            local flt_arr = ffi.cast("const float*", p + 4)
            local vec = {}
            for i = 0, n - 1 do
                vec[i + 1] = flt_arr[i]
            end
            return vec, expected_len
        end

        return {
            get_embedding_query = function(_text)
                return { 0.1, 0.2, 0.3 }
            end,
            get_embedding_passage = function(_text)
                return { 0.3, 0.2, 0.1 }
            end,
            cosine_similarity = cosine_similarity,
            file_exists = file_exists,
            vector_to_bin = vector_to_bin,
            bin_to_vector = bin_to_vector,
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
        generate_chat_with_tools_sync = function(_, _messages, _params, _tools, _tool_choice, _parallel_tool_calls)
            local step = nil
            if #CURRENT.agent_steps > 0 then
                step = table.remove(CURRENT.agent_steps, 1)
            elseif type(CURRENT.agent_default_step) == "table" then
                step = CURRENT.agent_default_step
            end
            if type(step) ~= "table" then
                return {
                    content = "FINAL_" .. tostring(case.id),
                    tool_calls = {
                        mk_tool_call("finish_turn", { message = "FINAL_" .. tostring(case.id), status = "completed" }, "finish_default"),
                    },
                }
            end
            local tool_calls = step.tool_calls or {}
            if #tool_calls == 0 and tostring(step.content or "") ~= "" then
                tool_calls = {
                    mk_tool_call("finish_turn", { message = tostring(step.content or ""), status = "completed" }, "finish_" .. tostring(case.id)),
                }
            end
            return {
                content = tostring(step.content or ""),
                tool_calls = tool_calls,
            }
        end,

        generate_chat_sync = function(_, _messages, _params)
            return "FALLBACK_" .. tostring(case.id)
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

        write_file = function(_, args_lua)
            local args = parse_lua_args(args_lua)
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "write_file"
            if tostring(args.path or "") == "" then
                return "write_file error: missing `path`"
            end
            return "[write_file] path=" .. tostring(args.path) .. " mode=" .. tostring(args.mode or "overwrite")
        end,

        apply_patch = function(_, args_lua)
            local args = parse_lua_args(args_lua)
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "apply_patch"
            if tostring(args.patch or "") == "" then
                return "apply_patch error: missing `patch`"
            end
            return "[apply_patch] ok"
        end,

        exec_command = function(_, args_lua)
            local args = parse_lua_args(args_lua)
            CURRENT.file_calls[#CURRENT.file_calls + 1] = "exec_command"
            local argv = args.argv or {}
            if type(argv) ~= "table" or #argv == 0 then
                return "exec_command error: missing `argv`"
            end
            return "[exec_command] argv=" .. table.concat(argv, " ")
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
    os.execute('rm -f "memory/v3/graph/session_state.lua"')
    os.execute('rm -f "memory/v3/graph/session_state.json"')
    os.execute('rm -rf "memory/experience_policy"')
    os.execute('rm -rf "memory/experience_policy_test"')
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

local function count_events(events, name)
    local n = 0
    for _, evt in ipairs(events or {}) do
        if evt.event == name then
            n = n + 1
        end
    end
    return n
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
        read_only = case.read_only == true,
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
    if case.expect_uploads_count ~= nil then
        ensure((tonumber(snapshot.uploads_count) or 0) == case.expect_uploads_count, case.id .. ": uploads count mismatch")
    end
    if case.expect_external_calls ~= nil then
        ensure((tonumber(CURRENT.external_calls) or 0) == case.expect_external_calls, case.id .. ": external call count mismatch")
    end
    if case.expect_writeback_saved ~= nil then
        ensure((tonumber(snapshot.writeback_saved) or 0) == case.expect_writeback_saved, case.id .. ": writeback_saved mismatch")
    end
    if case.expect_stop_reason ~= nil then
        ensure(tostring(snapshot.stop_reason or "") == tostring(case.expect_stop_reason), case.id .. ": stop_reason mismatch")
    end
    if case.expect_experience_retrieved ~= nil then
        ensure(#(snapshot.experience_retrieved_ids or {}) == tonumber(case.expect_experience_retrieved), case.id .. ": experience_retrieved mismatch")
    end
    if case.min_experience_retrieved ~= nil then
        ensure(#(snapshot.experience_retrieved_ids or {}) >= tonumber(case.min_experience_retrieved), case.id .. ": experience_retrieved too small")
    end
    if case.expect_experience_written ~= nil then
        ensure((snapshot.experience_written == true) == (case.expect_experience_written == true), case.id .. ": experience_written mismatch")
    end
    if case.experience_hints_contains ~= nil then
        ensure(tostring(snapshot.experience_hints or ""):find(tostring(case.experience_hints_contains), 1, true) ~= nil, case.id .. ": experience hints mismatch")
    end
    if case.final_contains ~= nil then
        ensure(tostring(output):find(tostring(case.final_contains), 1, true) ~= nil, case.id .. ": final text mismatch")
    end
    if case.min_tool_events ~= nil then
        local n = math.min(count_events(events, "tool_call"), count_events(events, "tool_result"))
        ensure(n >= case.min_tool_events, case.id .. ": tool events too small")
    end

    return {
        run_id = trace.run_id,
        trace = trace,
        snapshot = snapshot,
        output = output,
    }
end

local function count_experience_bins(root)
    local dir = string.format("%s/experiences", tostring(root))
    local proc = io.popen(string.format('ls -1A "%s" 2>/dev/null', dir))
    if not proc then
        return 0
    end
    local count = 0
    for line in proc:lines() do
        if line:sub(-4) == ".bin" then
            count = count + 1
        end
    end
    proc:close()
    return count
end

local function run_experience_policy_checks()
    local root_write = "memory/experience_policy_test/write"
    local write_case = {
        id = "exp_write_seed",
        category = "file",
        experience_root = root_write,
        input = "debug lua file policy",
        agent_steps = {
            mk_agent_step("need_file_seed", {
                mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "exp_seed_read"),
            }),
            mk_agent_step("seed_done", {}),
        },
        min_tool_executed = 1,
        expect_experience_written = true,
        expect_experience_retrieved = 0,
    }
    local write_result = run_case(write_case)
    ensure(count_experience_bins(root_write) == 1, "experience write seed count mismatch")

    local retrieve_case = {
        id = "exp_retrieve_hit",
        category = "no_tool",
        experience_root = root_write,
        input = "debug lua file policy",
        agent_steps = {
            mk_agent_step("retrieve_done", {}),
        },
        min_experience_retrieved = 1,
        expect_experience_written = true,
        experience_hints_contains = "task=coding",
    }
    local retrieve_result = run_case(retrieve_case)
    ensure(#(retrieve_result.snapshot.experience_retrieved_ids or {}) >= 1, "experience retrieval missing")

    local root_rank = "memory/experience_policy_test/rank"
    run_case({
        id = "exp_rank_fast",
        category = "file",
        experience_root = root_rank,
        input = "debug lua ranking path",
        agent_steps = {
            mk_agent_step("rank_fast_need", {
                mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "rank_fast_read"),
            }),
            mk_agent_step("rank_fast_done", {}),
        },
        min_tool_executed = 1,
        expect_experience_written = true,
    })
    run_case({
        id = "exp_rank_slow",
        category = "file",
        experience_root = root_rank,
        input = "debug lua ranking path",
        agent_steps = {
            mk_agent_step("rank_slow_need_a", {
                mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "rank_slow_read"),
            }),
            mk_agent_step("rank_slow_need_b", {
                mk_tool_call("list_files", { prefix = "download" }, "rank_slow_list"),
            }),
            mk_agent_step("rank_slow_done", {}),
        },
        min_tool_executed = 2,
        expect_experience_written = true,
    })
    local rank_result = run_case({
        id = "exp_rank_query",
        category = "no_tool",
        experience_root = root_rank,
        input = "debug lua ranking path",
        agent_steps = {
            mk_agent_step("rank_query_done", {}),
        },
        min_experience_retrieved = 2,
        experience_hints_contains = "task=coding",
    })
    ensure(tostring(rank_result.snapshot.experience_hints or ""):find("task=coding", 1, true) ~= nil,
        "experience ranking missing secondary path")

    local root_feedback = "memory/experience_policy_test/feedback"
    run_case({
        id = "exp_feedback_seed",
        category = "file",
        experience_root = root_feedback,
        input = "debug lua feedback path",
        agent_steps = {
            mk_agent_step("feedback_seed_need", {
                mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "feedback_seed_read"),
            }),
            mk_agent_step("feedback_seed_done", {}),
        },
        min_tool_executed = 1,
        expect_experience_written = true,
    })
    local fail_result = run_case({
        id = "exp_feedback_fail",
        category = "file",
        experience_root = root_feedback,
        input = "debug lua feedback path",
        remaining_steps = 1,
        agent_steps = {
            mk_agent_step("feedback_fail_need", {
                mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "feedback_fail_read"),
            }),
        },
        expect_stop_reason = "remaining_steps_exhausted",
        expect_experience_written = true,
        min_experience_retrieved = 1,
    })
    local fail_retrieved_id = ((fail_result.snapshot.experience_retrieved_ids or {})[1])
    ensure(type(fail_retrieved_id) == "string" and fail_retrieved_id ~= "", "feedback test missing retrieved id")
    local exp_adaptive = require("module.experience.adaptive")
    ensure(exp_adaptive.get_experience_utility(fail_retrieved_id) < 0.5, "negative feedback did not lower utility")

    local root_read_only = "memory/experience_policy_test/readonly"
    run_case({
        id = "exp_readonly_seed",
        category = "file",
        experience_root = root_read_only,
        input = "debug lua readonly path",
        agent_steps = {
            mk_agent_step("readonly_seed_need", {
                mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "readonly_seed_read"),
            }),
            mk_agent_step("readonly_seed_done", {}),
        },
        min_tool_executed = 1,
        expect_experience_written = true,
    })
    local before_read_only = count_experience_bins(root_read_only)
    local read_only_result = run_case({
        id = "exp_readonly_query",
        category = "no_tool",
        experience_root = root_read_only,
        input = "debug lua readonly path",
        read_only = true,
        agent_steps = {
            mk_agent_step("readonly_done", {}),
        },
        min_experience_retrieved = 1,
        expect_experience_written = false,
        experience_hints_contains = "task=coding",
    })
    ensure(count_experience_bins(root_read_only) == before_read_only, "read_only unexpectedly wrote experience")
    ensure(#(read_only_result.snapshot.experience_retrieved_ids or {}) >= 1, "read_only retrieval missing")

    local topic_src = assert(io.open("module/memory/topic.lua", "r"))
    local topic_text = topic_src:read("*a") or ""
    topic_src:close()
    ensure(topic_text:find("module.experience.bridge", 1, true) == nil, "topic bridge reference still present")
end

local function build_cases()
    local cases = {}

    -- Memory 20: recall only, no tools.
    for i = 1, 20 do
        cases[#cases + 1] = {
            id = string.format("memory_%02d", i),
            category = "memory",
            input = string.format("MEMORY_HIT case %d", i),
            agent_steps = {
                mk_agent_step("memory_answer_" .. tostring(i), {}),
            },
            expect_route = "respond",
            expect_recall = true,
            expect_tool_executed = 0,
            expect_tool_failed = 0,
            expect_writeback_saved = 1,
            expect_uploads_count = 0,
        }
    end

    local file_tool_calls = {
        mk_tool_call("list_files", { prefix = "download" }, "file_c1"),
        mk_tool_call("read_file", { path = "download/a.txt", max_chars = 80 }, "file_c2"),
        mk_tool_call("read_lines", { path = "download/a.txt", start_line = 1, max_lines = 20 }, "file_c3"),
        mk_tool_call("search_file", { path = "download/a.txt", pattern = "todo" }, "file_c4"),
        mk_tool_call("search_files", { prefix = "download", pattern = "todo", max_hits = 4 }, "file_c5"),
    }

    -- File 20
    for i = 1, 20 do
        local call = file_tool_calls[((i - 1) % #file_tool_calls) + 1]
        local case = {
            id = string.format("file_%02d", i),
            category = "file",
            input = string.format("file tool case %d", i),
            agent_steps = {
                mk_agent_step("need_file_tool_" .. tostring(i), { call }),
                mk_agent_step("file_done_" .. tostring(i), {}),
            },
            expect_route = "tool_loop",
            min_tool_executed = 1,
            expect_tool_failed = 0,
            expect_loops = 1,
            expect_writeback_saved = 1,
            expect_uploads_count = 0,
            min_tool_events = 1,
        }

        if i == 1 then
            case.agent_steps = {
                mk_agent_step("loop_forever_file", { call }),
            }
            case.agent_default_step = mk_agent_step("loop_forever_file", { call })
            case.expect_loops = 5
            case.min_tool_executed = 5
            case.expect_stop_reason = "tool_loop_max_exceeded"
            case.final_contains = "Sorry, need more steps"
        elseif i == 2 then
            case.remaining_steps = 1
            case.agent_steps = {
                mk_agent_step("need_but_no_steps", { call }),
            }
            case.expect_route = "tool_loop"
            case.expect_loops = 1
            case.expect_tool_executed = 1
            case.min_tool_executed = nil
            case.expect_stop_reason = "remaining_steps_exhausted"
            case.final_contains = "Sorry, need more steps"
            case.min_tool_events = 1
        elseif i == 20 then
            case.input = "读取我上传文件前几行"
            case.uploads = {
                { name = "a.txt", path = "./workspace/download/a.txt", tool_path = "download/a.txt", bytes = 5 },
            }
            case.expect_uploads_count = 1
            case.expect_loops = 1
        end

        cases[#cases + 1] = case
    end

    -- No-tool 10
    for i = 1, 10 do
        local case = {
            id = string.format("no_tool_%02d", i),
            category = "no_tool",
            input = string.format("plain response %d", i),
            agent_steps = {
                mk_agent_step("plain_done_" .. tostring(i), {}),
            },
            expect_route = "respond",
            expect_recall = false,
            expect_tool_executed = 0,
            expect_tool_failed = 0,
            expect_writeback_saved = 1,
            expect_uploads_count = 0,
        }

        if i == 1 then
            case.uploads = {
                { name = "a.txt", path = "./workspace/download/a.txt", tool_path = "download/a.txt", bytes = 5 },
            }
            case.agent_steps = {
                mk_agent_step("process_upload", { mk_tool_call("read_file", { path = "download/a.txt", max_chars = 40 }, "upload_read_1") }),
                mk_agent_step("upload_done", {}),
            }
            case.expect_route = "tool_loop"
            case.min_tool_executed = 1
            case.expect_tool_executed = nil
            case.expect_uploads_count = 1
            case.min_tool_events = 1
        end

        cases[#cases + 1] = case
    end

    -- External 10
    for i = 1, 10 do
        local case = {
            id = string.format("external_%02d", i),
            category = "external",
            input = string.format("external tool case %d", i),
            external_enabled = true,
            external_allowlist = { "web_search" },
            agent_steps = {
                mk_agent_step("need_external_" .. tostring(i), {
                    mk_tool_call("web_search", { query = "q" .. tostring(i) }, "ext_c1"),
                }),
                mk_agent_step("external_done_" .. tostring(i), {}),
            },
            expect_route = "tool_loop",
            min_tool_executed = 1,
            expect_tool_failed = 0,
            expect_external_calls = 1,
            expect_loops = 1,
            expect_writeback_saved = 1,
            expect_uploads_count = 0,
            min_tool_events = 1,
        }

        if i == 2 then
            case.agent_steps = {
                mk_agent_step("bad_external", {
                    mk_tool_call("not_allowlisted", { query = "x" }, "ext_bad1"),
                }),
                mk_agent_step("retry_external", {
                    mk_tool_call("web_search", { query = "x_fixed" }, "ext_good2"),
                }),
                mk_agent_step("done_after_retry", {}),
            }
            case.expect_external_calls = 1
            case.min_tool_failed = 1
            case.expect_tool_failed = nil
            case.expect_loops = 2
        elseif i == 3 then
            case.agent_steps = {
                mk_agent_step("parallel_two_external", {
                    mk_tool_call("web_search", { query = "a" }, "ext_p1"),
                    mk_tool_call("web_search", { query = "b" }, "ext_p2"),
                }),
                mk_agent_step("external_parallel_done", {}),
            }
            case.expect_external_calls = 2
            case.min_tool_executed = 2
            case.expect_loops = 1
            case.min_tool_events = 2
        end

        cases[#cases + 1] = case
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

local function run_v2_feature_checks()
    local contract_result = run_case({
        id = "v2_contract_repair",
        category = "no_tool",
        input = "contract repair check",
        agent_steps = {
            mk_agent_step("", {}),
            mk_agent_step("contract_repaired", {}),
        },
        expect_route = "respond",
        expect_tool_executed = 0,
        final_contains = "contract_repaired",
    })
    ensure((tonumber(contract_result.trace.repair_attempts) or 0) >= 1, "v2 contract repair missing repair attempt")

    local write_exec_result = run_case({
        id = "v2_write_exec",
        category = "file",
        input = "write and execute in workspace",
        agent_steps = {
            mk_agent_step("write_step", {
                mk_tool_call("write_file", {
                    path = "/mori/workspace/tmp.txt",
                    content = "hello",
                    mode = "overwrite",
                }, "v2_write"),
            }),
            mk_agent_step("exec_step", {
                mk_tool_call("exec_command", {
                    argv = { "pwd" },
                    workdir = "/mori/workspace",
                }, "v2_exec"),
            }),
            mk_agent_step("write_exec_done", {}),
        },
        expect_route = "tool_loop",
        min_tool_executed = 2,
        expect_loops = 2,
        min_tool_events = 2,
        final_contains = "write_exec_done",
    })
    local file_call_text = table.concat(CURRENT.file_calls or {}, ",")
    ensure(file_call_text:find("write_file", 1, true) ~= nil, "v2 write tool not executed")
    ensure(file_call_text:find("exec_command", 1, true) ~= nil, "v2 exec tool not executed")
    ensure((tonumber(write_exec_result.trace.tool_loops) or 0) == 2, "v2 write/exec loop count mismatch")

    local mixed_result = run_case({
        id = "v2_mixed_terminal",
        category = "file",
        input = "write file in workspace mixed terminal batch check",
        agent_steps = {
            mk_agent_step("bad_mixed", {
                mk_tool_call("write_file", {
                    path = "/mori/workspace/mixed.txt",
                    content = "bad",
                }, "v2_mixed_write"),
                mk_tool_call("finish_turn", {
                    message = "bad mixed batch",
                    status = "completed",
                }, "v2_mixed_finish"),
            }),
            mk_agent_step("recovered_write", {
                mk_tool_call("write_file", {
                    path = "/mori/workspace/recovered.txt",
                    content = "good",
                }, "v2_recovered_write"),
            }),
            mk_agent_step("mixed_done", {}),
        },
        min_tool_executed = 1,
        expect_loops = 1,
        final_contains = "mixed_done",
    })
    local write_count = 0
    for _, name in ipairs(CURRENT.file_calls or {}) do
        if name == "write_file" then
            write_count = write_count + 1
        end
    end
    ensure(write_count == 1, "mixed terminal batch should not execute the invalid write")
    ensure((tonumber(mixed_result.trace.repair_attempts) or 0) >= 1, "mixed terminal batch did not enter repair")

    local util = require("module.graph.util")
    ensure(util.normalize_tool_path("/mori/workspace/demo.txt") == "/mori/workspace/demo.txt", "workspace path normalization mismatch")
    ensure(util.normalize_tool_path("../escape.txt") == "", "parent traversal should be rejected")
    ensure(util.normalize_tool_path("/tmp/outside.txt") == "", "absolute physical path should be rejected")
    ensure(util.normalize_tool_path("workspace/legacy.txt") == "", "legacy workspace alias should be rejected")

    setup_stubs({
        id = "v2_resume",
        category = "file",
        input = "继续",
        agent_steps = {
            mk_agent_step("resume_completed", {}),
        },
    })
    local checkpoint_store = require("module.graph.checkpoint_store")
    local session_store = require("module.graph.session_store")
    local state_schema = require("module.graph.state_schema")
    local graph_runtime = require("module.graph.graph_runtime")

    local resume_run_id = "resume_fixture"
    local resume_state = state_schema.new_state({
        run_id = resume_run_id,
        user_input = "resume original task",
        read_only = false,
        uploads = {},
        conversation_history = {
            { role = "system", content = "SYSTEM_PROMPT" },
        },
        system_prompt = "SYSTEM_PROMPT",
        active_task = {
            task_id = "task_resume_fixture",
            goal = "resume original task",
            status = "open",
            carryover_summary = "pending resume",
            last_user_message = "resume original task",
            profile = "workspace",
        },
        working_memory = {
            current_plan = "resume pending planner",
            plan_step_index = 1,
            files_read_set = {},
            files_written_set = {},
            patches_applied = {},
            command_history_tail = {},
            last_tool_batch_summary = "",
            last_repair_error = "",
        },
        task_profile = "workspace",
    })
    resume_state.context.task_profile = "workspace"
    resume_state.session.active_task.profile = "workspace"
    local checkpoint_ok, checkpoint_err = checkpoint_store.save_checkpoint(
        resume_run_id,
        1,
        "context_node",
        resume_state,
        "planner_node"
    )
    ensure(checkpoint_ok == true, "resume checkpoint save failed: " .. tostring(checkpoint_err))
    local session_ok, session_err = session_store.save({
        last_run_id = resume_run_id,
        active_task = resume_state.session.active_task,
        working_memory = resume_state.working_memory,
        recovery = {
            resumable_run_id = resume_run_id,
            last_checkpoint_seq = 1,
            next_node = "planner_node",
            resumed_from_checkpoint = false,
        },
        stats = {
            files_read_count = 0,
            files_written_count = 0,
        },
    })
    ensure(session_ok == true, "resume session save failed: " .. tostring(session_err))

    local resumed_output = graph_runtime.run_turn({
        user_input = "继续",
        read_only = false,
        uploads = {},
        conversation_history = {
            { role = "system", content = "SYSTEM_PROMPT" },
        },
        stream_sink = nil,
    })
    local meta = _G.mori_last_response_meta or {}
    local session_snapshot = _G.mori_session_status or {}
    ensure(meta.resumed == true, "resume metadata missing resumed=true")
    ensure(tostring(resumed_output):find("resume_completed", 1, true) ~= nil, "resume output mismatch")
    ensure(tostring((((session_snapshot or {}).recovery or {}).resumable_run_id) or "") == "", "resume recovery should be cleared after success")
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
    run_experience_policy_checks()
    run_v2_feature_checks()

    run_consistency_check({
        id = "consistency",
        input = "MEMORY_HIT consistency check",
        agent_steps = {
            mk_agent_step("consistency_done", {}),
        },
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
