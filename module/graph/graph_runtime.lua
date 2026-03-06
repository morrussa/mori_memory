local state_schema = require("module.graph.state_schema")
local graph_builder = require("module.graph.graph_builder")
local checkpoint_store = require("module.graph.checkpoint_store")
local session_store = require("module.graph.session_store")
local trace_writer = require("module.graph.trace_writer")
local conversation_source = require("module.graph.conversation_source")
local util = require("module.graph.util")
local command = require("module.graph.command")
local config = require("module.config")

local M = {}

local _global_checkpoint_seq = 0

local function assert_luajit_only()
    if type(jit) ~= "table" then
        error("[GraphRuntime] LuaJIT is required")
    end
end

local function is_callable(obj)
    if type(obj) == "function" then
        return true
    end
    local ok, _ = pcall(function()
        return obj and obj.__call or (getmetatable(obj) or {}).__call
    end)
    if ok then
        return true
    end
    if obj ~= nil then
        local ok2, _ = pcall(function()
            obj({})
        end)
        return ok2
    end
    return false
end

local function emit_stream(stream_sink, event_name, payload)
    if not is_callable(stream_sink) then
        return
    end
    local ok, err = pcall(stream_sink, {
        event = event_name,
        data = payload or {},
    })
    if not ok then
        print(string.format("[GraphRuntime][WARN] stream emit failed: %s", tostring(err)))
    end
end

local function trace(run_id, event_name, payload)
    local ok, err = trace_writer.append(run_id, event_name, payload)
    if not ok then
        print(string.format("[GraphRuntime][WARN] trace append failed: %s", tostring(err)))
    end
end

local function checkpoint(state, node_name, next_node)
    state.checkpoint_meta = state.checkpoint_meta or { seq = 0, last_node = "" }
    _global_checkpoint_seq = _global_checkpoint_seq + 1
    local seq = _global_checkpoint_seq
    state.checkpoint_meta.seq = seq
    state.checkpoint_meta.last_node = node_name
    state.recovery = state.recovery or {}
    state.recovery.resumable_run_id = tostring(state.run_id or "")
    state.recovery.last_checkpoint_seq = seq
    state.recovery.next_node = tostring(next_node or "")

    checkpoint_store.mark_dirty(state.run_id, seq, node_name, next_node, state, nil)
end

local function summarize_node(state, node_name)
    return {
        node = node_name,
        route = (((state or {}).router_decision or {}).route) or nil,
        planner_calls = #((((state or {}).planner or {}).tool_calls) or {}),
        tool_executed = tonumber((((state or {}).tool_exec or {}).executed) or 0) or 0,
        tool_failed = tonumber((((state or {}).tool_exec or {}).failed) or 0) or 0,
        tool_executed_total = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0,
        tool_failed_total = tonumber((((state or {}).tool_exec or {}).failed_total) or 0) or 0,
        agent_iteration = tonumber((((state or {}).agent_loop or {}).iteration) or 0) or 0,
        remaining_steps = tonumber((((state or {}).agent_loop or {}).remaining_steps) or 0) or 0,
        task_status = tostring((((((state or {}).session or {}).active_task) or {}).status) or ""),
    }
end

local function chunk_text(text, chunk_chars)
    local s = tostring(text or "")
    local n = math.max(1, math.floor(tonumber(chunk_chars) or 32))
    local out = {}
    local buf = {}
    local count = 0
    for ch in s:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        buf[#buf + 1] = ch
        count = count + 1
        if count >= n then
            out[#out + 1] = table.concat(buf)
            buf = {}
            count = 0
        end
    end
    if #buf > 0 then
        out[#out + 1] = table.concat(buf)
    end
    return out
end

local function build_trace_summary(state)
    local metrics = (state or {}).metrics or {}
    local nodes = {}
    for name, duration in pairs(metrics.node_durations_ms or {}) do
        nodes[#nodes + 1] = {
            node = name,
            duration_ms = duration,
        }
    end
    table.sort(nodes, function(a, b)
        return tostring(a.node) < tostring(b.node)
    end)

    return {
        run_id = state.run_id,
        node_count = #nodes,
        nodes = nodes,
        tool_loops = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0,
        tool_executed = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0,
        tool_failed = tonumber((((state or {}).tool_exec or {}).failed_total) or 0) or 0,
        repair_attempts = tonumber((((state or {}).repair or {}).attempts) or 0) or 0,
        agent_iteration = tonumber((((state or {}).agent_loop or {}).iteration) or 0) or 0,
        remaining_steps = tonumber((((state or {}).agent_loop or {}).remaining_steps) or 0) or 0,
        stop_reason = tostring((((state or {}).termination or {}).stop_reason) or (((state or {}).agent_loop or {}).stop_reason) or ""),
    }
end

local function ensure_v2_shape(state, args, conversation_history, base_system_prompt)
    state.state_version = state_schema.STATE_VERSION
    state.input = state.input or {}
    state.input.message = tostring(state.input.message or args.user_input or "")
    state.input.read_only = args.read_only == true
    state.uploads = state.uploads or (args.uploads or {})

    state.messages = state.messages or {}
    state.messages.conversation_history = conversation_history or state.messages.conversation_history or {}
    state.messages.system_prompt = util.trim(base_system_prompt or state.messages.system_prompt or "")
    state.messages.runtime_messages = state.messages.runtime_messages or {}

    state.agent_loop = state.agent_loop or {
        remaining_steps = 25,
        pending_tool_calls = {},
        stop_reason = "",
        iteration = 0,
    }
    state.context = state.context or {
        memory_context = "",
        experience_context = "",
        tool_context = "",
        planner_context = "",
    }
    state.router_decision = state.router_decision or { route = "respond", raw = "", reason = "" }
    state.recall = state.recall or { triggered = false, context = "", score = nil }
    state.experience = state.experience or {
        query = {},
        retrieved = { items = {}, ids = {}, strategy = "" },
        hints = "",
        feedback = { effective_ids = {} },
        writeback = { written = false },
    }
    state.planner = state.planner or { raw = "", tool_calls = {}, errors = {}, force_reason = "", missing_terminal_signal = false }
    state.tool_exec = state.tool_exec or {
        loop_count = 0,
        executed = 0,
        failed = 0,
        executed_total = 0,
        failed_total = 0,
        results = {},
        all_results = {},
        context_fragments = {},
        read_files = {},
    }
    state.repair = state.repair or { attempts = 0, max_attempts = 2, last_error = "", retry_requested = false, pending = false }
    state.final_response = state.final_response or { message = "" }
    state.writeback = state.writeback or { facts = {}, saved = 0 }
    state.metrics = state.metrics or { started_at_ms = util.now_ms(), finished_at_ms = nil, node_durations_ms = {} }
    state.checkpoint_meta = state.checkpoint_meta or { seq = 0, last_node = "" }
    state.session = state.session or { mode = "single", active_task = {} }
    state.session.active_task = state.session.active_task or {}
    state.working_memory = state.working_memory or {
        current_plan = "",
        plan_step_index = 0,
        files_read_set = {},
        files_written_set = {},
        patches_applied = {},
        command_history_tail = {},
        last_tool_batch_summary = "",
        last_repair_error = "",
    }
    state.termination = state.termination or { finish_requested = false, final_message = "", final_status = "", stop_reason = "" }
    state.recovery = state.recovery or { resumable_run_id = "", last_checkpoint_seq = 0, next_node = "", resumed_from_checkpoint = false }
    state.stream_sink = args.stream_sink
end

local function build_new_state(args, conversation_history, base_system_prompt, session_state)
    local user_input = util.trim(args.user_input or "")
    local continue_request = util.is_continue_request(user_input)
    local carry_active_task = continue_request and type((session_state or {}).active_task) == "table" and util.trim(((session_state or {}).active_task or {}).goal or "") ~= ""
    local active_task = carry_active_task and (session_state.active_task or {}) or {
        task_id = "",
        goal = user_input,
        status = "open",
        carryover_summary = "",
        last_user_message = user_input,
        profile = "",
    }
    active_task.last_user_message = user_input
    if util.trim(active_task.goal or "") == "" then
        active_task.goal = user_input
    end
    if util.trim(active_task.status or "") == "" then
        active_task.status = "open"
    end

    local working_memory = carry_active_task and ((session_state or {}).working_memory or {}) or nil
    local state = state_schema.new_state({
        user_input = user_input,
        read_only = args.read_only == true,
        uploads = args.uploads or {},
        conversation_history = conversation_history,
        system_prompt = base_system_prompt,
        stream_sink = args.stream_sink,
        active_task = active_task,
        working_memory = working_memory,
        task_profile = util.trim(active_task.profile or ""),
    })
    return state, false, ""
end

local function maybe_resume_state(args, conversation_history, base_system_prompt, session_state)
    local recovery = ((session_state or {}).recovery) or {}
    local resumable_run_id = util.trim(recovery.resumable_run_id or "")
    if resumable_run_id == "" then
        return nil, false, "", "no_resumable_run"
    end

    if not util.is_continue_request(args.user_input or "") then
        return nil, false, "", "resume_not_requested"
    end

    local checkpoint, load_err = checkpoint_store.load_last_checkpoint(resumable_run_id)
    if not checkpoint then
        return nil, false, "", tostring(load_err or "checkpoint_load_failed")
    end
    if type(checkpoint.full_state) ~= "table" then
        return nil, false, "", "checkpoint_missing_full_state"
    end

    local state = checkpoint.full_state
    ensure_v2_shape(state, args, conversation_history, base_system_prompt)
    state.recovery.resumable_run_id = resumable_run_id
    state.recovery.last_checkpoint_seq = tonumber(checkpoint.seq) or 0
    state.recovery.next_node = util.trim(checkpoint.next_node or "")
    state.recovery.resumed_from_checkpoint = true
    state.session.active_task = state.session.active_task or {}
    state.session.active_task.last_user_message = tostring(args.user_input or "")
    if util.trim(state.session.active_task.status or "") == "" then
        state.session.active_task.status = "open"
    end

    local next_node = util.trim(checkpoint.next_node or "")
    if next_node == "" or next_node == "end" then
        next_node = "planner_node"
    end
    return state, true, next_node, ""
end

local function persist_session_snapshot(state, trace_summary)
    local snapshot = session_store.build_from_state(state, trace_summary)
    local ok, err = session_store.save(snapshot)
    if not ok then
        print(string.format("[GraphRuntime][WARN] session save failed: %s", tostring(err)))
    end
    _G.mori_session_status = snapshot
    return snapshot
end

function M.run_turn(args)
    assert_luajit_only()
    args = args or {}

    local user_input = util.trim(args.user_input or "")
    if user_input == "" then
        return ""
    end

    local cfg = config.get("graph", {})
    local stream_sink = args.stream_sink
    local conversation_history, base_system_prompt = conversation_source.resolve_conversation(
        args.conversation_history,
        args.system_prompt
    )

    local session_state = session_store.load()
    local state, resumed, current, resume_err = maybe_resume_state(args, conversation_history, base_system_prompt, session_state)
    if not state then
        state = build_new_state(args, conversation_history, base_system_prompt, session_state)
        resumed = false
        current = "ingest_node"
        if resume_err ~= "" and resume_err ~= "no_resumable_run" and resume_err ~= "resume_not_requested" then
            state.context = state.context or {}
            state.context.planner_context = "Previous resumable run could not be restored: " .. tostring(resume_err)
        end
    end

    ensure_v2_shape(state, args, conversation_history, base_system_prompt)
    state.repair.max_attempts = math.max(0, math.floor(tonumber((cfg.repair or {}).max_attempts) or tonumber(state.repair.max_attempts) or 2))
    local remaining_steps = nil
    if resumed then
        remaining_steps = tonumber(state.agent_loop.remaining_steps)
    else
        remaining_steps = tonumber(((cfg.agent or {}).remaining_steps) or state.agent_loop.remaining_steps or 25)
    end
    if remaining_steps == nil then
        remaining_steps = 25
    end
    state.agent_loop.remaining_steps = math.max(0, math.floor(remaining_steps))

    state_schema.assert_valid(state)
    local graph = graph_builder.build()
    if util.trim(current or "") == "" then
        current = graph.start_node
    end

    emit_stream(stream_sink, "run_start", {
        run_id = state.run_id,
        state_version = state.state_version,
        resumed = resumed,
    })
    emit_stream(stream_sink, "status", {
        run_id = state.run_id,
        phase = "running",
        resumed = resumed,
    })
    trace(state.run_id, "run_start", {
        state_version = state.state_version,
        resumed = resumed,
    })

    local guard = 0
    local max_nodes = math.max(20, math.floor(tonumber(cfg.max_nodes_per_run) or 128))

    while current and current ~= "end" do
        guard = guard + 1
        if guard > max_nodes then
            error(string.format("[GraphRuntime] node guard exceeded: %d", max_nodes))
        end

        local node = graph.nodes[current]
        if type(node) ~= "table" or type(node.run) ~= "function" then
            error(string.format("[GraphRuntime] node missing: %s", tostring(current)))
        end

        local node_start_ms = util.now_ms()
        emit_stream(stream_sink, "node_start", {
            run_id = state.run_id,
            node = current,
            seq = guard,
            resumed = resumed,
        })
        trace(state.run_id, "node_start", {
            node = current,
            seq = guard,
        })

        local node_result = node.run(state, {})
        local next_node_name, updates = graph.resolve_next(current, state, node_result)
        if updates then
            command.apply_update(state, updates)
        end

        state_schema.assert_valid(state)
        local node_end_ms = util.now_ms()
        state.metrics.node_durations_ms[current] = math.max(0, node_end_ms - node_start_ms)

        checkpoint(state, current, next_node_name)
        local trace_summary = build_trace_summary(state)
        persist_session_snapshot(state, trace_summary)

        local summary = summarize_node(state, current)
        summary.duration_ms = state.metrics.node_durations_ms[current]
        emit_stream(stream_sink, "node_end", summary)
        trace(state.run_id, "node_end", summary)

        if current == "tool_exec_node" then
            for _, row in ipairs((((state or {}).tool_exec or {}).results) or {}) do
                emit_stream(stream_sink, "tool_call", {
                    run_id = state.run_id,
                    call_id = row.call_id,
                    tool = row.tool,
                })
                emit_stream(stream_sink, "tool_result", {
                    run_id = state.run_id,
                    call_id = row.call_id,
                    tool = row.tool,
                    ok = row.ok == true,
                    result = row.result,
                    error = row.error,
                })
            end
        end

        current = next_node_name
    end

    state.metrics.finished_at_ms = util.now_ms()
    local final_text = tostring((((state or {}).final_response or {}).message) or "")

    if not state._streaming_sent and final_text ~= "" then
        local stream_chunk_chars = math.max(1, math.floor(tonumber((cfg.streaming or {}).token_chunk_chars) or 24))
        for _, chunk in ipairs(chunk_text(final_text, stream_chunk_chars)) do
            emit_stream(stream_sink, "token", {
                run_id = state.run_id,
                token = chunk,
            })
        end
    end

    local trace_summary = build_trace_summary(state)
    local checkpoint_ok, checkpoint_err = checkpoint_store.flush(true)
    if not checkpoint_ok then
        print(string.format("[GraphRuntime][WARN] checkpoint flush failed: %s", tostring(checkpoint_err)))
    end

    state.recovery.resumable_run_id = ""
    state.recovery.last_checkpoint_seq = 0
    state.recovery.next_node = ""

    local session_snapshot = persist_session_snapshot(state, trace_summary)
    local task_status = tostring((((((state or {}).session or {}).active_task) or {}).status) or (((state or {}).termination or {}).final_status) or "")

    _G.mori_last_run_id = state.run_id
    _G.mori_last_trace_summary = trace_summary
    _G.mori_last_response_meta = {
        task_status = task_status,
        resumed = resumed == true,
        final_status = tostring((((state or {}).termination or {}).final_status) or ""),
        stop_reason = tostring((((state or {}).termination or {}).stop_reason) or ""),
    }
    _G.mori_last_state_snapshot = {
        run_id = state.run_id,
        uploads_count = #((state.uploads) or {}),
        route = (((state or {}).router_decision or {}).route) or "",
        recall_triggered = (((state or {}).recall or {}).triggered) == true,
        experience_retrieved_ids = (((((state or {}).experience or {}).retrieved) or {}).ids) or {},
        experience_hints = tostring((((state or {}).experience or {}).hints) or ""),
        experience_written = (((((state or {}).experience or {}).writeback) or {}).written) == true,
        planner_calls = #((((state or {}).planner or {}).tool_calls) or {}),
        tool_results = (((state or {}).tool_exec or {}).results) or {},
        tool_executed = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0,
        tool_failed = tonumber((((state or {}).tool_exec or {}).failed_total) or 0) or 0,
        stop_reason = tostring((((state or {}).termination or {}).stop_reason) or ""),
        remaining_steps = tonumber((((state or {}).agent_loop or {}).remaining_steps) or 0) or 0,
        writeback_facts = (((state or {}).writeback or {}).facts) or {},
        writeback_saved = tonumber((((state or {}).writeback or {}).saved) or 0) or 0,
        session_status = session_snapshot,
    }

    emit_stream(stream_sink, "status", {
        run_id = state.run_id,
        phase = "completed",
        resumed = resumed,
        task_status = task_status,
    })
    emit_stream(stream_sink, "done", {
        run_id = state.run_id,
        message = final_text,
        trace = trace_summary,
        resumed = resumed,
        task_status = task_status,
    })
    trace(state.run_id, "done", {
        message_chars = #final_text,
        trace = trace_summary,
        resumed = resumed,
        task_status = task_status,
    })

    return final_text
end

return M
