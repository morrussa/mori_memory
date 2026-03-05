local state_schema = require("module.graph.state_schema")
local graph_builder = require("module.graph.graph_builder")
local checkpoint_store = require("module.graph.checkpoint_store")
local trace_writer = require("module.graph.trace_writer")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function assert_luajit_only()
    if type(jit) ~= "table" then
        error("[GraphRuntime] LuaJIT is required")
    end
end

local function emit_stream(stream_sink, event_name, payload)
    if type(stream_sink) ~= "function" then
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

local function checkpoint(state, node_name)
    state.checkpoint_meta = state.checkpoint_meta or { seq = 0, last_node = "" }
    local seq = (tonumber(state.checkpoint_meta.seq) or 0) + 1
    state.checkpoint_meta.seq = seq
    state.checkpoint_meta.last_node = node_name

    local ok, err = checkpoint_store.save_checkpoint(state.run_id, seq, node_name, state)
    if not ok then
        print(string.format("[GraphRuntime][WARN] checkpoint failed: %s", tostring(err)))
    end
end

local function summarize_node(state, node_name)
    local summary = {
        node = node_name,
        route = (((state or {}).router_decision or {}).route) or nil,
        planner_calls = #((((state or {}).planner or {}).tool_calls) or {}),
        tool_executed = tonumber((((state or {}).tool_exec or {}).executed) or 0) or 0,
        tool_failed = tonumber((((state or {}).tool_exec or {}).failed) or 0) or 0,
        tool_executed_total = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0,
        tool_failed_total = tonumber((((state or {}).tool_exec or {}).failed_total) or 0) or 0,
        repair_attempts = tonumber((((state or {}).repair or {}).attempts) or 0) or 0,
    }
    return summary
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
    }
end

function M.run_turn(args)
    assert_luajit_only()
    args = args or {}

    local user_input = util.trim(args.user_input or "")
    if user_input == "" then
        return ""
    end

    local cfg = graph_cfg()
    local stream_sink = args.stream_sink
    local conversation_history = args.conversation_history or {}
    local base_system_prompt = ""
    if type(conversation_history[1]) == "table" and tostring(conversation_history[1].role or "") == "system" then
        base_system_prompt = tostring(conversation_history[1].content or "")
    end

    local state = state_schema.new_state({
        user_input = user_input,
        read_only = args.read_only == true,
        uploads = args.uploads or {},
        conversation_history = conversation_history,
        system_prompt = base_system_prompt,
    })
    state.repair.max_attempts = math.max(0, math.floor(tonumber((cfg.repair or {}).max_attempts) or 2))

    state_schema.assert_valid(state)
    local graph = graph_builder.build()

    emit_stream(stream_sink, "run_start", {
        run_id = state.run_id,
        state_version = state.state_version,
    })
    emit_stream(stream_sink, "status", {
        run_id = state.run_id,
        phase = "running",
    })
    trace(state.run_id, "run_start", {
        state_version = state.state_version,
    })

    local current = graph.start_node
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
        })
        trace(state.run_id, "node_start", {
            node = current,
            seq = guard,
        })

        state = node.run(state, {
            add_to_history = args.add_to_history,
        })
        state_schema.assert_valid(state)

        local node_end_ms = util.now_ms()
        local duration = math.max(0, node_end_ms - node_start_ms)
        state.metrics.node_durations_ms[current] = duration

        checkpoint(state, current)

        local summary = summarize_node(state, current)
        summary.duration_ms = duration
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

        current = graph.next_node(current, state)
    end

    state.metrics.finished_at_ms = util.now_ms()
    local final_text = tostring((((state or {}).final_response or {}).message) or "")

    local stream_chunk_chars = math.max(1, math.floor(tonumber((cfg.streaming or {}).token_chunk_chars) or 24))
    for _, chunk in ipairs(chunk_text(final_text, stream_chunk_chars)) do
        emit_stream(stream_sink, "token", {
            run_id = state.run_id,
            token = chunk,
        })
    end

    local trace_summary = build_trace_summary(state)
    _G.mori_last_run_id = state.run_id
    _G.mori_last_trace_summary = trace_summary
    _G.mori_last_state_snapshot = {
        run_id = state.run_id,
        uploads_count = #((state.uploads) or {}),
        route = (((state or {}).router_decision or {}).route) or "",
        recall_triggered = (((state or {}).recall or {}).triggered) == true,
        planner_calls = #((((state or {}).planner or {}).tool_calls) or {}),
        tool_results = (((state or {}).tool_exec or {}).results) or {},
        tool_executed = tonumber((((state or {}).tool_exec or {}).executed_total) or 0) or 0,
        tool_failed = tonumber((((state or {}).tool_exec or {}).failed_total) or 0) or 0,
        writeback_facts = (((state or {}).writeback or {}).facts) or {},
        writeback_saved = tonumber((((state or {}).writeback or {}).saved) or 0) or 0,
    }
    emit_stream(stream_sink, "status", {
        run_id = state.run_id,
        phase = "completed",
    })
    emit_stream(stream_sink, "done", {
        run_id = state.run_id,
        message = final_text,
        trace = trace_summary,
    })
    trace(state.run_id, "done", {
        message_chars = #final_text,
        trace = trace_summary,
    })

    return final_text
end

return M
