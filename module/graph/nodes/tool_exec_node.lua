local util = require("module.graph.util")
local tool_registry = require("module.graph.tool_registry_v2")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function ensure_sequence(tbl)
    if type(tbl) ~= "table" then
        return {}
    end
    return tbl
end

local function push_tail(seq, item, max_items)
    seq[#seq + 1] = item
    local keep = math.max(1, math.floor(tonumber(max_items) or 8))
    while #seq > keep do
        table.remove(seq, 1)
    end
end

local function mark_set(tbl, key)
    if type(tbl) ~= "table" then
        return
    end
    local value = util.normalize_tool_path(key or "")
    if value ~= "" then
        tbl[value] = true
    end
end

local function update_working_memory_from_call(memory, row)
    if type(row) ~= "table" then
        return
    end
    local args = row.args or {}
    local tool_name = tostring(row.tool or "")

    if row.ok == true then
        if tool_name == "list_files" then
            mark_set(memory.files_read_set, args.prefix or util.workspace_virtual_root())
        elseif tool_name == "read_file" or tool_name == "read_lines" or tool_name == "search_file" then
            mark_set(memory.files_read_set, args.path)
        elseif tool_name == "search_files" then
            mark_set(memory.files_read_set, args.prefix or util.workspace_virtual_root())
        elseif tool_name == "write_file" then
            mark_set(memory.files_written_set, args.path)
        elseif tool_name == "apply_patch" then
            push_tail(memory.patches_applied, {
                patch = util.utf8_take(tostring(args.patch or args.diff or ""), 1200),
                result = util.utf8_take(tostring(row.result or ""), 200),
            }, 6)
        elseif tool_name == "exec_command" then
            push_tail(memory.command_history_tail, {
                argv = args.argv or {},
                workdir = util.normalize_tool_path(args.workdir or util.workspace_virtual_root()),
                ok = row.ok == true,
                result = util.utf8_take(tostring(row.result or row.error or ""), 320),
            }, 8)
        end
    elseif tool_name == "exec_command" then
        push_tail(memory.command_history_tail, {
            argv = args.argv or {},
            workdir = util.normalize_tool_path(args.workdir or util.workspace_virtual_root()),
            ok = false,
            result = util.utf8_take(tostring(row.error or ""), 320),
        }, 8)
    end
end

local function update_read_tracking(tool_exec, row)
    if type(row) ~= "table" or row.ok ~= true then
        return
    end

    local args = row.args or {}
    local tool_name = tostring(row.tool or "")
    if tool_name == "read_file" or tool_name == "read_lines" or tool_name == "search_file" then
        tool_exec.read_files = tool_exec.read_files or {}
        mark_set(tool_exec.read_files, args.path)
        tool_exec.read_evidence_total = (tonumber(tool_exec.read_evidence_total) or 0) + 1
    elseif tool_name == "search_files" then
        tool_exec.read_evidence_total = (tonumber(tool_exec.read_evidence_total) or 0) + 1
    end
end

local function append_runtime_tool_messages(state, results)
    state.messages = state.messages or {}
    state.messages.runtime_messages = state.messages.runtime_messages or {}
    for _, row in ipairs(results or {}) do
        if type(row) == "table" and row.is_control ~= true then
            local content = ""
            if row.ok == true then
                content = tostring(row.result or "")
            else
                content = "Error: " .. tostring(row.error or "tool_exec_failed")
            end
            state.messages.runtime_messages[#state.messages.runtime_messages + 1] = {
                role = "tool",
                name = tostring(row.tool or ""),
                tool_call_id = tostring(row.call_id or ""),
                content = content,
            }
        end
    end
end

function M.run(state, _ctx)
    state.tool_exec = state.tool_exec or {}
    state.planner = state.planner or {}
    state.repair = state.repair or {}
    state.termination = state.termination or {}
    state.context = state.context or {}
    state.working_memory = state.working_memory or {}

    state.tool_exec.loop_count = (tonumber(state.tool_exec.loop_count) or 0) + 1
    state.tool_exec.executed = 0
    state.tool_exec.failed = 0
    state.tool_exec.results = {}
    state.tool_exec.all_results = ensure_sequence(state.tool_exec.all_results)
    state.tool_exec.context_fragments = {}
    state.tool_exec.read_files = state.tool_exec.read_files or {}
    state.tool_exec.read_evidence_total = tonumber(state.tool_exec.read_evidence_total) or 0
    state.repair.pending = false
    state.repair.retry_requested = false

    local calls = ensure_sequence(state.planner.tool_calls)
    local result = tool_registry.execute_calls(calls, state)

    state.tool_exec.executed = tonumber(result.executed) or 0
    state.tool_exec.failed = tonumber(result.failed) or 0
    state.tool_exec.executed_total = (tonumber(state.tool_exec.executed_total) or 0) + state.tool_exec.executed
    state.tool_exec.failed_total = (tonumber(state.tool_exec.failed_total) or 0) + state.tool_exec.failed
    state.tool_exec.results = result.call_results or {}
    state.tool_exec.context_fragments = result.context_fragments or {}
    state.tool_exec.total_result_chars = (tonumber(state.tool_exec.total_result_chars) or 0) + (tonumber(result.total_result_chars) or 0)
    state.tool_exec.large_results = result.large_results or {}

    for _, row in ipairs(state.tool_exec.results or {}) do
        state.tool_exec.all_results[#state.tool_exec.all_results + 1] = row
        update_working_memory_from_call(state.working_memory, row)
        update_read_tracking(state.tool_exec, row)
    end
    append_runtime_tool_messages(state, state.tool_exec.results)

    if result.control_action == "finish" then
        state.termination.finish_requested = true
        state.termination.final_message = util.trim((((result or {}).control_data) or {}).message or "")
        state.termination.final_status = util.trim((((result or {}).control_data) or {}).status or "completed")
        state.termination.stop_reason = "finish_turn_called"
        state.agent_loop = state.agent_loop or {}
        state.agent_loop.stop_reason = "finish_turn_called"
    end

    if util.trim(result.protocol_error or "") ~= "" then
        state.repair.pending = true
        state.repair.last_error = tostring(result.protocol_error)
        state.working_memory.last_repair_error = tostring(result.protocol_error)
        state.context.planner_context = "Previous tool batch violated the planner/tool contract: " .. tostring(result.protocol_error)
    elseif state.tool_exec.failed > 0 then
        state.repair.pending = true
        state.repair.last_error = tostring(result.last_error or "tool_exec_failed")
        state.working_memory.last_repair_error = state.repair.last_error
        state.context.planner_context = "Previous tool batch failed. Repair the call or choose a different tool."
    else
        state.context.planner_context = ""
        state.working_memory.last_repair_error = ""
    end

    local max_chars = math.max(120, math.floor(tonumber((((graph_cfg() or {}).tools or {}).file_context_max_chars) or 1600)))
    local summary = util.trim(result.summary or "")
    if summary == "" and #(state.tool_exec.context_fragments or {}) > 0 then
        summary = table.concat(state.tool_exec.context_fragments, "\n\n")
    end
    summary = util.utf8_take(summary, max_chars)
    state.working_memory.last_tool_batch_summary = summary
    state.context.tool_context = summary
    state.planner.tool_calls = {}
    state.agent_loop = state.agent_loop or {}
    state.agent_loop.pending_tool_calls = {}
    return state
end

return M
