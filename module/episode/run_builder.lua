local util = require("module.graph.util")

local M = {}

local function trim(s)
    return util.trim(s or "")
end

local function sorted_set_keys(src)
    local out = {}
    for key, enabled in pairs(src or {}) do
        if enabled then
            out[#out + 1] = tostring(key)
        end
    end
    table.sort(out, function(a, b)
        return tostring(a) < tostring(b)
    end)
    return out
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function shallow_copy_map(src)
    local out = {}
    for k, v in pairs(src or {}) do
        out[k] = v
    end
    return out
end

local function copy_tool_results(rows)
    local out = {}
    for _, row in ipairs(rows or {}) do
        if type(row) == "table" then
            out[#out + 1] = {
                call_id = tostring(row.call_id or ""),
                tool = tostring(row.tool or ""),
                ok = row.ok == true,
                is_control = row.is_control == true,
                result_preview = util.utf8_take(trim(row.result or ""), 320),
                error_preview = util.utf8_take(trim(row.error or ""), 240),
            }
        end
    end
    return out
end

local function copy_uploads(rows)
    local out = {}
    for _, row in ipairs(rows or {}) do
        if type(row) == "table" then
            out[#out + 1] = {
                name = tostring(row.name or ""),
                path = tostring(row.path or ""),
                tool_path = tostring(row.tool_path or ""),
                bytes = tonumber(row.bytes) or 0,
            }
        end
    end
    return out
end

local function copy_patch_history(rows)
    local out = {}
    for _, row in ipairs(rows or {}) do
        if type(row) == "table" then
            out[#out + 1] = {
                patch = util.utf8_take(trim(row.patch or row.diff or ""), 1200),
                result = util.utf8_take(trim(row.result or ""), 240),
            }
        end
    end
    return out
end

local function copy_command_history(rows)
    local out = {}
    for _, row in ipairs(rows or {}) do
        if type(row) == "table" then
            out[#out + 1] = {
                argv = shallow_copy_array(row.argv or {}),
                workdir = trim(row.workdir or ""),
                ok = row.ok == true,
                result = util.utf8_take(trim(row.result or row.error or ""), 320),
            }
        end
    end
    return out
end

local function collect_tools(rows)
    local tools_used = {}
    local sequence = {}

    for _, row in ipairs(rows or {}) do
        if type(row) == "table" and row.is_control ~= true then
            local tool_name = trim(row.tool)
            if tool_name ~= "" then
                sequence[#sequence + 1] = tool_name
                tools_used[tool_name] = (tools_used[tool_name] or 0) + 1
            end
        end
    end

    return tools_used, sequence
end

local function collect_effective_ids(src)
    local out = {}
    for key, enabled in pairs(src or {}) do
        if enabled then
            out[#out + 1] = tostring(key)
        end
    end
    table.sort(out)
    return out
end

local function detect_success(state)
    local termination = ((state or {}).termination) or {}
    local final_status = trim(termination.final_status or (((state or {}).final_response or {}).status) or "")
    local stop_reason = trim(termination.stop_reason or (((state or {}).agent_loop or {}).stop_reason) or "")
    local final_text = trim((((state or {}).final_response or {}).message) or "")

    if final_status == "completed" and final_text ~= "" then
        return true
    end
    if stop_reason == "finish_turn_called" and final_text ~= "" then
        return true
    end
    return false
end

local function build_summary(active_task, status, stop_reason, tool_sequence)
    local parts = {
        string.format("goal=%s", trim((active_task or {}).goal)),
        string.format("status=%s", trim(status)),
        string.format("stop=%s", trim(stop_reason)),
    }

    if #tool_sequence > 0 then
        parts[#parts + 1] = "path=" .. table.concat(tool_sequence, ">")
    else
        parts[#parts + 1] = "path=direct"
    end

    return util.utf8_take(table.concat(parts, " | "), 320)
end

local function build_working_memory_snapshot(working_memory)
    local memory = working_memory or {}
    return {
        current_plan = trim(memory.current_plan or ""),
        plan_step_index = tonumber(memory.plan_step_index) or 0,
        files_read_set = shallow_copy_map(memory.files_read_set or {}),
        files_written_set = shallow_copy_map(memory.files_written_set or {}),
        patches_applied = copy_patch_history(memory.patches_applied or {}),
        command_history_tail = copy_command_history(memory.command_history_tail or {}),
        last_tool_batch_summary = util.utf8_take(trim(memory.last_tool_batch_summary or ""), 1600),
        last_repair_error = util.utf8_take(trim(memory.last_repair_error or ""), 240),
    }
end

function M.build_from_state(state)
    local active_task = ((((state or {}).session or {}).active_task) or {})
    local working_memory = ((state or {}).working_memory) or {}
    local tool_exec = ((state or {}).tool_exec) or {}
    local repair = ((state or {}).repair) or {}
    local writeback = ((state or {}).writeback) or {}
    local experience = ((state or {}).experience) or {}
    local episode_state = ((state or {}).episode) or {}
    local episode_current = (episode_state.current or {})

    local status = trim(active_task.status or (((state or {}).termination or {}).final_status) or "")
    local stop_reason = trim((((state or {}).termination or {}).stop_reason) or (((state or {}).agent_loop or {}).stop_reason) or "")
    local tools_used, tool_sequence = collect_tools(tool_exec.all_results or tool_exec.results or {})

    return {
        run_id = trim((state or {}).run_id or ""),
        task_id = trim(active_task.task_id or ""),
        goal = trim(active_task.goal or (((state or {}).input or {}).message) or ""),
        profile = trim(active_task.profile or (((state or {}).context or {}).task_profile) or ""),
        status = status,
        stop_reason = stop_reason,
        success = detect_success(state),
        read_only = (((state or {}).input or {}).read_only) == true,
        created_at = os.time(),
        created_at_ms = util.now_ms(),
        turn_index = tonumber(episode_current.turn_index) or 0,
        topic_anchor = trim(episode_current.topic_anchor or ""),
        user_input = util.utf8_take(trim((((state or {}).input or {}).message) or ""), 1600),
        final_text = util.utf8_take(trim((((state or {}).final_response or {}).message) or ""), 2200),
        summary = build_summary(active_task, status, stop_reason, tool_sequence),
        tool_sequence = shallow_copy_array(tool_sequence),
        tools_used = tools_used,
        tool_results = copy_tool_results(tool_exec.all_results or tool_exec.results or {}),
        files_read = sorted_set_keys(working_memory.files_read_set),
        files_written = sorted_set_keys(working_memory.files_written_set),
        uploads = copy_uploads((state or {}).uploads or {}),
        retrieved_policy_ids = shallow_copy_array((((experience or {}).retrieved or {}).ids) or {}),
        effective_policy_ids = collect_effective_ids((((experience or {}).feedback) or {}).effective_ids),
        policy_writeback = {
            written = (((experience or {}).writeback or {}).written) == true,
            policy_id = trim((((experience or {}).writeback or {}).policy_id) or ((((experience or {}).writeback or {}).experience_id) or "")),
        },
        memory_writeback = {
            items = shallow_copy_array(writeback.items or writeback.facts or {}),
            ingest_strategy = trim(writeback.ingest_strategy or "atomic_fact"),
            saved_count = tonumber(writeback.saved_count or writeback.saved) or 0,
        },
        working_memory_snapshot = build_working_memory_snapshot(working_memory),
        metrics = {
            loop_count = tonumber(tool_exec.loop_count) or 0,
            executed_total = tonumber(tool_exec.executed_total) or 0,
            failed_total = tonumber(tool_exec.failed_total) or 0,
            repair_attempts = tonumber(repair.attempts) or 0,
            read_evidence_total = tonumber(tool_exec.read_evidence_total) or 0,
            uploads_count = #(((state or {}).uploads) or {}),
        },
    }
end

return M
