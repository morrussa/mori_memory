local util = require("module.graph.util")
local persistence = require("module.persistence")

local M = {}

local ROOT = "memory/v3/graph"
local LUA_PATH = ROOT .. "/session_state.lua"
local JSON_PATH = ROOT .. "/session_state.json"

local function dirname(path)
    local dir = tostring(path or ""):match("^(.*)/[^/]+$")
    if util.trim(dir) == "" then
        return "."
    end
    return dir
end

local function copy_table(src)
    local out = {}
    if type(src) ~= "table" then
        return out
    end
    for k, v in pairs(src) do
        out[k] = v
    end
    return out
end

local function default_payload()
    return {
        api_version = "graph_v2",
        session_mode = "single",
        last_run_id = "",
        updated_at_ms = 0,
        active_task = {
            task_id = "",
            goal = "",
            status = "",
            carryover_summary = "",
            last_user_message = "",
            profile = "",
        },
        working_memory = {
            current_plan = "",
            plan_step_index = 0,
            files_read_set = {},
            files_written_set = {},
            patches_applied = {},
            command_history_tail = {},
            last_tool_batch_summary = "",
            last_repair_error = "",
        },
        recovery = {
            resumable_run_id = "",
            last_checkpoint_seq = 0,
            next_node = "",
            resumed_from_checkpoint = false,
        },
        stats = {
            files_read_count = 0,
            files_written_count = 0,
        },
        last_trace_summary = {},
    }
end

local function count_table_entries(tbl)
    local count = 0
    if type(tbl) ~= "table" then
        return 0
    end
    for _, _ in pairs(tbl) do
        count = count + 1
    end
    return count
end

local function normalize(payload)
    local out = default_payload()
    if type(payload) ~= "table" then
        return out
    end

    out.api_version = util.trim(payload.api_version or out.api_version)
    if out.api_version == "" then
        out.api_version = "graph_v2"
    end
    out.session_mode = util.trim(payload.session_mode or out.session_mode)
    if out.session_mode == "" then
        out.session_mode = "single"
    end
    out.last_run_id = util.trim(payload.last_run_id or "")
    out.updated_at_ms = tonumber(payload.updated_at_ms) or 0

    local active_task = copy_table(payload.active_task)
    out.active_task.task_id = util.trim(active_task.task_id or "")
    out.active_task.goal = util.trim(active_task.goal or "")
    out.active_task.status = util.trim(active_task.status or "")
    out.active_task.carryover_summary = util.trim(active_task.carryover_summary or "")
    out.active_task.last_user_message = util.trim(active_task.last_user_message or "")
    out.active_task.profile = util.trim(active_task.profile or "")

    local memory = copy_table(payload.working_memory)
    out.working_memory.current_plan = util.trim(memory.current_plan or "")
    out.working_memory.plan_step_index = tonumber(memory.plan_step_index) or 0
    out.working_memory.files_read_set = copy_table(memory.files_read_set)
    out.working_memory.files_written_set = copy_table(memory.files_written_set)
    out.working_memory.patches_applied = copy_table(memory.patches_applied)
    out.working_memory.command_history_tail = memory.command_history_tail or {}
    out.working_memory.last_tool_batch_summary = util.trim(memory.last_tool_batch_summary or "")
    out.working_memory.last_repair_error = util.trim(memory.last_repair_error or "")

    local recovery = copy_table(payload.recovery)
    out.recovery.resumable_run_id = util.trim(recovery.resumable_run_id or "")
    out.recovery.last_checkpoint_seq = tonumber(recovery.last_checkpoint_seq) or 0
    out.recovery.next_node = util.trim(recovery.next_node or "")
    out.recovery.resumed_from_checkpoint = recovery.resumed_from_checkpoint == true

    out.stats.files_read_count = tonumber((((payload or {}).stats or {}).files_read_count)) or count_table_entries(out.working_memory.files_read_set)
    out.stats.files_written_count = tonumber((((payload or {}).stats or {}).files_written_count)) or count_table_entries(out.working_memory.files_written_set)
    out.last_trace_summary = copy_table(payload.last_trace_summary)

    return out
end

function M.new_session_state()
    return default_payload()
end

function M.load()
    util.ensure_dir(ROOT)

    local f = io.open(LUA_PATH, "rb")
    if not f then
        return default_payload()
    end

    local raw = f:read("*a") or ""
    f:close()
    local parsed, _err = util.parse_lua_table_literal(raw)
    return normalize(parsed)
end

function M.save(payload)
    local normalized = normalize(payload)
    normalized.updated_at_ms = util.now_ms()

    util.ensure_dir(dirname(LUA_PATH))
    local ok, err = persistence.write_atomic(LUA_PATH, "wb", function(f)
        return f:write(util.encode_lua_value(normalized, 0))
    end)
    if not ok then
        return false, err
    end

    local json_ok, json_err = persistence.write_atomic(JSON_PATH, "wb", function(f)
        return f:write(util.json_encode(normalized, 0))
    end)
    if not json_ok then
        return false, json_err
    end

    return true
end

function M.build_from_state(state, trace_summary)
    local active_task = ((((state or {}).session or {}).active_task) or {})
    local working_memory = ((state or {}).working_memory) or {}
    local recovery = ((state or {}).recovery) or {}

    return normalize({
        last_run_id = tostring((state or {}).run_id or ""),
        active_task = active_task,
        working_memory = working_memory,
        recovery = recovery,
        stats = {
            files_read_count = count_table_entries(working_memory.files_read_set),
            files_written_count = count_table_entries(working_memory.files_written_set),
        },
        last_trace_summary = trace_summary or {},
    })
end

return M
