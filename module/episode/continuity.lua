local util = require("module.graph.util")
local store = require("module.episode.store")

local M = {}

local function trim(s)
    return util.trim(s or "")
end

local function join_tool_path(tool_sequence)
    if type(tool_sequence) ~= "table" or #tool_sequence == 0 then
        return "direct"
    end
    return table.concat(tool_sequence, ">")
end

local function format_episode_line(index, episode)
    local path = join_tool_path((episode or {}).tool_sequence)
    local summary = trim((episode or {}).summary or "")
    if summary == "" then
        summary = trim((episode or {}).final_text or (episode or {}).goal or "")
    end
    summary = util.utf8_take(summary, 220)

    local line = string.format(
        "%d. status=%s stop=%s path=%s",
        tonumber(index) or 0,
        trim((episode or {}).status or ""),
        trim((episode or {}).stop_reason or ""),
        path
    )
    if summary ~= "" then
        line = line .. " | " .. summary
    end

    local files = (episode or {}).files_read or {}
    if type(files) == "table" and #files > 0 then
        line = line .. " | files=" .. util.utf8_take(table.concat(files, ","), 180)
    end

    return line
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

local function push_tail(seq, item, max_items)
    if type(seq) ~= "table" or type(item) ~= "table" then
        return
    end
    seq[#seq + 1] = item
    local keep = math.max(1, math.floor(tonumber(max_items) or 8))
    while #seq > keep do
        table.remove(seq, 1)
    end
end

local function merge_flag_map(dst, src)
    for key, enabled in pairs(src or {}) do
        if enabled then
            dst[tostring(key)] = true
        end
    end
end

local function has_true_entries(tbl)
    for _, enabled in pairs(tbl or {}) do
        if enabled then
            return true
        end
    end
    return false
end

local function merge_snapshot(into, snapshot)
    if type(snapshot) ~= "table" then
        return
    end

    if trim(into.current_plan or "") == "" and trim(snapshot.current_plan or "") ~= "" then
        into.current_plan = tostring(snapshot.current_plan or "")
    end
    if tonumber(into.plan_step_index) == 0 and (tonumber(snapshot.plan_step_index) or 0) > 0 then
        into.plan_step_index = tonumber(snapshot.plan_step_index) or 0
    end
    if trim(into.last_tool_batch_summary or "") == "" and trim(snapshot.last_tool_batch_summary or "") ~= "" then
        into.last_tool_batch_summary = tostring(snapshot.last_tool_batch_summary or "")
    end
    if trim(into.last_repair_error or "") == "" and trim(snapshot.last_repair_error or "") ~= "" then
        into.last_repair_error = tostring(snapshot.last_repair_error or "")
    end

    merge_flag_map(into.files_read_set, snapshot.files_read_set or {})
    merge_flag_map(into.files_written_set, snapshot.files_written_set or {})

    for _, row in ipairs(snapshot.patches_applied or {}) do
        push_tail(into.patches_applied, row, 6)
    end
    for _, row in ipairs(snapshot.command_history_tail or {}) do
        push_tail(into.command_history_tail, row, 8)
    end
end

local function build_restored_working_memory(items)
    local restored = {
        current_plan = "",
        plan_step_index = 0,
        files_read_set = {},
        files_written_set = {},
        patches_applied = {},
        command_history_tail = {},
        last_tool_batch_summary = "",
        last_repair_error = "",
    }

    for idx = #items, 1, -1 do
        local episode = items[idx] or {}
        merge_snapshot(restored, (episode or {}).working_memory_snapshot)
    end

    local latest = items[1] or {}
    if trim(restored.current_plan or "") == "" then
        local path = join_tool_path((latest or {}).tool_sequence)
        if path ~= "direct" then
            restored.current_plan = "resume:" .. path
        elseif trim((latest or {}).summary or "") ~= "" then
            restored.current_plan = util.utf8_take(trim((latest or {}).summary or ""), 160)
        end
    end
    if trim(restored.last_tool_batch_summary or "") == "" then
        restored.last_tool_batch_summary = util.utf8_take(trim((latest or {}).final_text or (latest or {}).summary or ""), 800)
    end
    if trim(restored.last_repair_error or "") == "" and trim((latest or {}).stop_reason or "") ~= "" then
        local stop_reason = trim((latest or {}).stop_reason or "")
        if stop_reason ~= "finish_turn_called" then
            restored.last_repair_error = stop_reason
        end
    end

    return restored
end

local function build_working_memory_summary(restored)
    local read_count = 0
    local written_count = 0
    for _, enabled in pairs((restored or {}).files_read_set or {}) do
        if enabled then
            read_count = read_count + 1
        end
    end
    for _, enabled in pairs((restored or {}).files_written_set or {}) do
        if enabled then
            written_count = written_count + 1
        end
    end

    local parts = {}
    if trim((restored or {}).current_plan or "") ~= "" then
        parts[#parts + 1] = "plan=" .. tostring(restored.current_plan)
    end
    if read_count > 0 or written_count > 0 then
        parts[#parts + 1] = string.format("files=%d/%d", read_count, written_count)
    end
    if trim((restored or {}).last_repair_error or "") ~= "" then
        parts[#parts + 1] = "repair=" .. tostring(restored.last_repair_error)
    end
    if trim((restored or {}).last_tool_batch_summary or "") ~= "" then
        parts[#parts + 1] = "tool=" .. util.utf8_take(tostring(restored.last_tool_batch_summary), 120)
    end
    return table.concat(parts, " | ")
end

function M.build_for_task(task_id, opts)
    local task = trim(task_id)
    if task == "" then
        return {
            task_id = "",
            items = {},
            count = 0,
            latest_episode_id = "",
            latest_summary = "",
            latest_profile = "",
            summary = "",
        }
    end

    opts = opts or {}
    local limit = math.max(1, math.floor(tonumber(opts.limit) or 3))
    local items = store.get_recent_by_task(task, limit)

    local lines = {}
    for i, episode in ipairs(items or {}) do
        lines[#lines + 1] = format_episode_line(i, episode)
    end
    local restored = build_restored_working_memory(items or {})
    local working_memory_summary = build_working_memory_summary(restored)
    if working_memory_summary ~= "" then
        lines[#lines + 1] = "wm: " .. working_memory_summary
    end

    local latest = items[1] or {}
    return {
        task_id = task,
        items = items or {},
        count = #lines,
        latest_episode_id = trim(latest.id or ""),
        latest_summary = trim(latest.summary or ""),
        latest_profile = trim(latest.profile or ""),
        latest_goal = trim(latest.goal or ""),
        latest_status = trim(latest.status or ""),
        latest_user_input = trim(latest.user_input or ""),
        restored_working_memory = restored,
        summary = table.concat(lines, "\n"),
    }
end

return M
