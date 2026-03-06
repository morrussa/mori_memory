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

    local latest = items[1] or {}
    return {
        task_id = task,
        items = items or {},
        count = #lines,
        latest_episode_id = trim(latest.id or ""),
        latest_summary = trim(latest.summary or ""),
        latest_profile = trim(latest.profile or ""),
        summary = table.concat(lines, "\n"),
    }
end

return M
