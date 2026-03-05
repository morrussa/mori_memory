local util = require("module.graph.util")

local M = {}

local TRACE_ROOT = "memory/v3/graph/traces"

local function append_line(path, line)
    local f = io.open(path, "a")
    if not f then
        return false, "open_failed"
    end
    local ok, err = f:write(line)
    f:close()
    if not ok then
        return false, err
    end
    return true
end

local function trace_path(run_id)
    return string.format("%s/%s.jsonl", TRACE_ROOT, tostring(run_id))
end

function M.ensure_root()
    util.ensure_dir(TRACE_ROOT)
end

function M.append(run_id, event_name, payload)
    if util.trim(run_id) == "" then
        return false, "empty_run_id"
    end
    if util.trim(event_name) == "" then
        return false, "empty_event"
    end

    M.ensure_root()
    local row = {
        ts_ms = util.now_ms(),
        run_id = run_id,
        event = event_name,
        payload = payload or {},
    }
    local line = util.json_encode(row) .. "\n"
    return append_line(trace_path(run_id), line)
end

return M
