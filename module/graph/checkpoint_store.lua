local ffi = require("ffi")
local util = require("module.graph.util")

local M = {}

local CHECKPOINT_ROOT = "memory/v3/graph/checkpoints"
local MAGIC = 0x47524350 -- GRCP
local VERSION = 1

ffi.cdef[[
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint64_t timestamp_ms;
    uint32_t seq;
    uint32_t payload_len;
} mori_graph_checkpoint_header_t;
]]

local HEADER_SIZE = tonumber(ffi.sizeof("mori_graph_checkpoint_header_t"))

local function encode_payload(tbl)
    return util.encode_lua_value(tbl or {}, 0)
end

local function decode_payload(raw)
    local text = util.trim(raw)
    if text == "" then
        return {}
    end
    local chunk, load_err = load("return " .. text, "checkpoint_payload", "t", {})
    if not chunk then
        return nil, tostring(load_err or "load_failed")
    end
    local ok, parsed = pcall(chunk)
    if not ok then
        return nil, tostring(parsed or "eval_failed")
    end
    if type(parsed) ~= "table" then
        return nil, "payload_not_table"
    end
    return parsed
end

local function checkpoint_path(run_id, seq)
    return string.format("%s/%s_%06d.bin", CHECKPOINT_ROOT, tostring(run_id), tonumber(seq) or 0)
end

local function write_bytes(path, bytes)
    local tmp = path .. ".tmp"
    local f = io.open(tmp, "wb")
    if not f then
        return false, "open_failed"
    end
    local ok, err = f:write(bytes)
    f:close()
    if not ok then
        os.remove(tmp)
        return false, tostring(err or "write_failed")
    end
    os.rename(tmp, path)
    return true
end

function M.ensure_root()
    util.ensure_dir(CHECKPOINT_ROOT)
end

function M.save_checkpoint(run_id, seq, node_name, state)
    if util.trim(run_id) == "" then
        return false, "empty_run_id"
    end

    M.ensure_root()

    local snapshot = {
        run_id = run_id,
        seq = tonumber(seq) or 0,
        node = tostring(node_name or ""),
        state_version = tostring((state or {}).state_version or ""),
        router_decision = ((state or {}).router_decision or {}).route,
        recall_triggered = ((state or {}).recall or {}).triggered == true,
        planner_calls = #((((state or {}).agent_loop or {}).pending_tool_calls) or {}),
        agent_iteration = tonumber((((state or {}).agent_loop or {}).iteration) or 0) or 0,
        remaining_steps = tonumber((((state or {}).agent_loop or {}).remaining_steps) or 0) or 0,
        stop_reason = tostring((((state or {}).agent_loop or {}).stop_reason) or ""),
        tool_executed = tonumber((((state or {}).tool_exec or {}).executed) or 0) or 0,
        tool_failed = tonumber((((state or {}).tool_exec or {}).failed) or 0) or 0,
        repair_attempts = tonumber((((state or {}).repair or {}).attempts) or 0) or 0,
        final_message = (((state or {}).final_response or {}).message) or "",
    }

    local payload = encode_payload(snapshot)
    local payload_len = #payload

    local header = ffi.new("mori_graph_checkpoint_header_t")
    header.magic = MAGIC
    header.version = VERSION
    header.flags = 0
    header.timestamp_ms = util.now_ms()
    header.seq = tonumber(seq) or 0
    header.payload_len = payload_len

    local path = checkpoint_path(run_id, seq)
    local blob = ffi.string(header, HEADER_SIZE) .. payload
    return write_bytes(path, blob)
end

local function list_run_checkpoint_files(run_id)
    local escaped_root = CHECKPOINT_ROOT:gsub('"', '\\"')
    local escaped_id = tostring(run_id):gsub('"', '\\"')
    local cmd = string.format('ls -1 "%s/%s_"*.bin 2>/dev/null', escaped_root, escaped_id)
    local p = io.popen(cmd)
    if not p then
        return {}
    end
    local out = {}
    for line in p:lines() do
        if line and line ~= "" then
            out[#out + 1] = line
        end
    end
    p:close()
    table.sort(out)
    return out
end

function M.load_last_checkpoint(run_id)
    if util.trim(run_id) == "" then
        return nil, "empty_run_id"
    end
    local files = list_run_checkpoint_files(run_id)
    if #files == 0 then
        return nil, "checkpoint_not_found"
    end

    local path = files[#files]
    local f = io.open(path, "rb")
    if not f then
        return nil, "open_failed"
    end
    local raw = f:read("*a") or ""
    f:close()

    if #raw < HEADER_SIZE then
        return nil, "header_too_short"
    end

    local header = ffi.new("mori_graph_checkpoint_header_t")
    ffi.copy(header, raw, HEADER_SIZE)

    if tonumber(header.magic) ~= MAGIC then
        return nil, "invalid_magic"
    end
    if tonumber(header.version) ~= VERSION then
        return nil, "invalid_version"
    end

    local payload_len = tonumber(header.payload_len) or 0
    local payload = raw:sub(HEADER_SIZE + 1, HEADER_SIZE + payload_len)
    local parsed, err = decode_payload(payload)
    if not parsed then
        return nil, err
    end

    return {
        path = path,
        seq = tonumber(header.seq) or 0,
        timestamp_ms = tonumber(header.timestamp_ms) or 0,
        snapshot = parsed,
    }
end

return M
