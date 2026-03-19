local config = require("module.config")
local persistence = require("module.persistence")
local util = require("mori_memory.util")

local M = {
    next_seq = 0,
}

local function trim(s)
    return util.trim(s)
end

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function runtime_root()
    return tostring(config.get("disentangle.runtime.root", "memory/v4/runtime") or "memory/v4/runtime")
end

local function wal_path()
    return runtime_root() .. "/thread_wal.log"
end

local function normalize_record(record)
    record = type(record) == "table" and record or {}
    return {
        seq = math.max(0, math.floor(tonumber(record.seq) or 0)),
        turn = math.max(0, math.floor(tonumber(record.turn) or 0)),
        kind = trim(record.kind or "scope_state"),
        scope_key = trim(record.scope_key or ""),
        reason = trim(record.reason or ""),
        scope_state = type(record.scope_state) == "table" and record.scope_state or {},
    }
end

function M.path()
    return wal_path()
end

function M.set_next_seq(seq)
    M.next_seq = math.max(0, math.floor(tonumber(seq) or 0))
    return M.next_seq
end

function M.load_after(last_seq)
    last_seq = math.max(0, math.floor(tonumber(last_seq) or 0))
    local f = io.open(wal_path(), "r")
    if not f then
        M.next_seq = math.max(M.next_seq, last_seq)
        return {}
    end

    local out = {}
    for line in f:lines() do
        line = trim(line)
        if line ~= "" then
            local parsed = util.parse_lua_table_literal(line)
            if type(parsed) == "table" then
                local record = normalize_record(parsed)
                if record.seq > 0 then
                    M.next_seq = math.max(M.next_seq, record.seq)
                    if record.seq > last_seq then
                        out[#out + 1] = record
                    end
                end
            end
        end
    end
    f:close()

    table.sort(out, function(a, b)
        local sa = tonumber((a or {}).seq) or 0
        local sb = tonumber((b or {}).seq) or 0
        if sa ~= sb then
            return sa < sb
        end
        return (tonumber((a or {}).turn) or 0) < (tonumber((b or {}).turn) or 0)
    end)
    return out
end

function M.append(record)
    record = normalize_record(record)
    ensure_dir(runtime_root())

    if record.seq <= 0 then
        record.seq = math.max(0, M.next_seq) + 1
    end
    M.next_seq = math.max(M.next_seq, record.seq)

    local f, open_err = io.open(wal_path(), "a")
    if not f then
        return nil, "open_wal_failed: " .. tostring(open_err)
    end

    local encoded = util.encode_lua_value(record, 0)
    local ok_write, write_err = f:write(encoded, "\n")
    if not ok_write then
        pcall(function() f:close() end)
        return nil, "append_wal_failed: " .. tostring(write_err)
    end

    local close_ok, close_err = f:close()
    if close_ok == nil then
        return nil, "close_wal_failed: " .. tostring(close_err)
    end
    return record.seq
end

function M.reset(seq_floor)
    ensure_dir(runtime_root())
    M.next_seq = math.max(M.next_seq, math.max(0, math.floor(tonumber(seq_floor) or 0)))
    return persistence.write_atomic(wal_path(), "w", function(f)
        return f:write("")
    end)
end

return M
