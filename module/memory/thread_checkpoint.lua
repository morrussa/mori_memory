local config = require("module.config")
local persistence = require("module.persistence")
local util = require("mori_memory.util")

local M = {}

local VERSION = 1

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function runtime_root()
    return tostring(config.get("disentangle.runtime.root", "memory/v4/runtime") or "memory/v4/runtime")
end

local function checkpoint_path()
    return runtime_root() .. "/thread_checkpoint.lua"
end

function M.path()
    return checkpoint_path()
end

function M.load()
    local f = io.open(checkpoint_path(), "r")
    if not f then
        return {
            version = VERSION,
            last_seq = 0,
            saved_turn = 0,
            saved_at = 0,
            state = {},
        }, "missing_checkpoint"
    end

    local raw = f:read("*a")
    f:close()
    local parsed, err = util.parse_lua_table_literal(raw or "")
    if type(parsed) ~= "table" then
        return {
            version = VERSION,
            last_seq = 0,
            saved_turn = 0,
            saved_at = 0,
            state = {},
        }, err or "invalid_checkpoint"
    end

    return {
        version = math.max(1, math.floor(tonumber(parsed.version) or VERSION)),
        last_seq = math.max(0, math.floor(tonumber(parsed.last_seq) or 0)),
        saved_turn = math.max(0, math.floor(tonumber(parsed.saved_turn) or 0)),
        saved_at = math.max(0, math.floor(tonumber(parsed.saved_at) or 0)),
        state = type(parsed.state) == "table" and parsed.state or {},
    }
end

function M.save(state, last_seq, meta)
    meta = type(meta) == "table" and meta or {}
    ensure_dir(runtime_root())
    local payload = {
        version = VERSION,
        last_seq = math.max(0, math.floor(tonumber(last_seq) or 0)),
        saved_turn = math.max(0, math.floor(tonumber(meta.turn) or 0)),
        saved_at = math.max(0, math.floor(tonumber(os.time()) or 0)),
        state = type(state) == "table" and state or {},
    }
    return persistence.write_atomic(checkpoint_path(), "w", function(f)
        return f:write(util.encode_lua_value(payload, 0))
    end)
end

return M
