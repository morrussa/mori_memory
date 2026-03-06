local ffi = require("ffi")
local bit = bit or require("bit")  -- LuaJIT bit 库
local util = require("module.graph.util")
local persistence = require("module.persistence")

local M = {}

local CHECKPOINT_ROOT = "memory/v3/graph/checkpoints"
local MAGIC = 0x47524350 -- GRCP
local VERSION = 2  -- v2: 支持完整状态恢复

ffi.cdef[[
typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;      // bit0: has_full_state
    uint64_t timestamp_ms;
    uint32_t seq;
    uint32_t payload_len;
    uint32_t state_len;  // v2: 完整状态长度
    uint32_t next_node_len; // v2: 下一个节点名长度
} mori_graph_checkpoint_header_t;
]]

local HEADER_SIZE = tonumber(ffi.sizeof("mori_graph_checkpoint_header_t"))

-- ========== Saver 模式支持 ==========
M.dirty = false
M.pending = nil  -- { run_id, seq, node, next_node, state_snapshot, state_summary }
M.pending_queue = {}  -- 队列：存储所有待保存的 checkpoint

function M.mark_dirty(run_id, seq, node, next_node, state_snapshot, state_summary)
    M.dirty = true
    -- 将新的 checkpoint 添加到队列末尾
    M.pending_queue[#M.pending_queue + 1] = {
        run_id = run_id,
        seq = tonumber(seq) or 0,
        node = tostring(node or ""),
        next_node = tostring(next_node or ""),
        state_snapshot = state_snapshot,  -- 完整状态（用于恢复）
        state_summary = state_summary,     -- 摘要（用于trace）
    }
    -- 更新 pending 为最新的一个（用于 flush）
    M.pending = M.pending_queue[#M.pending_queue]
end

function M.clear_pending()
    M.dirty = false
    M.pending = nil
    M.pending_queue = {}
end

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

-- 序列化完整状态（排除不可序列化的字段）
local function serialize_state(state)
    if type(state) ~= "table" then
        return ""
    end
    -- 排除函数、stream_sink、userdata等不可序列化字段
    local safe_state = {}
    local exclude_keys = {
        stream_sink = true,
        _streaming_sent = true,
    }
    local seen = {}  -- 用于检测循环引用

    local function serialize_value(v, depth)
        depth = (depth or 0) + 1
        if depth > 32 then
            return nil  -- 防止过深递归
        end

        local vt = type(v)
        -- 排除不可序列化的类型
        if vt == "function" or vt == "thread" or vt == "userdata" then
            return nil
        end

        if vt == "table" then
            -- 检测循环引用
            if seen[v] then
                return nil
            end
            seen[v] = true

            local result = {}
            for k, val in pairs(v) do
                -- 排除特定的键
                if not exclude_keys[k] then
                    local serialized = serialize_value(val, depth)
                    if serialized ~= nil then
                        result[k] = serialized
                    end
                end
            end
            seen[v] = nil
            return result
        end

        return v
    end

    return serialize_value(state, 0) or ""
end

-- 提取状态摘要（用于trace和快速查看）
local function extract_summary(state)
    return {
        run_id = tostring((state or {}).run_id or ""),
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
end

-- 保存checkpoint（支持完整状态）
function M.save_checkpoint(run_id, seq, node_name, state, next_node)
    if util.trim(run_id) == "" then
        return false, "empty_run_id"
    end

    M.ensure_root()

    local summary = extract_summary(state)
    summary.seq = tonumber(seq) or 0
    summary.node = tostring(node_name or "")
    summary.next_node = tostring(next_node or "")

    local payload = encode_payload(summary)
    local payload_len = #payload

    local full_state = serialize_state(state)
    local state_len = #full_state
    local next_node_bytes = tostring(next_node or "")
    local next_node_len = #next_node_bytes

    local header = ffi.new("mori_graph_checkpoint_header_t")
    header.magic = MAGIC
    header.version = VERSION
    header.flags = state_len > 0 and 1 or 0  -- bit0: has_full_state
    header.timestamp_ms = util.now_ms()
    header.seq = tonumber(seq) or 0
    header.payload_len = payload_len
    header.state_len = state_len
    header.next_node_len = next_node_len

    local path = checkpoint_path(run_id, seq)
    local blob = ffi.string(header, HEADER_SIZE) .. payload .. full_state .. next_node_bytes

    -- 使用 persistence.lua 的原子写入
    local ok, err = persistence.write_atomic(path, "wb", function(f)
        return f:write(blob)
    end)
    if not ok then
        return false, err or "write_failed"
    end
    return true
end

-- ========== Saver 模式的 flush 接口 ==========
function M.flush(force)
    if not M.dirty and not force then
        return true
    end

    if #M.pending_queue == 0 then
        return true
    end

    -- 保存队列中的所有 checkpoint
    local saved_count = 0
    for i, p in ipairs(M.pending_queue) do
        local ok, err = M.save_checkpoint(
            p.run_id,
            p.seq,
            p.node,
            p.state_snapshot,
            p.next_node
        )

        if ok then
            saved_count = saved_count + 1
        else
            M.dirty = true  -- 确保下次会重试
            print(string.format("[CheckpointStore][ERROR] flush failed at index %d: %s", i, tostring(err)))
            return false, err
        end
    end

    -- 全部保存成功，清空队列
    M.dirty = false
    M.pending = nil
    M.pending_queue = {}

    if saved_count > 0 then
        print(string.format("[CheckpointStore] 已保存 %d 个 checkpoint", saved_count))
    end

    return true
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

    local version = tonumber(header.version) or 1
    local flags = tonumber(header.flags) or 0
    local payload_len = tonumber(header.payload_len) or 0
    local state_len = tonumber(header.state_len) or 0
    local next_node_len = tonumber(header.next_node_len) or 0

    -- 解析摘要
    local payload = raw:sub(HEADER_SIZE + 1, HEADER_SIZE + payload_len)
    local parsed, err = decode_payload(payload)
    if not parsed then
        return nil, err
    end

    local result = {
        path = path,
        seq = tonumber(header.seq) or 0,
        timestamp_ms = tonumber(header.timestamp_ms) or 0,
        snapshot = parsed,
        has_full_state = (bit.band(flags, 1) == 1) and state_len > 0,
        next_node = "",
    }

    -- v2: 解析完整状态和 next_node
    if version >= 2 then
        -- 解析完整状态
        if result.has_full_state then
            local state_start = HEADER_SIZE + payload_len + 1
            local state_end = state_start + state_len - 1
            local state_raw = raw:sub(state_start, state_end)
            if #state_raw > 0 then
                local state_parsed, state_err = decode_payload(state_raw)
                if state_parsed then
                    result.full_state = state_parsed
                else
                    print(string.format("[CheckpointStore][WARN] decode full_state failed: %s", tostring(state_err)))
                end
            end
        end

        -- 解析下一个节点（即使没有完整状态也要解析）
        if next_node_len > 0 then
            local next_start = HEADER_SIZE + payload_len + state_len + 1
            local next_end = next_start + next_node_len - 1
            local next_node_raw = raw:sub(next_start, next_end)
            if #next_node_raw > 0 then
                result.next_node = next_node_raw
            end
        end
    end

    return result
end

return M
