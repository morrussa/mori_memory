local util = require("module.graph.util")

local M = {}

M.STATE_VERSION = "v1"

local REQUIRED_KEYS = {
    "state_version",
    "run_id",
    "input",
    "uploads",
    "messages",
    "agent_loop",
    "context",
    "router_decision",
    "recall",
    "planner",
    "tool_exec",
    "repair",
    "final_response",
    "writeback",
    "metrics",
    "checkpoint_meta",
}

local function get_field(obj, key)
    if obj == nil then
        return nil
    end

    if type(obj) == "table" then
        local v = obj[key]
        if v ~= nil then
            return v
        end
        return obj[tostring(key)]
    end

    local ok, v = pcall(function()
        return obj[key]
    end)
    if ok and v ~= nil then
        return v
    end

    ok, v = pcall(function()
        return obj[tostring(key)]
    end)
    if ok and v ~= nil then
        return v
    end

    return nil
end

local function normalize_upload_item(item)
    if item == nil then
        return nil
    end
    local name = tostring(get_field(item, "name") or "")
    local path = tostring(get_field(item, "path") or "")
    local tool_path = tostring(get_field(item, "tool_path") or get_field(item, "toolPath") or "")
    local bytes = tonumber(get_field(item, "bytes")) or 0
    if name == "" and path == "" and tool_path == "" and bytes <= 0 then
        return nil
    end
    return {
        name = name,
        path = path,
        tool_path = tool_path,
        bytes = bytes,
    }
end

local function get_seq_item(seq, index1)
    local ok, v = pcall(function()
        return seq[index1]
    end)
    if ok and v ~= nil then
        return v
    end
    ok, v = pcall(function()
        return seq[index1 - 1]
    end)
    if ok then
        return v
    end
    return nil
end

local function clone_uploads(src)
    local out = {}
    if src == nil then
        return out
    end

    if type(src) == "table" then
        for _, item in ipairs(src) do
            local row = normalize_upload_item(item)
            if row then
                out[#out + 1] = row
            end
        end
        return out
    end

    -- Lupa may pass Python list/dict proxies as userdata.
    local len = 0
    local ok_len, len_or_err = pcall(function()
        return #src
    end)
    if ok_len then
        len = math.max(0, math.floor(tonumber(len_or_err) or 0))
    end

    if len > 0 then
        for i = 1, len do
            local row = normalize_upload_item(get_seq_item(src, i))
            if row then
                out[#out + 1] = row
            end
        end
        return out
    end

    -- Fallback probe (for proxies without reliable length).
    for i = 1, 64 do
        local item = get_seq_item(src, i)
        if item == nil then
            break
        end
        local row = normalize_upload_item(item)
        if row then
            out[#out + 1] = row
        end
    end
    return out
end

function M.new_state(args)
    args = args or {}
    local user_input = util.trim(args.user_input or "")
    local run_id = util.trim(args.run_id)
    if run_id == "" then
        run_id = util.new_run_id()
    end

    return {
        state_version = M.STATE_VERSION,
        run_id = run_id,
        input = {
            message = user_input,
            read_only = args.read_only == true,
        },
        uploads = clone_uploads(args.uploads),
        messages = {
            conversation_history = args.conversation_history or {},
            system_prompt = util.trim(args.system_prompt or ""),
            runtime_messages = {},
        },
        agent_loop = {
            remaining_steps = 25,
            pending_tool_calls = {},
            stop_reason = "",
            iteration = 0,
        },
        context = {
            memory_context = "",
            tool_context = "",
            planner_context = "",
        },
        router_decision = {
            route = "respond",
            raw = "",
            reason = "",
        },
        recall = {
            triggered = false,
            context = "",
            score = nil,
        },
        planner = {
            raw = "",
            tool_calls = {},
            errors = {},
            force_reason = "",
        },
        tool_exec = {
            loop_count = 0,
            executed = 0,
            failed = 0,
            executed_total = 0,
            failed_total = 0,
            results = {},
            context_fragments = {},
        },
        repair = {
            attempts = 0,
            max_attempts = 2,
            last_error = "",
        },
        final_response = {
            message = "",
        },
        writeback = {
            facts = {},
            saved = 0,
        },
        metrics = {
            started_at_ms = util.now_ms(),
            finished_at_ms = nil,
            node_durations_ms = {},
        },
        checkpoint_meta = {
            seq = 0,
            last_node = "",
        },
    }
end

function M.validate(state)
    if type(state) ~= "table" then
        return false, "state_not_table"
    end
    for _, key in ipairs(REQUIRED_KEYS) do
        if state[key] == nil then
            return false, "missing_" .. tostring(key)
        end
    end
    if tostring(state.state_version or "") ~= M.STATE_VERSION then
        return false, "invalid_state_version"
    end
    if util.trim(state.run_id or "") == "" then
        return false, "empty_run_id"
    end
    return true
end

function M.assert_valid(state)
    local ok, err = M.validate(state)
    if not ok then
        error("[GraphState] invalid: " .. tostring(err))
    end
end

return M
