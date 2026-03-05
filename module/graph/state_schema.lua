local util = require("module.graph.util")

local M = {}

M.STATE_VERSION = "v1"

local REQUIRED_KEYS = {
    "state_version",
    "run_id",
    "input",
    "uploads",
    "messages",
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

local function clone_uploads(src)
    local out = {}
    for _, item in ipairs(src or {}) do
        if type(item) == "table" then
            out[#out + 1] = {
                name = tostring(item.name or ""),
                path = tostring(item.path or ""),
                bytes = tonumber(item.bytes) or 0,
            }
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
