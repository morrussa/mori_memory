local util = require("module.graph.util")

local M = {}

M.STATE_VERSION = "v2"

local REQUIRED_KEYS = {
    "state_version",
    "run_id",
    "input",
    "uploads",
    "messages",
    "agent_loop",
    "task",
    "context",
    "router_decision",
    "recall",
    "episode",
    "planner",
    "tool_exec",
    "repair",
    "final_response",
    "writeback",
    "metrics",
    "checkpoint_meta",
    "session",
    "working_memory",
    "termination",
    "recovery",
}

local function normalize_string_list(raw, limit)
    local out = {}
    if type(raw) ~= "table" then
        return out
    end

    local max_items = math.max(0, math.floor(tonumber(limit) or 0))
    if max_items <= 0 then
        return out
    end

    for _, item in ipairs(raw) do
        local text = util.utf8_take(util.trim(item or ""), 180)
        if text ~= "" then
            out[#out + 1] = text
            if #out >= max_items then
                break
            end
        end
    end
    return out
end

local function normalize_task_contract(raw, fallback_goal)
    local contract = type(raw) == "table" and raw or {}
    local goal = util.utf8_take(util.trim(contract.goal or fallback_goal or ""), 320)
    local deliverables = normalize_string_list(contract.deliverables, 6)
    local acceptance_criteria = normalize_string_list(contract.acceptance_criteria, 6)
    local non_goals = normalize_string_list(contract.non_goals, 6)

    if goal ~= "" and #deliverables == 0 then
        deliverables[1] = goal
    end
    if goal ~= "" and #acceptance_criteria == 0 then
        acceptance_criteria[1] = "Address goal: " .. util.utf8_take(goal, 120)
        acceptance_criteria[2] = "Provide a result summary to the user"
    end

    return {
        goal = goal,
        deliverables = deliverables,
        acceptance_criteria = acceptance_criteria,
        non_goals = non_goals,
    }
end

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

    local session_active_task = args.active_task
    if type(session_active_task) ~= "table" then
        session_active_task = {}
    end
    local active_goal = util.trim(session_active_task.goal or user_input)
    local active_status = util.trim(session_active_task.status or "open")
    if active_status == "" then
        active_status = "open"
    end
    local active_contract = normalize_task_contract(session_active_task.contract, active_goal)
    if util.trim(active_contract.goal or "") ~= "" then
        active_goal = util.trim(active_contract.goal or "")
    end

    local restored_memory = args.working_memory
    if type(restored_memory) ~= "table" then
        restored_memory = {}
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
            tool_loop_max = 0,
            pending_tool_calls = {},
            stop_reason = "",
            iteration = 0,
        },
        task = {
            turn_units = { user_input },
            contract = normalize_task_contract(session_active_task.contract, active_goal),
            decision = {
                kind = "",
                confidence = 0,
                reasons = {},
                changed = false,
                previous_task_id = "",
                previous_goal = "",
                previous_status = "",
                target_task_id = "",
                updated_goal = "",
            },
        },
        context = {
            task_context = "",
            memory_context = "",
            policy_context = "",
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
            topic_anchor = "",
            predicted_topics = {},
            predicted_memories = {},
            predicted_nodes = {},
            selected_turns = {},
            selected_memories = {},
            fragments = {},
            adopted_memories = {},
        },
        episode = {
            current = {
                turn_index = 0,
                topic_anchor = "",
            },
            recent = {
                items = {},
                summary = "",
                count = 0,
                latest_episode_id = "",
            },
            writeback = {
                written = false,
                episode_id = "",
            },
        },
        planner = {
            raw = "",
            tool_calls = {},
            errors = {},
            force_reason = "",
            missing_terminal_signal = false,
            task_profile = util.trim(args.task_profile or session_active_task.profile or ""),
        },
        tool_exec = {
            loop_count = 0,
            executed = 0,
            failed = 0,
            executed_total = 0,
            failed_total = 0,
            read_evidence_total = 0,
            results = {},
            all_results = {},
            context_fragments = {},
            truncated_count = 0,
            total_result_chars = 0,
            large_results = {},
            read_files = {},
        },
        repair = {
            attempts = 0,
            max_attempts = 2,
            last_error = "",
            retry_requested = false,
            pending = false,
        },
        final_response = {
            message = "",
        },
        session = {
            mode = "single",
            active_task = {
                task_id = util.trim(session_active_task.task_id or ("task_" .. run_id)),
                goal = active_goal,
                status = active_status,
                carryover_summary = util.trim(session_active_task.carryover_summary or ""),
                last_user_message = user_input,
                profile = util.trim(session_active_task.profile or args.task_profile or ""),
                last_episode_id = util.trim(session_active_task.last_episode_id or ""),
                contract = active_contract,
            },
        },
        working_memory = {
            current_plan = util.trim(restored_memory.current_plan or ""),
            plan_step_index = tonumber(restored_memory.plan_step_index) or 0,
            files_read_set = restored_memory.files_read_set or {},
            files_written_set = restored_memory.files_written_set or {},
            patches_applied = restored_memory.patches_applied or {},
            command_history_tail = restored_memory.command_history_tail or {},
            last_tool_batch_summary = util.trim(restored_memory.last_tool_batch_summary or ""),
            last_repair_error = util.trim(restored_memory.last_repair_error or ""),
        },
        termination = {
            finish_requested = false,
            final_message = "",
            final_status = "",
            stop_reason = "",
        },
        recovery = {
            resumable_run_id = util.trim(args.resumable_run_id or ""),
            last_checkpoint_seq = tonumber(args.last_checkpoint_seq) or 0,
            next_node = util.trim(args.next_node or ""),
            resumed_from_checkpoint = args.resumed_from_checkpoint == true,
        },
        writeback = {
            items = {},
            ingest_strategy = "atomic_fact",
            saved_count = 0,
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
        stream_sink = args.stream_sink,
    }
end

M.normalize_task_contract = normalize_task_contract

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
