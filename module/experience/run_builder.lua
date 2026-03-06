local retriever = require("module.experience.retriever")
local policy = require("module.experience.policy")

local M = {}

local WRITE_TOOLS = {
    write_file = true,
    apply_patch = true,
    exec_command = true,
}

local READ_TOOLS = {
    list_files = true,
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
    code_outline = true,
    project_structure = true,
    code_symbols = true,
}

local FAILURE_STOP_REASONS = {
    model_call_failed = true,
    remaining_steps_exhausted = true,
    tool_loop_max_exceeded = true,
    repair_exhausted = true,
}

local function trim(s)
    return (tostring(s or ""):gsub("^%s*(.-)%s*$", "%1"))
end

local function copy_string_array(raw)
    local out = {}
    for _, item in ipairs(raw or {}) do
        local text = trim(item)
        if text ~= "" then
            out[#out + 1] = text
        end
    end
    return out
end

local function collect_tool_sequence(tool_results)
    local sequence = {}
    local has_read = false
    local has_write = false
    local first_read_idx = nil
    local first_write_idx = nil
    for _, row in ipairs(tool_results or {}) do
        if type(row) == "table" and row.is_control ~= true then
            local tool_name = trim(row.tool)
            if tool_name ~= "" then
                sequence[#sequence + 1] = tool_name
                local idx = #sequence
                if READ_TOOLS[tool_name] then
                    has_read = true
                    if first_read_idx == nil then
                        first_read_idx = idx
                    end
                end
                if WRITE_TOOLS[tool_name] then
                    has_write = true
                    if first_write_idx == nil then
                        first_write_idx = idx
                    end
                end
            end
        end
    end
    local read_before_write = has_write ~= true
        or (first_read_idx ~= nil and first_write_idx ~= nil and first_read_idx < first_write_idx)
    return sequence, has_read, has_write, read_before_write
end

local function detect_success(state)
    local stop_reason = trim((((state or {}).termination or {}).stop_reason) or (((state or {}).agent_loop or {}).stop_reason) or "")
    local final_status = trim((((state or {}).termination or {}).final_status) or (((state or {}).final_response or {}).status) or "")
    local final_text = trim((((state or {}).final_response or {}).message) or "")
    if final_status == "completed" and final_text ~= "" then
        return true
    end
    if stop_reason == "finish_turn_called" and final_text ~= "" then
        return true
    end
    if FAILURE_STOP_REASONS[stop_reason] then
        return false
    end
    return false
end

local function has_recent_episode_context(state)
    local recent = (((state or {}).episode or {}).recent) or {}
    local task_decision = ((((state or {}).task or {}).decision) or {})
    local kind = trim(task_decision.kind or "")
    return (tonumber(recent.count) or 0) > 0 and kind ~= "hard_shift"
end

local function infer_planner_mode(tool_sequence, has_read_ops, has_write_ops, read_before_write)
    if #(tool_sequence or {}) <= 0 then
        return "direct_first"
    end
    if has_write_ops and has_read_ops and read_before_write == true then
        return "evidence_first"
    end
    return "tool_first"
end

local function infer_recall_mode(runtime_recall_mode, recall_triggered, user_input)
    local mode = trim(runtime_recall_mode or "")
    local lower = tostring(user_input or ""):lower()
    if mode == "suppress" then
        return "suppress"
    end
    if lower:find("suppress", 1, true) ~= nil
        or lower:find("no recall", 1, true) ~= nil
        or lower:find("disable recall", 1, true) ~= nil
        or lower:find("不要回忆", 1, true) ~= nil
        or lower:find("不要检索记忆", 1, true) ~= nil then
        return "suppress"
    end
    if recall_triggered == true then
        return "force"
    end
    if mode == "force" then
        return "force"
    end
    return "auto"
end

function M.build_from_state(state)
    local user_input = trim((((state or {}).input or {}).message) or "")
    local features = retriever.extract_query_features(user_input)
    local task_profile = trim((((state or {}).context or {}).task_profile) or policy.detect_task_profile(state) or "general")
    local task_contract = ((((state or {}).task or {}).contract) or ((((state or {}).session or {}).active_task) or {}).contract) or {}
    local read_only = (((state or {}).input or {}).read_only) == true
    local uploads = (((state or {}).uploads) or {})
    local has_uploads = type(uploads) == "table" and #uploads > 0
    local contract_shape = policy.build_contract_shape(task_profile, user_input, task_contract, read_only)
    local policy_key = policy.build_policy_key({
        task_profile = task_profile,
        task_type = features.task_type,
        domain = features.domain,
        language = features.language,
        read_only = read_only,
        has_uploads = has_uploads,
        contract_shape = contract_shape,
    })

    local tool_exec = ((state or {}).tool_exec) or {}
    local repair = ((state or {}).repair) or {}
    local runtime_policy = (((state or {}).experience) or {}).runtime_policy or policy.default_runtime_policy()
    local success = detect_success(state)
    local tool_sequence, has_read_ops, has_write_ops, read_before_write = collect_tool_sequence(tool_exec.all_results or tool_exec.results or {})
    local mode = (#tool_sequence > 0) and "tool" or "direct"
    local recall_triggered = (((state or {}).recall or {}).triggered) == true
    local episode_continuity_used = has_recent_episode_context(state)
    local files_read_set = ((((state or {}).working_memory) or {}).files_read_set) or {}
    local candidate_ids = copy_string_array((((((state or {}).experience) or {}).retrieved) or {}).ids or {})
    local planner_mode_used = infer_planner_mode(tool_sequence, has_read_ops, has_write_ops, read_before_write)
    local recall_mode_used = infer_recall_mode(((((runtime_policy or {}).recall) or {}).mode), recall_triggered, user_input)
    local repair_mode_used = trim((((runtime_policy or {}).repair) or {}).mode or "normal")
    if repair_mode_used == "" then
        repair_mode_used = "normal"
    end
    local evidence_needed = has_read_ops
    for _, enabled in pairs(files_read_set) do
        if enabled then
            evidence_needed = true
            break
        end
    end

    local state_signature = {
        task_profile = task_profile ~= "" and task_profile or "general",
        task_type = trim(features.task_type or "general"),
        domain = trim(features.domain or "general"),
        language = trim(features.language or "unknown"),
        read_only = read_only,
        has_uploads = has_uploads,
        contract_shape = contract_shape,
    }

    local repair_attempts = tonumber(repair.attempts) or 0
    local loop_count = tonumber(tool_exec.loop_count) or 0
    local tool_executed = tonumber(tool_exec.executed_total) or tonumber(tool_exec.executed) or 0
    local tool_failed = tonumber(tool_exec.failed_total) or tonumber(tool_exec.failed) or 0

    return {
        kind = "graph_policy_observation",
        policy_key = policy_key,
        family_key = policy_key,
        task_profile = task_profile ~= "" and task_profile or "general",
        task_type = trim(features.task_type or "general"),
        domain = trim(features.domain or "general"),
        language = trim(features.language or "unknown"),
        read_only = read_only,
        has_uploads = has_uploads,
        contract_shape = contract_shape,
        context_signature = state_signature,
        state_signature = state_signature,
        candidate_ids = candidate_ids,
        success = success,
        mode = mode,
        planner_mode_used = planner_mode_used,
        recall_mode_used = recall_mode_used,
        repair_mode_used = repair_mode_used,
        tool_sequence = tool_sequence,
        recall_triggered = recall_triggered,
        episode_continuity_used = episode_continuity_used,
        evidence_needed = evidence_needed,
        has_write_ops = has_write_ops,
        read_before_write = read_before_write,
        force_read_before_write_used = has_write_ops == true and read_before_write == true,
        repair_attempts = repair_attempts,
        loop_count = loop_count,
        macro_patch = runtime_policy,
        cost_metrics = {
            loop_count = loop_count,
            repair_attempts = repair_attempts,
            tool_executed = tool_executed,
            tool_failed = tool_failed,
        },
        stop_reason = trim((((state or {}).termination or {}).stop_reason) or (((state or {}).agent_loop or {}).stop_reason) or ""),
    }
end

return M
