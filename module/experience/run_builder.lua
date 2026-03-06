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

local function collect_tool_sequence(tool_results)
    local sequence = {}
    local has_read = false
    local has_write = false
    for _, row in ipairs(tool_results or {}) do
        if type(row) == "table" and row.is_control ~= true then
            local tool_name = trim(row.tool)
            if tool_name ~= "" then
                sequence[#sequence + 1] = tool_name
                if READ_TOOLS[tool_name] then
                    has_read = true
                end
                if WRITE_TOOLS[tool_name] then
                    has_write = true
                end
            end
        end
    end
    return sequence, has_read, has_write
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
    local success = detect_success(state)
    local tool_sequence, has_read_ops, has_write_ops = collect_tool_sequence(tool_exec.all_results or tool_exec.results or {})
    local mode = (#tool_sequence > 0) and "tool" or "direct"
    local recall_triggered = (((state or {}).recall or {}).triggered) == true
    local episode_continuity_used = has_recent_episode_context(state)
    local files_read_set = ((((state or {}).working_memory) or {}).files_read_set) or {}
    local evidence_needed = has_read_ops
    for _, enabled in pairs(files_read_set) do
        if enabled then
            evidence_needed = true
            break
        end
    end

    return {
        kind = "graph_policy_observation",
        policy_key = policy_key,
        task_profile = task_profile ~= "" and task_profile or "general",
        task_type = trim(features.task_type or "general"),
        domain = trim(features.domain or "general"),
        language = trim(features.language or "unknown"),
        read_only = read_only,
        has_uploads = has_uploads,
        contract_shape = contract_shape,
        context_signature = {
            task_profile = task_profile ~= "" and task_profile or "general",
            task_type = trim(features.task_type or "general"),
            domain = trim(features.domain or "general"),
            language = trim(features.language or "unknown"),
            read_only = read_only,
            has_uploads = has_uploads,
            contract_shape = contract_shape,
        },
        success = success,
        mode = mode,
        tool_sequence = tool_sequence,
        recall_triggered = recall_triggered,
        episode_continuity_used = episode_continuity_used,
        evidence_needed = evidence_needed,
        has_write_ops = has_write_ops,
        read_before_write = has_write_ops ~= true or has_read_ops == true,
        repair_attempts = tonumber(repair.attempts) or 0,
        loop_count = tonumber(tool_exec.loop_count) or 0,
        stop_reason = trim((((state or {}).termination or {}).stop_reason) or (((state or {}).agent_loop or {}).stop_reason) or ""),
    }
end

return M
