local retriever = require("module.experience.retriever")

local M = {}

local FAILURE_STOP_REASONS = {
    model_call_failed = true,
    remaining_steps_exhausted = true,
    tool_loop_max_exceeded = true,
}

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function trim(s)
    return (tostring(s or ""):gsub("^%s*(.-)%s*$", "%1"))
end

local function collect_successful_tools(tool_results)
    local tools_used = {}
    local tool_sequence = {}

    for _, row in ipairs(tool_results or {}) do
        if type(row) == "table" and row.is_control ~= true and row.ok == true then
            local tool_name = trim(row.tool)
            if tool_name ~= "" then
                tool_sequence[#tool_sequence + 1] = tool_name
                tools_used[tool_name] = (tools_used[tool_name] or 0) + 1
            end
        end
    end

    return tools_used, tool_sequence
end

local function detect_success(state)
    local stop_reason = trim((((state or {}).agent_loop or {}).stop_reason) or "")
    local final_text = trim((((state or {}).final_response or {}).message) or "")

    if stop_reason == "finish_turn_called" then
        return true, stop_reason
    end
    if FAILURE_STOP_REASONS[stop_reason] then
        return false, stop_reason
    end
    if stop_reason == "" and final_text ~= "" then
        return true, "ok"
    end
    return false, (stop_reason ~= "" and stop_reason or "no_final_response")
end

local function compute_utility_prior(success, loop_count, failed_total, repair_attempts, read_evidence_total)
    local score = success and 0.80 or 0.30
    score = score - 0.05 * math.max((tonumber(loop_count) or 0) - 1, 0)
    score = score - 0.08 * math.min(tonumber(failed_total) or 0, 3)
    score = score - 0.06 * math.min(tonumber(repair_attempts) or 0, 2)

    if success and (tonumber(failed_total) or 0) == 0 and (tonumber(repair_attempts) or 0) == 0
        and (tonumber(read_evidence_total) or 0) > 0 then
        score = score + 0.04
    end

    return clamp(score, 0.05, 0.95)
end

function M.build_from_state(state)
    local user_input = trim((((state or {}).input or {}).message) or "")
    local query_features = retriever.extract_query_features(user_input)
    local task_type = trim(query_features.task_type or "general")
    local domain = trim(query_features.domain or task_type)
    if domain == "" then
        domain = "general"
    end
    local language = trim(query_features.language or "unknown")
    if language == "" then
        language = "unknown"
    end

    local read_only = (((state or {}).input or {}).read_only) == true
    local uploads_count = #((((state or {}).uploads) or {}))
    local tool_exec = ((state or {}).tool_exec) or {}
    local repair = ((state or {}).repair) or {}

    local tools_used, tool_sequence = collect_successful_tools(tool_exec.all_results or tool_exec.results or {})
    local path_text = (#tool_sequence > 0) and table.concat(tool_sequence, ">") or "direct"
    local mode = (#tool_sequence > 0) and "tool" or "direct"
    local success, outcome_reason = detect_success(state)
    local repair_flag = ((tonumber(repair.attempts) or 0) > 0) and 1 or 0

    local utility_prior = compute_utility_prior(
        success,
        tool_exec.loop_count,
        tool_exec.failed_total,
        repair.attempts,
        tool_exec.read_evidence_total
    )

    local context_signature = {
        domain = domain,
        language = language,
        read_only = read_only,
        has_uploads = uploads_count > 0,
        mode = mode,
    }

    local success_key = string.format(
        "task=%s|lang=%s|path=%s|repair=%d|mode=%s",
        task_type,
        language,
        path_text,
        repair_flag,
        mode
    )

    local experience = {
        type = success and "success" or "failure",
        task_type = task_type,
        domain = domain,
        language = language,
        context_signature = context_signature,
        tools_used = tools_used,
        description = string.format(
            "task=%s lang=%s path=%s mode=%s stop=%s",
            task_type,
            language,
            path_text,
            mode,
            outcome_reason
        ),
        patterns = {
            {
                key = "path:" .. path_text,
                type = mode,
            },
        },
        lessons = success
            and { "推荐路径:" .. path_text }
            or { "失败条件:stop=" .. outcome_reason .. ", path=" .. path_text },
        success_key = success_key,
        utility_prior = utility_prior,
        success_rate = success and 1.0 or 0.0,
        outcome = {
            success = success,
            reason = outcome_reason,
            mode = mode,
            loop_count = tonumber(tool_exec.loop_count) or 0,
            failed_total = tonumber(tool_exec.failed_total) or 0,
            repair_attempts = tonumber(repair.attempts) or 0,
            read_evidence_total = tonumber(tool_exec.read_evidence_total) or 0,
            uploads_count = uploads_count,
        },
        error_info = success and nil or {
            type = outcome_reason,
            count = math.max(1, tonumber(tool_exec.failed_total) or 0),
        },
    }

    return experience, {
        success = success,
        success_key = success_key,
        path = path_text,
        mode = mode,
    }
end

return M
