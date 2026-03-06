local util = require("module.graph.util")

local M = {}

local META_PATTERNS = {
    "现在进度",
    "当前进度",
    "进度怎样",
    "做到哪",
    "做到哪了",
    "当前任务",
    "什么任务",
    "你在做什么",
    "目前状态",
    "上一步",
    "刚才做了什么",
    "总结一下",
    "汇报一下",
    "状态如何",
    "status",
    "current task",
    "progress",
    "what are you doing",
    "what did you do",
}

local REFINE_PATTERNS = {
    "也",
    "顺手",
    "顺便",
    "另外",
    "再把",
    "加上",
    "补上",
    "改成",
    "换成",
    "改为",
    "不要",
    "别",
    "记得",
    "also",
    "plus",
    "instead",
    "change to",
    "add",
    "include",
    "remember to",
    "don't",
}

local function contains_any(text, patterns)
    local s = tostring(text or "")
    for _, pattern in ipairs(patterns or {}) do
        local candidate = tostring(pattern or "")
        if candidate ~= "" and s:find(candidate, 1, true) ~= nil then
            return true, candidate
        end
    end
    return false, ""
end

local function split_turn_units(text)
    local raw = util.trim(text or "")
    if raw == "" then
        return {}
    end

    local units = {}
    for part in raw:gmatch("[^\n]+") do
        local unit = util.trim(part)
        if unit ~= "" then
            units[#units + 1] = unit
        end
    end

    if #units == 0 then
        units[1] = raw
    end

    return units
end

local function blank_working_memory()
    return {
        current_plan = "",
        plan_step_index = 0,
        files_read_set = {},
        files_written_set = {},
        patches_applied = {},
        command_history_tail = {},
        last_tool_batch_summary = "",
        last_repair_error = "",
    }
end

local function has_working_memory_evidence(working_memory)
    local memory = type(working_memory) == "table" and working_memory or {}
    if util.trim(memory.current_plan or "") ~= "" then
        return true
    end
    if (tonumber(memory.plan_step_index) or 0) > 0 then
        return true
    end
    if util.trim(memory.last_tool_batch_summary or "") ~= "" then
        return true
    end
    if util.trim(memory.last_repair_error or "") ~= "" then
        return true
    end
    if next(memory.files_read_set or {}) ~= nil then
        return true
    end
    if next(memory.files_written_set or {}) ~= nil then
        return true
    end
    if next(memory.patches_applied or {}) ~= nil then
        return true
    end
    if next(memory.command_history_tail or {}) ~= nil then
        return true
    end
    return false
end

local function reset_episode_recent(state)
    state.episode = state.episode or {}
    state.episode.recent = {
        items = {},
        summary = "",
        count = 0,
        latest_episode_id = "",
    }
end

local function ensure_open_status(status)
    local current = util.trim(status or "")
    if current == "" or current == "completed" or current == "failed" then
        return "open"
    end
    if current == "waiting_user" then
        return "open"
    end
    return current
end

local function merge_goal(old_goal, new_input)
    local prev = util.trim(old_goal or "")
    local current = util.trim(new_input or "")
    if prev == "" then
        return current
    end
    if current == "" then
        return prev
    end

    local prev_lower = prev:lower()
    local current_lower = current:lower()
    if prev_lower == current_lower or prev_lower:find(current_lower, 1, true) ~= nil then
        return prev
    end
    if current_lower:find(prev_lower, 1, true) ~= nil then
        return current
    end
    return util.utf8_take(prev .. " | refine: " .. current, 320)
end

local function build_task_context(decision, active_task)
    local lines = {
        "[TaskDecision]",
        string.format("kind=%s", tostring((decision or {}).kind or "")),
        string.format("confidence=%.2f", tonumber((decision or {}).confidence) or 0),
        string.format("changed=%s", tostring((decision or {}).changed == true)),
    }

    local previous_task_id = util.trim((decision or {}).previous_task_id or "")
    if previous_task_id ~= "" then
        lines[#lines + 1] = "previous_task_id=" .. previous_task_id
    end

    for _, reason in ipairs((decision or {}).reasons or {}) do
        if util.trim(reason) ~= "" then
            lines[#lines + 1] = "reason=" .. tostring(reason)
        end
    end

    lines[#lines + 1] = string.format("active_task_id=%s", tostring((active_task or {}).task_id or ""))
    lines[#lines + 1] = string.format("active_goal=%s", tostring((active_task or {}).goal or ""))
    return table.concat(lines, "\n")
end

local function detect_meta_turn(user_input, active_task)
    if util.trim((active_task or {}).goal or "") == "" then
        return false, ""
    end
    return contains_any(tostring(user_input or ""):lower(), META_PATTERNS)
end

local function detect_refine_turn(user_input)
    return contains_any(tostring(user_input or ""):lower(), REFINE_PATTERNS)
end

local function is_terminal_status(status)
    local current = util.trim(status or ""):lower()
    return current == "completed" or current == "failed" or current == "abandoned"
end

local function is_resume_context(recovery)
    local resumed = ((recovery or {}).resumed_from_checkpoint) == true
    local run_id = util.trim(((recovery or {}).resumable_run_id) or "")
    return resumed and run_id ~= ""
end

local function looks_like_synthetic_seed(user_input, active_task, working_memory)
    local current_input = util.trim(user_input or "")
    if current_input == "" then
        return false
    end

    local task = type(active_task) == "table" and active_task or {}
    if util.trim(task.task_id or "") ~= "" then
        return false
    end
    if util.trim(task.goal or "") ~= current_input then
        return false
    end
    if util.trim(task.last_user_message or "") ~= current_input then
        return false
    end
    if util.trim(task.carryover_summary or "") ~= "" then
        return false
    end
    if util.trim(task.profile or "") ~= "" then
        return false
    end
    return not has_working_memory_evidence(working_memory)
end

local function classify_turn(user_input, active_task, working_memory, recovery)
    local reasons = {}
    local previous_goal = util.trim((active_task or {}).goal or "")
    local previous_task_id = util.trim((active_task or {}).task_id or "")
    local previous_status = util.trim((active_task or {}).status or "")
    local resume_context = is_resume_context(recovery)
    local has_previous_task = (previous_goal ~= "" or previous_task_id ~= "")
        and not looks_like_synthetic_seed(user_input, active_task, working_memory)
    local explicit_continue = util.is_continue_request(user_input)

    if not has_previous_task then
        return {
            kind = "hard_shift",
            confidence = 0.55,
            reasons = { "no_active_task" },
            changed = true,
            previous_task_id = previous_task_id,
            previous_goal = previous_goal,
            previous_status = previous_status,
        }
    end

    local is_meta, meta_reason = detect_meta_turn(user_input, active_task)
    if is_meta then
        reasons[#reasons + 1] = "meta:" .. tostring(meta_reason)
        return {
            kind = "meta_turn",
            confidence = 0.92,
            reasons = reasons,
            changed = false,
            previous_task_id = previous_task_id,
            previous_goal = previous_goal,
            previous_status = previous_status,
        }
    end

    if is_terminal_status(previous_status) and not explicit_continue then
        reasons[#reasons + 1] = "terminal_task_boundary"
        return {
            kind = "hard_shift",
            confidence = 0.90,
            reasons = reasons,
            changed = true,
            previous_task_id = previous_task_id,
            previous_goal = previous_goal,
            previous_status = previous_status,
        }
    end

    local continue_like = util.should_continue_task(
        user_input,
        active_task,
        working_memory,
        { has_resumable_run = resume_context }
    )

    if continue_like then
        local is_refine, refine_reason = detect_refine_turn(user_input)
        if explicit_continue then
            reasons[#reasons + 1] = "explicit_continue"
        else
            reasons[#reasons + 1] = "implicit_followup"
        end
        if resume_context then
            reasons[#reasons + 1] = "resumed_checkpoint"
        end

        if is_refine then
            reasons[#reasons + 1] = "refine:" .. tostring(refine_reason)
            return {
                kind = "same_task_refine",
                confidence = explicit_continue and 0.90 or 0.84,
                reasons = reasons,
                changed = true,
                previous_task_id = previous_task_id,
                previous_goal = previous_goal,
                previous_status = previous_status,
            }
        end

        return {
            kind = "same_task_step",
            confidence = explicit_continue and 0.95 or 0.82,
            reasons = reasons,
            changed = false,
            previous_task_id = previous_task_id,
            previous_goal = previous_goal,
            previous_status = previous_status,
        }
    end

    reasons[#reasons + 1] = "task_boundary_reset"
    return {
        kind = "hard_shift",
        confidence = 0.86,
        reasons = reasons,
        changed = true,
        previous_task_id = previous_task_id,
        previous_goal = previous_goal,
        previous_status = previous_status,
    }
end

function M.run(state, _ctx)
    state.session = state.session or { active_task = {} }
    state.session.active_task = state.session.active_task or {}
    state.working_memory = state.working_memory or blank_working_memory()
    state.task = state.task or {}

    local user_input = util.trim((((state or {}).input or {}).message) or "")
    local active_task = state.session.active_task
    local previous_task = util.shallow_copy(active_task)
    local decision = classify_turn(user_input, previous_task, state.working_memory or {}, state.recovery or {})

    if decision.kind == "same_task_step" then
        active_task.last_user_message = user_input
        active_task.status = ensure_open_status(active_task.status)
    elseif decision.kind == "same_task_refine" then
        active_task.goal = merge_goal(active_task.goal, user_input)
        active_task.last_user_message = user_input
        active_task.status = ensure_open_status(active_task.status)
    elseif decision.kind == "meta_turn" then
        active_task.last_user_message = user_input
        if util.trim(active_task.status or "") == "" then
            active_task.status = "open"
        end
    else
        state.session.active_task = {
            task_id = "task_" .. tostring(state.run_id or util.new_run_id()),
            goal = user_input,
            status = "open",
            carryover_summary = "",
            last_user_message = user_input,
            profile = "",
            last_episode_id = "",
        }
        state.working_memory = blank_working_memory()
        reset_episode_recent(state)
        active_task = state.session.active_task
    end

    state.task.turn_units = split_turn_units(user_input)
    state.task.decision = decision
    state.context = state.context or {}
    state.context.task_context = build_task_context(decision, active_task)
    return state
end

return M
