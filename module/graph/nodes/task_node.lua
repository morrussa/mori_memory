local util = require("module.graph.util")
local config = require("module.config")
local state_schema = require("module.graph.state_schema")

local M = {}

local ALLOWED_KINDS = {
    same_task_step = true,
    same_task_refine = true,
    hard_shift = true,
    meta_turn = true,
}

local DECISION_PROMPT = [[
You are the task continuity classifier for a single-session tool agent.
Return exactly one Lua table on a single line.

Allowed kinds:
- same_task_step
- same_task_refine
- hard_shift
- meta_turn

Schema:
{
  kind="same_task_step",
  confidence=0.91,
  reason="short reason",
  updated_goal="",
  deliverables={"..."},
  acceptance_criteria={"..."},
  non_goals={"..."}
}

Rules:
- Use same_task_step when the latest user turn should continue executing the current task contract.
- Use same_task_refine when the latest user turn changes or adds constraints to the current task contract.
- Use meta_turn when the latest user turn asks about status, progress, or what the agent is doing, without changing the contract.
- Use hard_shift when the latest user turn starts a new task or replaces the current contract.
- Rely on task contract continuity, unfinished work, artifact continuity, and episode continuity.
- Do not rely on shallow token overlap.
- For same_task_step and meta_turn, preserve the current contract unless a field is clearly missing.
- For same_task_refine and hard_shift, return the full updated contract.
- Confidence must be between 0 and 1.
- Output only the Lua table.
]]

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function has_method(runtime, method_name)
    if runtime == nil then
        return false
    end
    local ok, attr = pcall(function()
        return runtime[method_name]
    end)
    return ok and attr ~= nil
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

local function is_terminal_status(status)
    local current = util.trim(status or ""):lower()
    return current == "completed" or current == "failed" or current == "abandoned"
end

local function count_table_entries(tbl)
    local count = 0
    for _, enabled in pairs(tbl or {}) do
        if enabled then
            count = count + 1
        end
    end
    return count
end

local function summarize_contract(contract)
    local normalized = state_schema.normalize_task_contract(contract, "")
    local lines = {
        string.format("goal=%s", tostring(normalized.goal or "")),
    }
    for _, item in ipairs(normalized.deliverables or {}) do
        lines[#lines + 1] = "deliverable=" .. tostring(item)
    end
    for _, item in ipairs(normalized.acceptance_criteria or {}) do
        lines[#lines + 1] = "acceptance=" .. tostring(item)
    end
    for _, item in ipairs(normalized.non_goals or {}) do
        lines[#lines + 1] = "non_goal=" .. tostring(item)
    end
    return table.concat(lines, "\n")
end

local function summarize_working_memory(working_memory)
    local memory = type(working_memory) == "table" and working_memory or {}
    local lines = {
        string.format("current_plan=%s", tostring(memory.current_plan or "")),
        string.format("plan_step_index=%s", tostring(memory.plan_step_index or 0)),
        string.format("files_read=%d", count_table_entries(memory.files_read_set)),
        string.format("files_written=%d", count_table_entries(memory.files_written_set)),
    }
    if util.trim(memory.last_tool_batch_summary or "") ~= "" then
        lines[#lines + 1] = "last_tool_batch=" .. util.utf8_take(tostring(memory.last_tool_batch_summary or ""), 280)
    end
    if util.trim(memory.last_repair_error or "") ~= "" then
        lines[#lines + 1] = "last_repair_error=" .. util.utf8_take(tostring(memory.last_repair_error or ""), 220)
    end
    return table.concat(lines, "\n")
end

local function build_task_context(decision, active_task, contract)
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
    lines[#lines + 1] = "[TaskContract]"
    lines[#lines + 1] = summarize_contract(contract)
    return table.concat(lines, "\n")
end

local function normalize_reason_list(raw)
    local out = {}
    if type(raw) == "table" then
        for _, item in ipairs(raw) do
            local text = util.utf8_take(util.trim(item or ""), 120)
            if text ~= "" then
                out[#out + 1] = text
            end
        end
    elseif util.trim(raw or "") ~= "" then
        out[1] = util.utf8_take(util.trim(raw or ""), 120)
    end
    return out
end

local function has_list_items(raw)
    return type(raw) == "table" and #raw > 0
end

local function parse_task_output(raw)
    if type(raw) == "table" then
        return raw, ""
    end

    local text = util.trim(tostring(raw or ""))
    if text == "" then
        return nil, "task_decision_empty"
    end

    local parsed, err = util.parse_lua_table_literal(text)
    if not parsed then
        return nil, tostring(err or "task_decision_parse_failed")
    end
    return parsed, ""
end

local function build_decision_prompt(user_input, active_task, contract, working_memory, opts)
    opts = opts or {}
    local recovery = opts.recovery or {}
    local lines = {
        DECISION_PROMPT,
        "",
        "[TaskDecisionRequest]",
        "[LatestUserTurn]",
        tostring(user_input or ""),
        "",
        "[ActiveTask]",
        string.format("task_id=%s", tostring((active_task or {}).task_id or "")),
        string.format("goal=%s", tostring((active_task or {}).goal or "")),
        string.format("status=%s", tostring((active_task or {}).status or "")),
        string.format("profile=%s", tostring((active_task or {}).profile or "")),
        "",
        "[CurrentTaskContract]",
        summarize_contract(contract),
        "",
        "[WorkingMemory]",
        summarize_working_memory(working_memory),
        "",
        "[EpisodeContinuity]",
        tostring(opts.recent_episode_summary or ""),
        "",
        "[Recovery]",
        string.format("checkpoint_resume_available=%s", tostring(opts.checkpoint_available == true)),
        string.format("resumed_from_checkpoint=%s", tostring(((recovery or {}).resumed_from_checkpoint) == true)),
    }
    return table.concat(lines, "\n")
end

local function build_fallback_contract(kind, current_contract, user_input, updated_goal)
    local fallback_goal = util.trim(updated_goal or "")
    if fallback_goal == "" then
        if kind == "hard_shift" then
            fallback_goal = util.trim(user_input or "")
        else
            fallback_goal = util.trim((current_contract or {}).goal or user_input or "")
        end
    end

    if kind == "hard_shift" then
        return state_schema.normalize_task_contract({
            goal = fallback_goal,
            deliverables = { fallback_goal },
            acceptance_criteria = {
                "Address goal: " .. util.utf8_take(fallback_goal, 120),
                "Provide a result summary to the user",
            },
            non_goals = {},
        }, fallback_goal)
    end

    return state_schema.normalize_task_contract(current_contract or {}, fallback_goal)
end

local function fallback_decision(args)
    local active_task = type(args.active_task) == "table" and args.active_task or {}
    local contract = state_schema.normalize_task_contract(active_task.contract, active_task.goal or args.user_input or "")
    local previous_task_id = util.trim(active_task.task_id or "")
    local previous_goal = util.trim(contract.goal or active_task.goal or "")
    local previous_status = util.trim(active_task.status or "")
    local has_previous_task = previous_task_id ~= "" or previous_goal ~= ""

    if not has_previous_task then
        return {
            kind = "hard_shift",
            confidence = 0.40,
            reasons = { "fallback:no_active_task" },
            changed = true,
            previous_task_id = previous_task_id,
            previous_goal = previous_goal,
            previous_status = previous_status,
            target_task_id = "",
            updated_goal = util.trim(args.user_input or ""),
        }, build_fallback_contract("hard_shift", contract, args.user_input or "", args.user_input or "")
    end

    if is_terminal_status(previous_status) then
        return {
            kind = "hard_shift",
            confidence = 0.45,
            reasons = { "fallback:terminal_task" },
            changed = true,
            previous_task_id = previous_task_id,
            previous_goal = previous_goal,
            previous_status = previous_status,
            target_task_id = "",
            updated_goal = util.trim(args.user_input or ""),
        }, build_fallback_contract("hard_shift", contract, args.user_input or "", args.user_input or "")
    end

    if args.checkpoint_available == true then
        return {
            kind = "same_task_step",
            confidence = 0.35,
            reasons = { "fallback:checkpoint_available" },
            changed = false,
            previous_task_id = previous_task_id,
            previous_goal = previous_goal,
            previous_status = previous_status,
            target_task_id = previous_task_id,
            updated_goal = previous_goal,
        }, build_fallback_contract("same_task_step", contract, args.user_input or "", previous_goal)
    end

    return {
        kind = "same_task_step",
        confidence = 0.30,
        reasons = { "fallback:existing_task" },
        changed = false,
        previous_task_id = previous_task_id,
        previous_goal = previous_goal,
        previous_status = previous_status,
        target_task_id = previous_task_id,
        updated_goal = previous_goal,
    }, build_fallback_contract("same_task_step", contract, args.user_input or "", previous_goal)
end

function M.decide(args)
    args = args or {}

    local user_input = util.trim(args.user_input or "")
    local active_task = type(args.active_task) == "table" and args.active_task or {}
    local working_memory = type(args.working_memory) == "table" and args.working_memory or {}
    local recovery = type(args.recovery) == "table" and args.recovery or {}
    local current_contract = state_schema.normalize_task_contract(active_task.contract, active_task.goal or user_input)
    local prompt = build_decision_prompt(user_input, active_task, current_contract, working_memory, {
        recovery = recovery,
        recent_episode_summary = util.utf8_take(util.trim(args.recent_episode_summary or ""), 800),
        checkpoint_available = args.checkpoint_available == true,
    })

    local runtime = _G.py_pipeline
    if runtime == nil or not has_method(runtime, "generate_chat_sync") then
        local fallback, fallback_contract = fallback_decision({
            user_input = user_input,
            active_task = active_task,
            checkpoint_available = args.checkpoint_available == true,
        })
        return fallback, fallback_contract, prompt, "task_runtime_unavailable"
    end

    local cfg = graph_cfg().task or {}
    local params = {
        max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 384)),
        temperature = tonumber(cfg.temperature) or 0,
        seed = tonumber(cfg.seed) or 17,
    }

    local ok, raw = pcall(function()
        return runtime:generate_chat_sync(
            { { role = "user", content = prompt } },
            params
        )
    end)
    if not ok then
        local fallback, fallback_contract = fallback_decision({
            user_input = user_input,
            active_task = active_task,
            checkpoint_available = args.checkpoint_available == true,
        })
        return fallback, fallback_contract, prompt, tostring(raw or "task_model_call_failed")
    end

    local parsed, parse_err = parse_task_output(raw)
    if not parsed then
        local fallback, fallback_contract = fallback_decision({
            user_input = user_input,
            active_task = active_task,
            checkpoint_available = args.checkpoint_available == true,
        })
        return fallback, fallback_contract, prompt, parse_err
    end

    local previous_task_id = util.trim(active_task.task_id or "")
    local previous_goal = util.trim(current_contract.goal or active_task.goal or "")
    local previous_status = util.trim(active_task.status or "")
    local has_previous_task = previous_task_id ~= "" or previous_goal ~= ""

    local kind = util.trim(parsed.kind or "")
    if not ALLOWED_KINDS[kind] then
        kind = has_previous_task and "same_task_step" or "hard_shift"
    end
    if not has_previous_task and kind ~= "hard_shift" then
        kind = "hard_shift"
    end

    local updated_goal = util.utf8_take(util.trim(parsed.updated_goal or parsed.goal or ""), 320)
    if updated_goal == "" then
        if kind == "hard_shift" then
            updated_goal = user_input
        else
            updated_goal = previous_goal
        end
    end

    local contract_seed = {
        goal = updated_goal,
        deliverables = parsed.deliverables,
        acceptance_criteria = parsed.acceptance_criteria,
        non_goals = parsed.non_goals,
    }

    local merged_contract = nil
    if kind == "same_task_step" or kind == "meta_turn" then
        merged_contract = state_schema.normalize_task_contract({
            goal = updated_goal ~= "" and updated_goal or current_contract.goal,
            deliverables = has_list_items(parsed.deliverables) and parsed.deliverables or current_contract.deliverables,
            acceptance_criteria = has_list_items(parsed.acceptance_criteria) and parsed.acceptance_criteria or current_contract.acceptance_criteria,
            non_goals = has_list_items(parsed.non_goals) and parsed.non_goals or current_contract.non_goals,
        }, updated_goal ~= "" and updated_goal or current_contract.goal)
    elseif kind == "same_task_refine" then
        merged_contract = state_schema.normalize_task_contract({
            goal = updated_goal ~= "" and updated_goal or current_contract.goal,
            deliverables = has_list_items(parsed.deliverables) and parsed.deliverables or current_contract.deliverables,
            acceptance_criteria = has_list_items(parsed.acceptance_criteria) and parsed.acceptance_criteria or current_contract.acceptance_criteria,
            non_goals = has_list_items(parsed.non_goals) and parsed.non_goals or current_contract.non_goals,
        }, updated_goal ~= "" and updated_goal or current_contract.goal)
    else
        merged_contract = state_schema.normalize_task_contract(contract_seed, updated_goal ~= "" and updated_goal or user_input)
    end

    local reasons = normalize_reason_list(parsed.reasons)
    if #reasons == 0 then
        reasons = normalize_reason_list(parsed.reason)
    end
    if #reasons == 0 then
        reasons[1] = "model_task_decision"
    end

    local confidence = tonumber(parsed.confidence) or 0.5
    if confidence < 0 then
        confidence = 0
    elseif confidence > 1 then
        confidence = 1
    end

    local decision = {
        kind = kind,
        confidence = confidence,
        reasons = reasons,
        changed = kind == "same_task_refine" or kind == "hard_shift",
        previous_task_id = previous_task_id,
        previous_goal = previous_goal,
        previous_status = previous_status,
        target_task_id = kind == "hard_shift" and "" or previous_task_id,
        updated_goal = tostring(merged_contract.goal or ""),
    }

    return decision, merged_contract, prompt, util.trim(tostring(raw or ""))
end

function M.run(state, _ctx)
    state.session = state.session or { active_task = {} }
    state.session.active_task = state.session.active_task or {}
    state.working_memory = state.working_memory or blank_working_memory()
    state.task = state.task or {}

    local user_input = util.trim((((state or {}).input or {}).message) or "")
    local active_task = state.session.active_task
    local previous_task = util.shallow_copy(active_task)
    previous_task.contract = state_schema.normalize_task_contract(active_task.contract, active_task.goal or user_input)

    local decision, contract = M.decide({
        user_input = user_input,
        active_task = previous_task,
        working_memory = state.working_memory or {},
        recovery = state.recovery or {},
        recent_episode_summary = util.trim((((state or {}).episode or {}).recent or {}).summary or ""),
        checkpoint_available = false,
    })

    if decision.kind == "same_task_step" then
        active_task.goal = tostring((contract or {}).goal or active_task.goal or "")
        active_task.contract = contract
        active_task.last_user_message = user_input
        active_task.status = ensure_open_status(active_task.status)
    elseif decision.kind == "same_task_refine" then
        active_task.goal = tostring((contract or {}).goal or user_input)
        active_task.contract = contract
        active_task.last_user_message = user_input
        active_task.status = ensure_open_status(active_task.status)
    elseif decision.kind == "meta_turn" then
        active_task.goal = tostring((contract or {}).goal or active_task.goal or "")
        active_task.contract = contract
        active_task.last_user_message = user_input
        if util.trim(active_task.status or "") == "" then
            active_task.status = "open"
        end
    else
        state.session.active_task = {
            task_id = "task_" .. tostring(state.run_id or util.new_run_id()),
            goal = tostring((contract or {}).goal or user_input),
            status = "open",
            carryover_summary = "",
            last_user_message = user_input,
            profile = "",
            last_episode_id = "",
            contract = contract,
        }
        state.working_memory = blank_working_memory()
        reset_episode_recent(state)
        active_task = state.session.active_task
    end

    state.task.turn_units = split_turn_units(user_input)
    state.task.decision = decision
    state.task.contract = state_schema.normalize_task_contract(contract, active_task.goal or user_input)
    state.context = state.context or {}
    state.context.task_context = build_task_context(decision, active_task, state.task.contract)
    return state
end

return M
