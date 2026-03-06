local util = require("module.graph.util")
local config = require("module.config")
local memory = require("module.memory.store")

local M = {}

local ALLOWED_MODES = {
    companion = true,
    project = true,
    coding = true,
    knowledge = true,
    meta = true,
}

local MODE_PRESETS = {
    companion = {
        preferred_type = "User",
        allowed_types = { "User", "Person", "Preference", "Status", "Fact" },
        blocked_types = { "Artifact", "Project", "Constraint", "Decision" },
    },
    project = {
        preferred_type = "Project",
        allowed_types = { "Project", "Artifact", "Constraint", "Decision", "Status", "Fact" },
        blocked_types = { "User", "Person", "Preference" },
    },
    coding = {
        preferred_type = "Artifact",
        allowed_types = { "Artifact", "Project", "Constraint", "Decision", "Status", "Concept", "Fact" },
        blocked_types = { "User", "Person", "Preference" },
    },
    knowledge = {
        preferred_type = "Concept",
        allowed_types = { "Concept", "Fact", "Decision", "Status" },
        blocked_types = { "User", "Person", "Preference" },
    },
    meta = {
        preferred_type = "Status",
        allowed_types = { "Status", "Decision", "Project", "Artifact", "Constraint", "Fact" },
        blocked_types = { "User", "Person", "Preference" },
    },
}

local POLICY_PROMPT = [[
You are the recall policy compiler for a memory-first agent.
Return exactly one Lua table on a single line.

Allowed modes:
- companion
- project
- coding
- knowledge
- meta

Schema:
{
  mode="project",
  confidence=0.82,
  force=true,
  suppress=false,
  preferred_type="Project",
  allowed_types={"Project","Artifact","Constraint","Decision","Status","Fact"},
  blocked_types={"User","Person","Preference"},
  reason="short reason"
}

Rules:
- Choose the recall view needed before planning this turn.
- Use task continuity, active task contract, artifact continuity, working memory, and episode continuity.
- Do not rely on shallow keyword matching.
- force=true only when memory recall should run before planning even if the user did not explicitly ask.
- suppress=true only when long-term memory is clearly irrelevant for this turn.
- preferred_type must be empty or one of allowed_types.
- allowed_types and blocked_types must use only the configured types.
- coding mode is for code/files/artifacts/debugging/workspace execution.
- project mode is for long-running task continuity, decisions, commitments, and pending work.
- knowledge mode is for conceptual or explanatory turns not tied to a specific artifact.
- meta mode is for status/progress/process questions about the active task.
- companion mode is for user identity, relationship, or preference continuity.
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

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
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

local function lower_text(v)
    return tostring(v or ""):lower()
end

local function clone_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function normalize_type_list(raw, fallback)
    local filter = memory.build_type_filter(raw)
    local out = {}
    local seen = {}

    if filter then
        for _, type_name in ipairs(memory.get_allowed_type_names()) do
            if filter[type_name] and not seen[type_name] then
                seen[type_name] = true
                out[#out + 1] = type_name
            end
        end
    end

    if #out <= 0 then
        for _, type_name in ipairs(fallback or {}) do
            local canonical = memory.match_type_name(type_name) or memory.normalize_type_name(type_name)
            if canonical and not seen[canonical] then
                seen[canonical] = true
                out[#out + 1] = canonical
            end
        end
    end

    return out
end

local function list_to_lookup(rows)
    local out = {}
    for _, item in ipairs(rows or {}) do
        out[tostring(item)] = true
    end
    return out
end

local function summarize_contract(contract)
    contract = type(contract) == "table" and contract or {}
    local lines = {
        string.format("goal=%s", tostring(contract.goal or "")),
    }
    for _, item in ipairs(contract.deliverables or {}) do
        lines[#lines + 1] = "deliverable=" .. tostring(item)
    end
    for _, item in ipairs(contract.acceptance_criteria or {}) do
        lines[#lines + 1] = "acceptance=" .. tostring(item)
    end
    for _, item in ipairs(contract.non_goals or {}) do
        lines[#lines + 1] = "non_goal=" .. tostring(item)
    end
    return table.concat(lines, "\n")
end

local function summarize_working_memory(working_memory)
    local row = type(working_memory) == "table" and working_memory or {}
    local lines = {
        string.format("current_plan=%s", tostring(row.current_plan or "")),
        string.format("plan_step_index=%s", tostring(row.plan_step_index or 0)),
        string.format("files_read=%d", count_table_entries(row.files_read_set)),
        string.format("files_written=%d", count_table_entries(row.files_written_set)),
    }
    if util.trim(row.last_tool_batch_summary or "") ~= "" then
        lines[#lines + 1] = "last_tool_batch=" .. util.utf8_take(tostring(row.last_tool_batch_summary or ""), 240)
    end
    if util.trim(row.last_repair_error or "") ~= "" then
        lines[#lines + 1] = "last_repair_error=" .. util.utf8_take(tostring(row.last_repair_error or ""), 200)
    end
    return table.concat(lines, "\n")
end

local function fallback_policy(state)
    local task = ((state or {}).task) or {}
    local decision = (task.decision or {})
    local active_task = (((state or {}).session or {}).active_task) or {}
    local working_memory = ((state or {}).working_memory) or {}
    local episode_summary = util.trim((((state or {}).episode or {}).recent or {}).summary or "")

    local profile = lower_text(active_task.profile or (((state or {}).planner or {}).task_profile) or "")
    local has_plan = util.trim(working_memory.current_plan or "") ~= ""
    local has_episode = episode_summary ~= ""
    local has_artifact_state = count_table_entries(working_memory.files_read_set) > 0
        or count_table_entries(working_memory.files_written_set) > 0
    local kind = util.trim(decision.kind or "")

    local mode = "knowledge"
    if profile == "companion" or profile == "relationship" then
        mode = "companion"
    elseif kind == "meta_turn" then
        mode = "meta"
    elseif profile == "workspace" or profile == "coding" or has_artifact_state then
        mode = "coding"
    elseif kind == "same_task_step" or kind == "same_task_refine" then
        mode = "project"
    end

    local preset = MODE_PRESETS[mode] or MODE_PRESETS.project
    return {
        mode = mode,
        confidence = 0.35,
        force = (kind ~= "hard_shift") and (has_plan or has_episode or has_artifact_state),
        suppress = false,
        preferred_type = preset.preferred_type,
        allowed_types = clone_array(preset.allowed_types),
        blocked_types = clone_array(preset.blocked_types),
        reason = "fallback_recall_policy",
        raw = "",
        decided = true,
    }
end

local function build_prompt(state)
    local task = ((state or {}).task) or {}
    local decision = (task.decision or {})
    local active_task = (((state or {}).session or {}).active_task) or {}
    local contract = task.contract or active_task.contract or {}
    local working_memory = ((state or {}).working_memory) or {}
    local episode_summary = util.utf8_take(util.trim((((state or {}).episode or {}).recent or {}).summary or ""), 800)
    local latest_turn = tostring((((state or {}).input or {}).message) or "")

    local preset_lines = {}
    for _, mode in ipairs({ "companion", "project", "coding", "knowledge", "meta" }) do
        local preset = MODE_PRESETS[mode]
        preset_lines[#preset_lines + 1] = string.format(
            "%s preferred=%s allowed=%s blocked=%s",
            mode,
            tostring(preset.preferred_type or ""),
            table.concat(preset.allowed_types or {}, ", "),
            table.concat(preset.blocked_types or {}, ", ")
        )
    end

    return table.concat({
        POLICY_PROMPT,
        "",
        "[AllowedTypes]",
        util.encode_lua_value(memory.get_allowed_type_names(), 0),
        "",
        "[ModePresets]",
        table.concat(preset_lines, "\n"),
        "",
        "[LatestUserTurn]",
        latest_turn,
        "",
        "[TaskDecision]",
        string.format("kind=%s", tostring(decision.kind or "")),
        string.format("confidence=%.2f", tonumber(decision.confidence) or 0),
        string.format("changed=%s", tostring(decision.changed == true)),
        string.format("updated_goal=%s", tostring(decision.updated_goal or "")),
        "",
        "[ActiveTask]",
        string.format("task_id=%s", tostring(active_task.task_id or "")),
        string.format("goal=%s", tostring(active_task.goal or "")),
        string.format("status=%s", tostring(active_task.status or "")),
        string.format("profile=%s", tostring(active_task.profile or "")),
        "",
        "[TaskContract]",
        summarize_contract(contract),
        "",
        "[WorkingMemory]",
        summarize_working_memory(working_memory),
        "",
        "[RecentEpisodes]",
        episode_summary,
    }, "\n")
end

local function parse_policy_output(raw)
    local parsed, err = util.parse_lua_table_literal(util.trim(raw or ""))
    if not parsed then
        return nil, err
    end
    return parsed, nil
end

local function normalize_policy(parsed, fallback, raw)
    fallback = type(fallback) == "table" and fallback or fallback_policy({})
    parsed = type(parsed) == "table" and parsed or {}

    local mode = util.trim(parsed.mode or fallback.mode or "")
    if not ALLOWED_MODES[mode] then
        mode = fallback.mode or "project"
    end
    local preset = MODE_PRESETS[mode] or MODE_PRESETS.project

    local allowed_types = normalize_type_list(parsed.allowed_types, preset.allowed_types)
    local blocked_types = normalize_type_list(parsed.blocked_types, preset.blocked_types)
    local allowed_lookup = list_to_lookup(allowed_types)

    local filtered_blocked = {}
    for _, type_name in ipairs(blocked_types) do
        if not allowed_lookup[type_name] then
            filtered_blocked[#filtered_blocked + 1] = type_name
        end
    end
    blocked_types = filtered_blocked

    local preferred_type = memory.match_type_name(parsed.preferred_type) or memory.match_type_name(fallback.preferred_type) or ""
    if preferred_type ~= "" and not allowed_lookup[preferred_type] then
        preferred_type = ""
    end
    if preferred_type == "" then
        local preset_pref = memory.match_type_name(preset.preferred_type)
        if preset_pref and allowed_lookup[preset_pref] then
            preferred_type = preset_pref
        else
            preferred_type = tostring(allowed_types[1] or "")
        end
    end

    return {
        mode = mode,
        confidence = clamp(tonumber(parsed.confidence) or tonumber(fallback.confidence) or 0.35, 0.0, 1.0),
        force = parsed.force == true,
        suppress = parsed.suppress == true,
        preferred_type = preferred_type,
        allowed_types = allowed_types,
        blocked_types = blocked_types,
        reason = util.utf8_take(util.trim(parsed.reason or fallback.reason or ""), 160),
        raw = util.trim(raw or ""),
        decided = true,
    }
end

local function build_policy_context(policy)
    policy = type(policy) == "table" and policy or {}
    return table.concat({
        "[RecallPolicy]",
        string.format("mode=%s", tostring(policy.mode or "")),
        string.format("confidence=%.2f", tonumber(policy.confidence) or 0),
        string.format("force=%s", tostring(policy.force == true)),
        string.format("suppress=%s", tostring(policy.suppress == true)),
        string.format("preferred_type=%s", tostring(policy.preferred_type or "")),
        string.format("allowed_types=%s", table.concat(policy.allowed_types or {}, ", ")),
        string.format("blocked_types=%s", table.concat(policy.blocked_types or {}, ", ")),
        string.format("reason=%s", tostring(policy.reason or "")),
    }, "\n")
end

function M.run(state, _ctx)
    state.recall = state.recall or {}
    state.context = state.context or {}

    local fallback = fallback_policy(state)
    local runtime = _G.py_pipeline
    local policy = fallback

    if runtime ~= nil and has_method(runtime, "generate_chat_sync") then
        local cfg = ((graph_cfg().recall or {}).policy or {})
        local prompt = build_prompt(state)
        local ok, raw = pcall(function()
            return runtime:generate_chat_sync(
                { { role = "user", content = prompt } },
                {
                    max_tokens = math.max(96, math.floor(tonumber(cfg.max_tokens) or 192)),
                    temperature = tonumber(cfg.temperature) or 0,
                    seed = tonumber(cfg.seed) or 23,
                }
            )
        end)

        if ok then
            local parsed = parse_policy_output(raw)
            if parsed then
                policy = normalize_policy(parsed, fallback, raw)
            else
                policy = fallback
                policy.raw = util.trim(tostring(raw or ""))
            end
        end
    end

    state.recall.policy = policy
    state.context.policy_context = build_policy_context(policy)
    state.context.applied_policy = tostring(policy.mode or "")
    return state
end

return M
