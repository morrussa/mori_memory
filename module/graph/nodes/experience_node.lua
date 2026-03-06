local experience = require("module.experience")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function v2_cfg()
    return ((((config.settings or {}).experience or {}).v2) or {})
end

local function v2_enabled()
    local enabled = v2_cfg().enabled
    if enabled == nil then
        return true
    end
    return enabled == true
end

local function v2_hard_fallback_on_error()
    local value = v2_cfg().hard_fallback_on_error
    if value == nil then
        return true
    end
    return value == true
end

local function build_query(state)
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local features = experience.retriever.extract_query_features(user_input)
    local task_profile = util.trim(experience.policy.detect_task_profile(state) or "general")
    local task_contract = ((((state or {}).task or {}).contract) or ((((state or {}).session or {}).active_task) or {}).contract) or {}
    local read_only = (((state or {}).input or {}).read_only) == true
    local uploads = (((state or {}).uploads) or {})
    local has_uploads = type(uploads) == "table" and #uploads > 0
    local contract_shape = experience.policy.build_contract_shape(task_profile, user_input, task_contract, read_only)

    local query = {
        text = user_input,
        task_profile = task_profile ~= "" and task_profile or "general",
        task_type = util.trim(features.task_type or "general"),
        domain = util.trim(features.domain or "general"),
        language = util.trim(features.language or "unknown"),
        read_only = read_only,
        has_uploads = has_uploads,
        contract_shape = contract_shape,
    }
    query.policy_key = experience.policy.build_policy_key(query)
    query.family_key = query.policy_key
    query.context_signature = {
        task_profile = query.task_profile,
        task_type = query.task_type,
        domain = query.domain,
        language = query.language,
        read_only = query.read_only,
        has_uploads = query.has_uploads,
        contract_shape = query.contract_shape,
    }
    return query
end

local function apply_budget_policy(state, runtime_policy)
    state.agent_loop = state.agent_loop or {}
    local current_steps = tonumber(state.agent_loop.remaining_steps) or 25
    local current_loop_max = tonumber(state.agent_loop.tool_loop_max) or 0
    if current_loop_max <= 0 then
        current_loop_max = math.max(1, math.floor(tonumber((((config.settings or {}).graph or {}).tool_loop_max)) or 5))
    end

    local step_delta = tonumber((((runtime_policy or {}).budget) or {}).remaining_steps_delta) or 0
    local loop_delta = tonumber((((runtime_policy or {}).budget) or {}).tool_loop_max_delta) or 0
    state.agent_loop.remaining_steps = clamp(current_steps + step_delta, 4, 48)
    state.agent_loop.tool_loop_max = clamp(current_loop_max + loop_delta, 1, 12)
end

local function collect_candidate_ids(rows)
    local ids = {}
    for _, item in ipairs(rows or {}) do
        if item and item.id then
            ids[#ids + 1] = tostring(item.id)
        end
    end
    return ids
end

local function apply_default_auto_fallback(state, query, reason)
    local runtime_policy = experience.policy.default_runtime_policy()
    local fallback_reason = util.trim(reason or "experience_v2_error")
    local audit = table.concat({
        "version=v2",
        string.format("family=%s", tostring((query or {}).family_key or "")),
        "candidates=0",
        "recommendation=none",
        "fallback=default_auto",
        "reason=" .. fallback_reason,
    }, "\n")

    state.experience = state.experience or {}
    state.experience.version = "v2"
    state.experience.kind = "graph_policy"
    state.experience.query = query or {}
    state.experience.retrieved = {
        items = {},
        ids = {},
        strategy = "fallback_default_auto",
    }
    state.experience.candidates = {}
    state.experience.recommendation = { id = "", confidence = 0, reason = "fallback_default_auto", support = 0, accepted = false }
    state.experience.runtime_policy = runtime_policy
    state.experience.audit = audit
    state.experience.feedback = state.experience.feedback or { effective_ids = {} }
    state.experience.writeback = state.experience.writeback or { written = false, policy_id = "" }
    state.experience.behavior_match = state.experience.behavior_match or { selected_id = "", match_score = 0 }

    state.context = state.context or {}
    state.context.applied_policy = audit
    state.context.policy_context = audit
    state.context.experience_context = audit
    state.context.experience_prior = ""
    return state
end

local function apply_v2(state, query)
    local cfg = v2_cfg()
    local limit = math.max(1, math.floor(tonumber(cfg.candidate_limit) or 5))
    local candidates, meta = experience.retrieve_v2(query, {
        limit = limit,
    })
    meta = type(meta) == "table" and meta or {}
    local recommendation = type(meta.recommendation) == "table" and meta.recommendation or {}
    local min_support = math.max(1, math.floor(tonumber(cfg.min_support_for_recommend) or 3))
    local min_conf = clamp(tonumber(cfg.min_confidence_for_recommend) or 0.72, 0.0, 1.0)

    local runtime_policy = experience.policy.default_runtime_policy()
    local rec_id = util.trim(recommendation.id or "")
    local rec_conf = tonumber(recommendation.confidence) or 0
    local rec_support = tonumber(recommendation.support_count) or 0
    local accepted = rec_id ~= "" and rec_conf >= min_conf and rec_support >= min_support
    if accepted then
        runtime_policy = experience.policy.normalize_runtime_policy(recommendation.macro_patch or recommendation.patch or {})
    end
    runtime_policy = experience.policy.merge_runtime_policies({
        { patch = runtime_policy },
    })

    local policy_audit = experience.policy.summarize_runtime_policy(runtime_policy)
    local lines = {
        "version=v2",
        string.format("family=%s", tostring(meta.family_key or query.family_key or "")),
        string.format("candidates=%d", #(candidates or {})),
    }
    if rec_id ~= "" then
        lines[#lines + 1] = string.format(
            "recommendation=%s confidence=%.3f support=%d accepted=%s",
            rec_id,
            rec_conf,
            rec_support,
            tostring(accepted)
        )
    else
        lines[#lines + 1] = "recommendation=none"
    end
    if util.trim(policy_audit or "") ~= "" then
        lines[#lines + 1] = policy_audit
    end
    local audit = table.concat(lines, "\n")

    apply_budget_policy(state, runtime_policy)

    local candidate_summaries = {}
    for _, row in ipairs(candidates or {}) do
        candidate_summaries[#candidate_summaries + 1] = {
            id = tostring((row or {}).id or ""),
            score = tonumber((row or {}).score) or 0,
            support = tonumber((row or {}).support_count) or 0,
            patch_summary = experience.policy.summarize_runtime_policy(((row or {}).macro_patch) or {}),
            risk_flags = (row or {}).risk_flags or {},
        }
    end
    local prior = experience.compose_planner_prior(candidates, recommendation, meta.failure_warnings or {})

    state.experience = state.experience or {}
    state.experience.version = "v2"
    state.experience.kind = "graph_policy"
    state.experience.query = query
    state.experience.retrieved = {
        items = candidates or {},
        ids = collect_candidate_ids(candidates or {}),
        strategy = tostring(meta.strategy or ""),
    }
    state.experience.candidates = candidate_summaries
    state.experience.recommendation = {
        id = tostring(recommendation.id or ""),
        confidence = tonumber(recommendation.confidence) or 0,
        reason = tostring(recommendation.reason or ""),
        support = tonumber(recommendation.support_count) or 0,
        accepted = accepted,
    }
    state.experience.runtime_policy = runtime_policy
    state.experience.audit = audit
    state.experience.feedback = state.experience.feedback or { effective_ids = {} }
    state.experience.writeback = state.experience.writeback or { written = false, policy_id = "" }
    state.experience.behavior_match = state.experience.behavior_match or { selected_id = "", match_score = 0 }

    state.context = state.context or {}
    state.context.applied_policy = audit
    state.context.policy_context = audit
    state.context.experience_context = audit
    state.context.experience_prior = prior
    return state
end

function M.run(state, _ctx)
    experience.init()

    local query = build_query(state)
    if not v2_enabled() then
        return apply_default_auto_fallback(state, query, "experience_v2_disabled")
    end

    local ok, next_state = pcall(apply_v2, state, query)
    if ok and type(next_state) == "table" then
        return next_state
    end

    print(string.format("[ExperienceV2][WARN] fallback to default auto due to error: %s", tostring(next_state)))
    if v2_hard_fallback_on_error() then
        return apply_default_auto_fallback(state, query, tostring(next_state))
    end
    error(next_state)
end

return M
