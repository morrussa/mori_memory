local experience = require("module.experience")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
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

function M.run(state, _ctx)
    experience.init()

    local query = build_query(state)
    local items, strategy = experience.retrieve(query, {
        limit = 3,
    })
    local runtime_policy = experience.policy.merge_runtime_policies(items or {})
    local audit = experience.policy.summarize_runtime_policy(runtime_policy)

    apply_budget_policy(state, runtime_policy)

    local ids = {}
    for _, item in ipairs(items or {}) do
        if item and item.id then
            ids[#ids + 1] = item.id
        end
    end

    state.experience = state.experience or {}
    state.experience.kind = "graph_policy"
    state.experience.query = query
    state.experience.retrieved = {
        items = items or {},
        ids = ids,
        strategy = tostring((strategy or {}).strategy or ""),
    }
    state.experience.runtime_policy = runtime_policy
    state.experience.audit = audit
    state.experience.feedback = state.experience.feedback or { effective_ids = {} }
    state.experience.writeback = state.experience.writeback or { written = false, policy_id = "" }

    state.context = state.context or {}
    state.context.applied_policy = audit
    state.context.policy_context = audit
    state.context.experience_context = audit
    return state
end

return M
