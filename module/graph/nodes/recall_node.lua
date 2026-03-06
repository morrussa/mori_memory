local tool = require("module.tool")
local recall = require("module.memory.recall")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function normalize_policy(raw_policy)
    local policy = type(raw_policy) == "table" and raw_policy or {}
    return {
        mode = util.trim(policy.mode or ""),
        confidence = tonumber(policy.confidence) or 0,
        force = policy.force == true,
        suppress = policy.suppress == true,
        preferred_type = util.trim(policy.preferred_type or ""),
        allowed_types = type(policy.allowed_types) == "table" and policy.allowed_types or {},
        blocked_types = type(policy.blocked_types) == "table" and policy.blocked_types or {},
        reason = util.trim(policy.reason or ""),
        decided = policy.decided == true,
    }
end

function M.run(state, _ctx)
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local read_only = (((state or {}).input or {}).read_only) == true
    local policy = normalize_policy((((state or {}).recall or {}).policy) or {})

    local user_vec_q = tool.get_embedding_query(user_input)
    local memory_context = recall.check_and_retrieve(user_input, user_vec_q, {
        read_only = read_only,
        force = policy.force,
        suppress = policy.suppress,
        allowed_types = policy.allowed_types,
        blocked_types = policy.blocked_types,
        preferred_type = policy.preferred_type ~= "" and policy.preferred_type or nil,
        policy_decided = policy.decided,
    })

    memory_context = util.trim(memory_context)
    state.recall = state.recall or {}
    state.recall.policy = policy
    state.recall.triggered = memory_context ~= ""
    state.recall.context = memory_context
    state.recall.score = tonumber(policy.confidence) or nil

    state.context = state.context or {}
    state.context.memory_context = memory_context
    return state
end

return M
