local context_builder = require("module.graph.context_builder")
local util = require("module.graph.util")
local experience_policy = require("module.experience.policy")

local M = {}

local function detect_task_profile(state)
    local session = ((state or {}).session) or {}
    local active_task = session.active_task or {}
    local task_decision = ((((state or {}).task or {}).decision) or {})
    local existing = util.trim(active_task.profile or "")
    local decision_kind = util.trim(task_decision.kind or "")
    if (decision_kind == "same_task_step" or decision_kind == "same_task_refine" or decision_kind == "meta_turn")
        and existing ~= "" then
        return existing
    end
    return experience_policy.detect_task_profile(state)
end

function M.run(state, _ctx)
    state.context = state.context or {}
    state.session = state.session or { mode = "single", active_task = {} }
    state.session.active_task = state.session.active_task or {}

    local profile = detect_task_profile(state)
    state.context.task_profile = profile
    state.context.workspace_virtual_root = util.workspace_virtual_root()
    state.session.active_task.profile = profile

    local messages, meta = context_builder.build_chat_messages(state)
    state.messages = state.messages or {}
    state.messages.chat_messages = messages
    state.messages.runtime_messages = messages
    state.agent_loop = state.agent_loop or {}
    state.agent_loop.pending_tool_calls = state.agent_loop.pending_tool_calls or {}
    state.agent_loop.stop_reason = util.trim(state.agent_loop.stop_reason or "")
    state.agent_loop.iteration = tonumber(state.agent_loop.iteration) or 0
    state.metrics = state.metrics or {}
    state.metrics.context = meta
    return state
end

return M
