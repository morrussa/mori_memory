local M = {}

function M.run(state, _ctx)
    state.context = state.context or {}
    state.recall = state.recall or {}
    state.recall.reentry = state.recall.reentry or {}
    state.recall.reentry.pending = false
    state.recall.reentry.used = 0
    state.recall.reentry.kind = ""
    state.recall.reentry.phase = ""
    state.recall.reentry.reason = ""
    state.recall.reentry.source_error = ""
    state.recall.reentry.requested_by = ""
    state.recall.reentry.preferred_type = ""
    state.recall.reentry.allowed_types = {}
    state.recall.reentry.blocked_types = {}
    state.recall.reentry.anchors = {}
    state.recall.reentry.context = ""
    state.recall.reentry.last_kind = ""
    state.recall.reentry.last_phase = ""
    state.recall.reentry.last_source_error = ""
    state.recall.reentry.last_reason = ""
    state.recall.reentry.last_anchor_count = 0
    state.context.policy_context = ""
    state.context.applied_policy = ""
    state.context.memory_context = ""
    state.context.tool_context = ""
    state.context.planner_context = ""
    return state
end

return M
