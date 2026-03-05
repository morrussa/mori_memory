local M = {}

function M.run(state, _ctx)
    state.context = state.context or {}
    state.context.memory_context = ""
    state.context.tool_context = ""
    state.context.planner_context = ""
    return state
end

return M
