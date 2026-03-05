local context_builder = require("module.graph.context_builder")

local M = {}

function M.run(state, _ctx)
    local messages, meta = context_builder.build_chat_messages(state)
    state.messages = state.messages or {}
    state.messages.chat_messages = messages
    state.messages.runtime_messages = messages
    state.agent_loop = state.agent_loop or {}
    state.agent_loop.pending_tool_calls = {}
    state.agent_loop.stop_reason = ""
    state.agent_loop.iteration = 0
    state.metrics = state.metrics or {}
    state.metrics.context = meta
    return state
end

return M
