local context_builder = require("module.graph.context_builder")

local M = {}

function M.run(state, _ctx)
    local messages, meta = context_builder.build_chat_messages(state)
    state.messages = state.messages or {}
    state.messages.chat_messages = messages
    state.metrics = state.metrics or {}
    state.metrics.context = meta
    return state
end

return M
