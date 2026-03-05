local history = require("module.memory.history")
local topic = require("module.memory.topic")
local tool = require("module.tool")
local memory_core = require("module.graph.memory_core")

local M = {}

function M.run(state, _ctx)
    local read_only = (((state or {}).input or {}).read_only) == true
    if read_only then
        return state
    end

    local user_input = tostring((((state or {}).input or {}).message) or "")
    local final_text = tostring((((state or {}).final_response or {}).message) or "")

    local current_turn = (history.get_turn() or 0) + 1
    topic.add_turn(current_turn, user_input, tool.get_embedding_passage(user_input))

    history.add_history(user_input, final_text)
    topic.update_assistant(current_turn, final_text)

    local facts = (((state or {}).writeback or {}).facts) or {}
    local saved = memory_core.save_turn_memory(facts, current_turn)
    state.writeback = state.writeback or {}
    state.writeback.saved = tonumber(saved) or 0

    local summary = topic.get_summary and topic.get_summary(current_turn)
    if summary and summary ~= "" then
        state.context = state.context or {}
        state.context.topic_summary = summary
    end

    return state
end

return M
