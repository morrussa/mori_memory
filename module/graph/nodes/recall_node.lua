local tool = require("module.tool")
local recall = require("module.memory.recall")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

function M.run(state, _ctx)
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local read_only = (((state or {}).input or {}).read_only) == true

    local user_vec_q = tool.get_embedding_query(user_input)
    local recall_result = recall.check_and_retrieve(user_input, user_vec_q, {
        read_only = read_only,
        force = false,
        suppress = false,
    })
    if type(recall_result) ~= "table" then
        recall_result = {
            context = tostring(recall_result or ""),
        }
    end
    local memory_context = util.trim(recall_result.context or "")

    state.recall = {
        triggered = memory_context ~= "",
        context = memory_context,
        score = nil,
        topic_anchor = tostring(recall_result.topic_anchor or ""),
        predicted_memories = (recall_result.predicted_memories or {}),
        predicted_nodes = (recall_result.predicted_nodes or {}),
        selected_turns = (recall_result.selected_turns or {}),
        selected_memories = (recall_result.selected_memories or {}),
        fragments = (recall_result.fragments or {}),
        adopted_memories = (recall_result.adopted_memories or {}),
    }

    state.context = state.context or {}
    state.context.memory_context = memory_context
    return state
end

return M
