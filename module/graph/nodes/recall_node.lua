local tool = require("module.tool")
local recall = require("module.memory.recall")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

function M.run(state, _ctx)
    local cfg = graph_cfg().recall or {}
    local route = tostring((((state or {}).router_decision or {}).route) or "respond")
    local enable_on_respond = util.to_bool(cfg.enable_on_respond, false)

    if route ~= "tool_loop" and (not enable_on_respond) then
        state.recall = {
            triggered = false,
            context = "",
            score = nil,
        }
        return state
    end

    local user_input = tostring((((state or {}).input or {}).message) or "")
    local read_only = (((state or {}).input or {}).read_only) == true

    local user_vec_q = tool.get_embedding_query(user_input)
    local memory_context = recall.check_and_retrieve(user_input, user_vec_q, {
        read_only = read_only,
    })

    memory_context = util.trim(memory_context)
    state.recall = {
        triggered = memory_context ~= "",
        context = memory_context,
        score = nil,
    }

    state.context = state.context or {}
    state.context.memory_context = memory_context
    return state
end

return M
