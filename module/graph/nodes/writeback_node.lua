local memory_core = require("module.graph.memory_core")

local M = {}

function M.run(state, _ctx)
    state.writeback = state.writeback or { facts = {}, saved = 0 }
    local read_only = (((state or {}).input or {}).read_only) == true
    if read_only then
        state.writeback.facts = {}
        state.writeback.saved = 0
        return state
    end

    local user_input = tostring((((state or {}).input or {}).message) or "")
    local final_text = tostring((((state or {}).final_response or {}).message) or "")
    local facts = memory_core.extract_atomic_facts(user_input, final_text)
    state.writeback.facts = facts or {}
    state.writeback.saved = 0
    return state
end

return M
