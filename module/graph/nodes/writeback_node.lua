local memory_core = require("module.graph.memory_core")

local M = {}

function M.run(state, _ctx)
    state.writeback = state.writeback or { items = {}, ingest_strategy = "atomic_fact", saved_count = 0 }
    local read_only = (((state or {}).input or {}).read_only) == true
    if read_only then
        state.writeback.items = {}
        state.writeback.ingest_strategy = "atomic_fact"
        state.writeback.saved_count = 0
        return state
    end

    local user_input = tostring((((state or {}).input or {}).message) or "")
    local final_text = tostring((((state or {}).final_response or {}).message) or "")
    local items = memory_core.extract_atomic_facts(user_input, final_text)
    state.writeback.items = items or {}
    state.writeback.ingest_strategy = "main_atomic_fact"
    state.writeback.saved_count = 0
    return state
end

return M
