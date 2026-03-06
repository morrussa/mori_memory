local history = require("module.memory.history")
local topic = require("module.memory.topic")
local tool = require("module.tool")
local memory_core = require("module.graph.memory_core")
local experience = require("module.experience")

local M = {}

local function build_effective_ids(retrieved_items, current_experience)
    local effective_ids = {}
    if not current_experience or not current_experience.outcome or current_experience.outcome.success ~= true then
        return effective_ids
    end

    local success_key = tostring(current_experience.success_key or "")
    if success_key == "" then
        return effective_ids
    end

    for _, item in ipairs(retrieved_items or {}) do
        if tostring((item or {}).success_key or "") == success_key and item.id then
            effective_ids[item.id] = true
            break
        end
    end

    return effective_ids
end

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

    experience.init()

    local current_experience = experience.run_builder.build_from_state(state)
    local ok, exp_id = experience.add_experience(current_experience)

    state.experience = state.experience or {}
    state.experience.writeback = state.experience.writeback or { written = false }
    state.experience.feedback = state.experience.feedback or { effective_ids = {} }
    state.experience.writeback.written = ok == true
    state.experience.writeback.experience_id = ok and exp_id or nil

    local retrieved_items = ((((state or {}).experience or {}).retrieved or {}).items) or {}
    local effective_ids = build_effective_ids(retrieved_items, current_experience)
    state.experience.feedback.effective_ids = effective_ids

    if #retrieved_items > 0 then
        experience.record_utility_feedback(retrieved_items, effective_ids)
    end

    experience.adaptive.save_to_disk()
    experience.store.save()

    return state
end

return M
