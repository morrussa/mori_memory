local history = require("module.memory.history")
local topic = require("module.memory.topic")
local tool = require("module.tool")
local memory_core = require("module.graph.memory_core")
local experience = require("module.experience")
local episode = require("module.episode")
local util = require("module.graph.util")

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

    state.episode = state.episode or {}
    state.episode.current = state.episode.current or {}
    state.episode.current.turn_index = current_turn
    state.episode.current.topic_anchor = tostring((topic.get_topic_anchor and topic.get_topic_anchor(current_turn)) or "")

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
        local success = current_experience
            and current_experience.outcome
            and current_experience.outcome.success == true

        if success then
            local matched_items = {}
            for _, item in ipairs(retrieved_items) do
                if item and effective_ids[item.id] == true then
                    matched_items[#matched_items + 1] = item
                end
            end

            if #matched_items > 0 then
                experience.record_utility_feedback(matched_items, effective_ids)
            end
        else
            experience.record_utility_feedback(retrieved_items, {})
        end
    end

    experience.adaptive.save_to_disk()
    experience.store.save()

    episode.init()
    local current_episode = episode.run_builder.build_from_state(state)
    local episode_ok, episode_id = episode.add_episode(current_episode)
    local episode_saved = false
    if episode_ok then
        episode_saved = episode.store.save() == true
    end

    state.episode.writeback = state.episode.writeback or { written = false, episode_id = "" }
    state.episode.writeback.written = episode_ok == true and episode_saved == true
    state.episode.writeback.episode_id = (episode_ok == true and episode_saved == true) and tostring(episode_id or "") or ""

    if episode_ok == true and episode_saved == true then
        state.session = state.session or { active_task = {} }
        state.session.active_task = state.session.active_task or {}
        state.session.active_task.last_episode_id = tostring(episode_id or "")

        local carryover_parts = {
            tostring(current_episode.summary or ""),
        }
        local final_preview = util.trim(final_text)
        if final_preview ~= "" then
            carryover_parts[#carryover_parts + 1] = "reply=" .. util.utf8_take(final_preview, 220)
        end
        state.session.active_task.carryover_summary = util.utf8_take(table.concat(carryover_parts, " | "), 600)

        state.episode.recent = state.episode.recent or { items = {}, summary = "", count = 0, latest_episode_id = "" }
        state.episode.recent.items = { current_episode }
        state.episode.recent.summary = tostring(current_episode.summary or "")
        state.episode.recent.count = 1
        state.episode.recent.latest_episode_id = tostring(episode_id or "")
    end

    return state
end

return M
