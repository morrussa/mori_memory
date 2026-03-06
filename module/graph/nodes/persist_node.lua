local history = require("module.memory.history")
local topic = require("module.memory.topic")
local tool = require("module.tool")
local memory_core = require("module.graph.memory_core")
local experience = require("module.experience")
local episode = require("module.episode")
local util = require("module.graph.util")

local M = {}
local MATCH_THRESHOLD = 0.65

local function build_effective_ids(selected_id, match_score, success)
    local effective_ids = {}
    if success ~= true then
        return effective_ids
    end
    local id = util.trim(selected_id or "")
    if id == "" then
        return effective_ids
    end
    if (tonumber(match_score) or 0) < MATCH_THRESHOLD then
        return effective_ids
    end
    effective_ids[id] = true
    return effective_ids
end

local function pick_primary_policy_id(observed_ids, selected_id, match_score)
    local selected = util.trim(selected_id or "")
    if selected ~= "" and (tonumber(match_score) or 0) >= MATCH_THRESHOLD then
        return selected
    end
    if type(observed_ids) == "table" and #observed_ids > 0 then
        return tostring(observed_ids[1] or "")
    end
    return ""
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

    local items = (((state or {}).writeback or {}).items) or {}
    local saved = memory_core.save_ingest_items(items, current_turn)
    state.writeback = state.writeback or {}
    state.writeback.saved_count = tonumber(saved) or 0

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

    local observation = experience.run_builder.build_from_state(state)
    local retrieved_items = ((((state or {}).experience or {}).retrieved or {}).items) or {}
    local success = observation and observation.success == true

    state.experience = state.experience or {}
    state.experience.writeback = state.experience.writeback or { written = false, policy_id = "" }
    state.experience.feedback = state.experience.feedback or { effective_ids = {} }
    state.experience.behavior_match = state.experience.behavior_match or { selected_id = "", match_score = 0 }

    local selected_id, match_score = experience.match_behavior_to_candidate(observation, retrieved_items)
    if util.trim(selected_id or "") == "" and #(((observation or {}).candidate_ids) or {}) > 0 then
        selected_id, match_score = experience.match_behavior_to_candidate(observation, ((observation or {}).candidate_ids) or {})
    end

    state.experience.behavior_match.selected_id = tostring(selected_id or "")
    state.experience.behavior_match.match_score = tonumber(match_score) or 0
    observation.behavior_match = {
        selected_id = tostring(selected_id or ""),
        match_score = tonumber(match_score) or 0,
    }

    local ok, updated_candidate_ids = experience.observe_v2(observation)
    local policy_id = pick_primary_policy_id(updated_candidate_ids, selected_id, match_score)
    state.experience.writeback.written = ok == true
    state.experience.writeback.policy_id = ok and policy_id or ""

    local effective_ids = build_effective_ids(selected_id, match_score, success)
    state.experience.feedback.effective_ids = effective_ids
    if #retrieved_items > 0 then
        if success then
            experience.record_utility_feedback(retrieved_items, effective_ids)
        else
            experience.record_utility_feedback(retrieved_items, {})
        end
    end

    experience.save_all()

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
