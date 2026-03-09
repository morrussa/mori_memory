local history = require("module.memory.history")
local recall = require("module.memory.recall")
local topic = require("module.memory.topic")
local predictor = require("module.memory.topic_predictor")
local tool = require("module.tool")
local memory_core = require("module.graph.memory_core")
local episode = require("module.episode")
local util = require("module.graph.util")

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

    local recall_state = ((state or {}).recall) or {}
    local adopted_memories = recall.infer_adopted_memories(final_text, recall_state)
    local current_anchor = topic.get_stable_anchor and topic.get_stable_anchor(current_turn) or topic.get_topic_anchor(current_turn)
    state.recall = state.recall or {}
    state.recall.topic_anchor = tostring(current_anchor or "")
    state.recall.adopted_memories = adopted_memories
    if #((recall_state.selected_memories) or {}) > 0 or #adopted_memories > 0 then
        predictor.observe(current_anchor, nil, {
            retrieved_memories = (recall_state.selected_memories or {}),
            adopted_memories = adopted_memories,
        })
    end

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
