local util = require("mori_memory.util")

local history = require("module.memory.history")
local topic = require("module.memory.topic")
local topic_graph = require("module.memory.topic_graph")
local saver = require("module.memory.saver")

local M = {}

local _initialized = false

local function trim(s)
    return util.trim(s)
end

local function ensure_init()
    if _initialized then
        return
    end
    if history and history.load then
        history.load()
    end
    if topic and topic.init then
        topic.init()
    end
    if topic_graph and topic_graph.load then
        topic_graph.load()
    end
    _initialized = true
end

local function unique_sorted_numbers(arr)
    local tmp = {}
    for _, v in ipairs(arr or {}) do
        local n = tonumber(v)
        if n and n > 0 then
            tmp[math.floor(n)] = true
        end
    end
    local out = {}
    for k in pairs(tmp) do
        out[#out + 1] = k
    end
    table.sort(out)
    return out
end

local function build_selected_turn_transcript(turns, opts)
    turns = unique_sorted_numbers(turns)
    opts = type(opts) == "table" and opts or {}

    local max_turns = math.max(0, math.floor(tonumber(opts.max_selected_turns) or 6))
    if max_turns <= 0 or #turns <= 0 then
        return ""
    end
    while #turns > max_turns do
        table.remove(turns, 1)
    end

    local user_cap = math.max(40, math.floor(tonumber(opts.user_chars) or 180))
    local assistant_cap = math.max(40, math.floor(tonumber(opts.assistant_chars) or 220))

    local parts = {}
    for _, t in ipairs(turns) do
        local entry = history.get_by_turn(t)
        if entry then
            local user_text, ai_text = history.parse_entry(entry)
            local user_compact = util.utf8_take(trim(user_text), user_cap)
            local ai_compact = util.utf8_take(trim(ai_text), assistant_cap)
            if user_compact ~= "" or ai_compact ~= "" then
                parts[#parts + 1] = string.format(
                    "第%d轮\n用户：%s\n助手：%s",
                    t,
                    user_compact,
                    ai_compact
                )
            end
        end
    end
    if #parts <= 0 then
        return ""
    end
    return "【相关对话片段】\n" .. table.concat(parts, "\n\n")
end

function M.ingest_turn(meta)
    ensure_init()
    meta = type(meta) == "table" and meta or {}
    if meta.read_only == true then
        return { ok = true, skipped = true }
    end

    local user_input = trim(meta.user_input or meta.user or meta.text or "")
    local assistant_text = trim(meta.assistant or meta.assistant_text or meta.reply or "")
    local turn = tonumber(meta.turn)
    if not turn or turn <= 0 then
        turn = (history.get_turn() or 0) + 1
    end
    turn = math.floor(turn)

    if user_input ~= "" then
        local vec = meta.user_vec
        if type(vec) ~= "table" or #vec <= 0 then
            return { ok = false, error = "missing_user_vec", turn = turn }
        end
        topic.add_turn(turn, user_input, vec)
    end

    history.add_history(user_input, assistant_text)
    if assistant_text ~= "" then
        topic.update_assistant(turn, assistant_text)
    end

    local anchor = ""
    if topic.get_stable_anchor then
        anchor = tostring(topic.get_stable_anchor(turn) or "")
    end
    if anchor == "" and topic.get_topic_anchor then
        anchor = tostring(topic.get_topic_anchor(turn) or "")
    end
    if anchor ~= "" then
        topic_graph.observe_turn(turn, anchor)
    end

    if saver and saver.mark_dirty then
        saver.mark_dirty()
    end

    return {
        ok = true,
        turn = turn,
        topic_anchor = anchor,
    }
end

function M.compile_context(meta)
    ensure_init()
    meta = type(meta) == "table" and meta or {}

    local user_input = trim(meta.user_input or meta.user or meta.text or "")
    local current_turn = tonumber(meta.turn)
    if not current_turn or current_turn <= 0 then
        current_turn = (history.get_turn() or 0) + 1
    end
    current_turn = math.floor(current_turn)

    local current_anchor = trim(meta.topic_anchor or "")
    if current_anchor == "" then
        if topic.get_stable_anchor then
            current_anchor = trim(topic.get_stable_anchor(current_turn) or "")
        end
        if current_anchor == "" and topic.get_topic_anchor then
            current_anchor = trim(topic.get_topic_anchor(current_turn) or "")
        end
    end

    local query_vec = meta.query_vec
    if type(query_vec) ~= "table" or #query_vec <= 0 then
        query_vec = meta.user_vec
    end
    if type(query_vec) ~= "table" or #query_vec <= 0 then
        local rec = topic.get_topic_for_turn and topic.get_topic_for_turn(current_turn) or nil
        query_vec = (rec and rec.centroid) or {}
    end

    local retrieved = topic_graph.retrieve(query_vec, current_anchor, current_turn, {
        user_input = user_input,
    })

    local blocks = {}
    local ctx = trim((retrieved or {}).context or "")
    if ctx ~= "" then
        blocks[#blocks + 1] = { role = "system", content = ctx }
    end

    local transcript = build_selected_turn_transcript((retrieved or {}).selected_turns, meta)
    if transcript ~= "" then
        blocks[#blocks + 1] = { role = "system", content = transcript }
    end

    return blocks
end

function M.shutdown()
    if saver and saver.on_exit then
        saver.on_exit()
    end
end

return M
