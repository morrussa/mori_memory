local config = require("module.config")
local tool = require("module.tool")
local util = require("mori_memory.util")

local M = {}

local _state = {
    scopes = {},
}

local REPLY_CUE_PATTERNS = {
    "你刚才",
    "你之前",
    "刚才说",
    "刚刚说",
    "前面说",
    "上面说",
    "继续",
    "接着",
    "还是",
    "不是那个",
    "不是这个",
    "关于刚才",
    "reply",
    "continue",
    "earlier",
    "before",
}

local function deep_copy(value, seen, depth)
    if type(value) ~= "table" then
        return value
    end
    seen = seen or {}
    depth = math.max(0, math.floor(tonumber(depth) or 0))
    if depth > 24 then
        return {}
    end
    if seen[value] then
        return seen[value]
    end
    local out = {}
    seen[value] = out
    for k, v in pairs(value) do
        out[deep_copy(k, seen, depth + 1)] = deep_copy(v, seen, depth + 1)
    end
    return out
end

local function trim(s)
    return util.trim(s)
end

local function cfg()
    return ((config.settings or {}).disentangle or {})
end

local function clamp(v, lo, hi)
    v = tonumber(v) or 0.0
    lo = tonumber(lo) or v
    hi = tonumber(hi) or v
    if v < lo then
        return lo
    end
    if v > hi then
        return hi
    end
    return v
end

local function truthy(v)
    return v == true or v == 1 or v == "1" or v == "true"
end

local function list_includes(list, needle)
    if type(list) ~= "table" then
        return false
    end
    needle = trim(needle)
    if needle == "" then
        return false
    end
    if list[needle] ~= nil then
        return truthy(list[needle])
    end
    for _, v in ipairs(list) do
        if trim(v) == needle then
            return true
        end
    end
    return false
end

function M.enabled_for(meta)
    meta = type(meta) == "table" and meta or {}
    local c = cfg()
    if c.enabled ~= true then
        return false
    end
    local source = trim(meta.source or "")
    local allow = c.enable_sources or c.enabled_by_source
    if allow == nil then
        return true
    end
    return list_includes(allow, source)
end

local function count_streams(streams)
    local n = 0
    for _, stream in pairs(streams or {}) do
        if not truthy((stream or {}).orphan) and not truthy((stream or {}).ambient) then
            n = n + 1
        end
    end
    return n
end

local function ensure_scope(scope_key)
    scope_key = trim(scope_key)
    if scope_key == "" then
        scope_key = "global"
    end
    local scopes = _state.scopes
    local rec = scopes[scope_key]
    if not rec then
        rec = {
            next_stream_id = 1,
            next_segment_id = 1,
            streams = {},
            last_assigned = nil,
            streak = 0,
            local_sequence_until_turn = 0,
        }
        scopes[scope_key] = rec
    end
    rec.streams = type(rec.streams) == "table" and rec.streams or {}
    rec.next_stream_id = math.max(1, math.floor(tonumber(rec.next_stream_id) or 1))
    rec.next_segment_id = math.max(1, math.floor(tonumber(rec.next_segment_id) or 1))
    rec.streak = math.max(0, math.floor(tonumber(rec.streak) or 0))
    rec.local_sequence_until_turn = math.max(0, math.floor(tonumber(rec.local_sequence_until_turn) or 0))
    return scope_key, rec
end

local function normalize_user_id(meta)
    meta = type(meta) == "table" and meta or {}
    return trim(meta.user_id or meta.uid or "")
end

local function normalize_actor_key(meta)
    meta = type(meta) == "table" and meta or {}
    local user_id = normalize_user_id(meta)
    if user_id ~= "" then
        return user_id
    end
    return trim(meta.nickname or meta.nick or "")
end

local function normalize_alias(value)
    local alias = trim(tostring(value or "")):lower()
    if alias == "" then
        return ""
    end
    alias = alias:gsub("^[@#]+", "")
    alias = alias:gsub("[%s，。！？；：,%.!%?;:]+$", "")
    return trim(alias)
end

local function current_text(meta)
    meta = type(meta) == "table" and meta or {}
    return trim(meta.raw_user_input or meta.user_input or meta.user or meta.text or "")
end

local function normalized_text(meta)
    return current_text(meta):lower()
end

local function actor_aliases(meta)
    meta = type(meta) == "table" and meta or {}
    local out = {}
    local seen = {}
    local candidates = {
        meta.user_id,
        meta.uid,
        meta.nickname,
        meta.nick,
    }
    for _, raw in ipairs(candidates) do
        local alias = normalize_alias(raw)
        if alias ~= "" and not seen[alias] then
            seen[alias] = true
            out[#out + 1] = alias
        end
    end
    return out
end

local function extract_mentions(text)
    text = tostring(text or "")
    local out = {}
    local seen = {}
    for raw in text:gmatch("@([^%s，。！？；：,%.!%?;:%)%]%}]+)") do
        local alias = normalize_alias(raw)
        if alias ~= "" and not seen[alias] then
            seen[alias] = true
            out[#out + 1] = alias
        end
    end
    return out
end

local function contains_reply_cue(text)
    text = tostring(text or ""):lower()
    if text == "" then
        return false
    end
    for _, pattern in ipairs(REPLY_CUE_PATTERNS) do
        if text:find(pattern, 1, true) then
            return true
        end
    end
    return false
end

local function utf8_len(s)
    local n = 0
    for _ in tostring(s or ""):gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        n = n + 1
    end
    return n
end

local function compact_reaction_text(text)
    local compact = trim(tostring(text or "")):lower()
    compact = compact:gsub("%s+", "")
    compact = compact:gsub("[,%.!%?;:%-_/\\|`~%@#%^%*%(%)%[%]%{%}%<%>%+%=]+", "")
    compact = compact:gsub("，", "")
    compact = compact:gsub("。", "")
    compact = compact:gsub("！", "")
    compact = compact:gsub("？", "")
    compact = compact:gsub("；", "")
    compact = compact:gsub("：", "")
    compact = compact:gsub("、", "")
    compact = compact:gsub("…", "")
    compact = compact:gsub("～", "")
    compact = compact:gsub("·", "")
    return compact
end

local function looks_like_emote_block(text)
    local rest = trim(text)
    if rest == "" then
        return false
    end
    local matched = false
    while rest ~= "" do
        local seg = rest:match("^(%b[])")
        if not seg then
            break
        end
        matched = true
        rest = trim(rest:sub(#seg + 1))
    end
    return matched and rest == ""
end

local function is_reaction_like_text(text, opts)
    opts = type(opts) == "table" and opts or {}
    text = trim(text)
    if text == "" then
        return false
    end

    local char_len = utf8_len(text)
    local short_chars = math.max(1, math.floor(tonumber(opts.reaction_short_chars) or 6))
    local max_chars = math.max(short_chars, math.floor(tonumber(opts.reaction_max_chars) or 10))
    if char_len > max_chars then
        return false
    end

    if contains_reply_cue(text) then
        return false
    end
    if #extract_mentions(text) > 0 then
        return false
    end
    if looks_like_emote_block(text) then
        return true
    end

    local compact = compact_reaction_text(text)
    if compact == "" then
        return true
    end
    if char_len <= short_chars then
        return true
    end
    if compact:match("^[%d6８8?？!！~～哈啊呵哦哇呀啦]+$") then
        return true
    end
    if not text:find("[%s，。！？；：,%.!%?;:]", 1) then
        local unique = {}
        local unique_count = 0
        for ch in compact:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
            if ch ~= "" and not unique[ch] then
                unique[ch] = true
                unique_count = unique_count + 1
            end
        end
        if unique_count <= math.max(2, math.floor(char_len / 2)) then
            return true
        end
    end
    return false
end

local function table_size(map)
    local n = 0
    for key, value in pairs(map or {}) do
        if value and trim(key) ~= "" then
            n = n + 1
        end
    end
    return n
end

local function stream_alias_match(stream, text, mentions)
    stream = type(stream) == "table" and stream or {}
    text = tostring(text or "")
    mentions = type(mentions) == "table" and mentions or {}
    local aliases = type(stream.actor_aliases) == "table" and stream.actor_aliases or {}
    local addressee_hints = type(stream.addressee_hints) == "table" and stream.addressee_hints or {}
    local mention_set = {}
    local explicit_hits = 0
    local addressee_hits = 0

    for _, alias in ipairs(mentions) do
        mention_set[alias] = true
        if tonumber(aliases[alias]) and tonumber(aliases[alias]) > 0 then
            explicit_hits = explicit_hits + 1
        end
        if tonumber(addressee_hints[alias]) and tonumber(addressee_hints[alias]) > 0 then
            addressee_hits = addressee_hits + 1
        end
    end

    if explicit_hits > 0 or addressee_hits > 0 then
        return explicit_hits, addressee_hits
    end

    for alias in pairs(aliases) do
        if #alias >= 3 and text:find(alias, 1, true) then
            explicit_hits = explicit_hits + 1
        end
    end
    for alias in pairs(addressee_hints) do
        if not mention_set[alias] and #alias >= 3 and text:find(alias, 1, true) then
            addressee_hits = addressee_hits + 1
        end
    end
    return explicit_hits, addressee_hits
end

local function compute_stream_stability(stream, turn, opts)
    stream = type(stream) == "table" and stream or {}
    opts = type(opts) == "table" and opts or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    local stability_turns = math.max(1, math.floor(tonumber(opts.stability_turns) or 4))
    local recent_turns = math.max(1, math.floor(tonumber(opts.reply_recent_turns) or 6))
    local confirmed = math.max(0, math.floor(tonumber(stream.confirmed_turns) or 0))
    local participant_count = table_size(stream.participants)
    local confidence = clamp(tonumber(stream.confidence) or 0.0, 0.0, 1.25) / 1.25
    local age = math.max(0, turn - math.max(0, math.floor(tonumber(stream.last_turn) or 0)))
    local age_factor = 1.0 / (1.0 + (age / recent_turns))
    local turn_factor = clamp(confirmed / stability_turns, 0.0, 1.0)
    local participant_factor = clamp(participant_count / 2.0, 0.0, 1.0)
    return clamp(
        0.45 * turn_factor
        + 0.20 * participant_factor
        + 0.20 * confidence
        + 0.15 * age_factor,
        0.0,
        1.0
    )
end

local function score_stream(stream, vec, meta, turn, opts)
    stream = type(stream) == "table" and stream or {}
    vec = type(vec) == "table" and vec or {}
    opts = type(opts) == "table" and opts or {}
    if #vec <= 0 then
        return -1.0, {}
    end

    local centroid = type(stream.centroid) == "table" and stream.centroid or {}
    local head_vec = type(stream.head_vec) == "table" and stream.head_vec or centroid
    local tail_vec = type(stream.tail_vec) == "table" and stream.tail_vec or centroid
    if #centroid <= 0 and #head_vec <= 0 and #tail_vec <= 0 then
        return -1.0, {}
    end

    local sim_centroid = tonumber(tool.cosine_similarity(centroid, vec)) or 0.0
    local sim_head = tonumber(tool.cosine_similarity(head_vec, vec)) or sim_centroid
    local sim_tail = tonumber(tool.cosine_similarity(tail_vec, vec)) or sim_centroid
    local semantic_score = (tonumber(opts.centroid_weight) or 0.42) * sim_centroid
        + (tonumber(opts.tail_weight) or 0.38) * sim_tail
        + (tonumber(opts.head_weight) or 0.20) * sim_head
    local score = semantic_score

    local actor_key = normalize_actor_key(meta)
    local last_actor_key = trim(stream.last_actor_key or "")
    if actor_key ~= "" and last_actor_key ~= "" and actor_key == last_actor_key then
        score = score + (tonumber(opts.same_user_bonus) or 0.0)
    elseif actor_key ~= "" and tonumber((stream.participants or {})[actor_key]) and tonumber((stream.participants or {})[actor_key]) > 0 then
        score = score + (tonumber(opts.participant_bonus) or 0.0)
    end

    local text = normalized_text(meta)
    local mentions = extract_mentions(text)
    local explicit_hits, addressee_hits = stream_alias_match(stream, text, mentions)
    if explicit_hits > 0 then
        score = score + math.min(explicit_hits, 2) * (tonumber(opts.mention_bonus) or 0.0)
    end
    if addressee_hits > 0 then
        score = score + math.min(addressee_hits, 2) * (tonumber(opts.addressee_hint_bonus) or 0.0)
    end

    local has_reply_cue = contains_reply_cue(text)
    local age = math.max(0, (tonumber(turn) or 0) - (tonumber(stream.last_turn) or 0))
    if has_reply_cue and age <= math.max(1, math.floor(tonumber(opts.reply_recent_turns) or 6)) then
        local recency = 1.0 - clamp(age / math.max(1.0, tonumber(opts.reply_recent_turns) or 6.0), 0.0, 0.85)
        score = score + (tonumber(opts.reply_cue_bonus) or 0.0) * recency
    end

    local stability = compute_stream_stability(stream, turn, opts)
    score = score + (tonumber(opts.stability_bonus) or 0.0) * stability

    local age_penalty = tonumber(opts.age_penalty) or 0.0
    if age_penalty > 0.0 and age > 0 then
        score = score - age_penalty * math.min(age, 32)
    end

    return score, {
        sim_centroid = sim_centroid,
        sim_head = sim_head,
        sim_tail = sim_tail,
        stability = stability,
        explicit_hits = explicit_hits,
        addressee_hits = addressee_hits,
        has_reply_cue = has_reply_cue,
        semantic_score = semantic_score,
        age = age,
    }
end

local function row_attachable(row)
    if type(row) ~= "table" then
        return false
    end
    if tonumber(row.stream_id) == nil or tonumber(row.stream_id) <= 0 then
        return false
    end
    return type(row.stream) == "table" and not truthy((row.stream or {}).orphan)
end

local function row_age(row, turn)
    if type(row) ~= "table" then
        return math.huge
    end
    local features = type(row.features) == "table" and row.features or {}
    local age = tonumber(features.age)
    if age ~= nil then
        return math.max(0, math.floor(age))
    end
    local stream = type(row.stream) == "table" and row.stream or {}
    return math.max(0, math.floor(turn - (tonumber(stream.last_turn) or 0)))
end

local function choose_reaction_fallback(scope, best, dominant, meta, turn, opts)
    scope = type(scope) == "table" and scope or {}
    opts = type(opts) == "table" and opts or {}
    if opts.reaction_fallback_enabled == false then
        return nil, ""
    end
    if not is_reaction_like_text(current_text(meta), opts) then
        return nil, ""
    end

    local recent_turns = math.max(1, math.floor(tonumber(opts.reaction_recent_turns) or 3))
    local attach_score = tonumber(opts.reaction_attach_score) or 0.46
    local dominant_score = tonumber(opts.reaction_dominant_score) or 0.36
    local stability_min = tonumber(opts.reaction_stability_min) or 0.18

    if row_attachable(best)
        and row_age(best, turn) <= recent_turns
        and (tonumber(best.score) or -1.0) >= attach_score
    then
        return best, "attach_reaction"
    end

    if row_attachable(dominant)
        and row_age(dominant, turn) <= recent_turns
        and (
            (tonumber(dominant.score) or -1.0) >= dominant_score
            or (tonumber(dominant.stability) or 0.0) >= stability_min
        )
    then
        return dominant, "attach_recent"
    end

    return nil, ""
end

local function should_reset_topic(sim_centroid, opts)
    opts = type(opts) == "table" and opts or {}
    local reset_th = tonumber(opts.reset_threshold)
    if reset_th == nil then
        return false
    end
    return (tonumber(sim_centroid) or 0.0) < reset_th
end

local function prune_stale_streams(scope, turn, stale_turns, orphan_stale_turns)
    scope = type(scope) == "table" and scope or {}
    local streams = type(scope.streams) == "table" and scope.streams or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    stale_turns = math.max(0, math.floor(tonumber(stale_turns) or 0))
    orphan_stale_turns = math.max(1, math.floor(tonumber(orphan_stale_turns) or math.max(8, stale_turns)))
    if stale_turns <= 0 then
        return 0
    end
    local removed = 0
    for id, stream in pairs(streams) do
        local last_turn = tonumber((stream or {}).last_turn) or 0
        local pending = type((stream or {}).pending) == "table" and (stream or {}).pending or {}
        local local_pending = type((stream or {}).local_pending) == "table" and (stream or {}).local_pending or {}
        local age = turn - last_turn
        if truthy((stream or {}).orphan) and last_turn > 0 and age >= orphan_stale_turns then
            streams[id] = nil
            removed = removed + 1
        elseif last_turn > 0 and #pending <= 0 and #local_pending <= 0 and age >= stale_turns then
            streams[id] = nil
            removed = removed + 1
        end
    end
    scope.streams = streams
    return removed
end

local function make_anchor(topic_start_turn)
    topic_start_turn = math.max(1, math.floor(tonumber(topic_start_turn) or 0))
    return "S:" .. tostring(topic_start_turn)
end

local function make_thread_key(scope_key, thread_id)
    scope_key = trim(scope_key)
    if scope_key == "" then
        scope_key = "global"
    end
    thread_id = math.max(1, math.floor(tonumber(thread_id) or 0))
    return scope_key .. "$th:" .. tostring(thread_id)
end

local function alloc_segment_id(scope)
    scope = type(scope) == "table" and scope or {}
    local seg_id = math.max(1, math.floor(tonumber(scope.next_segment_id) or 1))
    scope.next_segment_id = seg_id + 1
    return seg_id
end

local function make_segment_key(scope_key, thread_id, segment_id)
    return make_thread_key(scope_key, thread_id) .. "/seg:" .. tostring(math.max(1, math.floor(tonumber(segment_id) or 0)))
end

local function make_sequence_key(scope_key, segment_id)
    scope_key = trim(scope_key)
    if scope_key == "" then
        scope_key = "global"
    end
    segment_id = math.max(1, math.floor(tonumber(segment_id) or 0))
    return scope_key .. "$seg:" .. tostring(segment_id)
end

local function ensure_stream_fields(stream, scope)
    stream = type(stream) == "table" and stream or {}
    scope = type(scope) == "table" and scope or {}
    local scope_key = trim(scope.scope_key or "")
    stream.pending = type(stream.pending) == "table" and stream.pending or {}
    stream.local_pending = type(stream.local_pending) == "table" and stream.local_pending or {}
    stream.participants = type(stream.participants) == "table" and stream.participants or {}
    stream.actor_aliases = type(stream.actor_aliases) == "table" and stream.actor_aliases or {}
    stream.addressee_hints = type(stream.addressee_hints) == "table" and stream.addressee_hints or {}
    stream.id = math.max(1, math.floor(tonumber(stream.id) or 0))
    stream.thread_id = math.max(1, math.floor(tonumber(stream.thread_id) or stream.id or 0))
    stream.thread_key = trim(stream.thread_key or "")
    if stream.thread_key == "" then
        stream.thread_key = make_thread_key(scope_key, stream.thread_id)
    end
    local seg_id = tonumber(stream.segment_id) or 0
    if seg_id <= 0 then
        stream.segment_id = alloc_segment_id(scope)
    else
        stream.segment_id = math.floor(seg_id)
    end
    stream.segment_key = make_segment_key(scope_key, stream.thread_id, stream.segment_id)
    stream.created_turn = math.max(0, math.floor(tonumber(stream.created_turn) or 0))
    stream.topic_start_turn = math.max(0, math.floor(tonumber(stream.topic_start_turn) or 0))
    stream.last_turn = math.max(0, math.floor(tonumber(stream.last_turn) or 0))
    stream.turn_count = math.max(0, math.floor(tonumber(stream.turn_count) or 0))
    stream.confirmed_turns = math.max(0, math.floor(tonumber(stream.confirmed_turns) or 0))
    stream.last_user_id = trim(stream.last_user_id or "")
    stream.last_actor_key = trim(stream.last_actor_key or "")
    stream.last_text = trim(stream.last_text or "")
    stream.head_vec = type(stream.head_vec) == "table" and stream.head_vec or {}
    stream.tail_vec = type(stream.tail_vec) == "table" and stream.tail_vec or {}
    stream.confidence = tonumber(stream.confidence) or 0.0
    stream.stability = clamp(tonumber(stream.stability) or 0.0, 0.0, 1.0)
    stream.orphan = truthy(stream.orphan)
    stream.ambient = truthy(stream.ambient)
    return stream
end

local function create_stream(scope, scope_key, turn, vec, meta, opts)
    scope = type(scope) == "table" and scope or {}
    meta = type(meta) == "table" and meta or {}
    opts = type(opts) == "table" and opts or {}
    local new_id = tonumber(scope.next_stream_id) or 1
    scope.next_stream_id = new_id + 1
    local stream = {
        id = new_id,
        thread_id = new_id,
        created_turn = turn,
        topic_start_turn = turn,
        window = type(vec) == "table" and #vec > 0 and { vec } or {},
        centroid = type(vec) == "table" and #vec > 0 and vec or {},
        last_turn = turn,
        last_user_id = normalize_user_id(meta),
        last_actor_key = normalize_actor_key(meta),
        last_text = current_text(meta),
        pending = {},
        local_pending = {},
        participants = {},
        actor_aliases = {},
        addressee_hints = {},
        head_vec = type(vec) == "table" and #vec > 0 and vec or {},
        tail_vec = type(vec) == "table" and #vec > 0 and vec or {},
        turn_count = 0,
        confirmed_turns = 0,
        confidence = 0.0,
        stability = 0.0,
        orphan = opts.orphan == true,
        ambient = opts.ambient == true,
    }
    stream = ensure_stream_fields(stream, scope)
    scope.streams[new_id] = stream
    return stream
end

local function find_ambient_stream(scope)
    scope = type(scope) == "table" and scope or {}
    for id, stream in pairs(scope.streams or {}) do
        if truthy((stream or {}).ambient) then
            return tonumber(id) or tonumber((stream or {}).id) or 0, stream
        end
    end
    return 0, nil
end

local function selection_from_stream(sel, stream, scope_key)
    if type(sel) ~= "table" or type(stream) ~= "table" then
        return sel
    end
    sel.stream_id = tonumber(stream.id) or tonumber(stream.thread_id) or 0
    sel.thread_id = tonumber(stream.thread_id) or sel.stream_id or 0
    sel.thread_key = trim(stream.thread_key or "")
    sel.segment_id = tonumber(stream.segment_id) or 0
    sel.segment_key = trim(stream.segment_key or "")
    sel.topic_start_turn = tonumber(stream.topic_start_turn) or 0
    sel.anchor = make_anchor(sel.topic_start_turn)
    sel.sequence_key = make_sequence_key(scope_key, sel.segment_id)
    sel.orphaned = truthy(stream.orphan)
    sel.ambient = truthy(stream.ambient)
    return sel
end

local function trim_pending_queue(queue, cap)
    queue = type(queue) == "table" and queue or {}
    cap = math.max(1, math.floor(tonumber(cap) or 1))
    while #queue > cap do
        table.remove(queue, 1)
    end
    return queue
end

local function promote_local_pending(stream, selection)
    stream = type(stream) == "table" and stream or {}
    stream.pending = type(stream.pending) == "table" and stream.pending or {}
    stream.local_pending = type(stream.local_pending) == "table" and stream.local_pending or {}
    if #stream.local_pending <= 0 then
        return false
    end
    selection = type(selection) == "table" and selection or {}
    for _, event in ipairs(stream.local_pending) do
        if type(event) == "table" then
            event.pending_only = false
            event.local_only = false
            event.orphaned = false
            event.thread_id = tonumber(selection.thread_id) or tonumber(stream.thread_id) or 0
            event.thread_key = trim(selection.thread_key or stream.thread_key or "")
            event.segment_id = tonumber(selection.segment_id) or tonumber(stream.segment_id) or 0
            event.segment_key = trim(selection.segment_key or stream.segment_key or "")
            if trim(event.sequence_key or "") == "" then
                event.sequence_key = trim(selection.sequence_key or "")
            end
            if trim(event.anchor or "") == "" then
                event.anchor = trim(selection.anchor or "")
            end
            stream.pending[#stream.pending + 1] = event
        end
    end
    stream.local_pending = {}
    return true
end

local function collect_pending_rows(stream)
    stream = type(stream) == "table" and stream or {}
    local rows = {}
    for _, event in ipairs(stream.pending or {}) do
        if type(event) == "table" then
            rows[#rows + 1] = event
        end
    end
    for _, event in ipairs(stream.local_pending or {}) do
        if type(event) == "table" then
            rows[#rows + 1] = event
        end
    end
    table.sort(rows, function(a, b)
        local ta = math.max(0, math.floor(tonumber((a or {}).turn) or 0))
        local tb = math.max(0, math.floor(tonumber((b or {}).turn) or 0))
        if ta ~= tb then
            return ta < tb
        end
        return tostring((a or {}).sequence_key or "") < tostring((b or {}).sequence_key or "")
    end)
    return rows
end

local function default_selection()
    return {
        stream_id = nil,
        thread_id = 0,
        thread_key = "",
        segment_id = 0,
        segment_key = "",
        mode = "single",
        anchor = "",
        topic_start_turn = 0,
        sequence_key = "",
        use_local_sequence = false,
        segment_boundary = false,
        pending_only = false,
        local_only = false,
        orphaned = false,
        ambient = false,
        decision = "",
        confidence = 0.0,
        stability = 0.0,
        best_score = -1.0,
        second_score = -1.0,
        is_new = false,
        merged = false,
        dropped = false,
        reason = "",
        score_debug = {},
    }
end

function M.select(scope_key, query_vec, meta, turn)
    meta = type(meta) == "table" and meta or {}
    query_vec = type(query_vec) == "table" and query_vec or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))

    local sel = default_selection()
    if #query_vec <= 0 or turn <= 0 then
        sel.dropped = true
        sel.reason = "drop_missing_vec"
        return sel
    end

    local c = cfg()
    local opts = {
        max_streams = math.max(1, math.floor(tonumber(c.max_streams) or 6)),
        window_size = math.max(1, math.floor(tonumber(c.window_size) or 4)),
        assign_threshold = tonumber(c.assign_threshold) or 0.80,
        pending_threshold = tonumber(c.pending_threshold),
        pending_margin = math.max(0.0, tonumber(c.pending_margin) or 0.06),
        same_user_bonus = tonumber(c.same_user_bonus) or 0.06,
        participant_bonus = tonumber(c.participant_bonus) or 0.03,
        mention_bonus = tonumber(c.mention_bonus) or 0.05,
        addressee_hint_bonus = tonumber(c.addressee_hint_bonus) or 0.04,
        reply_cue_bonus = tonumber(c.reply_cue_bonus) or 0.05,
        reply_recent_turns = math.max(1, math.floor(tonumber(c.reply_recent_turns) or 6)),
        centroid_weight = tonumber(c.centroid_weight) or 0.42,
        tail_weight = tonumber(c.tail_weight) or 0.38,
        head_weight = tonumber(c.head_weight) or 0.20,
        stability_bonus = tonumber(c.stability_bonus) or 0.04,
        stability_turns = math.max(1, math.floor(tonumber(c.stability_turns) or 4)),
        age_penalty = tonumber(c.age_penalty) or 0.01,
        stale_turns = math.max(0, math.floor(tonumber(c.stale_turns) or 60)),
        orphan_stale_turns = math.max(1, math.floor(tonumber(c.orphan_stale_turns) or math.max(8, math.floor((tonumber(c.stale_turns) or 60) / 3)))),
        reaction_fallback_enabled = c.reaction_fallback_enabled ~= false,
        reaction_short_chars = math.max(1, math.floor(tonumber(c.reaction_short_chars) or 6)),
        reaction_max_chars = math.max(1, math.floor(tonumber(c.reaction_max_chars) or 10)),
        reaction_attach_score = tonumber(c.reaction_attach_score) or 0.46,
        reaction_dominant_score = tonumber(c.reaction_dominant_score) or 0.36,
        reaction_recent_turns = math.max(1, math.floor(tonumber(c.reaction_recent_turns) or 3)),
        reaction_stability_min = tonumber(c.reaction_stability_min) or 0.18,
        ambient_enabled = c.ambient_enabled ~= false,
        ambient_local_pending_cap = math.max(1, math.floor(tonumber(c.ambient_local_pending_cap) or 12)),
        local_pending_cap = math.max(1, math.floor(tonumber(c.local_pending_cap) or 4)),
        commit_idle_turns = math.max(1, math.floor(tonumber(c.commit_idle_turns) or 2)),
        reset_threshold = c.reset_threshold,
        merge_idle_turns = math.max(0, math.floor(tonumber(c.merge_idle_turns) or 8)),
        merge_streak_turns = math.max(0, math.floor(tonumber(c.merge_streak_turns) or 4)),
        reset_on_merge = c.reset_on_merge ~= false,
    }
    if opts.pending_threshold == nil then
        opts.pending_threshold = math.max(0.50, opts.assign_threshold - 0.08)
    end

    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
    scope.scope_key = scope_key
    prune_stale_streams(scope, turn, opts.stale_turns, opts.orphan_stale_turns)

    local streams = scope.streams
    local best = nil
    local second = nil
    local dominant = nil

    for id, stream in pairs(streams or {}) do
        if truthy((stream or {}).ambient) then
            streams[id] = ensure_stream_fields(stream, scope)
        else
        local score, features = score_stream(stream, query_vec, meta, turn, opts)
        local row = {
            stream_id = tonumber(id) or tonumber((stream or {}).id) or 0,
            score = score,
            sim_reset = math.max(
                tonumber((features or {}).sim_tail) or -1.0,
                tonumber((features or {}).sim_centroid) or -1.0
            ),
            stability = tonumber((features or {}).stability) or 0.0,
            features = features or {},
            stream = stream,
        }
        if tonumber(scope.last_assigned) == tonumber(row.stream_id) then
            dominant = row
        end
        if not best or (row.score or -1e9) > (best.score or -1e9) then
            second = best
            best = row
        elseif not second or (row.score or -1e9) > (second.score or -1e9) then
            second = row
        end
        end
    end

    sel.best_score = best and (tonumber(best.score) or -1.0) or -1.0
    sel.second_score = second and (tonumber(second.score) or -1.0) or -1.0

    local stream_count = count_streams(streams)
    local can_create = stream_count < opts.max_streams
    local best_margin = sel.best_score - (second and (tonumber(second.score) or -1.0) or -1.0)

    local assign_ok = best and (tonumber(best.score) or -1.0) >= opts.assign_threshold
    local pending_ok = best and (tonumber(best.score) or -1.0) >= opts.pending_threshold
    local ambiguous = pending_ok and second and best_margin < opts.pending_margin
    local reaction_row, reaction_reason = nil, ""
    if not assign_ok then
        reaction_row, reaction_reason = choose_reaction_fallback(scope, best, dominant, meta, turn, opts)
    end
    local ambient_id, ambient_stream = 0, nil
    if opts.ambient_enabled and stream_count > 0 and not assign_ok and reaction_row == nil and is_reaction_like_text(current_text(meta), opts) then
        ambient_id, ambient_stream = find_ambient_stream(scope)
        if ambient_id <= 0 or type(ambient_stream) ~= "table" then
            ambient_stream = create_stream(scope, scope_key, turn, query_vec, meta, { ambient = true })
            ambient_id = tonumber(ambient_stream.id) or 0
        else
            ambient_stream = ensure_stream_fields(ambient_stream, scope)
            scope.streams[ambient_id] = ambient_stream
        end
    end

    if assign_ok then
        local stream_id = tonumber(best.stream_id) or 0
        if stream_id > 0 and streams[stream_id] then
            streams[stream_id] = ensure_stream_fields(streams[stream_id], scope)
            if truthy(streams[stream_id].orphan) then
                streams[stream_id].orphan = false
                sel.reason = "promote_orphan"
            end
            sel.stream_id = stream_id
            sel.confidence = tonumber(best.score) or 0.0
            sel.stability = tonumber(best.stability) or 0.0
            sel.score_debug = deep_copy(best.features or {})
            sel.decision = "attach"
            if sel.reason == "" then
                sel.reason = "assign"
            end

            if should_reset_topic(tonumber(best.sim_reset) or 0.0, opts) then
                local s = streams[stream_id]
                sel.segment_boundary = true
                s.segment_id = alloc_segment_id(scope)
                s.segment_key = make_segment_key(scope_key, tonumber(s.thread_id) or stream_id, tonumber(s.segment_id) or 0)
                s.topic_start_turn = turn
                s.window = { query_vec }
                s.centroid = query_vec
                s.head_vec = query_vec
                s.tail_vec = query_vec
                s.last_turn = turn
                s.last_user_id = normalize_user_id(meta)
                s.last_actor_key = normalize_actor_key(meta)
                s.last_text = current_text(meta)
                sel.reason = "reset_topic"
            end
        end
    elseif reaction_row and streams[tonumber(reaction_row.stream_id) or 0] then
        local stream_id = tonumber(reaction_row.stream_id) or 0
        streams[stream_id] = ensure_stream_fields(streams[stream_id], scope)
        if truthy(streams[stream_id].orphan) then
            streams[stream_id].orphan = false
        end
        sel.stream_id = stream_id
        sel.confidence = clamp((tonumber(reaction_row.score) or 0.0) + 0.04, 0.0, 1.25)
        sel.stability = tonumber(reaction_row.stability) or tonumber((streams[stream_id] or {}).stability) or 0.0
        sel.score_debug = deep_copy(reaction_row.features or {})
        sel.score_debug.reaction_fallback = reaction_reason
        sel.decision = "attach"
        sel.reason = reaction_reason ~= "" and reaction_reason or "attach_reaction"
    elseif ambient_id > 0 and ambient_stream then
        sel.stream_id = ambient_id
        sel.pending_only = true
        sel.local_only = true
        sel.ambient = true
        sel.confidence = math.max(0.0, tonumber(best and best.score or 0.0) or 0.0)
        sel.stability = tonumber(ambient_stream.stability) or 0.0
        sel.score_debug = {
            ambient = true,
            best_score = sel.best_score,
            second_score = sel.second_score,
            reaction_like = true,
        }
        sel.decision = "ambient"
        sel.reason = "ambient_attach"
    elseif pending_ok and (ambiguous or not can_create) then
        local stream_id = best and (tonumber(best.stream_id) or 0) or 0
        if stream_id > 0 and streams[stream_id] then
            streams[stream_id] = ensure_stream_fields(streams[stream_id], scope)
            sel.stream_id = stream_id
            sel.confidence = tonumber(best.score) or 0.0
            sel.stability = tonumber(best.stability) or 0.0
            sel.score_debug = deep_copy(best.features or {})
            sel.pending_only = true
            sel.local_only = true
            sel.decision = "keep_pending"
            sel.reason = "keep_pending"
        else
            sel.dropped = true
            sel.decision = "drop"
            sel.reason = "drop_pending_missing_stream"
        end
    elseif can_create then
        local stream = create_stream(scope, scope_key, turn, query_vec, meta, { orphan = false })
        sel.stream_id = tonumber(stream.id) or 0
        sel.is_new = true
        sel.confidence = 1.0
        sel.stability = tonumber(stream.stability) or 0.0
        sel.decision = "open"
        sel.reason = "create"
    else
        local orphan_stream = create_stream(scope, scope_key, turn, query_vec, meta, { orphan = true })
        orphan_stream.last_turn = turn
        orphan_stream.last_user_id = normalize_user_id(meta)
        orphan_stream.last_actor_key = normalize_actor_key(meta)
        orphan_stream.last_text = current_text(meta)
        sel.stream_id = tonumber(orphan_stream.id) or 0
        sel.confidence = best and (tonumber(best.score) or 0.0) or 0.0
        sel.stability = best and (tonumber(best.stability) or 0.0) or 0.0
        sel.score_debug = best and deep_copy(best.features or {}) or {}
        sel.pending_only = true
        sel.local_only = true
        sel.orphaned = true
        sel.is_new = true
        sel.decision = "orphan"
        sel.reason = "orphan_overflow"
    end

    stream_count = count_streams(streams)
    sel.mode = stream_count > 1 and "split" or "single"

    if sel.stream_id and streams[sel.stream_id] then
        local s = streams[sel.stream_id]
        ensure_stream_fields(s, scope)
        selection_from_stream(sel, s, scope_key)
        if (tonumber(sel.stability) or 0.0) <= 0.0 then
            sel.stability = tonumber(s.stability) or 0.0
        end
    end

    -- Update assignment streak for merge decisions.
    if sel.stream_id and sel.pending_only ~= true then
        if tonumber(scope.last_assigned) == tonumber(sel.stream_id) then
            scope.streak = math.max(0, tonumber(scope.streak) or 0) + 1
        else
            scope.last_assigned = sel.stream_id
            scope.streak = 1
        end
    else
        scope.streak = 0
    end

    -- Merge: if one stream dominates recently and others are idle.
    if sel.mode == "split"
        and opts.merge_streak_turns > 0
        and (tonumber(scope.streak) or 0) >= opts.merge_streak_turns
        and scope.last_assigned
    then
        local keep_id = tonumber(scope.last_assigned) or 0
        if keep_id > 0 and streams[keep_id] then
            local ok_merge = true
            for id, s in pairs(streams) do
                if tonumber(id) ~= keep_id then
                    local age = turn - (tonumber((s or {}).last_turn) or 0)
                    if age < opts.merge_idle_turns then
                        ok_merge = false
                        break
                    end
                end
            end
            if ok_merge then
                for id in pairs(streams) do
                    if tonumber(id) ~= keep_id then
                        streams[id] = nil
                    end
                end
                local keep = ensure_stream_fields(streams[keep_id], scope)
                keep.segment_id = alloc_segment_id(scope)
                keep.segment_key = make_segment_key(scope_key, tonumber(keep.thread_id) or keep_id, tonumber(keep.segment_id) or 0)
                sel.segment_boundary = true
                if opts.reset_on_merge then
                    if keep then
                        keep.topic_start_turn = turn
                        keep.window = { query_vec }
                        keep.centroid = query_vec
                        keep.head_vec = query_vec
                        keep.tail_vec = query_vec
                        keep.last_turn = turn
                        keep.last_user_id = normalize_user_id(meta)
                        keep.last_actor_key = normalize_actor_key(meta)
                        keep.last_text = current_text(meta)
                    end
                end
                scope.last_assigned = keep_id
                scope.streak = 1
                sel.merged = true
                sel.mode = "single"
                sel.stream_id = keep_id
                selection_from_stream(sel, keep, scope_key)
                sel.reason = "merge"
            end
        end
    end

    if count_streams(streams) > 1 then
        local hold = math.max(opts.merge_idle_turns, opts.commit_idle_turns)
        scope.local_sequence_until_turn = math.max(tonumber(scope.local_sequence_until_turn) or 0, turn + hold)
    elseif sel.segment_boundary then
        scope.local_sequence_until_turn = math.max(tonumber(scope.local_sequence_until_turn) or 0, turn + opts.commit_idle_turns)
    end

    sel.use_local_sequence = sel.pending_only == true
        or count_streams(streams) > 1
        or turn <= (tonumber(scope.local_sequence_until_turn) or 0)
    if (tonumber(sel.segment_id) or 0) > 0 then
        sel.sequence_key = make_sequence_key(scope_key, sel.segment_id)
        sel.segment_key = make_segment_key(scope_key, tonumber(sel.thread_id) or tonumber(sel.stream_id) or 0, sel.segment_id)
    end

    for id, stream in pairs(streams) do
        stream = ensure_stream_fields(stream, scope)
        streams[id] = stream
        local age = turn - (tonumber(stream.last_turn) or 0)
        if truthy(stream.orphan)
            and age >= opts.orphan_stale_turns
            and #collect_pending_rows(stream) <= 0
        then
            streams[id] = nil
        end
    end

    if trim(sel.reason or "") == "" then
        if sel.decision == "attach" then
            sel.reason = "assign"
        elseif sel.decision == "ambient" then
            sel.reason = "ambient_attach"
        elseif sel.decision == "keep_pending" then
            sel.reason = "keep_pending"
        elseif sel.decision == "open" then
            sel.reason = "create"
        elseif sel.decision == "orphan" then
            sel.reason = "orphan_overflow"
        elseif sel.decision == "drop" then
            sel.reason = "drop"
        else
            sel.reason = "noop"
        end
    end

    return sel
end

function M.observe(scope_key, selection, vec, meta, turn)
    selection = type(selection) == "table" and selection or {}
    vec = type(vec) == "table" and vec or {}
    meta = type(meta) == "table" and meta or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    if #vec <= 0 or turn <= 0 then
        return false
    end
    if selection.dropped == true then
        return false
    end
    if selection.pending_only == true and selection.orphaned ~= true and selection.ambient ~= true then
        return false
    end

    local c = cfg()
    local window_size = math.max(1, math.floor(tonumber(c.window_size) or 4))
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
    scope.scope_key = scope_key
    local streams = scope.streams
    local sid = tonumber(selection.stream_id) or 0
    if sid <= 0 or not streams[sid] then
        return false
    end
    local s = ensure_stream_fields(streams[sid], scope)
    streams[sid] = s
    s.window = type(s.window) == "table" and s.window or {}
    s.window[#s.window + 1] = vec
    while #s.window > window_size do
        table.remove(s.window, 1)
    end
    s.centroid = tool.average_vectors(s.window)
    if type(s.head_vec) ~= "table" or #s.head_vec <= 0 then
        s.head_vec = vec
    end
    s.tail_vec = vec
    s.last_turn = turn
    s.last_user_id = normalize_user_id(meta)
    s.last_actor_key = normalize_actor_key(meta)
    s.last_text = current_text(meta)
    s.turn_count = math.max(0, math.floor(tonumber(s.turn_count) or 0)) + 1
    s.confirmed_turns = math.max(0, math.floor(tonumber(s.confirmed_turns) or 0)) + 1
    if s.last_actor_key ~= "" then
        s.participants[s.last_actor_key] = (tonumber(s.participants[s.last_actor_key]) or 0) + 1
    end
    for _, alias in ipairs(actor_aliases(meta)) do
        s.actor_aliases[alias] = (tonumber(s.actor_aliases[alias]) or 0) + 1
    end
    for _, alias in ipairs(extract_mentions(s.last_text)) do
        s.addressee_hints[alias] = (tonumber(s.addressee_hints[alias]) or 0) + 1
    end
    local prev_confidence = tonumber(s.confidence) or 0.0
    local current_confidence = clamp(tonumber(selection.confidence) or 0.0, 0.0, 1.25)
    if prev_confidence <= 0.0 then
        s.confidence = current_confidence
    else
        s.confidence = 0.75 * prev_confidence + 0.25 * current_confidence
    end
    s.stability = compute_stream_stability(s, turn, {
        stability_turns = tonumber(c.stability_turns) or 4,
        reply_recent_turns = tonumber(c.reply_recent_turns) or 6,
    })
    return true
end

local function flush_pending(stream, out, keep_last)
    stream = type(stream) == "table" and stream or {}
    stream.pending = type(stream.pending) == "table" and stream.pending or {}
    out = type(out) == "table" and out or {}
    keep_last = math.max(0, math.floor(tonumber(keep_last) or 0))
    while #stream.pending > keep_last do
        out[#out + 1] = table.remove(stream.pending, 1)
    end
    return out
end

local function flush_ready_streams(scope, current_stream_id, turn, idle_turns, out)
    scope = type(scope) == "table" and scope or {}
    local streams = type(scope.streams) == "table" and scope.streams or {}
    current_stream_id = tonumber(current_stream_id) or 0
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    idle_turns = math.max(1, math.floor(tonumber(idle_turns) or 1))
    out = type(out) == "table" and out or {}
    for id, stream in pairs(streams) do
        stream = ensure_stream_fields(stream, scope)
        streams[id] = stream
        if #stream.pending > 0 then
            local keep_last = 1
            if tonumber(id) ~= current_stream_id then
                local age = turn - (tonumber(stream.last_turn) or 0)
                if age >= idle_turns then
                    keep_last = 0
                end
            end
            flush_pending(stream, out, keep_last)
        end
    end
    return out
end

function M.stage(scope_key, selection, event, turn)
    selection = type(selection) == "table" and selection or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    local c = cfg()
    local idle_turns = math.max(1, math.floor(tonumber(c.commit_idle_turns) or 2))
    local chunk_turns = math.max(1, math.floor(tonumber(c.commit_chunk_turns) or 2))
    local local_pending_cap = math.max(1, math.floor(tonumber(c.local_pending_cap) or 4))
    local ambient_local_pending_cap = math.max(1, math.floor(tonumber(c.ambient_local_pending_cap) or 12))
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
    scope.scope_key = scope_key
    local out = {}
    local current_stream_id = 0
    if type(event) == "table" then
        current_stream_id = tonumber(selection.stream_id) or 0
    end

    out = flush_ready_streams(scope, current_stream_id, turn, idle_turns, out)

    local sid = tonumber(selection.stream_id) or 0
    if sid <= 0 or selection.dropped == true or type(event) ~= "table" then
        return out
    end
    local stream = scope.streams[sid]
    if type(stream) ~= "table" then
        return out
    end
    stream = ensure_stream_fields(stream, scope)
    scope.streams[sid] = stream

    if selection.segment_boundary == true then
        flush_pending(stream, out, 0)
    end

    event.turn = math.max(0, math.floor(tonumber(event.turn) or turn))
    event.stream_id = sid
    event.thread_id = tonumber(selection.thread_id) or tonumber(stream.thread_id) or sid
    event.thread_key = trim(selection.thread_key or stream.thread_key or "")
    event.segment_id = tonumber(selection.segment_id) or tonumber(stream.segment_id) or 0
    event.segment_key = trim(selection.segment_key or stream.segment_key or "")
    event.sequence_key = trim(selection.sequence_key or "")
    event.anchor = trim(event.anchor or selection.anchor or "")
    event.pending_only = selection.pending_only == true
    event.local_only = selection.local_only == true
    event.orphaned = selection.orphaned == true
    event.ambient = selection.ambient == true

    if selection.pending_only == true then
        stream.local_pending[#stream.local_pending + 1] = event
        trim_pending_queue(stream.local_pending, selection.ambient == true and ambient_local_pending_cap or local_pending_cap)
        stream.last_turn = turn
        stream.last_user_id = trim(stream.last_user_id or "")
        return out
    end

    promote_local_pending(stream, selection)
    stream.pending[#stream.pending + 1] = event

    if selection.use_local_sequence == true then
        flush_pending(stream, out, 1)
    elseif #stream.pending >= chunk_turns then
        flush_pending(stream, out, 0)
    else
        stream.last_turn = turn
    end

    return out
end

function M.pending_context(scope_key, selection, opts)
    selection = type(selection) == "table" and selection or {}
    opts = type(opts) == "table" and opts or {}
    local sid = tonumber(selection.stream_id) or 0
    if sid <= 0 then
        return ""
    end
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
    scope.scope_key = scope_key
    local stream = scope.streams[sid]
    if type(stream) ~= "table" then
        return ""
    end
    stream = ensure_stream_fields(stream, scope)
    local pending = collect_pending_rows(stream)
    if #pending <= 0 then
        return ""
    end

    local cap = math.max(1, math.floor(tonumber(opts.pending_context_turns) or tonumber(cfg().pending_context_turns) or 2))
    local user_cap = math.max(40, math.floor(tonumber(opts.user_chars) or 140))
    local assistant_cap = math.max(40, math.floor(tonumber(opts.assistant_chars) or 180))
    local start_idx = math.max(1, #pending - cap + 1)
    local parts = {}
    for i = start_idx, #pending do
        local event = pending[i]
        if type(event) == "table" then
            local user_text = util.utf8_take(trim(event.user_input or event.embed_input or ""), user_cap)
            local assistant_text = util.utf8_take(trim(event.assistant_text or ""), assistant_cap)
            if user_text ~= "" or assistant_text ~= "" then
                parts[#parts + 1] = string.format(
                    "第%d轮\n用户：%s\n助手：%s",
                    math.max(0, math.floor(tonumber(event.turn) or 0)),
                    user_text,
                    assistant_text
                )
            end
        end
    end
    if #parts <= 0 then
        return ""
    end
    local title = selection.ambient == true and "【直播氛围上下文】"
        or (selection.orphaned == true and "【未确认线程上下文】" or "【未确认分段上下文】")
    return title .. "\n" .. table.concat(parts, "\n\n")
end

function M.flush_all()
    local out = {}
    for _, scope in pairs(_state.scopes or {}) do
        local streams = type((scope or {}).streams) == "table" and (scope or {}).streams or {}
        for _, stream in pairs(streams) do
            stream = ensure_stream_fields(stream, scope)
            flush_pending(stream, out, 0)
        end
    end
    return out
end

local function export_stream(scope_key, stream, scope)
    scope = type(scope) == "table" and scope or { scope_key = scope_key }
    stream = ensure_stream_fields(deep_copy(stream), scope)
    return {
        id = tonumber(stream.id) or 0,
        thread_id = tonumber(stream.thread_id) or tonumber(stream.id) or 0,
        thread_key = trim(stream.thread_key or ""),
        segment_id = tonumber(stream.segment_id) or 0,
        segment_key = trim(stream.segment_key or ""),
        created_turn = math.max(0, math.floor(tonumber(stream.created_turn) or 0)),
        topic_start_turn = math.max(0, math.floor(tonumber(stream.topic_start_turn) or 0)),
        last_turn = math.max(0, math.floor(tonumber(stream.last_turn) or 0)),
        last_user_id = trim(stream.last_user_id or ""),
        last_actor_key = trim(stream.last_actor_key or ""),
        last_text = trim(stream.last_text or ""),
        turn_count = math.max(0, math.floor(tonumber(stream.turn_count) or 0)),
        confirmed_turns = math.max(0, math.floor(tonumber(stream.confirmed_turns) or 0)),
        participants = deep_copy(stream.participants),
        actor_aliases = deep_copy(stream.actor_aliases),
        addressee_hints = deep_copy(stream.addressee_hints),
        head_vec = deep_copy(stream.head_vec),
        tail_vec = deep_copy(stream.tail_vec),
        confidence = tonumber(stream.confidence) or 0.0,
        stability = clamp(tonumber(stream.stability) or 0.0, 0.0, 1.0),
        orphan = truthy(stream.orphan),
        ambient = truthy(stream.ambient),
        pending = deep_copy(stream.pending),
        local_pending = deep_copy(stream.local_pending),
        window = deep_copy(stream.window),
        centroid = deep_copy(stream.centroid),
    }
end

function M.export_scope_state(scope_key)
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
    scope.scope_key = scope_key
    local out = {
        scope_key = scope_key,
        next_stream_id = math.max(1, math.floor(tonumber(scope.next_stream_id) or 1)),
        next_segment_id = math.max(1, math.floor(tonumber(scope.next_segment_id) or 1)),
        last_assigned = tonumber(scope.last_assigned) or nil,
        streak = math.max(0, math.floor(tonumber(scope.streak) or 0)),
        local_sequence_until_turn = math.max(0, math.floor(tonumber(scope.local_sequence_until_turn) or 0)),
        streams = {},
    }
    for id, stream in pairs(scope.streams or {}) do
        local sid = tonumber(id) or tonumber((stream or {}).id) or 0
        if sid > 0 then
            out.streams[sid] = export_stream(scope_key, stream, scope)
        end
    end
    return out
end

function M.export_state()
    local out = {
        version = 1,
        scopes = {},
    }
    for scope_key in pairs((_state or {}).scopes or {}) do
        out.scopes[scope_key] = M.export_scope_state(scope_key)
    end
    return out
end

function M.import_scope_state(scope_key, scope_state)
    scope_state = type(scope_state) == "table" and scope_state or {}
    scope_key = trim(scope_key or scope_state.scope_key or "")
    if scope_key == "" then
        scope_key = "global"
    end

    local scope = {
        scope_key = scope_key,
        next_stream_id = math.max(1, math.floor(tonumber(scope_state.next_stream_id) or 1)),
        next_segment_id = math.max(1, math.floor(tonumber(scope_state.next_segment_id) or 1)),
        streams = {},
        last_assigned = tonumber(scope_state.last_assigned) or nil,
        streak = math.max(0, math.floor(tonumber(scope_state.streak) or 0)),
        local_sequence_until_turn = math.max(0, math.floor(tonumber(scope_state.local_sequence_until_turn) or 0)),
    }

    for id, stream in pairs(scope_state.streams or {}) do
        local sid = tonumber(id) or tonumber((stream or {}).id) or 0
        if sid > 0 then
            local row = deep_copy(stream)
            row.id = sid
            scope.streams[sid] = ensure_stream_fields(row, scope)
        end
    end

    _state.scopes[scope_key] = scope
    return true
end

function M.import_state(state)
    state = type(state) == "table" and state or {}
    _state.scopes = {}
    for scope_key, scope_state in pairs(state.scopes or {}) do
        M.import_scope_state(scope_key, scope_state)
    end
    return true
end

function M.reset_runtime()
    _state.scopes = {}
    return true
end

return M
