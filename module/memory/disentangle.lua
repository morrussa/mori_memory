local config = require("module.config")
local tool = require("module.tool")
local util = require("mori_memory.util")

local M = {}

local _state = {
    scopes = {},
}

local function trim(s)
    return util.trim(s)
end

local function cfg()
    return ((config.settings or {}).disentangle or {})
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
    for _ in pairs(streams or {}) do
        n = n + 1
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

local function score_stream(stream, vec, meta, turn, opts)
    stream = type(stream) == "table" and stream or {}
    vec = type(vec) == "table" and vec or {}
    opts = type(opts) == "table" and opts or {}
    local centroid = stream.centroid
    if type(centroid) ~= "table" or #centroid <= 0 or #vec <= 0 then
        return -1.0, 0.0
    end

    local sim_centroid = tonumber(tool.cosine_similarity(centroid, vec)) or 0.0
    local score = sim_centroid

    local user_id = normalize_user_id(meta)
    local last_user_id = trim(stream.last_user_id or "")
    if user_id ~= "" and last_user_id ~= "" and user_id == last_user_id then
        score = score + (tonumber(opts.same_user_bonus) or 0.0)
    end

    local age_penalty = tonumber(opts.age_penalty) or 0.0
    local age = math.max(0, (tonumber(turn) or 0) - (tonumber(stream.last_turn) or 0))
    if age_penalty > 0.0 and age > 0 then
        score = score - age_penalty * math.min(age, 32)
    end

    return score, sim_centroid
end

local function should_reset_topic(sim_centroid, opts)
    opts = type(opts) == "table" and opts or {}
    local reset_th = tonumber(opts.reset_threshold)
    if reset_th == nil then
        return false
    end
    return (tonumber(sim_centroid) or 0.0) < reset_th
end

local function prune_stale_streams(scope, turn, stale_turns)
    scope = type(scope) == "table" and scope or {}
    local streams = type(scope.streams) == "table" and scope.streams or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    stale_turns = math.max(0, math.floor(tonumber(stale_turns) or 0))
    if stale_turns <= 0 then
        return 0
    end
    local removed = 0
    for id, stream in pairs(streams) do
        local last_turn = tonumber((stream or {}).last_turn) or 0
        local pending = type((stream or {}).pending) == "table" and (stream or {}).pending or {}
        if last_turn > 0 and #pending <= 0 and (turn - last_turn) >= stale_turns then
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

local function alloc_segment_id(scope)
    scope = type(scope) == "table" and scope or {}
    local seg_id = math.max(1, math.floor(tonumber(scope.next_segment_id) or 1))
    scope.next_segment_id = seg_id + 1
    return seg_id
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
    stream.pending = type(stream.pending) == "table" and stream.pending or {}
    local seg_id = tonumber(stream.segment_id) or 0
    if seg_id <= 0 then
        stream.segment_id = alloc_segment_id(scope)
    else
        stream.segment_id = math.floor(seg_id)
    end
    return stream
end

local function default_selection()
    return {
        stream_id = nil,
        segment_id = 0,
        mode = "single",
        anchor = "",
        topic_start_turn = 0,
        sequence_key = "",
        use_local_sequence = false,
        segment_boundary = false,
        confidence = 0.0,
        best_score = -1.0,
        second_score = -1.0,
        is_new = false,
        merged = false,
        dropped = false,
        reason = "",
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
        max_streams = math.max(1, math.floor(tonumber(c.max_streams) or 3)),
        window_size = math.max(1, math.floor(tonumber(c.window_size) or 4)),
        assign_threshold = tonumber(c.assign_threshold) or 0.80,
        same_user_bonus = tonumber(c.same_user_bonus) or 0.06,
        age_penalty = tonumber(c.age_penalty) or 0.01,
        stale_turns = math.max(0, math.floor(tonumber(c.stale_turns) or 60)),
        commit_idle_turns = math.max(1, math.floor(tonumber(c.commit_idle_turns) or 2)),
        reset_threshold = c.reset_threshold,
        merge_idle_turns = math.max(0, math.floor(tonumber(c.merge_idle_turns) or 8)),
        merge_streak_turns = math.max(0, math.floor(tonumber(c.merge_streak_turns) or 4)),
        reset_on_merge = c.reset_on_merge ~= false,
    }

    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
    prune_stale_streams(scope, turn, opts.stale_turns)

    local streams = scope.streams
    local best = nil
    local second = nil

    for id, stream in pairs(streams or {}) do
        local score, sim_centroid = score_stream(stream, query_vec, meta, turn, opts)
        local row = {
            stream_id = tonumber(id) or tonumber((stream or {}).id) or 0,
            score = score,
            sim_centroid = sim_centroid,
            stream = stream,
        }
        if not best or (row.score or -1e9) > (best.score or -1e9) then
            second = best
            best = row
        elseif not second or (row.score or -1e9) > (second.score or -1e9) then
            second = row
        end
    end

    sel.best_score = best and (tonumber(best.score) or -1.0) or -1.0
    sel.second_score = second and (tonumber(second.score) or -1.0) or -1.0

    local stream_count = count_streams(streams)
    local can_create = stream_count < opts.max_streams

    local assign_ok = best and (tonumber(best.score) or -1.0) >= opts.assign_threshold

    if assign_ok then
        local stream_id = tonumber(best.stream_id) or 0
        if stream_id > 0 and streams[stream_id] then
            streams[stream_id] = ensure_stream_fields(streams[stream_id], scope)
            sel.stream_id = stream_id
            sel.confidence = tonumber(best.score) or 0.0
            sel.reason = "assign"

            if should_reset_topic(tonumber(best.sim_centroid) or 0.0, opts) then
                local s = streams[stream_id]
                sel.segment_boundary = true
                s.segment_id = alloc_segment_id(scope)
                s.topic_start_turn = turn
                s.window = { query_vec }
                s.centroid = query_vec
                s.last_turn = turn
                s.last_user_id = normalize_user_id(meta)
                sel.reason = "reset_topic"
            end
        end
    elseif can_create then
        local new_id = tonumber(scope.next_stream_id) or 1
        scope.next_stream_id = new_id + 1
        streams[new_id] = {
            id = new_id,
            segment_id = alloc_segment_id(scope),
            created_turn = turn,
            topic_start_turn = turn,
            window = { query_vec },
            centroid = query_vec,
            last_turn = turn,
            last_user_id = normalize_user_id(meta),
            pending = {},
        }
        sel.stream_id = new_id
        sel.is_new = true
        sel.confidence = 1.0
        sel.reason = "create"
    else
        sel.dropped = true
        sel.confidence = best and (tonumber(best.score) or 0.0) or 0.0
        sel.reason = "drop_full"
    end

    stream_count = count_streams(streams)
    sel.mode = stream_count > 1 and "split" or "single"

    if sel.stream_id and streams[sel.stream_id] then
        local s = streams[sel.stream_id]
        ensure_stream_fields(s, scope)
        sel.topic_start_turn = tonumber(s.topic_start_turn) or 0
        sel.segment_id = tonumber(s.segment_id) or 0
        sel.anchor = make_anchor(sel.topic_start_turn)
    end

    -- Update assignment streak for merge decisions.
    if sel.stream_id then
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
                sel.segment_boundary = true
                if opts.reset_on_merge then
                    if keep then
                        keep.topic_start_turn = turn
                        keep.window = { query_vec }
                        keep.centroid = query_vec
                        keep.last_turn = turn
                        keep.last_user_id = normalize_user_id(meta)
                    end
                end
                scope.last_assigned = keep_id
                scope.streak = 1
                sel.merged = true
                sel.mode = "single"
                sel.stream_id = keep_id
                sel.segment_id = tonumber(keep.segment_id) or 0
                sel.topic_start_turn = tonumber(streams[keep_id].topic_start_turn) or turn
                sel.anchor = make_anchor(sel.topic_start_turn)
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

    sel.use_local_sequence = count_streams(streams) > 1 or turn <= (tonumber(scope.local_sequence_until_turn) or 0)
    if sel.use_local_sequence and (tonumber(sel.segment_id) or 0) > 0 then
        sel.sequence_key = make_sequence_key(scope_key, sel.segment_id)
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

    local c = cfg()
    local window_size = math.max(1, math.floor(tonumber(c.window_size) or 4))
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
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
    s.last_turn = turn
    s.last_user_id = normalize_user_id(meta)
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
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
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
    event.segment_id = tonumber(selection.segment_id) or tonumber(stream.segment_id) or 0
    event.sequence_key = trim(selection.sequence_key or "")
    event.anchor = trim(event.anchor or selection.anchor or "")
    stream.pending[#stream.pending + 1] = event

    if selection.use_local_sequence == true then
        flush_pending(stream, out, 1)
    else
        flush_pending(stream, out, 0)
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
    local stream = scope.streams[sid]
    if type(stream) ~= "table" then
        return ""
    end
    stream = ensure_stream_fields(stream, scope)
    local pending = type(stream.pending) == "table" and stream.pending or {}
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
    return "【未确认分段上下文】\n" .. table.concat(parts, "\n\n")
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

return M
