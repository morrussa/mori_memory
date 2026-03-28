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

local function cfg()
    return ((config.settings or {}).disentangle or {})
end

local function router_opts()
    local c = cfg()
    return {
        recent_window_size = math.max(8, math.floor(tonumber(c.recent_window_size) or 64)),
        top_k = math.max(1, math.floor(tonumber(c.top_k) or 1)),
        local_bfs_turns = math.max(1, math.floor(tonumber(c.local_bfs_turns) or 6)),
        semantic_weight = tonumber(c.semantic_weight) or 0.42,
        same_user_bonus = tonumber(c.same_user_bonus) or 0.06,
        mention_bonus = tonumber(c.mention_bonus) or 0.15,
        age_penalty = tonumber(c.age_penalty) or 0.01,
        strong_edge_score = tonumber(c.strong_edge_score) or 0.12,
        read_anchor_score = tonumber(c.read_anchor_score) or 0.12,
        read_traverse_score = tonumber(c.read_traverse_score) or 0.12,
        noise_outgoing_score = tonumber(c.noise_outgoing_score) or 0.06,
    }
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

local function actor_alias_map(meta)
    meta = type(meta) == "table" and meta or {}
    local out = {}
    local candidates = {
        meta.user_id,
        meta.uid,
        meta.nickname,
        meta.nick,
    }
    for _, raw in ipairs(candidates) do
        local alias = normalize_alias(raw)
        if alias ~= "" then
            out[alias] = true
        end
    end
    return out
end

local function extract_mentions(text)
    text = tostring(text or "")
    local out = {}
    for raw in text:gmatch("@([^%s，。！？；：,%.!%?;:%)%]%}]+)") do
        local alias = normalize_alias(raw)
        if alias ~= "" then
            out[alias] = true
        end
    end
    return out
end

local function default_selection()
    return {
        stream_id = nil,
        thread_id = 0,
        thread_key = "",
        segment_id = 0,
        segment_key = "",
        mode = "dag",
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
        selected_turns = {},
    }
end

local function make_recent_window(cap)
    return {
        cap = math.max(1, math.floor(tonumber(cap) or 1)),
        head = 1,
        count = 0,
        items = {},
    }
end

local function ring_list(window)
    window = type(window) == "table" and window or make_recent_window(1)
    local out = {}
    local count = math.max(0, math.floor(tonumber(window.count) or 0))
    local cap = math.max(1, math.floor(tonumber(window.cap) or 1))
    local head = math.max(1, math.floor(tonumber(window.head) or 1))
    for offset = 0, count - 1 do
        local idx = ((head - 1 + offset) % cap) + 1
        local turn = tonumber(window.items[idx])
        if turn and turn > 0 then
            out[#out + 1] = math.floor(turn)
        end
    end
    return out
end

local function ring_push(window, turn)
    window = type(window) == "table" and window or make_recent_window(1)
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    if turn <= 0 then
        return nil
    end

    local cap = math.max(1, math.floor(tonumber(window.cap) or 1))
    local count = math.max(0, math.floor(tonumber(window.count) or 0))
    local head = math.max(1, math.floor(tonumber(window.head) or 1))

    if count < cap then
        local idx = ((head - 1 + count) % cap) + 1
        window.items[idx] = turn
        window.count = count + 1
        window.head = head
        return nil
    end

    local evicted = tonumber(window.items[head]) or 0
    window.items[head] = turn
    window.head = (head % cap) + 1
    window.count = cap
    if evicted > 0 then
        return math.floor(evicted)
    end
    return nil
end

local function make_component_anchor(component_id)
    return string.format("D:%d", math.max(1, math.floor(tonumber(component_id) or 1)))
end

local function make_thread_key(scope_key, component_id)
    scope_key = trim(scope_key)
    if scope_key == "" then
        scope_key = "global"
    end
    return string.format("%s$dag:%d", scope_key, math.max(1, math.floor(tonumber(component_id) or 1)))
end

local function make_segment_key(scope_key, component_id, root_turn)
    return string.format(
        "%s/seg:%d",
        make_thread_key(scope_key, component_id),
        math.max(1, math.floor(tonumber(root_turn) or 1))
    )
end

local function recent_turns(scope)
    local out = {}
    local seen = {}
    for _, turn in ipairs(ring_list((scope or {}).recent_window)) do
        if not seen[turn] then
            seen[turn] = true
            if type((scope or {}).nodes) == "table" and type(scope.nodes[turn]) == "table" then
                out[#out + 1] = turn
            end
        end
    end
    return out
end

local function ensure_scope(scope_key)
    scope_key = trim(scope_key)
    if scope_key == "" then
        scope_key = "global"
    end

    local scopes = _state.scopes
    local scope = scopes[scope_key]
    if type(scope) ~= "table" then
        scope = {
            scope_key = scope_key,
            next_component_id = 1,
            nodes = {},
            recent_window = make_recent_window(router_opts().recent_window_size),
            ready_events = {},
        }
        scopes[scope_key] = scope
    end

    scope.scope_key = scope_key
    scope.next_component_id = math.max(1, math.floor(tonumber(scope.next_component_id) or 1))
    scope.nodes = type(scope.nodes) == "table" and scope.nodes or {}
    scope.ready_events = type(scope.ready_events) == "table" and scope.ready_events or {}
    scope.recent_window = type(scope.recent_window) == "table" and scope.recent_window or make_recent_window(router_opts().recent_window_size)

    local cap = math.max(1, math.floor(tonumber(scope.recent_window.cap) or router_opts().recent_window_size))
    local expected_cap = router_opts().recent_window_size
    if cap ~= expected_cap then
        local turns = recent_turns(scope)
        scope.recent_window = make_recent_window(expected_cap)
        local start_idx = math.max(1, #turns - expected_cap + 1)
        for idx = start_idx, #turns do
            ring_push(scope.recent_window, turns[idx])
        end
    end

    return scope_key, scope
end

local function alloc_component_id(scope)
    scope = type(scope) == "table" and scope or {}
    local value = math.max(1, math.floor(tonumber(scope.next_component_id) or 1))
    scope.next_component_id = value + 1
    return value
end

local function node_stability(node)
    node = type(node) == "table" and node or {}
    local total = 0.0
    local count = 0
    for _, score in pairs(node.edges_out or {}) do
        total = total + math.max(0.0, tonumber(score) or 0.0)
        count = count + 1
    end
    for _, score in pairs(node.edges_in or {}) do
        total = total + math.max(0.0, tonumber(score) or 0.0)
        count = count + 1
    end
    if count <= 0 then
        return 0.0
    end
    return clamp(total / count, 0.0, 1.0)
end

local function strong_edge(score, opts)
    opts = type(opts) == "table" and opts or router_opts()
    return (tonumber(score) or -1e9) >= (tonumber(opts.strong_edge_score) or 0.12)
end

local function build_node(turn, vec, meta)
    meta = type(meta) == "table" and meta or {}
    return {
        turn = math.max(0, math.floor(tonumber(turn) or 0)),
        actor_key = normalize_actor_key(meta),
        text = current_text(meta),
        vec = deep_copy(vec),
        alias_map = actor_alias_map(meta),
        mention_map = extract_mentions(current_text(meta)),
        edges_out = {},
        edges_in = {},
        primary_parent_turn = 0,
        primary_parent_score = -1.0,
        component_id = 0,
        root_turn = 0,
        anchor = "",
        thread_key = "",
        thread_id = 0,
        sequence_key = "",
        segment_id = 0,
        segment_key = "",
        in_window = true,
        source = tostring(meta.source or ""),
        user_input = trim(meta.user_input or meta.user or meta.text or ""),
        embed_input = current_text(meta),
        assistant_text = trim(meta.assistant or meta.assistant_text or meta.reply or ""),
        event = nil,
    }
end

local function mentions_node(node_new, node_old)
    node_new = type(node_new) == "table" and node_new or {}
    node_old = type(node_old) == "table" and node_old or {}
    local mentions = type(node_new.mention_map) == "table" and node_new.mention_map or {}
    local old_actor = normalize_alias(node_old.actor_key)
    if old_actor ~= "" and mentions[old_actor] == true then
        return true
    end
    for alias in pairs(node_old.alias_map or {}) do
        if mentions[alias] == true then
            return true
        end
    end
    return false
end

local function score_edge(node_new, node_old, opts)
    opts = type(opts) == "table" and opts or router_opts()
    local score = (tonumber(opts.semantic_weight) or 0.42)
        * (tonumber(tool.cosine_similarity(node_new.vec or {}, node_old.vec or {})) or 0.0)
    if trim(node_new.actor_key or "") ~= "" and trim(node_new.actor_key or "") == trim(node_old.actor_key or "") then
        score = score + (tonumber(opts.same_user_bonus) or 0.06)
    end
    if mentions_node(node_new, node_old) then
        score = score + (tonumber(opts.mention_bonus) or 0.15)
    end
    score = score - (tonumber(opts.age_penalty) or 0.01) * math.max(0, (tonumber(node_new.turn) or 0) - (tonumber(node_old.turn) or 0))
    return score
end

local function sort_score_rows(rows)
    table.sort(rows, function(a, b)
        local sa = tonumber((a or {}).score) or -1e9
        local sb = tonumber((b or {}).score) or -1e9
        if sa ~= sb then
            return sa > sb
        end
        return (tonumber((a or {}).turn) or 0) > (tonumber((b or {}).turn) or 0)
    end)
    return rows
end

local function add_edge(node_child, node_parent, score)
    node_child = type(node_child) == "table" and node_child or {}
    node_parent = type(node_parent) == "table" and node_parent or {}
    score = tonumber(score) or 0.0
    if (tonumber(node_child.turn) or 0) <= 0 or (tonumber(node_parent.turn) or 0) <= 0 then
        return false
    end
    node_child.edges_out[node_parent.turn] = score
    node_parent.edges_in[node_child.turn] = score
    return true
end

local function neighbor_rows(scope, node, opts, sealed_only)
    scope = type(scope) == "table" and scope or {}
    node = type(node) == "table" and node or {}
    opts = type(opts) == "table" and opts or router_opts()
    local out = {}
    local seen = {}

    local function visit(turn, score)
        turn = tonumber(turn)
        score = tonumber(score) or 0.0
        if not turn or turn <= 0 or seen[turn] or not strong_edge(score, opts) then
            return
        end
        local other = scope.nodes[turn]
        if type(other) ~= "table" then
            return
        end
        if sealed_only and truthy(other.in_window) then
            return
        end
        seen[turn] = true
        out[#out + 1] = {
            turn = turn,
            score = score,
            node = other,
        }
    end

    for turn, score in pairs(node.edges_out or {}) do
        visit(turn, score)
    end
    for turn, score in pairs(node.edges_in or {}) do
        visit(turn, score)
    end
    return sort_score_rows(out)
end

local function bfs_turns(scope, start_turn, opts)
    scope = type(scope) == "table" and scope or {}
    start_turn = math.max(0, math.floor(tonumber(start_turn) or 0))
    opts = type(opts) == "table" and opts or router_opts()
    if start_turn <= 0 or type(scope.nodes[start_turn]) ~= "table" then
        return {}
    end

    local out = {}
    local visited = {}
    local queue = { start_turn }
    local limit = math.max(1, math.floor(tonumber(opts.local_bfs_turns) or 6))
    local head = 1

    while head <= #queue and #out < limit do
        local turn = tonumber(queue[head]) or 0
        head = head + 1
        if turn > 0 and not visited[turn] then
            visited[turn] = true
            local node = scope.nodes[turn]
            if type(node) == "table" then
                out[#out + 1] = turn
                for _, row in ipairs(neighbor_rows(scope, node, opts, false)) do
                    local next_turn = tonumber(row.turn) or 0
                    if next_turn > 0 and not visited[next_turn] then
                        queue[#queue + 1] = next_turn
                    end
                end
            end
        end
    end

    table.sort(out)
    return out
end

local function collect_component_turns(scope, seed_turn, opts, sealed_only)
    scope = type(scope) == "table" and scope or {}
    seed_turn = math.max(0, math.floor(tonumber(seed_turn) or 0))
    opts = type(opts) == "table" and opts or router_opts()
    if seed_turn <= 0 or type(scope.nodes[seed_turn]) ~= "table" then
        return {}
    end

    local out = {}
    local visited = {}
    local queue = { seed_turn }
    local head = 1

    while head <= #queue do
        local turn = tonumber(queue[head]) or 0
        head = head + 1
        if turn > 0 and not visited[turn] then
            local node = scope.nodes[turn]
            if type(node) == "table" and (not sealed_only or not truthy(node.in_window)) then
                visited[turn] = true
                out[#out + 1] = turn
                for _, row in ipairs(neighbor_rows(scope, node, opts, sealed_only)) do
                    local next_turn = tonumber(row.turn) or 0
                    if next_turn > 0 and not visited[next_turn] then
                        queue[#queue + 1] = next_turn
                    end
                end
            end
        end
    end

    table.sort(out)
    return out
end

local function remove_node(scope, turn)
    scope = type(scope) == "table" and scope or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    local node = scope.nodes[turn]
    if type(node) ~= "table" then
        return false
    end
    for parent_turn in pairs(node.edges_out or {}) do
        local parent = scope.nodes[tonumber(parent_turn)]
        if type(parent) == "table" then
            parent.edges_in[turn] = nil
        end
    end
    for child_turn in pairs(node.edges_in or {}) do
        local child = scope.nodes[tonumber(child_turn)]
        if type(child) == "table" then
            child.edges_out[turn] = nil
        end
    end
    scope.nodes[turn] = nil
    return true
end

local function noise_node(node, opts)
    node = type(node) == "table" and node or {}
    opts = type(opts) == "table" and opts or router_opts()
    local strong_in = 0
    local max_out = 0.0
    for _, score in pairs(node.edges_in or {}) do
        if strong_edge(score, opts) then
            strong_in = strong_in + 1
        end
    end
    for _, score in pairs(node.edges_out or {}) do
        max_out = math.max(max_out, tonumber(score) or 0.0)
    end
    return strong_in <= 0 and max_out < (tonumber(opts.noise_outgoing_score) or 0.06)
end

local function enqueue_component_events(scope, turns, opts)
    scope = type(scope) == "table" and scope or {}
    turns = type(turns) == "table" and turns or {}
    opts = type(opts) == "table" and opts or router_opts()
    if #turns <= 0 then
        return
    end

    local min_turn = turns[1]
    local component_id = 0
    for _, turn in ipairs(turns) do
        local node = scope.nodes[turn]
        if type(node) == "table" then
            min_turn = math.min(min_turn, math.max(1, math.floor(tonumber(node.turn) or min_turn)))
            component_id = math.max(component_id, math.floor(tonumber(node.component_id) or 0))
        end
    end

    if #turns == 1 then
        local only = scope.nodes[turns[1]]
        if type(only) == "table" and noise_node(only, opts) then
            remove_node(scope, turns[1])
            return
        end
    end

    for _, turn in ipairs(turns) do
        local node = scope.nodes[turn]
        if type(node) == "table" and type(node.event) == "table" then
            local row = deep_copy(node.event)
            row.turn = math.max(0, math.floor(tonumber(row.turn) or node.turn or 0))
            row.thread_id = math.max(1, math.floor(tonumber(row.thread_id) or node.thread_id or component_id or 1))
            row.thread_key = trim(row.thread_key or node.thread_key or make_thread_key(scope.scope_key, row.thread_id))
            row.segment_id = math.max(1, math.floor(tonumber(row.segment_id) or node.segment_id or min_turn or 1))
            row.segment_key = trim(row.segment_key or node.segment_key or make_segment_key(scope.scope_key, row.thread_id, min_turn))
            row.sequence_key = trim(row.sequence_key or node.sequence_key or row.thread_key)
            row.anchor = trim(row.anchor or node.anchor or make_component_anchor(row.thread_id))
            row.memory_scope = trim(row.memory_scope or "thread")
            scope.ready_events[#scope.ready_events + 1] = row
        end
        remove_node(scope, turn)
    end
end

local function finalize_evicted(scope, turn, opts)
    scope = type(scope) == "table" and scope or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    opts = type(opts) == "table" and opts or router_opts()
    local node = scope.nodes[turn]
    if type(node) ~= "table" then
        return false
    end
    node.in_window = false
    enqueue_component_events(scope, collect_component_turns(scope, turn, opts, true), opts)
    return true
end

local function drain_ready_events(scope)
    scope = type(scope) == "table" and scope or {}
    local out = {}
    for idx, row in ipairs(scope.ready_events or {}) do
        out[idx] = row
    end
    scope.ready_events = {}
    return out
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

    local opts = router_opts()
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k

    local query_node = build_node(turn, query_vec, meta)
    local rows = {}
    for _, recent_turn in ipairs(recent_turns(scope)) do
        local node = scope.nodes[recent_turn]
        if type(node) == "table" then
            rows[#rows + 1] = {
                turn = recent_turn,
                node = node,
                score = score_edge(query_node, node, opts),
            }
        end
    end
    sort_score_rows(rows)

    local best = rows[1]
    local second = rows[2]
    sel.best_score = best and (tonumber(best.score) or -1.0) or -1.0
    sel.second_score = second and (tonumber(second.score) or -1.0) or -1.0

    if best and (tonumber(best.score) or -1e9) >= (tonumber(opts.read_anchor_score) or 0.12) then
        local node = best.node
        sel.thread_id = math.max(0, math.floor(tonumber(node.component_id) or 0))
        sel.thread_key = trim(node.thread_key or "")
        sel.segment_id = math.max(0, math.floor(tonumber(node.segment_id) or 0))
        sel.segment_key = trim(node.segment_key or "")
        sel.sequence_key = trim(node.sequence_key or sel.thread_key)
        sel.anchor = trim(node.anchor or "")
        sel.topic_start_turn = math.max(0, math.floor(tonumber(node.root_turn) or node.turn or 0))
        sel.use_local_sequence = true
        sel.local_only = sel.anchor == ""
        sel.decision = "attach"
        sel.reason = "local_anchor"
        sel.confidence = clamp(best.score, 0.0, 1.25)
        sel.stability = node_stability(node)
        sel.selected_turns = bfs_turns(scope, best.turn, opts)
    else
        sel.decision = "open"
        sel.reason = "no_local_anchor"
        sel.use_local_sequence = #rows > 0
        sel.selected_turns = {}
    end

    sel.score_debug = {
        recent_window_count = #rows,
        best_turn = best and best.turn or 0,
        best_score = sel.best_score,
        second_score = sel.second_score,
        selected_turns = #sel.selected_turns,
    }

    return sel
end

function M.observe(scope_key, selection, vec, meta, turn)
    selection = type(selection) == "table" and selection or {}
    vec = type(vec) == "table" and vec or {}
    meta = type(meta) == "table" and meta or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    if #vec <= 0 or turn <= 0 or selection.dropped == true then
        return false
    end

    local opts = router_opts()
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k

    local node = build_node(turn, vec, meta)
    local rows = {}
    for _, recent_turn in ipairs(recent_turns(scope)) do
        local old = scope.nodes[recent_turn]
        if type(old) == "table" and old.turn ~= turn then
            rows[#rows + 1] = {
                turn = recent_turn,
                node = old,
                score = score_edge(node, old, opts),
            }
        end
    end
    sort_score_rows(rows)

    local best = rows[1]
    local second = rows[2]
    selection.best_score = best and (tonumber(best.score) or -1.0) or -1.0
    selection.second_score = second and (tonumber(second.score) or -1.0) or -1.0

    local edge_cap = math.min(math.max(1, math.floor(tonumber(opts.top_k) or 1)), #rows)
    for idx = 1, edge_cap do
        add_edge(node, rows[idx].node, rows[idx].score)
    end
    if best then
        node.primary_parent_turn = math.max(0, math.floor(tonumber(best.turn) or 0))
        node.primary_parent_score = tonumber(best.score) or -1.0
    end

    if best and strong_edge(best.score, opts) and math.max(0, math.floor(tonumber((best.node or {}).component_id) or 0)) > 0 then
        node.component_id = math.max(1, math.floor(tonumber(best.node.component_id) or 1))
        node.root_turn = math.max(1, math.floor(tonumber(best.node.root_turn) or best.node.turn or turn))
        node.anchor = trim(best.node.anchor or "")
        selection.is_new = false
        selection.decision = "attach"
        selection.reason = "attach"
    else
        node.component_id = alloc_component_id(scope)
        node.root_turn = turn
        selection.is_new = true
        selection.decision = "open"
        selection.reason = best and "weak_edge_new_component" or "new_component"
    end

    if node.anchor == "" then
        node.anchor = make_component_anchor(node.component_id)
    end
    node.thread_id = node.component_id
    node.thread_key = make_thread_key(scope_key, node.component_id)
    node.sequence_key = node.thread_key
    node.segment_id = node.root_turn
    node.segment_key = make_segment_key(scope_key, node.component_id, node.root_turn)
    node.in_window = true

    scope.nodes[turn] = node
    local evicted_turn = ring_push(scope.recent_window, turn)

    selection.thread_id = node.thread_id
    selection.thread_key = node.thread_key
    selection.segment_id = node.segment_id
    selection.segment_key = node.segment_key
    selection.sequence_key = node.sequence_key
    selection.anchor = node.anchor
    selection.topic_start_turn = node.root_turn
    selection.use_local_sequence = #rows > 0
    selection.local_only = false
    selection.pending_only = false
    selection.orphaned = false
    selection.ambient = false
    selection.mode = "dag"
    selection.confidence = clamp(best and best.score or 1.0, 0.0, 1.25)
    selection.stability = node_stability(node)
    selection.score_debug = {
        recent_window_count = #rows,
        primary_parent_turn = node.primary_parent_turn,
        primary_parent_score = node.primary_parent_score,
        component_id = node.component_id,
        root_turn = node.root_turn,
        top_k = edge_cap,
    }

    if evicted_turn and evicted_turn ~= turn then
        finalize_evicted(scope, evicted_turn, opts)
    end
    return true
end

function M.stage(scope_key, selection, event, turn)
    selection = type(selection) == "table" and selection or {}
    turn = math.max(0, math.floor(tonumber(turn) or 0))
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k

    local out = drain_ready_events(scope)

    if type(event) ~= "table" then
        return out
    end

    local node_turn = math.max(0, math.floor(tonumber(event.turn) or turn))
    local node = scope.nodes[node_turn]
    if type(node) ~= "table" then
        return out
    end

    local row = deep_copy(event)
    row.turn = node_turn
    row.thread_id = math.max(1, math.floor(tonumber(row.thread_id) or node.thread_id or node.component_id or 1))
    row.thread_key = trim(row.thread_key or selection.thread_key or node.thread_key or make_thread_key(scope_key, row.thread_id))
    row.segment_id = math.max(1, math.floor(tonumber(row.segment_id) or selection.segment_id or node.segment_id or node.root_turn or 1))
    row.segment_key = trim(row.segment_key or selection.segment_key or node.segment_key or make_segment_key(scope_key, row.thread_id, row.segment_id))
    row.sequence_key = trim(row.sequence_key or selection.sequence_key or node.sequence_key or row.thread_key)
    row.anchor = trim(row.anchor or selection.anchor or node.anchor or make_component_anchor(row.thread_id))
    row.pending_only = false
    row.local_only = false
    row.orphaned = false
    row.ambient = false
    row.memory_scope = trim(row.memory_scope or "thread")

    node.anchor = row.anchor
    node.thread_id = row.thread_id
    node.thread_key = row.thread_key
    node.sequence_key = row.sequence_key
    node.segment_id = row.segment_id
    node.segment_key = row.segment_key
    node.event = row

    selection.anchor = row.anchor
    selection.thread_id = row.thread_id
    selection.thread_key = row.thread_key
    selection.segment_id = row.segment_id
    selection.segment_key = row.segment_key
    selection.sequence_key = row.sequence_key

    return out
end

function M.pending_context(scope_key, selection, opts)
    return ""
end

function M.flush_all()
    local out = {}
    local opts = router_opts()
    for scope_key in pairs(_state.scopes or {}) do
        local _, scope = ensure_scope(scope_key)
        local turns = {}
        for turn in pairs(scope.nodes or {}) do
            turns[#turns + 1] = math.floor(tonumber(turn) or 0)
        end
        table.sort(turns)
        for _, turn in ipairs(turns) do
            local node = scope.nodes[turn]
            if type(node) == "table" then
                node.in_window = false
            end
        end

        local seen = {}
        for _, turn in ipairs(turns) do
            if not seen[turn] and type(scope.nodes[turn]) == "table" then
                local component_turns = collect_component_turns(scope, turn, opts, true)
                for _, comp_turn in ipairs(component_turns) do
                    seen[comp_turn] = true
                end
                enqueue_component_events(scope, component_turns, opts)
            end
        end

        scope.recent_window = make_recent_window(opts.recent_window_size)
        local ready = drain_ready_events(scope)
        for _, row in ipairs(ready) do
            out[#out + 1] = row
        end
    end
    return out
end

local function export_node(node)
    node = type(node) == "table" and node or {}
    return {
        turn = math.max(0, math.floor(tonumber(node.turn) or 0)),
        actor_key = trim(node.actor_key or ""),
        text = trim(node.text or ""),
        vec = deep_copy(node.vec),
        alias_map = deep_copy(node.alias_map),
        mention_map = deep_copy(node.mention_map),
        edges_out = deep_copy(node.edges_out),
        edges_in = deep_copy(node.edges_in),
        primary_parent_turn = math.max(0, math.floor(tonumber(node.primary_parent_turn) or 0)),
        primary_parent_score = tonumber(node.primary_parent_score) or -1.0,
        component_id = math.max(0, math.floor(tonumber(node.component_id) or 0)),
        root_turn = math.max(0, math.floor(tonumber(node.root_turn) or 0)),
        anchor = trim(node.anchor or ""),
        thread_key = trim(node.thread_key or ""),
        thread_id = math.max(0, math.floor(tonumber(node.thread_id) or 0)),
        sequence_key = trim(node.sequence_key or ""),
        segment_id = math.max(0, math.floor(tonumber(node.segment_id) or 0)),
        segment_key = trim(node.segment_key or ""),
        in_window = truthy(node.in_window),
        source = tostring(node.source or ""),
        user_input = trim(node.user_input or ""),
        embed_input = trim(node.embed_input or ""),
        assistant_text = trim(node.assistant_text or ""),
        event = deep_copy(node.event),
    }
end

function M.export_scope_state(scope_key)
    local scope_k, scope = ensure_scope(scope_key)
    scope_key = scope_k
    local out = {
        scope_key = scope_key,
        next_component_id = math.max(1, math.floor(tonumber(scope.next_component_id) or 1)),
        recent_window = {
            cap = math.max(1, math.floor(tonumber((scope.recent_window or {}).cap) or router_opts().recent_window_size)),
            head = math.max(1, math.floor(tonumber((scope.recent_window or {}).head) or 1)),
            count = math.max(0, math.floor(tonumber((scope.recent_window or {}).count) or 0)),
            items = deep_copy((scope.recent_window or {}).items),
        },
        nodes = {},
    }
    for turn, node in pairs(scope.nodes or {}) do
        local key = math.max(1, math.floor(tonumber(turn) or 1))
        out.nodes[key] = export_node(node)
    end
    return out
end

function M.export_state()
    local out = {
        version = 2,
        scopes = {},
    }
    for scope_key in pairs((_state or {}).scopes or {}) do
        out.scopes[scope_key] = M.export_scope_state(scope_key)
    end
    return out
end

local function import_recent_window(scope, scope_state)
    local window_state = type(scope_state.recent_window) == "table" and scope_state.recent_window or {}
    local cap = math.max(1, math.floor(tonumber(window_state.cap) or router_opts().recent_window_size))
    local count = math.max(0, math.floor(tonumber(window_state.count) or 0))
    local head = math.max(1, math.floor(tonumber(window_state.head) or 1))
    scope.recent_window = make_recent_window(cap)
    scope.recent_window.head = head
    scope.recent_window.count = math.min(count, cap)
    scope.recent_window.items = deep_copy(window_state.items or {})
end

function M.import_scope_state(scope_key, scope_state)
    scope_state = type(scope_state) == "table" and scope_state or {}
    scope_key = trim(scope_key or scope_state.scope_key or "")
    if scope_key == "" then
        scope_key = "global"
    end

    if type(scope_state.streams) == "table" and scope_state.nodes == nil then
        _state.scopes[scope_key] = {
            scope_key = scope_key,
            next_component_id = 1,
            nodes = {},
            recent_window = make_recent_window(router_opts().recent_window_size),
            ready_events = {},
        }
        return true
    end

    local scope = {
        scope_key = scope_key,
        next_component_id = math.max(1, math.floor(tonumber(scope_state.next_component_id) or 1)),
        nodes = {},
        recent_window = make_recent_window(router_opts().recent_window_size),
        ready_events = {},
    }

    for turn, node_state in pairs(scope_state.nodes or {}) do
        local node_turn = math.max(1, math.floor(tonumber(turn) or tonumber((node_state or {}).turn) or 1))
        local node = export_node(node_state)
        node.turn = node_turn
        scope.nodes[node_turn] = node
    end

    import_recent_window(scope, scope_state)
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
