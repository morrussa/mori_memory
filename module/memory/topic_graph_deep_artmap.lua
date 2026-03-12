local tool = require("module.tool")

local M = {}

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function copy_vec(vec)
    local out = {}
    for i = 1, #(vec or {}) do
        out[i] = tonumber(vec[i]) or 0.0
    end
    return out
end

local function normalize(vec)
    local sum = 0.0
    for i = 1, #(vec or {}) do
        local v = tonumber(vec[i]) or 0.0
        sum = sum + v * v
    end
    if sum <= 0.0 then
        return copy_vec(vec)
    end
    local scale = 1.0 / math.sqrt(sum)
    local out = {}
    for i = 1, #(vec or {}) do
        out[i] = (tonumber(vec[i]) or 0.0) * scale
    end
    return out
end

local function blend_vec(base, vec, beta)
    if type(base) ~= "table" or #base <= 0 then
        return normalize(vec or {})
    end
    if type(vec) ~= "table" or #vec <= 0 then
        return normalize(base)
    end
    beta = clamp(tonumber(beta) or 0.2, 0.01, 1.0)
    local dim = math.max(#base, #vec)
    local out = {}
    for i = 1, dim do
        local a = tonumber(base[i]) or 0.0
        local b = tonumber(vec[i]) or 0.0
        out[i] = a * (1.0 - beta) + b * beta
    end
    return normalize(out)
end

local function safe_sim(a, b)
    if type(a) ~= "table" or type(b) ~= "table" or #a <= 0 or #b <= 0 then
        return 0.0
    end
    return tonumber(tool.cosine_similarity(a, b)) or 0.0
end

local function ensure_member(cat, mem_id)
    cat.member_set = cat.member_set or {}
    cat.members = cat.members or {}
    local mem = tonumber(mem_id)
    if not mem or mem <= 0 or cat.member_set[mem] then
        return false
    end
    cat.member_set[mem] = true
    cat.members[#cat.members + 1] = mem
    return true
end

local function sorted_numeric_keys(map)
    local out = {}
    for key in pairs(map or {}) do
        local n = tonumber(key)
        if n and n > 0 then
            out[#out + 1] = n
        end
    end
    table.sort(out)
    return out
end

local function rebuild_lookups(state)
    state.category_lookup = {}
    state.bundle_lookup = {}
    for _, cat in ipairs(state.categories or {}) do
        cat.members = cat.members or {}
        cat.member_set = cat.member_set or {}
        for _, mem_id in ipairs(cat.members) do
            cat.member_set[tonumber(mem_id)] = true
        end
        cat.exemplars = cat.exemplars or {}
        cat.exemplar_scores = cat.exemplar_scores or {}
        state.category_lookup[tonumber(cat.id)] = cat
    end
    for _, bundle in ipairs(state.bundles or {}) do
        bundle.category_ids = bundle.category_ids or {}
        bundle.neighbors = bundle.neighbors or {}
        state.bundle_lookup[tonumber(bundle.id)] = bundle
    end
end

function M.ensure_state(state)
    state = type(state) == "table" and state or {}
    state.categories = type(state.categories) == "table" and state.categories or {}
    state.bundles = type(state.bundles) == "table" and state.bundles or {}
    state.memory_to_category = type(state.memory_to_category) == "table" and state.memory_to_category or {}
    state.next_category_id = math.max(1, math.floor(tonumber(state.next_category_id) or 1))
    state.next_bundle_id = math.max(1, math.floor(tonumber(state.next_bundle_id) or 1))
    state.last_bundle_id = tonumber(state.last_bundle_id) or -1
    state.last_bundle_turn = math.max(0, math.floor(tonumber(state.last_bundle_turn) or 0))
    rebuild_lookups(state)
    return state
end

function M.new_state()
    return M.ensure_state({})
end

local function best_category(state, vec)
    local best, best_sim
    for _, cat in ipairs(state.categories or {}) do
        local sim = safe_sim(vec, cat.centroid)
        if (not best) or sim > best_sim then
            best = cat
            best_sim = sim
        end
    end
    return best, best_sim or -1.0
end

local function best_bundle(state, vec)
    local best, best_sim
    for _, bundle in ipairs(state.bundles or {}) do
        local sim = safe_sim(vec, bundle.centroid)
        if (not best) or sim > best_sim then
            best = bundle
            best_sim = sim
        end
    end
    return best, best_sim or -1.0
end

local function category_by_id(state, id)
    return state.category_lookup[tonumber(id)]
end

local function bundle_by_id(state, id)
    return state.bundle_lookup[tonumber(id)]
end

local function ensure_bundle_category(bundle, category_id)
    category_id = tonumber(category_id)
    if not category_id or category_id <= 0 then
        return false
    end
    bundle.category_ids = bundle.category_ids or {}
    for _, existing in ipairs(bundle.category_ids) do
        if tonumber(existing) == category_id then
            return false
        end
    end
    bundle.category_ids[#bundle.category_ids + 1] = category_id
    return true
end

local function remember_exemplar(cat, mem_id, score, cap)
    mem_id = tonumber(mem_id)
    if not mem_id or mem_id <= 0 then
        return
    end
    cap = math.max(1, math.floor(tonumber(cap) or 12))
    cat.exemplars = cat.exemplars or {}
    cat.exemplar_scores = cat.exemplar_scores or {}
    local existing = nil
    for i = 1, #(cat.exemplars or {}) do
        if tonumber(cat.exemplars[i]) == mem_id then
            existing = i
            break
        end
    end
    if existing then
        cat.exemplar_scores[existing] = math.max(tonumber(cat.exemplar_scores[existing]) or -1e9, tonumber(score) or 0.0)
    else
        cat.exemplars[#cat.exemplars + 1] = mem_id
        cat.exemplar_scores[#cat.exemplar_scores + 1] = tonumber(score) or 0.0
    end
    local rows = {}
    for i = 1, #(cat.exemplars or {}) do
        local id = tonumber(cat.exemplars[i])
        if id and id > 0 then
            rows[#rows + 1] = {
                mem_id = id,
                score = tonumber(cat.exemplar_scores[i]) or 0.0,
            }
        end
    end
    table.sort(rows, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.mem_id or 0) < (b.mem_id or 0)
    end)
    cat.exemplars = {}
    cat.exemplar_scores = {}
    for i = 1, math.min(cap, #rows) do
        cat.exemplars[i] = rows[i].mem_id
        cat.exemplar_scores[i] = rows[i].score
    end
end

local function keys_sorted_by_score(map)
    local rows = {}
    for key, score in pairs(map or {}) do
        local id = tonumber(key)
        if id and id > 0 then
            rows[#rows + 1] = {
                id = id,
                score = tonumber(score) or 0.0,
            }
        end
    end
    table.sort(rows, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)
    local out = {}
    for i = 1, #rows do
        out[i] = rows[i].id
    end
    return out
end

local function normalize_weight_map(map, cap)
    local rows = {}
    local total = 0.0
    cap = math.max(1, math.floor(tonumber(cap) or 8))
    for key, weight in pairs(map or {}) do
        local id = tonumber(key)
        local w = tonumber(weight) or 0.0
        if id and id > 0 and w > 0.0 then
            total = total + w
            rows[#rows + 1] = { id = id, weight = w }
        end
    end
    if #rows <= 0 then
        return {}
    end
    table.sort(rows, function(a, b)
        if (a.weight or 0.0) ~= (b.weight or 0.0) then
            return (a.weight or 0.0) > (b.weight or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)
    local out = {}
    local kept_total = 0.0
    for i = 1, math.min(cap, #rows) do
        kept_total = kept_total + rows[i].weight
        out[rows[i].id] = rows[i].weight
    end
    if kept_total > 1e-6 then
        for key, weight in pairs(out) do
            out[key] = weight / kept_total
        end
    end
    return out
end

local function lookup_facet_map(facet_lookup, mem_id)
    if type(facet_lookup) ~= "function" then
        return {}
    end
    local raw = facet_lookup(mem_id)
    if type(raw) ~= "table" then
        return {}
    end
    local out = {}
    local rows = raw
    if #rows > 0 then
        for _, row in ipairs(rows) do
            local facet_id = tonumber((row or {}).id or (row or {}).facet_id or row[1])
            local weight = tonumber((row or {}).weight or row[2]) or 0.0
            if facet_id and facet_id > 0 and weight > 0.0 then
                out[facet_id] = math.max(tonumber(out[facet_id]) or 0.0, weight)
            end
        end
        return normalize_weight_map(out, #rows)
    end
    for key, weight in pairs(raw) do
        local facet_id = tonumber(key)
        local facet_weight = tonumber(weight) or 0.0
        if facet_id and facet_id > 0 and facet_weight > 0.0 then
            out[facet_id] = math.max(tonumber(out[facet_id]) or 0.0, facet_weight)
        end
    end
    return normalize_weight_map(out, 8)
end

local function select_memories_for_context(memory_ids, query_vec, query_map, budget, semantic_scores, local_scores, group_ids, memory_lookup, facet_lookup, opts)
    budget = math.max(1, math.floor(tonumber(budget) or 1))
    semantic_scores = type(semantic_scores) == "table" and semantic_scores or {}
    local_scores = type(local_scores) == "table" and local_scores or {}
    group_ids = type(group_ids) == "table" and group_ids or {}
    opts = type(opts) == "table" and opts or {}
    local saturation_threshold = clamp(tonumber(opts.saturation_threshold) or 0.95, -1.0, 0.999)

    local seen = {}
    local unique_ids = {}
    for _, mem_id in ipairs(memory_ids or {}) do
        mem_id = tonumber(mem_id)
        if mem_id and mem_id > 0 and not seen[mem_id] then
            seen[mem_id] = true
            unique_ids[#unique_ids + 1] = mem_id
        end
    end
    if #unique_ids <= 0 then
        return {}
    end

    local local_max = 0.0
    for _, score in pairs(local_scores) do
        local s = math.max(0.0, tonumber(score) or 0.0)
        if s > local_max then
            local_max = s
        end
    end
    local query_total_weight = 0.0
    for _, weight in pairs(query_map or {}) do
        query_total_weight = query_total_weight + math.max(0.0, tonumber(weight) or 0.0)
    end
    query_total_weight = math.max(1e-6, query_total_weight)

    local cache = {}
    for _, mem_id in ipairs(unique_ids) do
        local mem_vec = memory_lookup and memory_lookup(mem_id) or nil
        cache[mem_id] = {
            semantic = math.max(0.0, tonumber(semantic_scores[mem_id]) or safe_sim(query_vec, mem_vec)),
            local_score = (local_max > 1e-6) and (math.max(0.0, tonumber(local_scores[mem_id]) or 0.0) / local_max) or 0.0,
            map = lookup_facet_map(facet_lookup, mem_id),
            group_id = tonumber(group_ids[mem_id]) or -1,
            vec = mem_vec,
        }
    end

    local remaining = {}
    local selected = {}
    local selected_ids = {}
    local best_cover = {}
    local group_counts = {}
    for _, mem_id in ipairs(unique_ids) do
        remaining[mem_id] = true
    end
    for facet_id in pairs(query_map or {}) do
        best_cover[tonumber(facet_id) or facet_id] = 0.0
    end

    while next(remaining) and #selected < budget do
        local best_mem = nil
        local best_score = -1e9
        for _, mem_id in ipairs(unique_ids) do
            if remaining[mem_id] then
                local info = cache[mem_id]
                local coverage_gain = 0.0
                local mem_map = info.map or {}
                if next(query_map or {}) and next(mem_map or {}) then
                    for facet_id, query_weight in pairs(query_map or {}) do
                        local current = tonumber(best_cover[facet_id]) or 0.0
                        local updated = tonumber(mem_map[facet_id]) or 0.0
                        if updated > current then
                            coverage_gain = coverage_gain + (tonumber(query_weight) or 0.0) * (updated - current)
                        end
                    end
                    coverage_gain = coverage_gain / query_total_weight
                end

                local max_pair_excess = 0.0
                for _, selected_id in ipairs(selected_ids) do
                    local other = cache[selected_id]
                    local sim = safe_sim(info.vec, other and other.vec)
                    if sim > saturation_threshold then
                        max_pair_excess = math.max(
                            max_pair_excess,
                            (sim - saturation_threshold) / math.max(1e-6, 1.0 - saturation_threshold)
                        )
                    end
                end

                local group_id = tonumber(info.group_id) or -1
                local group_repeat = (group_id >= 0) and (tonumber(group_counts[group_id]) or 0) or 0
                local group_bonus = (group_id >= 0 and group_repeat <= 0) and 0.08 or 0.0
                local group_penalty = (group_repeat > 0) and (0.05 * (group_repeat / (group_repeat + 1.0))) or 0.0
                local semantic = tonumber(info.semantic) or 0.0
                local local_score = tonumber(info.local_score) or 0.0
                local context_score
                if next(query_map or {}) then
                    context_score = coverage_gain
                        + 0.50 * semantic
                        + 0.22 * local_score
                        + group_bonus
                        - 0.14 * max_pair_excess
                        - group_penalty
                else
                    context_score = 0.72 * semantic
                        + 0.28 * local_score
                        + group_bonus
                        - 0.14 * max_pair_excess
                        - group_penalty
                end

                if context_score > best_score then
                    best_score = context_score
                    best_mem = mem_id
                end
            end
        end

        if not best_mem then
            break
        end

        selected[#selected + 1] = {
            mem_idx = best_mem,
            score = best_score,
        }
        selected_ids[#selected_ids + 1] = best_mem
        remaining[best_mem] = nil

        local info = cache[best_mem] or {}
        for facet_id in pairs(query_map or {}) do
            best_cover[facet_id] = math.max(
                tonumber(best_cover[facet_id]) or 0.0,
                tonumber(((info.map or {})[facet_id])) or 0.0
            )
        end
        local group_id = tonumber(info.group_id) or -1
        if group_id >= 0 then
            group_counts[group_id] = (tonumber(group_counts[group_id]) or 0) + 1
        end
    end

    return selected
end

local function exemplar_rows_for_category(cat, query_vec, memory_lookup, exemplar_cap)
    local rows = {}
    exemplar_cap = math.max(1, math.floor(tonumber(exemplar_cap) or 12))
    if type(cat) ~= "table" then
        return rows
    end
    local seen = {}
    for i = 1, math.min(exemplar_cap, #(cat.exemplars or {})) do
        local mem_id = tonumber(cat.exemplars[i])
        if mem_id and mem_id > 0 and not seen[mem_id] then
            seen[mem_id] = true
            local mem_vec = memory_lookup and memory_lookup(mem_id) or nil
            rows[#rows + 1] = {
                mem_id = mem_id,
                exemplar_score = math.max(
                    tonumber(cat.exemplar_scores[i]) or -1.0,
                    safe_sim(query_vec, mem_vec)
                ),
            }
        end
    end
    if #rows <= 0 then
        for _, mem_id in ipairs(cat.members or {}) do
            mem_id = tonumber(mem_id)
            if mem_id and mem_id > 0 and not seen[mem_id] then
                seen[mem_id] = true
                local mem_vec = memory_lookup and memory_lookup(mem_id) or nil
                rows[#rows + 1] = {
                    mem_id = mem_id,
                    exemplar_score = safe_sim(query_vec, mem_vec),
                }
            end
        end
    end
    table.sort(rows, function(a, b)
        if (a.exemplar_score or 0.0) ~= (b.exemplar_score or 0.0) then
            return (a.exemplar_score or 0.0) > (b.exemplar_score or 0.0)
        end
        return (a.mem_id or 0) < (b.mem_id or 0)
    end)
    while #rows > exemplar_cap do
        table.remove(rows)
    end
    return rows
end

local function collect_topoart_candidates(state, query_vec, memory_lookup, facet_lookup, opts)
    opts = type(opts) == "table" and opts or {}
    local query_categories = math.max(1, math.floor(tonumber(opts.query_categories) or 4))
    local max_results = math.max(1, math.floor(tonumber(opts.max_results) or 8))
    local exemplar_cap = math.max(1, math.floor(tonumber(opts.exemplars) or 12))
    local query_map = type(opts.query_map) == "table" and opts.query_map or {}

    local category_hits = {}
    local proto_score = 0.0
    for _, cat in ipairs(state.categories or {}) do
        local cat_id = tonumber(cat.id) or 0
        local score = safe_sim(query_vec, cat.centroid)
        proto_score = math.max(proto_score, score)
        category_hits[#category_hits + 1] = {
            id = cat_id,
            score = score,
        }
    end
    table.sort(category_hits, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)

    local selected_categories = {}
    for i = 1, math.min(query_categories, #category_hits) do
        selected_categories[#selected_categories + 1] = tonumber(category_hits[i].id) or 0
    end

    local local_scores = {}
    local semantic_scores = {}
    local group_ids = {}
    local category_ids = {}
    for i = 1, #selected_categories do
        local cat_id = tonumber(selected_categories[i]) or 0
        local cat = category_by_id(state, cat_id)
        local cat_score = safe_sim(query_vec, (cat or {}).centroid)
        for _, row in ipairs(exemplar_rows_for_category(cat, query_vec, memory_lookup, exemplar_cap)) do
            local mem_id = tonumber(row.mem_id) or 0
            if mem_id > 0 then
                local mem_vec = memory_lookup and memory_lookup(mem_id) or nil
                local semantic = safe_sim(query_vec, mem_vec)
                local local_score = 0.76 * math.max(0.0, cat_score) + 0.24 * math.max(0.0, tonumber(row.exemplar_score) or 0.0)
                if local_score > (tonumber(local_scores[mem_id]) or -1e9) then
                    local_scores[mem_id] = local_score
                    group_ids[mem_id] = tonumber((cat or {}).bundle_id) or cat_id
                    category_ids[mem_id] = cat_id
                end
                if semantic > (tonumber(semantic_scores[mem_id]) or -1e9) then
                    semantic_scores[mem_id] = semantic
                end
            end
        end
    end

    local memory_ids = keys_sorted_by_score(local_scores)
    local budget = math.max(4, max_results * 4)
    local ranked = select_memories_for_context(
        memory_ids,
        query_vec,
        query_map,
        budget,
        semantic_scores,
        local_scores,
        group_ids,
        memory_lookup,
        facet_lookup,
        {
            saturation_threshold = tonumber(opts.saturation_threshold) or 0.95,
        }
    )
    local out = {}
    for _, item in ipairs(ranked) do
        local mem_id = tonumber(item.mem_idx) or 0
        out[#out + 1] = {
            mem_idx = mem_id,
            score = tonumber(item.score) or 0.0,
            category_id = tonumber(category_ids[mem_id]) or 0,
            bundle_id = tonumber(group_ids[mem_id]) or 0,
        }
    end
    return out, {
        score = proto_score,
        bundle_ids = {},
        category_ids = selected_categories,
        route = "topoart",
    }
end

function M.add_memory(state, vec, mem_id, turn, opts)
    state = M.ensure_state(state)
    opts = type(opts) == "table" and opts or {}
    vec = normalize(vec or {})
    local category_vigilance = clamp(tonumber(opts.category_vigilance) or 0.88, -1.0, 1.0)
    local category_beta = clamp(tonumber(opts.category_beta) or 0.28, 0.01, 1.0)
    local bundle_vigilance = clamp(tonumber(opts.bundle_vigilance) or 0.82, -1.0, 1.0)
    local bundle_beta = clamp(tonumber(opts.bundle_beta) or 0.18, 0.01, 1.0)
    local temporal_window = math.max(0, math.floor(tonumber(opts.temporal_link_window) or 8))
    local exemplar_cap = math.max(1, math.floor(tonumber(opts.exemplars) or 12))

    local cat, cat_sim = best_category(state, vec)
    if (not cat) or cat_sim < category_vigilance then
        cat = {
            id = state.next_category_id,
            centroid = copy_vec(vec),
            members = {},
            member_set = {},
            exemplars = {},
            exemplar_scores = {},
            support = 0,
            bundle_id = 0,
            last_turn = math.max(0, math.floor(tonumber(turn) or 0)),
        }
        state.next_category_id = state.next_category_id + 1
        state.categories[#state.categories + 1] = cat
        state.category_lookup[cat.id] = cat
    else
        cat.centroid = blend_vec(cat.centroid, vec, category_beta)
        cat.last_turn = math.max(cat.last_turn or 0, math.floor(tonumber(turn) or 0))
    end
    if ensure_member(cat, mem_id) then
        cat.support = (tonumber(cat.support) or 0) + 1
    end
    remember_exemplar(cat, mem_id, safe_sim(vec, cat.centroid), exemplar_cap)
    state.memory_to_category[tonumber(mem_id)] = tonumber(cat.id)

    local bundle = bundle_by_id(state, cat.bundle_id)
    local bundle_sim = bundle and safe_sim(cat.centroid, bundle.centroid) or -1.0
    if (not bundle) or bundle_sim < bundle_vigilance then
        local best, best_sim = best_bundle(state, cat.centroid)
        if best and best_sim >= bundle_vigilance then
            bundle = best
            bundle.centroid = blend_vec(bundle.centroid, cat.centroid, bundle_beta)
        else
            bundle = {
                id = state.next_bundle_id,
                centroid = copy_vec(cat.centroid),
                category_ids = {},
                recall_prior = 0.0,
                adopt_prior = 0.0,
                neighbors = {},
                last_turn = math.max(0, math.floor(tonumber(turn) or 0)),
            }
            state.next_bundle_id = state.next_bundle_id + 1
            state.bundles[#state.bundles + 1] = bundle
            state.bundle_lookup[bundle.id] = bundle
        end
    end
    ensure_bundle_category(bundle, cat.id)
    bundle.centroid = blend_vec(bundle.centroid, cat.centroid, bundle_beta)
    bundle.last_turn = math.max(bundle.last_turn or 0, math.floor(tonumber(turn) or 0))
    cat.bundle_id = tonumber(bundle.id)

    local prev_bundle = tonumber(state.last_bundle_id) or -1
    if temporal_window > 0
        and prev_bundle > 0
        and prev_bundle ~= tonumber(bundle.id)
        and math.abs((math.floor(tonumber(turn) or 0)) - (tonumber(state.last_bundle_turn) or 0)) <= temporal_window then
        M.link_bundles(state, prev_bundle, tonumber(bundle.id), 1.0)
    end
    state.last_bundle_id = tonumber(bundle.id) or -1
    state.last_bundle_turn = math.max(0, math.floor(tonumber(turn) or 0))

    return {
        category_id = tonumber(cat.id) or 0,
        bundle_id = tonumber(bundle.id) or 0,
        similarity = tonumber(cat_sim) or 0.0,
    }
end

function M.collect_candidates(state, query_vec, memory_lookup, facet_lookup, opts)
    state = M.ensure_state(state)
    if type(facet_lookup) == "table" and opts == nil then
        opts = facet_lookup
        facet_lookup = nil
    end
    opts = type(opts) == "table" and opts or {}
    query_vec = normalize(query_vec or {})
    local query_bundles = math.max(1, math.floor(tonumber(opts.query_bundles) or 2))
    local query_margin = math.max(0.0, tonumber(opts.query_margin) or 0.06)
    local neighbor_topk = math.max(0, math.floor(tonumber(opts.neighbor_topk) or 2))
    local max_results = math.max(1, math.floor(tonumber(opts.max_results) or 8))
    local query_categories = math.max(1, math.floor(tonumber(opts.query_categories) or 4))
    local exemplar_cap = math.max(1, math.floor(tonumber(opts.exemplars) or 12))
    local bundle_vigilance = clamp(tonumber(opts.bundle_vigilance) or 0.82, -1.0, 1.0)
    local prior_weight = math.max(0.0, tonumber(opts.bundle_prior_weight) or 0.18)
    local query_map = type(opts.query_map) == "table" and opts.query_map or {}

    local bundle_hits = {}
    local best_bundle_score = nil
    for _, bundle in ipairs(state.bundles or {}) do
        local score = safe_sim(query_vec, bundle.centroid)
            + prior_weight * (
                0.65 * (tonumber(bundle.recall_prior) or 0.0)
                + 1.00 * (tonumber(bundle.adopt_prior) or 0.0)
            )
        bundle_hits[#bundle_hits + 1] = {
            id = tonumber(bundle.id) or 0,
            score = score,
        }
        if best_bundle_score == nil or score > best_bundle_score then
            best_bundle_score = score
        end
    end
    table.sort(bundle_hits, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)
    if #bundle_hits <= 0 then
        return collect_topoart_candidates(state, query_vec, memory_lookup, facet_lookup, opts)
    end
    if (best_bundle_score or -1e9) < bundle_vigilance then
        return collect_topoart_candidates(state, query_vec, memory_lookup, facet_lookup, opts)
    end

    local selected_bundles = {}
    local selected_bundle_set = {}
    selected_bundles[#selected_bundles + 1] = tonumber(bundle_hits[1].id) or 0
    selected_bundle_set[tonumber(bundle_hits[1].id) or 0] = true
    local bundle_floor = math.max(bundle_vigilance, (best_bundle_score or -1e9) - query_margin)
    for i = 2, #bundle_hits do
        local hit = bundle_hits[i]
        if #selected_bundles >= query_bundles then
            break
        end
        if (tonumber(hit.score) or -1e9) < bundle_floor then
            break
        end
        if not selected_bundle_set[tonumber(hit.id) or 0] then
            selected_bundles[#selected_bundles + 1] = tonumber(hit.id) or 0
            selected_bundle_set[tonumber(hit.id) or 0] = true
        end
    end

    local neighbor_pool = {}
    for _, bundle_id in ipairs(selected_bundles) do
        local bundle = bundle_by_id(state, bundle_id)
        local ranked_neighbors = {}
        for nid, strength in pairs((bundle or {}).neighbors or {}) do
            ranked_neighbors[#ranked_neighbors + 1] = {
                id = tonumber(nid) or 0,
                strength = tonumber(strength) or 0.0,
            }
        end
        table.sort(ranked_neighbors, function(a, b)
            if (a.strength or 0.0) ~= (b.strength or 0.0) then
                return (a.strength or 0.0) > (b.strength or 0.0)
            end
            return (a.id or 0) < (b.id or 0)
        end)
        for i = 1, math.min(neighbor_topk, #ranked_neighbors) do
            local nid = tonumber(ranked_neighbors[i].id) or 0
            if nid > 0 and not selected_bundle_set[nid] then
                local neighbor = bundle_by_id(state, nid)
                if neighbor then
                    local score = (tonumber(ranked_neighbors[i].strength) or 0.0)
                        + 0.35 * safe_sim(query_vec, neighbor.centroid)
                        + prior_weight * (
                            0.65 * (tonumber(neighbor.recall_prior) or 0.0)
                            + 1.00 * (tonumber(neighbor.adopt_prior) or 0.0)
                        )
                    neighbor_pool[#neighbor_pool + 1] = {
                        id = nid,
                        score = score,
                    }
                end
            end
        end
    end
    table.sort(neighbor_pool, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)
    for _, row in ipairs(neighbor_pool) do
        local bundle_id = tonumber(row.id) or 0
        if #selected_bundles >= query_bundles then
            break
        end
        if bundle_id > 0 and not selected_bundle_set[bundle_id] then
            selected_bundles[#selected_bundles + 1] = bundle_id
            selected_bundle_set[bundle_id] = true
        end
    end

    local category_budget = math.max(query_categories, #selected_bundles)
    local bundle_category_rows = {}
    local candidate_categories = {}
    local spill_categories = {}
    local seen_categories = {}
    local bundle_query_scores = {}
    local proto_score = 0.0
    for _, row in ipairs(bundle_hits) do
        bundle_query_scores[tonumber(row.id) or 0] = tonumber(row.score) or 0.0
    end
    for _, bundle_id in ipairs(selected_bundles) do
        local bundle = bundle_by_id(state, bundle_id)
        local bundle_query_score = math.max(
            safe_sim(query_vec, (bundle or {}).centroid),
            tonumber(bundle_query_scores[bundle_id]) or safe_sim(query_vec, (bundle or {}).centroid)
        )
        proto_score = math.max(proto_score, bundle_query_score)
        local rows = {}
        for _, cat_id in ipairs((bundle or {}).category_ids or {}) do
            local cat = category_by_id(state, cat_id)
            if cat then
                rows[#rows + 1] = {
                    category_id = tonumber(cat_id) or 0,
                    score = safe_sim(query_vec, cat.centroid),
                }
            end
        end
        table.sort(rows, function(a, b)
            if (a.score or 0.0) ~= (b.score or 0.0) then
                return (a.score or 0.0) > (b.score or 0.0)
            end
            return (a.category_id or 0) < (b.category_id or 0)
        end)
        bundle_category_rows[bundle_id] = rows
    end

    if next(bundle_category_rows) then
        local per_bundle_budget = math.max(1, math.floor((category_budget + #selected_bundles - 1) / math.max(1, #selected_bundles)))
        for _, bundle_id in ipairs(selected_bundles) do
            local rows = bundle_category_rows[bundle_id] or {}
            local kept = 0
            for _, row in ipairs(rows) do
                local cat_id = tonumber(row.category_id) or 0
                if cat_id > 0 and not seen_categories[cat_id] then
                    local packed = {
                        category_id = cat_id,
                        bundle_id = bundle_id,
                        score = tonumber(row.score) or 0.0,
                    }
                    if kept < per_bundle_budget then
                        candidate_categories[#candidate_categories + 1] = packed
                        seen_categories[cat_id] = true
                        kept = kept + 1
                    else
                        spill_categories[#spill_categories + 1] = packed
                    end
                end
            end
        end
        if #candidate_categories < category_budget then
            table.sort(spill_categories, function(a, b)
                if (a.score or 0.0) ~= (b.score or 0.0) then
                    return (a.score or 0.0) > (b.score or 0.0)
                end
                return (a.category_id or 0) < (b.category_id or 0)
            end)
            for _, row in ipairs(spill_categories) do
                local cat_id = tonumber(row.category_id) or 0
                if cat_id > 0 and not seen_categories[cat_id] then
                    candidate_categories[#candidate_categories + 1] = row
                    seen_categories[cat_id] = true
                    if #candidate_categories >= category_budget then
                        break
                    end
                end
            end
        end
    end

    local local_scores = {}
    local semantic_scores = {}
    local group_ids = {}
    local category_ids = {}
    for i = 1, math.min(category_budget, #candidate_categories) do
        local row = candidate_categories[i]
        local cat_id = tonumber(row.category_id) or 0
        local bundle_id = tonumber(row.bundle_id) or 0
        local cat_score = tonumber(row.score) or 0.0
        local cat = category_by_id(state, cat_id)
        proto_score = math.max(proto_score, safe_sim(query_vec, (cat or {}).centroid))
        local bundle_query_score = tonumber(bundle_query_scores[bundle_id]) or cat_score
        for _, exemplar in ipairs(exemplar_rows_for_category(cat, query_vec, memory_lookup, exemplar_cap)) do
            local mem_id = tonumber(exemplar.mem_id) or 0
            if mem_id > 0 then
                local mem_vec = memory_lookup and memory_lookup(mem_id) or nil
                local semantic = safe_sim(query_vec, mem_vec)
                local local_score = 0.55 * math.max(0.0, cat_score)
                    + 0.25 * math.max(0.0, bundle_query_score)
                    + 0.20 * math.max(0.0, tonumber(exemplar.exemplar_score) or 0.0)
                if local_score > (tonumber(local_scores[mem_id]) or -1e9) then
                    local_scores[mem_id] = local_score
                    group_ids[mem_id] = bundle_id
                    category_ids[mem_id] = cat_id
                end
                if semantic > (tonumber(semantic_scores[mem_id]) or -1e9) then
                    semantic_scores[mem_id] = semantic
                end
            end
        end
    end

    local memory_ids = keys_sorted_by_score(local_scores)
    local budget = math.max(4, max_results * 4)
    local ranked = select_memories_for_context(
        memory_ids,
        query_vec,
        query_map,
        budget,
        semantic_scores,
        local_scores,
        group_ids,
        memory_lookup,
        facet_lookup,
        {
            saturation_threshold = tonumber(opts.saturation_threshold) or 0.95,
        }
    )
    local out = {}
    for _, item in ipairs(ranked) do
        local mem_id = tonumber(item.mem_idx) or 0
        out[#out + 1] = {
            mem_idx = mem_id,
            score = tonumber(item.score) or 0.0,
            category_id = tonumber(category_ids[mem_id]) or 0,
            bundle_id = tonumber(group_ids[mem_id]) or 0,
        }
    end
    return out, {
        score = proto_score,
        bundle_ids = selected_bundles,
        category_ids = sorted_numeric_keys(seen_categories),
        route = "deep_artmap",
    }
end

function M.reinforce_memory(state, mem_id, kind, turn, opts)
    state = M.ensure_state(state)
    opts = type(opts) == "table" and opts or {}
    mem_id = tonumber(mem_id)
    if not mem_id or mem_id <= 0 then
        return false
    end
    local category_id = tonumber(state.memory_to_category[mem_id])
    local cat = category_by_id(state, category_id)
    local bundle = bundle_by_id(state, (cat or {}).bundle_id)
    if not bundle then
        return false
    end
    local recall_lr = clamp(tonumber(opts.recall_lr) or 0.10, 0.0, 1.0)
    local adopt_lr = clamp(tonumber(opts.adopt_lr) or 0.18, 0.0, 1.0)
    if tostring(kind) == "adopt" then
        local prior = tonumber(bundle.adopt_prior) or 0.0
        bundle.adopt_prior = clamp(prior + adopt_lr * (1.0 - prior), 0.0, 4.0)
    else
        local prior = tonumber(bundle.recall_prior) or 0.0
        bundle.recall_prior = clamp(prior + recall_lr * (1.0 - prior), 0.0, 4.0)
    end
    bundle.last_turn = math.max(bundle.last_turn or 0, math.floor(tonumber(turn) or 0))
    return true
end

function M.reinforce_bundle(state, bundle_id, kind, turn, opts)
    state = M.ensure_state(state)
    opts = type(opts) == "table" and opts or {}
    bundle_id = tonumber(bundle_id)
    if not bundle_id or bundle_id <= 0 then
        return false
    end
    local bundle = bundle_by_id(state, bundle_id)
    if not bundle then
        return false
    end
    local recall_lr = clamp(tonumber(opts.recall_lr) or 0.10, 0.0, 1.0)
    local adopt_lr = clamp(tonumber(opts.adopt_lr) or 0.18, 0.0, 1.0)
    if tostring(kind) == "adopt" then
        local prior = tonumber(bundle.adopt_prior) or 0.0
        bundle.adopt_prior = clamp(prior + adopt_lr * (1.0 - prior), 0.0, 4.0)
    else
        local prior = tonumber(bundle.recall_prior) or 0.0
        bundle.recall_prior = clamp(prior + recall_lr * (1.0 - prior), 0.0, 4.0)
    end
    bundle.last_turn = math.max(bundle.last_turn or 0, math.floor(tonumber(turn) or 0))
    return true
end

function M.link_bundles(state, a_id, b_id, amount)
    state = M.ensure_state(state)
    a_id = tonumber(a_id)
    b_id = tonumber(b_id)
    amount = math.max(0.0, tonumber(amount) or 1.0)
    if not a_id or not b_id or a_id <= 0 or b_id <= 0 or a_id == b_id or amount <= 0.0 then
        return false
    end
    local a = bundle_by_id(state, a_id)
    local b = bundle_by_id(state, b_id)
    if not a or not b then
        return false
    end
    a.neighbors = a.neighbors or {}
    b.neighbors = b.neighbors or {}
    a.neighbors[b_id] = (tonumber(a.neighbors[b_id]) or 0.0) + amount
    b.neighbors[a_id] = (tonumber(b.neighbors[a_id]) or 0.0) + amount
    return true
end

function M.decay(state, factor)
    state = M.ensure_state(state)
    factor = clamp(tonumber(factor) or 1.0, 0.0, 1.0)
    if factor >= 0.999999 then
        return state
    end
    for _, bundle in ipairs(state.bundles or {}) do
        bundle.recall_prior = (tonumber(bundle.recall_prior) or 0.0) * factor
        bundle.adopt_prior = (tonumber(bundle.adopt_prior) or 0.0) * factor
        local drop = {}
        for nid, weight in pairs(bundle.neighbors or {}) do
            local value = (tonumber(weight) or 0.0) * factor
            bundle.neighbors[nid] = value
            if value < 0.02 then
                drop[#drop + 1] = nid
            end
        end
        for _, nid in ipairs(drop) do
            bundle.neighbors[nid] = nil
        end
    end
    return state
end

return M
