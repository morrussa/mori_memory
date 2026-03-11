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

local function rebuild_lookups(state)
    state.category_lookup = {}
    state.bundle_lookup = {}
    for _, cat in ipairs(state.categories or {}) do
        cat.members = cat.members or {}
        cat.member_set = cat.member_set or {}
        for _, mem_id in ipairs(cat.members) do
            cat.member_set[tonumber(mem_id)] = true
        end
        state.category_lookup[tonumber(cat.id)] = cat
    end
    for _, bundle in ipairs(state.bundles or {}) do
        bundle.category_ids = bundle.category_ids or {}
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

function M.add_memory(state, vec, mem_id, turn, opts)
    state = M.ensure_state(state)
    opts = type(opts) == "table" and opts or {}
    vec = normalize(vec or {})
    local category_vigilance = clamp(tonumber(opts.category_vigilance) or 0.88, -1.0, 1.0)
    local category_beta = clamp(tonumber(opts.category_beta) or 0.28, 0.01, 1.0)
    local bundle_vigilance = clamp(tonumber(opts.bundle_vigilance) or 0.82, -1.0, 1.0)
    local bundle_beta = clamp(tonumber(opts.bundle_beta) or 0.18, 0.01, 1.0)

    local cat, cat_sim = best_category(state, vec)
    if (not cat) or cat_sim < category_vigilance then
        cat = {
            id = state.next_category_id,
            centroid = copy_vec(vec),
            members = {},
            member_set = {},
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

    return {
        category_id = tonumber(cat.id) or 0,
        bundle_id = tonumber(bundle.id) or 0,
        similarity = tonumber(cat_sim) or 0.0,
    }
end

function M.collect_candidates(state, query_vec, memory_lookup, opts)
    state = M.ensure_state(state)
    opts = type(opts) == "table" and opts or {}
    query_vec = normalize(query_vec or {})
    local query_bundles = math.max(1, math.floor(tonumber(opts.query_bundles) or 2))
    local query_margin = math.max(0.0, tonumber(opts.query_margin) or 0.06)
    local neighbor_topk = math.max(0, math.floor(tonumber(opts.neighbor_topk) or 2))
    local max_results = math.max(1, math.floor(tonumber(opts.max_results) or 8))

    local bundle_hits = {}
    local best_bundle_score = nil
    for _, bundle in ipairs(state.bundles or {}) do
        local score = safe_sim(query_vec, bundle.centroid)
            + 0.10 * (tonumber(bundle.recall_prior) or 0.0)
            + 0.16 * (tonumber(bundle.adopt_prior) or 0.0)
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

    local active_bundles = {}
    local active_bundle_set = {}
    for i = 1, #bundle_hits do
        local hit = bundle_hits[i]
        if i <= query_bundles or ((best_bundle_score or -1e9) - (hit.score or -1e9)) <= query_margin then
            active_bundles[#active_bundles + 1] = hit.id
            active_bundle_set[hit.id] = true
        end
    end

    local category_hits = {}
    for _, cat in ipairs(state.categories or {}) do
        local sim = safe_sim(query_vec, cat.centroid)
        if active_bundle_set[tonumber(cat.bundle_id) or 0] then
            sim = sim + 0.08
        end
        category_hits[#category_hits + 1] = {
            id = tonumber(cat.id) or 0,
            score = sim,
        }
    end
    table.sort(category_hits, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)

    local active_categories = {}
    local active_category_set = {}
    for _, bundle_id in ipairs(active_bundles) do
        local bundle = bundle_by_id(state, bundle_id)
        for _, cat_id in ipairs((bundle or {}).category_ids or {}) do
            if not active_category_set[cat_id] then
                active_category_set[cat_id] = true
                active_categories[#active_categories + 1] = cat_id
            end
        end
    end
    for i = 1, math.min(neighbor_topk, #category_hits) do
        local cat_id = category_hits[i].id
        if not active_category_set[cat_id] then
            active_category_set[cat_id] = true
            active_categories[#active_categories + 1] = cat_id
        end
    end

    local memory_scores = {}
    local memory_debug = {}
    for _, cat_id in ipairs(active_categories) do
        local cat = category_by_id(state, cat_id)
        local bundle = bundle_by_id(state, (cat or {}).bundle_id)
        for _, mem_id in ipairs((cat or {}).members or {}) do
            local mem_vec = memory_lookup and memory_lookup(mem_id) or nil
            local score = safe_sim(query_vec, mem_vec)
            score = score
                + 0.10 * (tonumber((bundle or {}).recall_prior) or 0.0)
                + 0.16 * (tonumber((bundle or {}).adopt_prior) or 0.0)
            local current = memory_scores[mem_id]
            if current == nil or score > current then
                memory_scores[mem_id] = score
                memory_debug[mem_id] = {
                    mem_idx = tonumber(mem_id) or 0,
                    score = score,
                    category_id = tonumber(cat_id) or 0,
                    bundle_id = tonumber((cat or {}).bundle_id) or 0,
                }
            end
        end
    end

    local out = {}
    for mem_id, item in pairs(memory_debug) do
        item.mem_idx = tonumber(mem_id) or tonumber(item.mem_idx) or 0
        out[#out + 1] = item
    end
    table.sort(out, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.mem_idx or 0) < (b.mem_idx or 0)
    end)

    while #out > max_results do
        table.remove(out)
    end

    return out, {
        bundle_ids = active_bundles,
        category_ids = active_categories,
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

function M.decay(state, factor)
    state = M.ensure_state(state)
    factor = clamp(tonumber(factor) or 1.0, 0.0, 1.0)
    if factor >= 0.999999 then
        return state
    end
    for _, bundle in ipairs(state.bundles or {}) do
        bundle.recall_prior = (tonumber(bundle.recall_prior) or 0.0) * factor
        bundle.adopt_prior = (tonumber(bundle.adopt_prior) or 0.0) * factor
    end
    return state
end

return M
