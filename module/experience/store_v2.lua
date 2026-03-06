local util = require("module.graph.util")
local config = require("module.config")
local persistence = require("module.persistence")
local policy = require("module.experience.policy")

local M = {}

M.candidates = {}
M.index = nil
M._loaded = false
M._dirty = false
M._dirty_ids = {}

local ALLOWED_PLANNER_MODES = {
    auto = true,
    direct_first = true,
    tool_first = true,
    evidence_first = true,
}

local ALLOWED_RECALL_MODES = {
    auto = true,
    force = true,
    suppress = true,
}

local ALLOWED_REPAIR_MODES = {
    normal = true,
    eager = true,
    fail_fast = true,
}

local ALLOWED_CHAIN_TOOLS = {
    list_files = true,
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
    code_outline = true,
    project_structure = true,
    code_symbols = true,
    write_file = true,
    apply_patch = true,
    exec_command = true,
    web_search = true,
    web_extractor = true,
    amap_weather = true,
}

local function trim(s)
    return util.trim(s or "")
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function shallow_copy_map(src)
    local out = {}
    for k, v in pairs(src or {}) do
        out[k] = v
    end
    return out
end

local function normalize_signature(src)
    local out = {}
    for k, v in pairs(src or {}) do
        if type(v) ~= "table" then
            out[tostring(k)] = v
        end
    end
    return out
end

local function ratio(numerator, denominator)
    local den = tonumber(denominator) or 0
    if den <= 0 then
        return 0.0
    end
    return (tonumber(numerator) or 0) / den
end

local function storage_root()
    local cfg = ((config.settings or {}).experience or {}).storage or {}
    local root = tostring(cfg.root or "memory/experience_graph_policy")
    if root == "" then
        root = "memory/experience_graph_policy"
    end
    return root
end

local function items_dir()
    return storage_root() .. "/items"
end

local function index_path()
    return storage_root() .. "/v2_index.lua"
end

local function candidate_path(candidate_id)
    return string.format("%s/%s.lua", items_dir(), tostring(candidate_id or "unknown"))
end

local function v2_cfg()
    return (((config.settings or {}).experience or {}).v2) or {}
end

local function default_index()
    return {
        version = "v2",
        ids = {},
        by_family = {},
        by_family_patch = {},
        updated_at_ms = 0,
    }
end

local function make_candidate_id()
    return string.format("gcand_%d_%04x", util.now_ms(), math.random(0, 0xffff))
end

local function sorted_risk_flags(src)
    local out = {}
    local seen = {}
    if type(src) == "table" then
        for key, enabled in pairs(src) do
            if enabled == true then
                local flag = tostring(key or "")
                if flag ~= "" and not seen[flag] then
                    seen[flag] = true
                    out[#out + 1] = flag
                end
            elseif type(key) == "number" then
                local flag = trim(enabled)
                if flag ~= "" and not seen[flag] then
                    seen[flag] = true
                    out[#out + 1] = flag
                end
            end
        end
    end
    table.sort(out)
    return out
end

local function normalize_budget_value(raw_value, allowed_values, fallback)
    local val = tonumber(raw_value)
    if not val then
        return fallback
    end
    local nearest = allowed_values[1]
    local nearest_diff = math.huge
    for _, item in ipairs(allowed_values) do
        local diff = math.abs(item - val)
        if diff < nearest_diff then
            nearest = item
            nearest_diff = diff
        end
    end
    return nearest
end

local function normalize_preferred_chain(raw)
    local out = {}
    local seen = {}
    for _, tool_name in ipairs(raw or {}) do
        local name = trim(tool_name)
        if name ~= "" and ALLOWED_CHAIN_TOOLS[name] == true and not seen[name] then
            seen[name] = true
            out[#out + 1] = name
            if #out >= 3 then
                break
            end
        end
    end
    return out
end

local function normalize_macro_patch(raw)
    local normalized = policy.normalize_runtime_policy(raw or {})

    local planner_mode = trim((((normalized or {}).planner) or {}).mode or "auto")
    if ALLOWED_PLANNER_MODES[planner_mode] ~= true then
        normalized.planner.mode = "auto"
    end

    local recall_mode = trim((((normalized or {}).recall) or {}).mode or "auto")
    if ALLOWED_RECALL_MODES[recall_mode] ~= true then
        normalized.recall.mode = "auto"
    end

    local repair_mode = trim((((normalized or {}).repair) or {}).mode or "normal")
    if ALLOWED_REPAIR_MODES[repair_mode] ~= true then
        normalized.repair.mode = "normal"
    end

    local budget = ((normalized or {}).budget) or {}
    budget.remaining_steps_delta = normalize_budget_value(budget.remaining_steps_delta, { -4, 0, 4 }, 0)
    budget.tool_loop_max_delta = normalize_budget_value(budget.tool_loop_max_delta, { 0, 2 }, 0)
    normalized.budget = budget

    local planner = ((normalized or {}).planner) or {}
    planner.preferred_tool_chain = normalize_preferred_chain(planner.preferred_tool_chain or {})
    if type(planner.force_read_before_write) ~= "boolean" then
        planner.force_read_before_write = nil
    end
    planner.avoid_tools = {}
    normalized.planner = planner

    return policy.normalize_runtime_policy(normalized)
end

local function patch_key(macro_patch)
    local encoded = util.encode_lua_value(normalize_macro_patch(macro_patch or {}), 0)
    return trim(encoded)
end

local function normalize_candidate(raw)
    local row = type(raw) == "table" and raw or {}
    local candidate = {
        id = trim(row.id),
        family_key = trim(row.family_key),
        state_signature = normalize_signature(row.state_signature),
        macro_patch = normalize_macro_patch(row.macro_patch or row.patch),
        risk_flags = sorted_risk_flags(row.risk_flags or {}),
        support_count = math.max(0, math.floor(tonumber(row.support_count) or 0)),
        success_count = math.max(0, math.floor(tonumber(row.success_count) or 0)),
        failure_count = math.max(0, math.floor(tonumber(row.failure_count) or 0)),
        avg_cost = math.max(0, tonumber(row.avg_cost) or 0),
        utility = clamp(tonumber(row.utility) or ratio(row.success_count, row.support_count), 0.0, 1.0),
        consecutive_failures = math.max(0, math.floor(tonumber(row.consecutive_failures) or 0)),
        created_at = tonumber(row.created_at) or os.time(),
        updated_at = tonumber(row.updated_at) or os.time(),
        source = trim(row.source or ""),
    }
    if candidate.id == "" then
        candidate.id = make_candidate_id()
    end
    return candidate
end

local function candidate_risky(row)
    local support = tonumber((row or {}).support_count) or 0
    local failures = tonumber((row or {}).failure_count) or 0
    local failure_rate = ratio(failures, support)
    return support >= 6 and failure_rate > 0.45
end

local function patch_summary(row)
    return policy.summarize_runtime_policy((row or {}).macro_patch or {})
end

local function normalize_family_key(query)
    local key = trim((query or {}).family_key)
    if key ~= "" then
        return key
    end
    key = trim((query or {}).policy_key)
    if key ~= "" then
        return key
    end
    return policy.build_policy_key(query or {})
end

local function similarity_value(left, right)
    if left == nil and right == nil then
        return 1.0
    end
    if type(left) == "boolean" or type(right) == "boolean" then
        return (left == right) and 1.0 or 0.0
    end
    return (trim(left) ~= "" and trim(left) == trim(right)) and 1.0 or 0.0
end

local function state_similarity(candidate_sig, query_sig)
    local candidate = normalize_signature(candidate_sig)
    local query = normalize_signature(query_sig)
    local weights = {
        task_profile = 0.20,
        task_type = 0.20,
        domain = 0.15,
        language = 0.15,
        read_only = 0.10,
        has_uploads = 0.10,
        contract_shape = 0.10,
    }

    local score = 0.0
    local total = 0.0
    for key, weight in pairs(weights) do
        score = score + similarity_value(candidate[key], query[key]) * weight
        total = total + weight
    end
    if total <= 0 then
        return 0.0
    end
    return clamp(score / total, 0.0, 1.0)
end

local function recency_score(updated_at)
    local half_life_days = math.max(1, math.floor(tonumber((v2_cfg() or {}).recency_half_life_days) or 30))
    local now = os.time()
    local age_seconds = math.max(0, now - (tonumber(updated_at) or now))
    local age_days = age_seconds / 86400.0
    local decay = math.exp(-0.69314718056 * age_days / half_life_days)
    return clamp(decay, 0.0, 1.0)
end

local function utility_score(candidate_id, candidate_utility)
    local base = clamp(tonumber(candidate_utility) or 0.5, 0.0, 1.0)
    local ok_adaptive, adaptive = pcall(require, "module.experience.adaptive")
    if not ok_adaptive or not adaptive then
        return base
    end

    local learned = clamp(tonumber(adaptive.get_experience_utility(candidate_id)) or 0.5, 0.0, 1.0)
    local seen = math.max(0, tonumber(adaptive.get_experience_utility_count(candidate_id)) or 0)
    local confidence_count = math.max(1, math.floor(tonumber((((config.settings or {}).experience or {}).retriever or {}).utility_confidence_count) or 6))
    local alpha = clamp(seen / confidence_count, 0.0, 1.0)
    return clamp(base * (1.0 - alpha) + learned * alpha, 0.0, 1.0)
end

local function rebuild_index()
    local idx = default_index()
    local ids = {}

    for candidate_id, candidate in pairs(M.candidates) do
        ids[#ids + 1] = candidate_id
        local family_key = trim((candidate or {}).family_key)
        if family_key ~= "" then
            idx.by_family[family_key] = idx.by_family[family_key] or {}
            idx.by_family[family_key][#idx.by_family[family_key] + 1] = candidate_id
            local key = family_key .. "|" .. patch_key((candidate or {}).macro_patch)
            idx.by_family_patch[key] = candidate_id
        end
    end

    table.sort(ids, function(a, b)
        local ta = tonumber(((M.candidates[a] or {}).updated_at)) or 0
        local tb = tonumber(((M.candidates[b] or {}).updated_at)) or 0
        if ta == tb then
            return tostring(a) < tostring(b)
        end
        return ta > tb
    end)
    idx.ids = ids

    for _, seq in pairs(idx.by_family) do
        table.sort(seq, function(a, b)
            local ta = tonumber(((M.candidates[a] or {}).updated_at)) or 0
            local tb = tonumber(((M.candidates[b] or {}).updated_at)) or 0
            if ta == tb then
                return tostring(a) < tostring(b)
            end
            return ta > tb
        end)
    end

    idx.updated_at_ms = util.now_ms()
    M.index = idx
end

local function mark_dirty(candidate_id)
    M._dirty = true
    if trim(candidate_id) ~= "" then
        M._dirty_ids[candidate_id] = true
    end
end

local function load_candidates_from_index(raw_index)
    local parsed = type(raw_index) == "table" and raw_index or {}
    local ids = shallow_copy_array(parsed.ids or {})
    for _, candidate_id in ipairs(ids) do
        local f = io.open(candidate_path(candidate_id), "rb")
        if f then
            local raw = f:read("*a") or ""
            f:close()
            local candidate = util.parse_lua_table_literal(raw)
            if type(candidate) == "table" then
                local normalized = normalize_candidate(candidate)
                M.candidates[normalized.id] = normalized
            end
        end
    end
end

local function update_risk_flags(row)
    local flags = {}
    if candidate_risky(row) then
        flags[#flags + 1] = "high_failure_rate"
    end
    if (tonumber(row.consecutive_failures) or 0) >= 3 then
        flags[#flags + 1] = "consecutive_failures"
    end
    table.sort(flags)
    row.risk_flags = flags
end

local function cost_from_observation(observation)
    local metrics = ((observation or {}).cost_metrics) or {}
    if type(metrics) == "table" then
        local loops = math.max(0, tonumber(metrics.loop_count) or 0)
        local repairs = math.max(0, tonumber(metrics.repair_attempts) or 0)
        local tools = math.max(0, tonumber(metrics.tool_executed) or 0)
        local failed = math.max(0, tonumber(metrics.tool_failed) or 0)
        return loops + repairs * 1.5 + tools * 0.4 + failed * 0.8
    end
    local loop_count = math.max(0, tonumber((observation or {}).loop_count) or 0)
    local repair_attempts = math.max(0, tonumber((observation or {}).repair_attempts) or 0)
    return loop_count + repair_attempts * 1.5
end

local function discrete_patch_from_observation(observation)
    local raw = ((observation or {}).macro_patch) or {}
    local normalized = normalize_macro_patch(raw)
    local planner_mode = trim((((observation or {}).planner_mode_used) or "")
        or ((((normalized or {}).planner) or {}).mode)
        or "auto")
    if ALLOWED_PLANNER_MODES[planner_mode] == true then
        normalized.planner.mode = planner_mode
    else
        normalized.planner.mode = "auto"
    end

    local recall_mode = trim((((observation or {}).recall_mode_used) or "")
        or ((((normalized or {}).recall) or {}).mode)
        or "auto")
    if ALLOWED_RECALL_MODES[recall_mode] == true then
        normalized.recall.mode = recall_mode
    else
        normalized.recall.mode = "auto"
    end

    local repair_mode = trim((((observation or {}).repair_mode_used) or "")
        or ((((normalized or {}).repair) or {}).mode)
        or "normal")
    if ALLOWED_REPAIR_MODES[repair_mode] == true then
        normalized.repair.mode = repair_mode
    else
        normalized.repair.mode = "normal"
    end

    if (((observation or {}).force_read_before_write_used) == true) then
        normalized.planner.force_read_before_write = true
    end

    local metrics = ((observation or {}).cost_metrics) or {}
    local loop_count = tonumber((observation or {}).loop_count) or tonumber(metrics.loop_count) or 0
    local repair_attempts = tonumber((observation or {}).repair_attempts) or tonumber(metrics.repair_attempts) or 0
    local budget = ((normalized or {}).budget) or {}
    local step_delta = tonumber(budget.remaining_steps_delta) or 0
    local loop_delta = tonumber(budget.tool_loop_max_delta) or 0
    if step_delta == 0 and loop_delta == 0 then
        if loop_count >= 3 or repair_attempts >= 2 then
            budget.remaining_steps_delta = 4
            budget.tool_loop_max_delta = 2
        end
    end
    normalized.budget = budget

    return normalize_macro_patch(normalized)
end

local function candidate_from_id(candidate_id)
    local id = trim(candidate_id)
    if id == "" then
        return nil
    end
    return M.candidates[id]
end

local function mode_of_candidate(row)
    return trim((((row or {}).macro_patch or {}).planner or {}).mode or "auto")
end

local function diversify_candidates(sorted_rows)
    if #sorted_rows <= 2 then
        return sorted_rows
    end

    local out = {}
    local used = {}
    local first = sorted_rows[1]
    out[#out + 1] = first
    used[first.id] = true

    local first_mode = mode_of_candidate(first)
    local second = nil
    for i = 2, #sorted_rows do
        local row = sorted_rows[i]
        if mode_of_candidate(row) ~= first_mode then
            second = row
            break
        end
    end
    if second then
        out[#out + 1] = second
        used[second.id] = true
    end

    for _, row in ipairs(sorted_rows) do
        if not used[row.id] then
            out[#out + 1] = row
        end
    end
    return out
end

local function risk_penalty(row)
    if candidate_risky(row) then
        return 0.35
    end
    return 0.0
end

local function candidate_score(row, query)
    local context_similarity = state_similarity((row or {}).state_signature, (query or {}).state_signature or (query or {}).context_signature)
    local utility = utility_score((row or {}).id, (row or {}).utility)
    local success_rate = ratio((row or {}).success_count, (row or {}).support_count)
    local recency = recency_score((row or {}).updated_at)
    local penalty = risk_penalty(row)

    return clamp(
        0.45 * context_similarity
            + 0.35 * utility
            + 0.15 * success_rate
            + 0.05 * recency
            - penalty,
        0.0,
        1.0
    ), {
        context_similarity = context_similarity,
        utility = utility,
        success_rate = success_rate,
        recency = recency,
        risk_penalty = penalty,
    }
end

local function enforce_non_risky_top1(rows)
    if #rows <= 1 then
        return rows
    end
    if not candidate_risky(rows[1]) then
        return rows
    end
    for i = 2, #rows do
        if not candidate_risky(rows[i]) then
            rows[1], rows[i] = rows[i], rows[1]
            break
        end
    end
    return rows
end

function M.init()
    util.ensure_dir(storage_root())
    util.ensure_dir(items_dir())
    M.load()
end

function M.load()
    M.candidates = {}
    M.index = default_index()
    M._dirty = false
    M._dirty_ids = {}

    util.ensure_dir(storage_root())
    util.ensure_dir(items_dir())

    local f = io.open(index_path(), "rb")
    if not f then
        M._loaded = true
        return
    end
    local raw = f:read("*a") or ""
    f:close()

    local parsed = util.parse_lua_table_literal(raw)
    if type(parsed) ~= "table" then
        M._loaded = true
        return
    end

    load_candidates_from_index(parsed)
    rebuild_index()
    M._loaded = true
end

function M.save()
    if M._dirty ~= true then
        return true
    end

    util.ensure_dir(storage_root())
    util.ensure_dir(items_dir())

    for candidate_id in pairs(M._dirty_ids or {}) do
        local row = M.candidates[candidate_id]
        if row then
            local ok, err = persistence.write_atomic(candidate_path(candidate_id), "wb", function(f)
                return f:write(util.encode_lua_value(row, 0))
            end)
            if not ok then
                return false, err
            end
        end
    end

    local ok, err = persistence.write_atomic(index_path(), "wb", function(f)
        return f:write(util.encode_lua_value(M.index or default_index(), 0))
    end)
    if not ok then
        return false, err
    end

    M._dirty = false
    M._dirty_ids = {}
    return true
end

function M.retrieve_v2(query, options)
    options = options or {}
    local family_key = normalize_family_key(query or {})
    local limit = math.max(1, math.floor(tonumber(options.limit) or tonumber((v2_cfg() or {}).candidate_limit) or 5))
    local ids = shallow_copy_array((((M.index or {}).by_family or {})[family_key]) or {})
    local rows = {}

    for _, candidate_id in ipairs(ids) do
        local row = candidate_from_id(candidate_id)
        if row then
            local score, breakdown = candidate_score(row, query or {})
            local item = normalize_candidate(row)
            item.score = score
            item.score_breakdown = breakdown
            rows[#rows + 1] = item
        end
    end

    table.sort(rows, function(a, b)
        local sa = tonumber((a or {}).score) or 0
        local sb = tonumber((b or {}).score) or 0
        if sa == sb then
            local ta = tonumber((a or {}).updated_at) or 0
            local tb = tonumber((b or {}).updated_at) or 0
            if ta == tb then
                return tostring((a or {}).id or "") < tostring((b or {}).id or "")
            end
            return ta > tb
        end
        return sa > sb
    end)

    rows = enforce_non_risky_top1(rows)
    rows = diversify_candidates(rows)

    while #rows > limit do
        table.remove(rows)
    end

    local warnings = {}
    for _, row in ipairs(rows) do
        if candidate_risky(row) then
            warnings[#warnings + 1] = string.format(
                "candidate=%s has high failure risk (support=%d failure_rate=%.2f)",
                tostring(row.id),
                tonumber(row.support_count) or 0,
                ratio(row.failure_count, row.support_count)
            )
        end
        for _, flag in ipairs((row or {}).risk_flags or {}) do
            if flag ~= "high_failure_rate" then
                warnings[#warnings + 1] = string.format("candidate=%s risk=%s", tostring(row.id), tostring(flag))
            end
        end
    end

    local recommendation = nil
    if #rows > 0 then
        local best = rows[1]
        recommendation = {
            id = tostring(best.id or ""),
            confidence = clamp(tonumber(best.score) or 0, 0.0, 1.0),
            reason = candidate_risky(best) and "best_non_risky_candidate" or "top_score_candidate",
            support_count = tonumber(best.support_count) or 0,
            macro_patch = normalize_macro_patch(best.macro_patch or {}),
            patch_summary = patch_summary(best),
        }
    end

    return rows, {
        strategy = (#rows > 0) and "family_topk" or "empty",
        family_key = family_key,
        failure_warnings = warnings,
        recommendation = recommendation,
        candidate_summaries = (function()
            local out = {}
            for _, row in ipairs(rows) do
                out[#out + 1] = {
                    id = tostring(row.id or ""),
                    score = tonumber(row.score) or 0,
                    support = tonumber(row.support_count) or 0,
                    patch_summary = patch_summary(row),
                    risk_flags = shallow_copy_array((row or {}).risk_flags or {}),
                }
            end
            return out
        end)(),
    }
end

function M.match_behavior_to_candidate(observation, candidates_or_ids)
    local planner_mode_used = trim((observation or {}).planner_mode_used or "auto")
    local recall_mode_used = trim((observation or {}).recall_mode_used or "auto")
    local repair_mode_used = trim((observation or {}).repair_mode_used or "normal")
    local force_read_used = (((observation or {}).force_read_before_write_used) == true)
    local tool_sequence = shallow_copy_array((observation or {}).tool_sequence or {})
    local success = ((observation or {}).success) == true

    local resolved = {}
    for _, item in ipairs(candidates_or_ids or {}) do
        if type(item) == "table" and trim(item.id) ~= "" then
            resolved[#resolved + 1] = item
        else
            local row = candidate_from_id(item)
            if row then
                resolved[#resolved + 1] = row
            end
        end
    end

    local best_id = ""
    local best_score = 0.0
    for _, row in ipairs(resolved) do
        local patch = ((row or {}).macro_patch) or {}
        local score = 0.0
        local planner_mode = trim((((patch or {}).planner) or {}).mode or "auto")
        local recall_mode = trim((((patch or {}).recall) or {}).mode or "auto")
        local repair_mode = trim((((patch or {}).repair) or {}).mode or "normal")
        local force_read = ((((patch or {}).planner) or {}).force_read_before_write) == true
        local preferred_chain = ((((patch or {}).planner) or {}).preferred_tool_chain) or {}

        if planner_mode_used == planner_mode then
            score = score + 0.35
        elseif planner_mode_used == "direct_first" and planner_mode == "auto" then
            score = score + 0.20
        elseif planner_mode_used ~= "direct_first" and planner_mode == "tool_first" then
            score = score + 0.10
        end

        if recall_mode_used == recall_mode then
            score = score + 0.15
        end
        if repair_mode_used == repair_mode then
            score = score + 0.10
        end
        if force_read_used == force_read then
            score = score + 0.15
        end

        if #preferred_chain > 0 and #tool_sequence > 0 and tostring(preferred_chain[1]) == tostring(tool_sequence[1]) then
            score = score + 0.15
        end

        local success_rate = ratio((row or {}).success_count, (row or {}).support_count)
        if success then
            score = score + 0.10 * success_rate
        else
            score = score + 0.10 * (1.0 - success_rate)
        end

        score = clamp(score, 0.0, 1.0)
        if score > best_score then
            best_score = score
            best_id = tostring((row or {}).id or "")
        end
    end

    return best_id, best_score
end

function M.observe_v2(observation)
    if type(observation) ~= "table" then
        return false, "invalid_observation"
    end

    local family_key = normalize_family_key(observation)
    if family_key == "" then
        return false, "missing_family_key"
    end

    local state_signature = normalize_signature((observation or {}).state_signature or (observation or {}).context_signature or {})
    local macro_patch = discrete_patch_from_observation(observation)
    local key = family_key .. "|" .. patch_key(macro_patch)
    local candidate_id = ((((M.index or {}).by_family_patch) or {})[key]) or ""
    local row = candidate_from_id(candidate_id)
    if not row then
        row = normalize_candidate({
            id = make_candidate_id(),
            family_key = family_key,
            state_signature = state_signature,
            macro_patch = macro_patch,
            risk_flags = {},
            support_count = 0,
            success_count = 0,
            failure_count = 0,
            avg_cost = 0,
            utility = 0.5,
            consecutive_failures = 0,
            created_at = os.time(),
            updated_at = os.time(),
            source = "v2_runtime",
        })
    end

    local success = ((observation or {}).success) == true
    local support_before = tonumber(row.support_count) or 0
    row.support_count = support_before + 1
    if success then
        row.success_count = (tonumber(row.success_count) or 0) + 1
        row.consecutive_failures = 0
    else
        row.failure_count = (tonumber(row.failure_count) or 0) + 1
        row.consecutive_failures = (tonumber(row.consecutive_failures) or 0) + 1
    end

    local cost = cost_from_observation(observation)
    if support_before <= 0 then
        row.avg_cost = cost
    else
        row.avg_cost = ((tonumber(row.avg_cost) or 0) * support_before + cost) / math.max(1, row.support_count)
    end

    local learning_rate = clamp(tonumber((((config.settings or {}).experience or {}).adaptive or {}).utility_learning_rate) or 0.10, 0.01, 0.5)
    local target = success and 1.0 or 0.0
    row.utility = clamp((tonumber(row.utility) or 0.5) + learning_rate * (target - (tonumber(row.utility) or 0.5)), 0.0, 1.0)
    row.updated_at = os.time()
    if next(state_signature) ~= nil then
        row.state_signature = state_signature
    end
    row.macro_patch = macro_patch
    update_risk_flags(row)

    M.candidates[row.id] = row
    mark_dirty(row.id)

    -- Success can slightly lift utilities of similar candidates in the same family.
    if success then
        local family_ids = shallow_copy_array((((M.index or {}).by_family or {})[family_key]) or {})
        for _, sib_id in ipairs(family_ids) do
            local sib = candidate_from_id(sib_id)
            if sib and sib.id ~= row.id then
                local sim = state_similarity(sib.state_signature, state_signature)
                if sim >= 0.80 then
                    sib.utility = clamp((tonumber(sib.utility) or 0.5) + 0.03 * (1.0 - (tonumber(sib.utility) or 0.5)), 0.0, 1.0)
                    M.candidates[sib.id] = sib
                    mark_dirty(sib.id)
                end
            end
        end
    end

    rebuild_index()
    return true, { row.id }, row
end

function M.get_candidate(candidate_id)
    local row = candidate_from_id(candidate_id)
    if not row then
        return nil
    end
    return normalize_candidate(row)
end

function M.list_candidates_by_family(family_key)
    local out = {}
    local key = trim(family_key)
    for _, candidate_id in ipairs(((((M.index or {}).by_family) or {})[key]) or {}) do
        local row = candidate_from_id(candidate_id)
        if row then
            out[#out + 1] = normalize_candidate(row)
        end
    end
    return out
end

function M.get_stats()
    local total = 0
    local risky = 0
    for _, row in pairs(M.candidates) do
        total = total + 1
        if candidate_risky(row) then
            risky = risky + 1
        end
    end
    return {
        total_candidates = total,
        risky_candidates = risky,
        storage_root = storage_root(),
    }
end

return M
