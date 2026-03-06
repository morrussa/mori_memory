local util = require("module.graph.util")
local config = require("module.config")
local persistence = require("module.persistence")
local policy = require("module.experience.policy")

local M = {}

M.policies = {}
M.index = nil
M._loaded = false
M._dirty = false
M._dirty_ids = {}

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
    return storage_root() .. "/index.lua"
end

local function policy_path(policy_id)
    return string.format("%s/%s.lua", items_dir(), tostring(policy_id or "unknown"))
end

local function default_index()
    return {
        ids = {},
        by_key = {},
        updated_at_ms = 0,
    }
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

local function generate_id()
    return string.format("gpol_%d_%04x", util.now_ms(), math.random(0, 0xffff))
end

local function normalize_tool_path_stats(src)
    local out = {}
    for path_key, row in pairs(src or {}) do
        if type(row) == "table" then
            out[tostring(path_key)] = {
                seen_count = tonumber(row.seen_count) or 0,
                success_count = tonumber(row.success_count) or 0,
                failure_count = tonumber(row.failure_count) or 0,
                steps = tonumber(row.steps) or 0,
            }
        end
    end
    return out
end

local function normalize_tool_failure_stats(src)
    local out = {}
    for tool_name, row in pairs(src or {}) do
        if type(row) == "table" then
            out[tostring(tool_name)] = {
                seen_count = tonumber(row.seen_count) or 0,
                success_count = tonumber(row.success_count) or 0,
                failure_count = tonumber(row.failure_count) or 0,
            }
        end
    end
    return out
end

local function default_stats()
    return {
        seen_count = 0,
        success_count = 0,
        failure_count = 0,
        direct_seen_count = 0,
        direct_success_count = 0,
        tool_seen_count = 0,
        tool_success_count = 0,
        multi_tool_success_count = 0,
        recall_success_count = 0,
        no_recall_direct_success_count = 0,
        episode_prefer_success_count = 0,
        evidence_success_count = 0,
        write_success_count = 0,
        read_before_write_success_count = 0,
        repair_seen_count = 0,
        repair_recovered_success_count = 0,
        repair_failure_count = 0,
        successful_loop_sum = 0,
        tool_path_stats = {},
        tool_failure_stats = {},
        last_stop_reason = "",
    }
end

local function normalize_context_signature(src)
    local out = {}
    for k, v in pairs(src or {}) do
        if type(v) ~= "table" then
            out[tostring(k)] = v
        end
    end
    return out
end

local function normalize_policy(raw)
    local row = type(raw) == "table" and raw or {}
    local normalized = {
        id = tostring(row.id or ""),
        kind = "graph_policy",
        policy_key = tostring(row.policy_key or ""),
        task_profile = tostring(row.task_profile or "general"),
        task_type = tostring(row.task_type or "general"),
        domain = tostring(row.domain or "general"),
        language = tostring(row.language or "unknown"),
        read_only = row.read_only == true,
        has_uploads = row.has_uploads == true,
        contract_shape = tostring(row.contract_shape or "general"),
        context_signature = normalize_context_signature(row.context_signature),
        created_at = tonumber(row.created_at) or os.time(),
        updated_at = tonumber(row.updated_at) or os.time(),
        stats = default_stats(),
        patch = policy.normalize_runtime_policy(row.patch),
    }

    local stats = type(row.stats) == "table" and row.stats or {}
    for key, value in pairs(default_stats()) do
        if type(value) ~= "table" then
            normalized.stats[key] = tonumber(stats[key]) or value
        end
    end
    normalized.stats.tool_path_stats = normalize_tool_path_stats(stats.tool_path_stats)
    normalized.stats.tool_failure_stats = normalize_tool_failure_stats(stats.tool_failure_stats)
    normalized.stats.last_stop_reason = tostring(stats.last_stop_reason or "")

    if normalized.id == "" then
        normalized.id = generate_id()
    end
    if normalized.policy_key == "" then
        normalized.policy_key = policy.build_policy_key({
            task_profile = normalized.task_profile,
            task_type = normalized.task_type,
            domain = normalized.domain,
            language = normalized.language,
            read_only = normalized.read_only,
            has_uploads = normalized.has_uploads,
            contract_shape = normalized.contract_shape,
        })
    end

    return normalized
end

local function rebuild_index()
    local out = default_index()
    local ids = {}
    for policy_id, row in pairs(M.policies) do
        ids[#ids + 1] = policy_id
        local key = tostring((row or {}).policy_key or "")
        if key ~= "" then
            out.by_key[key] = tostring(policy_id)
        end
    end

    table.sort(ids, function(a, b)
        local left = M.policies[a] or {}
        local right = M.policies[b] or {}
        local ta = tonumber(left.updated_at) or 0
        local tb = tonumber(right.updated_at) or 0
        if ta == tb then
            return tostring(a) < tostring(b)
        end
        return ta > tb
    end)

    out.ids = ids
    out.updated_at_ms = util.now_ms()
    M.index = out
end

local function policy_from_observation(observation)
    return normalize_policy({
        id = generate_id(),
        policy_key = observation.policy_key,
        task_profile = observation.task_profile,
        task_type = observation.task_type,
        domain = observation.domain,
        language = observation.language,
        read_only = observation.read_only,
        has_uploads = observation.has_uploads,
        contract_shape = observation.contract_shape,
        context_signature = observation.context_signature,
        stats = default_stats(),
        patch = policy.default_runtime_policy(),
    })
end

local function ratio(numerator, denominator)
    local den = tonumber(denominator) or 0
    if den <= 0 then
        return 0.0
    end
    return (tonumber(numerator) or 0) / den
end

local function pick_preferred_tool_chain(stats)
    local best_key = ""
    local best_success = -1
    local best_rate = -1
    for path_key, row in pairs((stats or {}).tool_path_stats or {}) do
        local success_count = tonumber((row or {}).success_count) or 0
        local failure_count = tonumber((row or {}).failure_count) or 0
        local seen_count = tonumber((row or {}).seen_count) or (success_count + failure_count)
        local success_rate = ratio(success_count, seen_count)
        if path_key ~= "direct" and success_count >= 2 and success_rate >= 0.70 then
            if success_count > best_success or (success_count == best_success and success_rate > best_rate) then
                best_key = tostring(path_key)
                best_success = success_count
                best_rate = success_rate
            end
        end
    end
    if best_key == "" then
        return {}
    end
    local out = {}
    for item in tostring(best_key):gmatch("[^>]+") do
        out[#out + 1] = item
    end
    return out
end

local function recompute_patch(row)
    local stats = row.stats or default_stats()
    local patch = policy.default_runtime_policy()

    if tonumber(stats.success_count) >= 3 and ratio(stats.recall_success_count, stats.success_count) >= 0.75 then
        patch.recall.mode = "force"
    elseif tonumber(stats.success_count) >= 3 and ratio(stats.no_recall_direct_success_count, stats.success_count) >= 0.80 then
        patch.recall.mode = "suppress"
        patch.context.include_memory = false
    end

    if tonumber(stats.episode_prefer_success_count) >= 2 then
        patch.episode.mode = "prefer"
        patch.context.include_episode = true
    end

    local tool_success_rate = ratio(stats.tool_success_count, stats.tool_seen_count)
    local direct_success_rate = ratio(stats.direct_success_count, stats.direct_seen_count)
    if tonumber(stats.tool_success_count) >= 2 and tool_success_rate >= direct_success_rate + 0.15 then
        patch.planner.mode = "tool_first"
    elseif tonumber(stats.direct_success_count) >= 2 and direct_success_rate >= tool_success_rate + 0.15 then
        patch.planner.mode = "direct_first"
    end

    if tonumber(stats.success_count) >= 3 and ratio(stats.evidence_success_count, stats.success_count) >= 0.70 then
        patch.planner.mode = "evidence_first"
    end

    patch.planner.preferred_tool_chain = pick_preferred_tool_chain(stats)

    for tool_name, tool_stats in pairs(stats.tool_failure_stats or {}) do
        local failure_count = tonumber((tool_stats or {}).failure_count) or 0
        local success_rate = ratio((tool_stats or {}).success_count, (tool_stats or {}).seen_count)
        if failure_count >= 2 and success_rate < 0.35 then
            patch.planner.avoid_tools[tostring(tool_name)] = true
        end
    end

    if tonumber(stats.write_success_count) >= 3
        and ratio(stats.read_before_write_success_count, stats.write_success_count) >= 0.80 then
        patch.planner.force_read_before_write = true
    else
        patch.planner.force_read_before_write = nil
    end

    local repair_seen_count = tonumber(stats.repair_seen_count) or 0
    if repair_seen_count > 0 and ratio(stats.repair_recovered_success_count, repair_seen_count) >= 0.65 then
        patch.repair.mode = "eager"
    end
    if repair_seen_count >= 3 and ratio(stats.repair_failure_count, repair_seen_count) >= 0.75 then
        patch.repair.mode = "fail_fast"
    end

    local avg_loop = ratio(stats.successful_loop_sum, stats.success_count)
    if avg_loop >= 3.0 then
        patch.budget.remaining_steps_delta = 4
    elseif avg_loop <= 1.2 and tonumber(stats.direct_success_count) > tonumber(stats.tool_success_count) then
        patch.budget.remaining_steps_delta = -4
    end

    if tonumber(stats.multi_tool_success_count) >= 2
        and tonumber(stats.multi_tool_success_count) > tonumber(stats.direct_success_count) then
        patch.budget.tool_loop_max_delta = 2
    end

    row.patch = policy.normalize_runtime_policy(patch)
end

local function record_tool_path(stats, observation)
    local path_key = (#(observation.tool_sequence or {}) > 0)
        and table.concat(observation.tool_sequence, ">")
        or "direct"
    local bucket = stats.tool_path_stats[path_key]
    if not bucket then
        bucket = {
            seen_count = 0,
            success_count = 0,
            failure_count = 0,
            steps = #(observation.tool_sequence or {}),
        }
        stats.tool_path_stats[path_key] = bucket
    end
    bucket.seen_count = bucket.seen_count + 1
    bucket.steps = #(observation.tool_sequence or {})
    if observation.success == true then
        bucket.success_count = bucket.success_count + 1
    else
        bucket.failure_count = bucket.failure_count + 1
    end
end

local function record_tool_failures(stats, observation)
    local seen_tools = {}
    for _, tool_name in ipairs(observation.tool_sequence or {}) do
        local key = tostring(tool_name)
        if key ~= "" and not seen_tools[key] then
            seen_tools[key] = true
            local bucket = stats.tool_failure_stats[key]
            if not bucket then
                bucket = {
                    seen_count = 0,
                    success_count = 0,
                    failure_count = 0,
                }
                stats.tool_failure_stats[key] = bucket
            end
            bucket.seen_count = bucket.seen_count + 1
            if observation.success == true then
                bucket.success_count = bucket.success_count + 1
            else
                bucket.failure_count = bucket.failure_count + 1
            end
        end
    end
end

local function apply_observation(row, observation)
    local stats = row.stats or default_stats()
    stats.seen_count = stats.seen_count + 1
    stats.last_stop_reason = tostring(observation.stop_reason or "")

    if observation.mode == "tool" then
        stats.tool_seen_count = stats.tool_seen_count + 1
    else
        stats.direct_seen_count = stats.direct_seen_count + 1
    end

    if tonumber(observation.repair_attempts) and tonumber(observation.repair_attempts) > 0 then
        stats.repair_seen_count = stats.repair_seen_count + 1
    end

    if observation.success == true then
        stats.success_count = stats.success_count + 1
        stats.successful_loop_sum = stats.successful_loop_sum + math.max(0, tonumber(observation.loop_count) or 0)

        if observation.mode == "tool" then
            stats.tool_success_count = stats.tool_success_count + 1
            if #(observation.tool_sequence or {}) > 1 then
                stats.multi_tool_success_count = stats.multi_tool_success_count + 1
            end
        else
            stats.direct_success_count = stats.direct_success_count + 1
        end

        if observation.recall_triggered == true then
            stats.recall_success_count = stats.recall_success_count + 1
        end
        if observation.mode == "direct" and observation.recall_triggered ~= true then
            stats.no_recall_direct_success_count = stats.no_recall_direct_success_count + 1
        end
        if observation.episode_continuity_used == true then
            stats.episode_prefer_success_count = stats.episode_prefer_success_count + 1
        end
        if observation.evidence_needed == true then
            stats.evidence_success_count = stats.evidence_success_count + 1
        end
        if observation.has_write_ops == true then
            stats.write_success_count = stats.write_success_count + 1
            if observation.read_before_write == true then
                stats.read_before_write_success_count = stats.read_before_write_success_count + 1
            end
        end
        if tonumber(observation.repair_attempts) and tonumber(observation.repair_attempts) > 0 then
            stats.repair_recovered_success_count = stats.repair_recovered_success_count + 1
        end
    else
        stats.failure_count = stats.failure_count + 1
        if tonumber(observation.repair_attempts) and tonumber(observation.repair_attempts) > 0 then
            stats.repair_failure_count = stats.repair_failure_count + 1
        end
    end

    record_tool_path(stats, observation)
    record_tool_failures(stats, observation)
    row.updated_at = os.time()
    row.stats = stats
    recompute_patch(row)
end

function M.init()
    util.ensure_dir(storage_root())
    util.ensure_dir(items_dir())
    M.load()
end

function M.load()
    M.policies = {}
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

    local parsed, _err = util.parse_lua_table_literal(raw)
    if type(parsed) ~= "table" then
        M._loaded = true
        return
    end

    local ids = shallow_copy_array(parsed.ids or {})
    for _, id in ipairs(ids) do
        local path = policy_path(id)
        local item_file = io.open(path, "rb")
        if item_file then
            local item_raw = item_file:read("*a") or ""
            item_file:close()
            local item = util.parse_lua_table_literal(item_raw)
            if type(item) == "table" then
                local normalized = normalize_policy(item)
                M.policies[normalized.id] = normalized
            end
        end
    end

    rebuild_index()
    M._loaded = true
end

function M.save()
    if M._dirty ~= true then
        return true
    end

    util.ensure_dir(storage_root())
    util.ensure_dir(items_dir())

    for policy_id in pairs(M._dirty_ids or {}) do
        local row = M.policies[policy_id]
        if row then
            local ok, err = persistence.write_atomic(policy_path(policy_id), "wb", function(f)
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

function M.upsert_observation(observation)
    if type(observation) ~= "table" then
        return false, "invalid_observation"
    end

    local key = tostring(observation.policy_key or "")
    if key == "" then
        return false, "missing_policy_key"
    end

    local existing_id = ((M.index or {}).by_key or {})[key]
    local row = existing_id and M.policies[existing_id] or nil
    if not row then
        row = policy_from_observation(observation)
        existing_id = row.id
    end

    apply_observation(row, observation)
    M.policies[row.id] = row
    M._dirty_ids[row.id] = true
    M._dirty = true
    rebuild_index()
    return true, row.id, row
end

function M.add(observation)
    return M.upsert_observation(observation)
end

function M.get(policy_id)
    return M.policies[tostring(policy_id or "")]
end

function M.get_by_key(policy_key)
    local key = tostring(policy_key or "")
    local policy_id = ((M.index or {}).by_key or {})[key]
    if not policy_id then
        return nil
    end
    return M.policies[policy_id]
end

function M.list()
    local out = {}
    for _, policy_id in ipairs(((M.index or {}).ids) or {}) do
        local row = M.policies[policy_id]
        if row then
            out[#out + 1] = row
        end
    end
    return out
end

function M.retrieve(query, options)
    options = options or {}
    local min_score = tonumber(options.min_score) or tonumber((((config.settings or {}).experience or {}).retriever or {}).relevance_gate) or 0.45
    local limit = math.max(1, math.floor(tonumber(options.limit) or 3))
    local out = {}

    for _, row in ipairs(M.list()) do
        local score = policy.compute_query_score(row, query or {})
        if score >= min_score then
            local item = shallow_copy_map(row)
            item.stats = shallow_copy_map(row.stats or {})
            item.stats.tool_path_stats = normalize_tool_path_stats((((row or {}).stats) or {}).tool_path_stats)
            item.stats.tool_failure_stats = normalize_tool_failure_stats((((row or {}).stats) or {}).tool_failure_stats)
            item.patch = policy.normalize_runtime_policy(row.patch)
            item.score = score
            out[#out + 1] = item
        end
    end

    table.sort(out, function(a, b)
        local sa = tonumber(a.score) or 0
        local sb = tonumber(b.score) or 0
        if sa == sb then
            local ta = tonumber(a.updated_at) or 0
            local tb = tonumber(b.updated_at) or 0
            if ta == tb then
                return tostring(a.id) < tostring(b.id)
            end
            return ta > tb
        end
        return sa > sb
    end)

    while #out > limit do
        table.remove(out)
    end
    return out
end

function M.get_stats()
    local total = 0
    for _ in pairs(M.policies) do
        total = total + 1
    end
    return {
        total_policies = total,
        storage_root = storage_root(),
    }
end

return M
