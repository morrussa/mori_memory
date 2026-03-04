local function clear_mods()
    local mods = {
        "module.config",
        "module.tool",
        "module.memory.store",
        "module.memory.history",
        "module.memory.cluster",
        "module.memory.topic",
        "module.memory.adaptive",
        "module.memory.heat",
        "module.memory.recall",
    }
    for _, m in ipairs(mods) do
        package.loaded[m] = nil
        package.preload[m] = nil
    end
end

clear_mods()

local stats = {
    begin_turn = 0,
    reserve_preload_io = 0,
    adaptive_add_counter = 0,
    adaptive_update_after_recall = 0,
    adaptive_mark_dirty = 0,
    heat_enqueue_cold_rescue = 0,
}

package.preload["module.config"] = function()
    return {
        settings = {
            cluster = {
                supercluster_topn_query = 2,
            },
            ai_query = {
                max_memory = 3,
                max_turns = 3,
                recall_base = 0.1,
                history_search_bonus = 4.0,
                explicit_recall_bonus = 2.0,
                context_link_bonus = 1.0,
                min_sim_gate = 0.20,
                power_suppress = 1.20,
                memory_drop_sim = 0.10,
                topic_sim_threshold = 0.60,
                use_topic_buckets = false,
                smart_preload_enabled = true,
                preload_budget_per_query = 5,
                preload_max_io_per_turn = 8,
                preload_use_vector_prediction = true,
                soft_gate_enabled = true,
                soft_gate_margin = 0.10,
                refinement_enabled = true,
                refinement_start_turn = 1,
                expected_recall_enabled = true,
                learning_curve_enabled = false,
                keyword_queries = 2,
            },
        },
    }
end

package.preload["module.tool"] = function()
    local M = {}
    function M.get_embedding_query(_)
        return { 1.0, 0.0 }
    end
    function M.to_ptr_vec(vec)
        return vec
    end
    function M.cosine_similarity(a, b)
        if type(a) ~= "table" or type(b) ~= "table" then return 0.0 end
        local dot, na, nb = 0.0, 0.0, 0.0
        local n = math.min(#a, #b)
        for i = 1, n do
            local x = tonumber(a[i]) or 0.0
            local y = tonumber(b[i]) or 0.0
            dot = dot + x * y
            na = na + x * x
            nb = nb + y * y
        end
        if na <= 0 or nb <= 0 then return 0.0 end
        return dot / math.sqrt(na * nb)
    end
    return M
end

package.preload["module.memory.store"] = function()
    local M = {}
    function M.begin_turn(_)
        stats.begin_turn = stats.begin_turn + 1
    end
    function M.reserve_preload_io(_, _)
        stats.reserve_preload_io = stats.reserve_preload_io + 1
        return 0
    end
    function M.get_total_lines()
        return 1
    end
    function M.return_mem_vec(line)
        if tonumber(line) == 1 then
            return { 1.0, 0.0 }
        end
        return nil
    end
    function M.get_turns(line)
        if tonumber(line) == 1 then
            return { 1 }
        end
        return {}
    end
    function M.get_heat_by_index(_)
        return 10
    end
    function M.iter_topic_lines(_, _)
        return { 1 }
    end
    return M
end

package.preload["module.memory.history"] = function()
    local M = {}
    function M.get_turn()
        return 5
    end
    function M.get_by_turn(turn)
        if tonumber(turn) == 1 then
            return "dummy"
        end
        return nil
    end
    function M.parse_entry(_)
        return "用户历史", "助手历史"
    end
    return M
end

package.preload["module.memory.cluster"] = function()
    local M = {}
    M.clusters = {
        [1] = {
            centroid = { 1.0, 0.0 },
            members = { 1 },
            hot_members = { 1 },
            cold_members = {},
        },
    }
    function M.cluster_count()
        return 1
    end
    function M.super_candidate_clusters(_, _)
        return { 1 }, 0
    end
    function M.get_cluster_ids()
        return { 1 }
    end
    function M.find_sim_in_cluster(_, _, _)
        return {
            { index = 1, similarity = 0.95 },
        }
    end
    function M.get_cluster_id_for_line(line)
        if tonumber(line) == 1 then
            return 1
        end
        return nil
    end
    return M
end

package.preload["module.memory.topic"] = function()
    local M = {}
    M.active_topic = { start = 1 }
    M.topics = {}
    function M.get_topic_for_turn(_)
        return {
            is_active = true,
            centroid = { 1.0, 0.0 },
            topic_idx = nil,
        }
    end
    function M.get_topic_anchor(_)
        return "A:1"
    end
    return M
end

package.preload["module.memory.adaptive"] = function()
    local M = {}
    M.state = {
        learned_min_gate = 0.2,
    }
    function M.get_min_sim_gate(base)
        return tonumber(base) or 0.2
    end
    function M.get_route_score(_)
        return 0.0
    end
    function M.add_counter(_, _)
        stats.adaptive_add_counter = stats.adaptive_add_counter + 1
    end
    function M.update_after_recall(_)
        stats.adaptive_update_after_recall = stats.adaptive_update_after_recall + 1
    end
    function M.mark_dirty()
        stats.adaptive_mark_dirty = stats.adaptive_mark_dirty + 1
    end
    return M
end

package.preload["module.memory.heat"] = function()
    local M = {}
    function M.enqueue_cold_rescue(_, _, _, _)
        stats.heat_enqueue_cold_rescue = stats.heat_enqueue_cold_rescue + 1
    end
    return M
end

local recall = require("module.memory.recall")

recall._last_topic_anchor = "A:1"
recall._same_topic_streak = 7
recall._streak_sim_sum = 4.2
recall._streak_sim_count = 6
recall._prev_query_vec = { 1.0, 0.0 }
recall._last_recall_attempt_turn = 3
recall._topic_cache_anchor = "A:1"
recall._topic_cache_mem = { [99] = true }
recall._preload_cache_anchor = "A:1"
recall._preload_cache_clusters = { [77] = true }
recall._preload_cache_mem = { [88] = true }
recall._consecutive_empty_count = 2
recall._soft_gate_pass_count = 5

local before = {
    last_topic_anchor = recall._last_topic_anchor,
    same_topic_streak = recall._same_topic_streak,
    streak_sim_sum = recall._streak_sim_sum,
    streak_sim_count = recall._streak_sim_count,
    last_recall_attempt_turn = recall._last_recall_attempt_turn,
    topic_cache_anchor = recall._topic_cache_anchor,
    preload_cache_anchor = recall._preload_cache_anchor,
    consecutive_empty_count = recall._consecutive_empty_count,
    soft_gate_pass_count = recall._soft_gate_pass_count,
}

local ctx = recall.check_and_retrieve("之前的方案是什么", { 1.0, 0.0 }, { read_only = true })
assert(type(ctx) == "string", "read_only recall should return string context")
assert(ctx:find("相关记忆", 1, true), "read_only recall should still retrieve memory context")

assert(stats.begin_turn == 0, "read_only must not call memory.begin_turn")
assert(stats.reserve_preload_io == 0, "read_only must not reserve preload IO")
assert(stats.adaptive_add_counter == 0, "read_only must not mutate adaptive counters")
assert(stats.adaptive_update_after_recall == 0, "read_only must not run refinement update")
assert(stats.adaptive_mark_dirty == 0, "read_only must not mark adaptive dirty")
assert(stats.heat_enqueue_cold_rescue == 0, "read_only must not enqueue cold rescue")

assert(recall._last_topic_anchor == before.last_topic_anchor, "read_only must not update last_topic_anchor")
assert(recall._same_topic_streak == before.same_topic_streak, "read_only must not update same_topic_streak")
assert(recall._streak_sim_sum == before.streak_sim_sum, "read_only must not update streak_sim_sum")
assert(recall._streak_sim_count == before.streak_sim_count, "read_only must not update streak_sim_count")
assert(recall._last_recall_attempt_turn == before.last_recall_attempt_turn, "read_only must not update recall_attempt_turn")
assert(recall._topic_cache_anchor == before.topic_cache_anchor, "read_only must not clear topic cache anchor")
assert(recall._preload_cache_anchor == before.preload_cache_anchor, "read_only must not clear preload cache anchor")
assert(recall._consecutive_empty_count == before.consecutive_empty_count, "read_only must not adjust gate empty counter")
assert(recall._soft_gate_pass_count == before.soft_gate_pass_count, "read_only must not update soft gate counter")
assert(recall._topic_cache_mem[99] == true, "read_only must not rewrite topic cache content")
assert(recall._preload_cache_clusters[77] == true, "read_only must not rewrite preload cluster cache")
assert(recall._preload_cache_mem[88] == true, "read_only must not rewrite preload memory cache")

print("RECALL_READONLY_NO_SIDE_EFFECT_PASS")
