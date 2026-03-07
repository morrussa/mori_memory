package.path = "./?.lua;./?/init.lua;" .. package.path

local function run_case(case)
    package.loaded["module.memory.recall"] = nil
    package.loaded["module.tool"] = {
        cosine_similarity = function(a, b)
            local sum = 0
            local n = math.max(#(a or {}), #(b or {}))
            for i = 1, n do
                sum = sum + (tonumber((a or {})[i]) or 0) * (tonumber((b or {})[i]) or 0)
            end
            return sum
        end,
        to_ptr_vec = function(v) return v end,
        get_embedding_query = function(_) return { 1, 0 } end,
    }

    package.loaded["module.memory.store"] = {
        get_total_lines = function() return case.memory_count end,
        begin_turn = function(_) end,
        build_type_filter = function(_) return nil end,
        match_type_name = function(v) return v end,
        type_matches = function(_, _, _) return true end,
        reserve_preload_io = function(_, _) return 0 end,
        iter_topic_lines = function(_, _) return {} end,
        get_turns = function(line) return case.mem_turns[line] or {} end,
        get_type_name = function(line) return case.mem_types[line] or "Fact" end,
        get_default_type_name = function() return "Fact" end,
        get_type_meta = function(type_name)
            local meta = (case.type_meta or {})[tostring(type_name)] or {}
            local out = { name = tostring(type_name) }
            for k, v in pairs(meta) do
                out[k] = v
            end
            return out
        end,
        get_heat_by_index = function(_) return 1 end,
        return_mem_vec = function(_) return { 1, 0 } end,
    }

    package.loaded["module.memory.history"] = {
        get_turn = function() return case.history_turn end,
        get_by_turn = function(turn)
            return string.format("user%d\31assistant%d", turn, turn)
        end,
        parse_entry = function(entry)
            local user, assistant = tostring(entry):match("^(.-)\31(.*)$")
            return user or "", assistant or ""
        end,
    }

    package.loaded["module.config"] = {
        settings = {
            ai_query = {
                memory_drop_sim = 0.50,
                min_sim_gate = 0.0,
                max_memory = 16,
                max_turns = 1,
                keyword_weight = 1.0,
                keyword_queries = 1,
                keyword_perf_mode = "lossless",
                power_suppress = 1.0,
                learning_curve_enabled = false,
                smart_preload_enabled = false,
                soft_gate_enabled = false,
                expected_recall_enabled = false,
                refinement_enabled = false,
                persistent_explore_enabled = false,
                use_topic_buckets = false,
                compiled_context_max_topics = 1,
                compiled_context_max_turns_per_topic = 1,
                compiled_context_neighbor_window = 0,
                compiled_context_summary_chars = 80,
                compiled_context_user_chars = 40,
                compiled_context_ai_chars = 40,
                supercluster_topn_query = 1,
                topic_sim_threshold = 0.7,
                topic_coverage_alpha = 0.45,
                topic_coverage_beta = 1.0,
                topic_coverage_min_sim = 0.50,
            },
            graph = {
                recall = {
                    legacy_trigger_enabled = false,
                },
            },
        },
    }

    package.loaded["module.memory.cluster"] = {
        clusters = {
            [1] = { centroid = { 1, 0 }, members = {} },
        },
        super_candidate_clusters = function(_, _) return { 1 }, 0 end,
        find_sim_in_cluster = function(_, _, _)
            local out = {}
            for _, item in ipairs(case.sim_results or {}) do
                out[#out + 1] = { index = item.index, similarity = item.sim }
            end
            return out
        end,
        get_cluster_ids = function() return { 1 } end,
        cluster_count = function() return 1 end,
        get_cluster_id_for_line = function(_) return 1 end,
        get_cluster_type_affinity = function(_, _) return 0 end,
    }

    local topic_recs = case.topics
    package.loaded["module.memory.topic"] = {
        topics = topic_recs,
        active_topic = { start = nil },
        get_topic_for_turn = function(turn)
            for idx, rec in ipairs(topic_recs) do
                if turn >= rec.start and turn <= rec.end_ then
                    return { is_active = false, centroid = rec.centroid, topic_idx = idx }
                end
            end
            return nil
        end,
        get_topic_anchor = function(turn)
            for idx, rec in ipairs(topic_recs) do
                if turn >= rec.start and turn <= rec.end_ then
                    return "C:" .. tostring(idx)
                end
            end
            return nil
        end,
    }

    package.loaded["module.memory.adaptive"] = {
        state = nil,
        get_min_sim_gate = function(v) return v end,
        add_counter = function(_, _) end,
        get_route_score = function(_) return 0 end,
        update_after_recall = function(_) end,
        mark_dirty = function() end,
    }

    package.loaded["module.memory.heat"] = {
        enqueue_cold_rescue = function(_, _, _, _) end,
    }

    local recall = require("module.memory.recall")
    local out = recall.check_and_retrieve(case.user_input, { 1, 0 }, { force = true, policy_decided = true })
    assert(out:find(case.expected_line, 1, true), "missing expected line: " .. tostring(case.expected_line))
end

local canonical_type_meta = {
    User = { class = "stable", recent_weight = 0.00 },
    Concept = { class = "stable", recent_weight = 0.00 },
    Constraint = { class = "stable", recent_weight = 0.00, head_visible = true, head_bias = 0.18 },
    Project = { class = "rolling", recent_weight = 0.10 },
    Preference = { class = "rolling", recent_weight = 0.10 },
    Decision = { class = "rolling", recent_weight = 0.08, current_topic_boost = 0.12 },
    Artifact = { class = "mode_sensitive", recent_weight = 0.04, coding_recent_weight = 0.28, coding_frontier_bias = 0.18 },
    Status = { class = "frontier", recent_weight = 0.38, frontier_bias = 0.22 },
    Blocker = { class = "frontier", recent_weight = 0.46, frontier_bias = 0.30 },
    Attempt = { class = "frontier", recent_weight = 0.52, frontier_bias = 0.08, stale_penalty = 0.18 },
    Verification = { class = "frontier", recent_weight = 0.48, frontier_bias = 0.26 },
    Fact = { class = "stable", recent_weight = 0.02 },
}

run_case({
    user_input = "继续看一下当前进度状态",
    memory_count = 2,
    history_turn = 5,
    mem_turns = {
        [1] = { 1 },
        [2] = { 5 },
    },
    mem_types = {
        [1] = "User",
        [2] = "Status",
    },
    type_meta = canonical_type_meta,
    sim_results = {
        { index = 1, sim = 0.88 },
        { index = 2, sim = 0.70 },
    },
    topics = {
        { start = 1, end_ = 5, centroid = { 1, 0 }, summary = "Work topic" },
    },
    expected_line = "主题1 | 来源turn: 5",
})

run_case({
    user_input = "把这个约束记住",
    memory_count = 2,
    history_turn = 4,
    mem_turns = {
        [1] = { 1 },
        [2] = { 4 },
    },
    mem_types = {
        [1] = "Constraint",
        [2] = "Concept",
    },
    type_meta = canonical_type_meta,
    sim_results = {
        { index = 2, sim = 0.74 },
        { index = 1, sim = 0.70 },
    },
    topics = {
        { start = 1, end_ = 4, centroid = { 1, 0 }, summary = "Constraint topic" },
    },
    expected_line = "主题1 | 来源turn: 1",
})

print("test_recall_type_order: ok")
