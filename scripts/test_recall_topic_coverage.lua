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
        get_topic_anchor = function(line)
            if case.mem_topic_anchors then
                return case.mem_topic_anchors[line]
            end
            return nil
        end,
        get_type_name = function(_) return "Fact" end,
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
                keyword_weight = 0.55,
                keyword_queries = 2,
                keyword_noise_mix = 0.20,
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

    local call_no = 0
    package.loaded["module.memory.cluster"] = {
        clusters = {
            [1] = { centroid = { 1, 0 }, members = {} },
        },
        super_candidate_clusters = function(_, _) return { 1 }, 0 end,
        find_sim_in_cluster = function(_, _, _)
            call_no = call_no + 1
            local out = {}
            for _, item in ipairs(case.query_results[call_no] or {}) do
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

    math.randomseed(7)
    local recall = require("module.memory.recall")
    local out = recall.check_and_retrieve(case.name, { 1, 0 }, { force = true, policy_decided = true })
    assert(out:find(case.expected_line, 1, true), "missing expected line: " .. tostring(case.expected_line))
    if case.reject_line then
        assert(not out:find(case.reject_line, 1, true), "unexpected line present: " .. tostring(case.reject_line))
    end
end

run_case({
    name = "coverage_reward_promotes_topic",
    memory_count = 3,
    history_turn = 10,
    mem_turns = {
        [1] = { 1 },
        [2] = { 2 },
        [3] = { 10 },
    },
    query_results = {
        {
            { index = 3, sim = 0.90 },
            { index = 1, sim = 0.82 },
        },
        {
            { index = 2, sim = 0.88 },
            { index = 3, sim = 0.20 },
        },
    },
    topics = {
        { start = 1, end_ = 2, centroid = { 1, 0 }, summary = "Topic A" },
        { start = 10, end_ = 10, centroid = { 1, 0 }, summary = "Topic B" },
    },
    expected_line = "主题1 | 来源turn: 1",
    reject_line = "来源turn: 10",
})

run_case({
    name = "length_without_coverage_does_not_win",
    memory_count = 6,
    history_turn = 30,
    mem_turns = {
        [1] = { 20 },
        [2] = { 21 },
        [3] = { 22 },
        [4] = { 23 },
        [5] = { 24 },
        [6] = { 30 },
    },
    query_results = {
        {
            { index = 6, sim = 0.74 },
            { index = 5, sim = 0.58 },
            { index = 4, sim = 0.57 },
            { index = 3, sim = 0.56 },
            { index = 2, sim = 0.55 },
            { index = 1, sim = 0.54 },
        },
        {
            { index = 6, sim = 0.10 },
            { index = 1, sim = 0.20 },
        },
    },
    topics = {
        { start = 20, end_ = 24, centroid = { 1, 0 }, summary = "Long topic" },
        { start = 30, end_ = 30, centroid = { 1, 0 }, summary = "Short topic" },
    },
    expected_line = "主题1 | 来源turn: 30",
    reject_line = "来源turn: 24",
})

run_case({
    name = "coverage_bonus_requires_extra_dimension",
    memory_count = 3,
    history_turn = 10,
    mem_turns = {
        [1] = { 1 },
        [2] = { 2 },
        [3] = { 10 },
    },
    query_results = {
        {
            { index = 3, sim = 0.80 },
            { index = 1, sim = 0.70 },
        },
        {
            { index = 2, sim = 0.70 },
            { index = 3, sim = 0.20 },
        },
    },
    topics = {
        { start = 1, end_ = 2, centroid = { 1, 0 }, summary = "Wide topic" },
        { start = 10, end_ = 10, centroid = { 1, 0 }, summary = "Single-hit topic" },
    },
    expected_line = "主题1 | 来源turn: 1",
    reject_line = "来源turn: 10",
})

run_case({
    name = "coverage_is_not_broadcast_across_topics",
    memory_count = 2,
    history_turn = 12,
    mem_turns = {
        [1] = { 1, 10 },
        [2] = { 10 },
    },
    mem_topic_anchors = {
        [1] = "C:1",
        [2] = "C:2",
    },
    query_results = {
        {
            { index = 2, sim = 0.84 },
            { index = 1, sim = 0.75 },
        },
        {
            { index = 1, sim = 0.74 },
            { index = 2, sim = 0.10 },
        },
    },
    topics = {
        { start = 1, end_ = 1, centroid = { 1, 0 }, summary = "Owner topic" },
        { start = 10, end_ = 10, centroid = { 1, 0 }, summary = "Borrowed topic" },
    },
    expected_line = "主题1 | 来源turn: 1",
    reject_line = "来源turn: 10",
})

print("test_recall_topic_coverage: ok")
