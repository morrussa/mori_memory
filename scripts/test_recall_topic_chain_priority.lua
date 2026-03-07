package.path = "./?.lua;./?/init.lua;" .. package.path

local function run_case(case)
    package.loaded["module.memory.recall"] = nil
    package.loaded["module.tool"] = {
        cosine_similarity = function(a, b)
            local sum = 0
            local na = 0
            local nb = 0
            local n = math.max(#(a or {}), #(b or {}))
            for i = 1, n do
                local va = tonumber((a or {})[i]) or 0
                local vb = tonumber((b or {})[i]) or 0
                sum = sum + va * vb
                na = na + va * va
                nb = nb + vb * vb
            end
            if na <= 0 or nb <= 0 then
                return 0
            end
            return sum / math.sqrt(na * nb)
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
        get_topic_anchor = function(_) return nil end,
        get_type_name = function(_) return "Fact" end,
        get_default_type_name = function() return "Fact" end,
        get_type_meta = function(_) return { name = "Fact", class = "stable", recent_weight = 0.0 } end,
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
                memory_drop_sim = 0.0,
                min_sim_gate = 0.0,
                max_memory = 8,
                max_turns = 1,
                keyword_weight = 1.0,
                keyword_queries = 1,
                keyword_noise_mix = 0.0,
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
                topic_coverage_alpha = 0.0,
                topic_coverage_beta = 1.0,
                topic_coverage_min_sim = 0.0,
                topic_chain_candidate_topn = 3,
                topic_chain_candidate_min_score = 0.4,
                topic_chain_recent_score_ratio = 0.88,
                topic_chain_recent_score_margin = 0.12,
                topic_chain_rep_bonus = 0.08,
                topic_chain_member_penalty = 0.04,
                topic_chain_current_bonus = 0.0,
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
    local chain_map = case.chain_map or {}
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
        find_related_topics = function(ref, _opts)
            return chain_map[tostring(ref or "")] or {}
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
    name = "recent_topic_wins_inside_chain_when_quality_close",
    memory_count = 2,
    history_turn = 30,
    mem_turns = {
        [1] = { 10 },
        [2] = { 20 },
    },
    query_results = {
        {
            { index = 1, sim = 0.91 },
            { index = 2, sim = 0.88 },
        },
    },
    topics = {
        { start = 10, end_ = 10, centroid = { 1, 0 }, summary = "Older topic" },
        { start = 20, end_ = 20, centroid = { 1, 0 }, summary = "Newer topic" },
    },
    chain_map = {
        ["S:10"] = { { key = "S:20", score = 0.82 } },
        ["S:20"] = { { key = "S:10", score = 0.82 } },
    },
    expected_line = "主题1 | 来源turn: 20",
    reject_line = "来源turn: 10",
})

run_case({
    name = "older_topic_kept_when_newer_is_not_good_enough",
    memory_count = 2,
    history_turn = 30,
    mem_turns = {
        [1] = { 10 },
        [2] = { 20 },
    },
    query_results = {
        {
            { index = 1, sim = 0.96 },
            { index = 2, sim = 0.70 },
        },
    },
    topics = {
        { start = 10, end_ = 10, centroid = { 1, 0 }, summary = "Older topic" },
        { start = 20, end_ = 20, centroid = { 1, 0 }, summary = "Newer topic" },
    },
    chain_map = {
        ["S:10"] = { { key = "S:20", score = 0.82 } },
        ["S:20"] = { { key = "S:10", score = 0.82 } },
    },
    expected_line = "主题1 | 来源turn: 10",
    reject_line = "来源turn: 20",
})

print("test_recall_topic_chain_priority: ok")
