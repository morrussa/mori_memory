local M = {}

-- Memory-core configuration only.
-- This repo intentionally omits any agent/graph runtime settings.

local DEFAULT_SETTINGS = {
    -- 两个记忆如果相似度 > merge_limit，则合并为一个
    merge_limit = 0.95,
    -- embedding 向量维度（仅用于外部约定；memory 内部以实际 vec 长度为准）
    dim = 1024,

    -- topic_graph.retrieve(...) 会用到的保底参数
    ai_query = {
        max_memory = 5,
        max_turns = 10,
    },

    topic_graph = {
        storage = {
            root = "memory/v4/topic_graph",
        },
        seed_topics = 2,
        expand_budget = 6,
        max_return_topics = 4,
        per_topic_evidence = 3,
        loaded_cap = 24,
        load_budget = 4,
        bridge_topk = 8,
        max_bridge_hops = 2,
        transition_lr = 0.12,
        recall_lr = 0.10,
        adopt_lr = 0.18,
        decay = 0.995,
        min_bridge_score = 0.08,
        query_semantic_weight = 1.00,
        bridge_weight = 0.55,
        resident_bonus = 0.08,
        current_topic_bonus = 0.12,
        family_topk = 2,
        family_similarity = 0.84,
        family_member_limit = 1,
        self_excite_penalty = 0.14,
        family_revisit_penalty = 0.10,
        family_escape_bonus = 0.06,
        facet_slots = 6,
        context_similarity_saturation_threshold = 0.95,
        retrieve_max_memories = 8,
        retrieve_max_turns = 10,
        topic_cap = 64,
        next_cap = 32,
        recent_weight = 0.55,
        topic_prior_weight = 1.00,
        flow_runtime_ttl = 128,
        flow_runtime_cap = 256,
        activation = {
            -- Topic activation is a non-exclusive temporal signal layer.
            -- It does not change primary memory->topic ownership.
            decay = 0.985,
            max_strength = 8.0,
            bind_inject = 1.0,
            recall_inject = 0.10,
            adopt_inject = 0.18,
            open_threshold = 0.25,
            active_threshold = 0.25,
            close_threshold = 0.12,
            windows_cap = 128,
            -- Conservative retrieval prior from topic activation.
            prior_weight = 0.08,
            prior_max_bonus = 0.10,
            prior_semantic_gate = 0.35,
            prior_recency_turns = 64,
            prior_closed_window_factor = 0.60,
        },
        deep_artmap = {
            category_vigilance = 0.88,
            category_beta = 0.28,
            bundle_vigilance = 0.82,
            bundle_beta = 0.18,
            query_bundles = 2,
            query_margin = 0.06,
            neighbor_topk = 2,
            query_categories = 4,
            exemplars = 12,
            temporal_link_window = 8,
            bundle_prior_weight = 0.18,
            bundle_recall_lr = 0.10,
            bundle_adopt_lr = 0.18,
            bundle_edge_feedback = 0.14,
            bundle_decay = 0.996,
        },
        topic_hnsw = {
            enabled = true,
            k = 48,
            m = 16,
            ef_construction = 80,
            ef_search = 32,
            max_elements = 2048,
        },
        legacy = {
            topic_predictor_path = "memory/topic_predictor.txt",
            adaptive_state_path = "memory/adaptive_state.txt",
        },
    },

    topic = {
        make_cluster1 = 4,
        make_cluster2 = 3,
        topic_limit = 0.62,
        break_limit = 0.48,
        confirm_limit = 0.55,
        min_topic_length = 2,
        summary_max_tokens = 192,
        allow_llm_summary = false,
        -- Memory core no longer owns embeddings; keep rebuild disabled by default.
        rebuild = false,
        summary_variant_weights = {
            full = 1.00,
            slight = 0.72,
            heavy = 0.40,
            none = 0.00,
        },
        summary_compress_ratio_slight = 0.65,
        summary_compress_ratio_heavy = 0.30,
    },

    -- 仅用于 legacy 导入（可选）
    storage_v3 = {
        root = "memory/v3",
    },

    -- Anti-poisoning / multi-user robustness (optional).
    -- These are best-effort heuristics: they only take effect if caller passes
    -- event identity fields (e.g. source/user_id/nickname/room_id) into meta.
    guard = {
        enabled = true,
        -- File stores a Lua table literal (no code execution).
        grudge_path = "memory/grudge.lua",
        scope_strategy = "source_room",
        -- Optional per-source override, e.g. { bilibili = "source_room_user" }.
        scope_strategy_by_source = {},
        -- Prefix topic anchors with scope key for non-default sources.
        anchor_scope_prefix = true,
        default_credit = 1.0,
        default_credit_by_source = {
            stdin = 1.0,
            bilibili = 0.35,
            system = 1.0,
        },
        max_users = 2048,
        note_once = true,
        note_threshold = 0.65,
        -- When credit drops below this threshold, the user is temporarily blocked
        -- from writing into memory/topic/history. Unblock happens only after the
        -- cooldown window passes.
        block_threshold = 0.25,
        block_duration_s = 3600,
        restore_threshold = 0.75,
        credit_decay = 0.985,
        credit_bonus = 0.02,
        credit_penalty = 0.40,
        -- Thresholds that influence memory/topic behaviors.
        allow_recall_threshold = 0.70,
        allow_history_threshold = 0.70,
        allow_topic_threshold = 0.60,
        allow_memory_write_threshold = 0.75,
        -- When enabled, topic boundaries can be forced on scope changes.
        topic_scope_isolation = true,
    },

    -- Multi-stream conversation disentanglement (optional).
    -- This helps when inputs are noisy / interleaved (e.g., live chat) and the
    -- single active topic anchor becomes unstable, harming recall & writes.
    disentangle = {
        -- Master switch. Keep this enabled and restrict by `enable_sources`.
        enabled = true,
        -- If set, only run disentanglement for these sources.
        -- Accepts either an array {"bilibili"} or a map {bilibili=true}.
        enable_sources = { "bilibili" },

        -- Max parallel streams per scope (upper bound).
        max_streams = 6,
        -- Rolling window size for stream centroid.
        window_size = 4,

        -- Stream assignment thresholds (cosine similarity-ish).
        assign_threshold = 0.80,
        -- Below `assign_threshold` but above this threshold, ambiguous turns
        -- stay in local pending instead of immediately opening/committing.
        pending_threshold = 0.72,
        -- If best-vs-second-best margin is smaller than this, prefer
        -- `keep_pending` over eager routing.
        pending_margin = 0.06,

        -- Heuristics.
        same_user_bonus = 0.06,
        participant_bonus = 0.03,
        mention_bonus = 0.05,
        addressee_hint_bonus = 0.04,
        reply_cue_bonus = 0.05,
        reply_recent_turns = 6,
        centroid_weight = 0.42,
        tail_weight = 0.38,
        head_weight = 0.20,
        stability_bonus = 0.04,
        stability_turns = 4,
        age_penalty = 0.01,
        stale_turns = 60,
        orphan_stale_turns = 20,
        local_pending_cap = 4,
        -- Real live rooms contain many short reaction-like messages. When they
        -- are recent and low-information, prefer attaching them to a recent
        -- stable stream instead of spawning endless orphan threads.
        reaction_fallback_enabled = true,
        reaction_short_chars = 6,
        reaction_max_chars = 10,
        reaction_attach_score = 0.46,
        reaction_dominant_score = 0.36,
        reaction_recent_turns = 3,
        reaction_stability_min = 0.18,
        ambient_enabled = true,
        ambient_local_pending_cap = 12,
        -- In split/local-sequence mode, keep the newest turn in each segment
        -- pending until another turn confirms it or it idles out.
        commit_idle_turns = 2,
        commit_chunk_turns = 2,
        pending_context_turns = 2,

        -- If set, assign-but-reset the stream topic when centroid similarity
        -- drops below this threshold (big reset).
        reset_threshold = 0.62,

        -- Merge back to single stream when one stream dominates for
        -- `merge_streak_turns` consecutive turns and other streams have been
        -- idle for `merge_idle_turns`.
        merge_idle_turns = 8,
        merge_streak_turns = 4,
        reset_on_merge = true,
        runtime = {
            root = "memory/v4/runtime",
            checkpoint_interval_turns = 24,
        },
    },
}

local function deep_copy(value, seen)
    if type(value) ~= "table" then
        return value
    end
    seen = seen or {}
    if seen[value] then
        return seen[value]
    end
    local out = {}
    seen[value] = out
    for k, v in pairs(value) do
        out[deep_copy(k, seen)] = deep_copy(v, seen)
    end
    return out
end

local function get_by_path(root, path)
    if type(root) ~= "table" then
        return nil
    end
    if type(path) ~= "string" or path == "" then
        return root
    end
    local cur = root
    for seg in path:gmatch("[^%.]+") do
        if type(cur) ~= "table" then
            return nil
        end
        cur = cur[seg]
        if cur == nil then
            return nil
        end
    end
    return cur
end

M.defaults = deep_copy(DEFAULT_SETTINGS)
M.settings = deep_copy(DEFAULT_SETTINGS)

function M.reset()
    M.settings = deep_copy(M.defaults)
    return M.settings
end

function M.get(path, fallback)
    local v = get_by_path(M.settings, path)
    if v == nil then
        return fallback
    end
    return v
end

function M.get_default(path, fallback)
    local v = get_by_path(M.defaults, path)
    if v == nil then
        return fallback
    end
    return v
end

return M
