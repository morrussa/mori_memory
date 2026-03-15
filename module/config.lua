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

