package.path = "./?.lua;./?/init.lua;" .. package.path

package.loaded["module.memory.topic_predictor"] = nil

package.loaded["module.config"] = {
    settings = {
        ai_query = {
            topic_activation_memory_topn = 4,
            topic_activation_chain_topn = 1,
            topic_activation_chain_min_score = 0.0,
            topic_activation_topic_weight = 1.0,
            topic_activation_chain_weight = 0.5,
            topic_activation_resident_weight = 0.2,
            topic_activation_recent_weight = 0.8,
            topic_activation_query_weight = 0.3,
            topic_activation_min_score = 0.0,
            topic_activation_recall_weight = 0.4,
            topic_activation_adopted_weight = 1.6,
            topic_activation_topic_cap = 16,
            topic_activation_next_cap = 8,
            topic_activation_transition_cap = 4,
            topic_activation_cross_topic_next_weight = 0.4,
        },
    },
}

package.loaded["module.persistence"] = {
    write_atomic = function(_, _, writer)
        local sink = {
            write = function() return true end,
        }
        return writer(sink)
    end,
}

package.loaded["module.memory.saver"] = {
    mark_dirty = function() end,
}

package.loaded["module.tool"] = {
    cosine_similarity = function(a, b)
        local dot = 0.0
        local na = 0.0
        local nb = 0.0
        local dim = math.max(#(a or {}), #(b or {}))
        for i = 1, dim do
            local va = tonumber((a or {})[i]) or 0.0
            local vb = tonumber((b or {})[i]) or 0.0
            dot = dot + va * vb
            na = na + va * va
            nb = nb + vb * vb
        end
        if na <= 0.0 or nb <= 0.0 then
            return 0.0
        end
        return dot / math.sqrt(na * nb)
    end,
}

package.loaded["module.memory.topic"] = {
    get_stable_anchor = function(turn)
        return "S:" .. tostring(turn or 1)
    end,
    get_topic_fingerprint = function(key)
        return { key = tostring(key or "") }
    end,
    get_topic_chain = function(key, _opts)
        if tostring(key or "") == "S:1" then
            return {
                { key = "S:2", score = 0.6 },
            }
        end
        return {}
    end,
}

package.loaded["module.memory.store"] = {
    iter_topic_lines = function(topic_key)
        local buckets = {
            ["S:1"] = { 11, 12, 13 },
            ["S:2"] = { 21 },
        }
        return buckets[tostring(topic_key or "")] or {}
    end,
    return_mem_vec = function(mem_idx)
        local vectors = {
            [11] = { 1.0, 0.0 },
            [12] = { 0.95, 0.05 },
            [13] = { 0.9, 0.1 },
            [21] = { 0.2, 0.8 },
        }
        return vectors[tonumber(mem_idx)]
    end,
}

package.loaded["module.memory.ghsom"] = {
    get_node_for_line = function(mem_idx)
        local mapping = {
            [11] = 1,
            [12] = 1,
            [13] = 2,
            [21] = 3,
        }
        return mapping[tonumber(mem_idx)]
    end,
}

local predictor = require("module.memory.topic_predictor")
predictor.reset_defaults()

predictor.observe("S:1", nil, {
    retrieved_memories = { 11, 12 },
    adopted_memories = { 12 },
})
predictor.observe("S:1", { 12, 13 })

local pred = predictor.predict("S:1", {
    query_vec = { 1.0, 0.0 },
})

assert(type(pred) == "table", "predict should return a table")
assert(#(pred.lines or {}) >= 2, "predict should return activated memories")
assert(pred.lines[1] == 12, "adopted memory should outrank recall-only memory")
assert((pred.memory_scores or {})[12] > (pred.memory_scores or {})[11], "assistant adoption feedback should strengthen the target memory")
assert((pred.memory_scores or {})[13] ~= nil, "recent transition memory should be preserved")
assert(((pred.node_scores or {})[1] or 0.0) > ((pred.node_scores or {})[3] or 0.0), "same-topic node should outrank chain-only node")

print("test_topic_predictor: ok")
