package.path = "./?.lua;./?/init.lua;" .. package.path

package.loaded["module.memory.topic"] = nil

package.loaded["module.memory.store"] = {
    iter_topic_lines = function(topic_key)
        local buckets = {
            ["S:1"] = { 11, 12, 13, 14 },
            ["S:3"] = { 21, 22, 23 },
            ["S:5"] = { 31, 32, 33 },
        }
        return buckets[tostring(topic_key or "")] or {}
    end,
    get_cluster_id = function(line)
        local mapping = {
            [11] = 1,
            [12] = 1,
            [13] = 1,
            [14] = 2,
            [21] = 1,
            [22] = 1,
            [23] = 3,
            [31] = 4,
            [32] = 4,
            [33] = 5,
        }
        return mapping[tonumber(line)]
    end,
}

package.loaded["module.memory.history"] = {
    get_turn = function() return 6 end,
    get_turn_text = function(_) return "" end,
}

package.loaded["module.config"] = {
    settings = {
        topic = {
            make_cluster1 = 4,
            make_cluster2 = 3,
            topic_limit = 0.62,
            break_limit = 0.48,
            confirm_limit = 0.55,
            min_topic_length = 2,
            summary_max_tokens = 64,
            rebuild = false,
            fingerprint_topk = 3,
            chain_topn = 3,
            chain_min_score = 0.20,
            chain_centroid_weight = 0.55,
            chain_hist_weight = 0.45,
            chain_dominant_bonus = 0.05,
        },
    },
}

package.loaded["module.tool"] = {
    average_vectors = function(vectors)
        local out = {}
        local count = #(vectors or {})
        if count <= 0 then
            return out
        end
        local dim = #(vectors[1] or {})
        for i = 1, dim do
            local sum = 0.0
            for _, vec in ipairs(vectors) do
                sum = sum + (tonumber((vec or {})[i]) or 0.0)
            end
            out[i] = sum / count
        end
        return out
    end,
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

local topic = require("module.memory.topic")

topic.topics = {
    { start = 1, end_ = 2, summary = "alpha", centroid = { 1.0, 0.0 } },
    { start = 3, end_ = 4, summary = "beta", centroid = { 0.96, 0.04 } },
    { start = 5, end_ = 6, summary = "gamma", centroid = { 0.0, 1.0 } },
}

local fp = topic.get_topic_fingerprint("S:1")
assert(fp.memory_count == 4, "fingerprint should count topic memories")
assert(fp.cluster_count == 2, "fingerprint should count unique clusters")
assert(fp.dominant_cluster == 1, "dominant cluster should be cluster 1")
assert(fp.top_clusters[1].cluster_id == 1 and fp.top_clusters[1].hits == 3, "top cluster histogram mismatch")
assert(math.abs((fp.top_clusters[1].weight or 0.0) - 0.75) < 1e-6, "top cluster weight mismatch")

local chain = topic.get_topic_chain("S:1", { topn = 2, min_score = 0.0 })
assert(#chain == 2, "chain should return the requested number of neighbors")
assert(chain[1].key == "S:3", "nearest topic should be the similar histogram/centroid topic")
assert(chain[2].key == "S:5", "distant topic should rank behind the similar topic")
assert((chain[1].fingerprint_overlap or 0.0) > (chain[2].fingerprint_overlap or 0.0), "histogram overlap should separate candidates")
assert((chain[1].score or 0.0) > (chain[2].score or 0.0), "chain score ordering mismatch")

print("test_topic_chain: ok")
