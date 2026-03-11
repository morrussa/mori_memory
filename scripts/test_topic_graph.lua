#!/usr/bin/env luajit

package.path = package.path .. ";./?.lua;./module/?.lua;./module/memory/?.lua;./module/graph/?.lua"

local function norm(vec)
    local sum = 0.0
    for i = 1, #vec do
        sum = sum + vec[i] * vec[i]
    end
    sum = math.sqrt(sum)
    for i = 1, #vec do
        vec[i] = vec[i] / sum
    end
    return vec
end

local root = string.format("/tmp/mori_topic_graph_retrieval_%d_%d", os.time(), math.random(1000, 9999))

package.loaded["module.memory.history"] = {
    get_turn = function()
        return 9
    end,
    get_by_turn = function(turn)
        local rows = {
            [1] = { user = "alpha start", ai = "alpha reply" },
            [2] = { user = "alpha detail", ai = "alpha answer" },
            [3] = { user = "beta branch", ai = "beta reply" },
        }
        return rows[tonumber(turn)]
    end,
    parse_entry = function(entry)
        return entry.user, entry.ai
    end,
}

package.loaded["module.memory.topic"] = {
    get_stable_anchor = function(turn)
        if tonumber(turn) and tonumber(turn) >= 3 then
            return "topic:b"
        end
        return "topic:a"
    end,
    get_topic_anchor = function(turn)
        if tonumber(turn) and tonumber(turn) >= 3 then
            return "topic:b"
        end
        return "topic:a"
    end,
}

local config = require("module.config")
config.settings.topic_graph = config.settings.topic_graph or {}
config.settings.topic_graph.storage = { root = root }
config.settings.topic_graph.topic_hnsw = config.settings.topic_graph.topic_hnsw or {}
config.settings.topic_graph.topic_hnsw.enabled = true
config.settings.topic_graph.max_return_topics = 2
config.settings.topic_graph.per_topic_evidence = 2

package.loaded["module.memory.topic_graph"] = nil
package.loaded["module.memory.store"] = nil

local store = require("module.memory.store")
local topic_graph = require("module.memory.topic_graph")

assert(store.load())

local vec_a1 = norm({1, 0, 0, 0, 0, 0, 0, 0})
local vec_a2 = norm({0.96, 0.04, 0, 0, 0, 0, 0, 0})
local vec_b1 = norm({0, 1, 0, 0, 0, 0, 0, 0})

local line_a1 = assert(store.add_memory(vec_a1, 1, { topic_anchor = "topic:a" }))
local line_a2 = assert(store.add_memory(vec_a2, 2, { topic_anchor = "topic:a" }))
local line_b1 = assert(store.add_memory(vec_b1, 3, { topic_anchor = "topic:b" }))

topic_graph.observe_turn(4, "topic:a")
topic_graph.observe_feedback("topic:a", {
    selected_memories = { line_b1 },
    predicted_topics = { "topic:b" },
}, { line_b1 }, 4)

local same_topic = topic_graph.retrieve(vec_a1, "topic:a", 5, {})
assert(type(same_topic) == "table", "same-topic retrieve should return table")
assert(#(same_topic.predicted_topics or {}) >= 1, "same-topic retrieve should return predicted topics")
assert(#(same_topic.selected_memories or {}) >= 1, "same-topic retrieve should return memories")

local bridge_query = topic_graph.retrieve(vec_a1, "topic:a", 6, {})
local has_topic_b = false
for _, anchor in ipairs(bridge_query.predicted_topics or {}) do
    if tostring(anchor) == "topic:b" then
        has_topic_b = true
        break
    end
end
assert(has_topic_b, "bridge-expanded topics should include topic:b")

local adopted = topic_graph.retrieve(vec_b1, "topic:a", 7, {})
local selected = {}
for _, mem_id in ipairs(adopted.selected_memories or {}) do
    selected[tonumber(mem_id)] = true
end
assert(selected[line_b1] == true, "topic:b memory should be retrievable after feedback")

print("test_topic_graph: ok")
