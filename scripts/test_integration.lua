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

local root = os.getenv("MORI_TEST_ROOT")
if not root or root == "" then
    root = string.format("/tmp/mori_topic_graph_integration_%d_%d", os.time(), math.random(1000, 9999))
end

local config = require("module.config")
config.settings.topic_graph = config.settings.topic_graph or {}
config.settings.topic_graph.storage = { root = root .. "/topic_graph" }
config.settings.topic_graph.topic_hnsw = config.settings.topic_graph.topic_hnsw or {}
config.settings.topic_graph.topic_hnsw.enabled = true

local store = require("module.memory.store")
local topic_graph = require("module.memory.topic_graph")

print("=== store + topic_graph integration ===")
print("root:", root)

assert(store.load())

local vec_a = norm({1, 0, 0, 0, 0, 0, 0, 0})
local vec_b = norm({0, 1, 0, 0, 0, 0, 0, 0})
local vec_c = norm({0, 0, 1, 0, 0, 0, 0, 0})

local line_a = assert(store.add_memory(vec_a, 201, { topic_anchor = "topic:a", text = "alpha topic lua" }))
local line_b = assert(store.add_memory(vec_b, 202, { topic_anchor = "topic:b", text = "beta topic python" }))
local line_c = assert(store.add_memory(vec_c, 203, { topic_anchor = "topic:c", text = "gamma topic graph" }))
assert(line_a == 1 and line_b == 2 and line_c == 3, "unexpected line allocation")

local merged = assert(store.add_memory(norm({0.999, 0.001, 0, 0, 0, 0, 0, 0}), 204, {
    topic_anchor = "topic:a",
    text = "alpha topic lua repeated",
}))
assert(merged == 1, "reachable duplicate should merge into line 1")
assert(store.get_total_lines() == 3, "merge should not create a new line")
assert(store.get_cluster_id(1) > 0, "deep_artmap should assign a category id")

topic_graph.observe_turn(205, "topic:a")
topic_graph.observe_feedback("topic:a", {
    selected_memories = { line_b },
    predicted_topics = { "topic:b" },
}, { line_b }, 205)

local result = topic_graph.retrieve(vec_b, "topic:a", 206, {})
assert(type(result) == "table", "retrieve should return a table")
assert(#(result.predicted_topics or {}) >= 1, "retrieve should return topics")
assert(#(result.selected_memories or {}) >= 1, "retrieve should return memories")

local ok_save, err_save = store.save_to_disk()
assert(ok_save, err_save)

package.loaded["module.memory.topic_graph"] = nil
package.loaded["module.memory.store"] = nil

local store2 = require("module.memory.store")
assert(store2.load())
assert(store2.get_total_lines() == 3, "reload total mismatch")
local hits = store2.find_similar_all_fast(vec_a, 1)
assert(#hits == 1 and hits[1].index == 1, "reloaded nearest hit mismatch")

print("integration-ok")
