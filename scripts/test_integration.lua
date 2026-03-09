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
    root = string.format("/tmp/mori_store_integration_%d_%d", os.time(), math.random(1000, 9999))
end
os.execute(string.format('mkdir -p "%s/shards"', root))

local config = require("module.config")
config.settings.storage_v3 = config.settings.storage_v3 or {}
config.settings.storage_v3.root = root
config.settings.storage_v3.max_memory = 128

local store = require("module.memory.store")
local ghsom = require("module.memory.ghsom")

print("=== store + GHSOM integration ===")
print("root:", root)

store.load()
ghsom.load()

local vectors = {
    norm({1, 0, 0, 0, 0, 0, 0, 0}),
    norm({0, 1, 0, 0, 0, 0, 0, 0}),
    norm({0, 0, 1, 0, 0, 0, 0, 0}),
}

for i, vec in ipairs(vectors) do
    local line = assert(store.add_memory(vec, 200 + i))
    assert(line == i, "unexpected line allocation")
end

local merged = assert(store.add_memory(norm({0.999, 0.001, 0, 0, 0, 0, 0, 0}), 300))
assert(merged == 1, "reachable duplicate should merge into line 1")
assert(store.get_total_lines() == 3, "merge should not create a new line")

local fast_hits = store.find_similar_all_fast(vectors[1], 2)
assert(#fast_hits >= 1, "expected hot fast search hits")

local node_hits = ghsom.probe_nodes(vectors[1], 2, { only_hot = true })
assert(#node_hits >= 1, "expected reachable hot nodes")

local ok1, err1 = store.save_to_disk()
assert(ok1, err1)
local ok2, err2 = ghsom.save_to_disk()
assert(ok2, err2)

store.load()
ghsom.load()
assert(store.get_total_lines() == 3, "reload total mismatch")

print("integration-ok")
