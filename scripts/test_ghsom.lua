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
    root = string.format("/tmp/mori_ghsom_test_%d_%d", os.time(), math.random(1000, 9999))
end
os.execute(string.format('mkdir -p "%s/shards"', root))

local config = require("module.config")
config.settings.storage_v3 = config.settings.storage_v3 or {}
config.settings.storage_v3.root = root
config.settings.storage_v3.max_memory = 128

local store = require("module.memory.store")
local ghsom = require("module.memory.ghsom")

print("=== GHSOM smoke test ===")
print("root:", root)

store.load()
ghsom.load()

local vec_a = norm({1, 0, 0, 0, 0, 0, 0, 0})
local vec_b = norm({0, 1, 0, 0, 0, 0, 0, 0})
local vec_c = norm({0.98, 0.02, 0, 0, 0, 0, 0, 0})

local line_a = assert(store.add_memory(vec_a, 100))
local line_b = assert(store.add_memory(vec_b, 101))
local merged = assert(store.add_memory(vec_c, 102))

assert(line_a == 1, "first line mismatch")
assert(line_b == 2, "second line mismatch")
assert(merged == line_a, "expected vec_c to merge into line_a")
assert(store.get_total_lines() == 2, "unexpected memory count after merge")

local node_id = assert(ghsom.get_node_for_line(line_a), "missing node mapping")
local path = ghsom.get_hierarchical_path(line_a)
assert(#path >= 1, "hierarchical path missing")

local hits = ghsom.find_sim_in_node(vec_a, node_id, {
    only_hot = true,
    max_results = 4,
})
assert(#hits >= 1, "expected at least one hot hit")
assert(ghsom.is_hot(line_a), "newly added line should be active")

ghsom.activate_lines({ line_a }, { mode = "replace" })
assert(ghsom.is_hot(line_a), "line_a should stay active after replace activation")
assert(not ghsom.is_hot(line_b), "replace activation should clear unrelated lines")

local active_hits = ghsom.find_sim_in_node(vec_a, node_id, {
    only_hot = true,
    max_results = 4,
})
assert(#active_hits >= 1 and active_hits[1].index == line_a, "active-only scan should prefer activated line")

local plan = ghsom.plan_probe_budget(vec_a, {
    max_nodes = 1,
    total_scan_budget = 6,
    per_node_floor = 2,
    predicted_nodes = { [node_id] = 1.0 },
})
assert(#plan == 1 and plan[1].id == node_id, "budget plan should keep the predicted node")
assert((plan[1].scan_limit or 0) >= 2, "budget plan should allocate scan budget")
assert(plan[1].prefer_active == true, "predicted node should scan active memories first")

local ok1, err1 = store.save_to_disk()
assert(ok1, err1)
local ok2, err2 = ghsom.save_to_disk()
assert(ok2, err2)

store.load()
ghsom.load()
assert(store.get_total_lines() == 2, "reload count mismatch")
assert(ghsom.get_node_for_line(line_a), "reload node mapping missing")

print("ghsom-smoke-ok")
