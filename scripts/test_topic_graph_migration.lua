#!/usr/bin/env luajit

package.path = package.path .. ";./?.lua;./module/?.lua;./module/memory/?.lua;./module/graph/?.lua"

local ffi = require("ffi")

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

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function write_legacy_index(path, rows, dim, next_line)
    local f = assert(io.open(path, "wb"))
    f:write("MID3")
    local header = ffi.new("uint32_t[4]", 3, #rows, dim, next_line)
    f:write(ffi.string(header, 16))
    for _, row in ipairs(rows) do
        local topic = tostring(row.topic_anchor or "")
        local topic_len = #topic
        local turns = row.turns or {}
        local rec_size = 4 + 4 + 4 + 2 + 2 + topic_len + 2 + (#turns * 4)
        local buf = ffi.new("uint8_t[?]", rec_size)
        local base = 0
        ffi.cast("uint32_t*", buf + base)[0] = rec_size
        base = base + 4
        ffi.cast("uint32_t*", buf + base)[0] = row.line
        base = base + 4
        ffi.cast("int32_t*", buf + base)[0] = row.cluster_id
        base = base + 4
        ffi.cast("uint16_t*", buf + base)[0] = 0
        base = base + 2
        ffi.cast("uint16_t*", buf + base)[0] = topic_len
        base = base + 2
        if topic_len > 0 then
            ffi.copy(buf + base, topic, topic_len)
            base = base + topic_len
        end
        ffi.cast("uint16_t*", buf + base)[0] = #turns
        base = base + 2
        if #turns > 0 then
            local tptr = ffi.cast("uint32_t*", buf + base)
            for i = 1, #turns do
                tptr[i - 1] = turns[i]
            end
        end
        f:write(ffi.string(buf, rec_size))
    end
    f:close()
end

local function write_legacy_shard(path, cid, line, vec)
    local f = assert(io.open(path, "wb"))
    f:write("SHD3")
    local dim = #vec
    local header = ffi.new("uint32_t[4]", 1, cid, 1, dim)
    f:write(ffi.string(header, 16))
    local buf = ffi.new("uint8_t[?]", 4 + dim * 4)
    ffi.cast("uint32_t*", buf)[0] = line
    local vptr = ffi.cast("float*", buf + 4)
    for i = 1, dim do
        vptr[i - 1] = vec[i]
    end
    f:write(ffi.string(buf, 4 + dim * 4))
    f:close()
end

local root = string.format("/tmp/mori_topic_graph_migration_%d_%d", os.time(), math.random(1000, 9999))
local legacy_root = root .. "/legacy_v3"
local new_root = root .. "/topic_graph"
local predictor_path = root .. "/topic_predictor.txt"
local adaptive_path = root .. "/adaptive_state.txt"

ensure_dir(legacy_root)
ensure_dir(legacy_root .. "/shards")

local vec_a = norm({1, 0, 0, 0})
local vec_b = norm({0, 1, 0, 0})

local manifest = assert(io.open(legacy_root .. "/manifest.txt", "w"))
manifest:write("version=V3\n")
manifest:write("dim=4\n")
manifest:write("next_line=3\n")
manifest:write("count=2\n")
manifest:write("current_turn=2\n")
manifest:close()

write_legacy_index(legacy_root .. "/memory_index.bin", {
    { line = 1, cluster_id = 101, topic_anchor = "topic:a", turns = {1} },
    { line = 2, cluster_id = 102, topic_anchor = "topic:b", turns = {2} },
}, 4, 3)
write_legacy_shard(legacy_root .. "/shards/cluster_101.bin", 101, 1, vec_a)
write_legacy_shard(legacy_root .. "/shards/cluster_102.bin", 102, 2, vec_b)

local predictor = assert(io.open(predictor_path, "w"))
predictor:write("TPR2\n")
predictor:write("TOPIC\ttopic:a\t1\t1.2000000000\n")
predictor:write("TOPIC\ttopic:b\t2\t1.3000000000\n")
predictor:write("TRANS\ttopic:a\ttopic:b\t1.5000000000\n")
predictor:write("NEXT\t1\t2\t0.7000000000\n")
predictor:close()

local adaptive = assert(io.open(adaptive_path, "w"))
adaptive:write("ADPT1\n")
adaptive:write("STAT\tmerge_deferred\t1\t0\n")
adaptive:close()

package.loaded["module.memory.history"] = {
    get_turn = function() return 2 end,
    get_by_turn = function(turn)
        local rows = {
            [1] = { user = "legacy a", ai = "reply a" },
            [2] = { user = "legacy b", ai = "reply b" },
        }
        return rows[tonumber(turn)]
    end,
    parse_entry = function(entry)
        return entry.user, entry.ai
    end,
}

package.loaded["module.memory.topic"] = {
    get_stable_anchor = function(turn)
        if tonumber(turn) and tonumber(turn) >= 2 then
            return "topic:b"
        end
        return "topic:a"
    end,
    get_topic_anchor = function(turn)
        if tonumber(turn) and tonumber(turn) >= 2 then
            return "topic:b"
        end
        return "topic:a"
    end,
}

local config = require("module.config")
config.settings.storage_v3 = { root = legacy_root }
config.settings.topic_graph = config.settings.topic_graph or {}
config.settings.topic_graph.storage = { root = new_root }
config.settings.topic_graph.legacy = {
    topic_predictor_path = predictor_path,
    adaptive_state_path = adaptive_path,
}
config.settings.topic_graph.topic_hnsw = config.settings.topic_graph.topic_hnsw or {}
config.settings.topic_graph.topic_hnsw.enabled = true

package.loaded["module.memory.topic_graph"] = nil
package.loaded["module.memory.store"] = nil

local store = require("module.memory.store")
local topic_graph = require("module.memory.topic_graph")

assert(store.load())
assert(store.get_total_lines() == 2, "migration should import both legacy memories")
assert(store.get_topic_anchor(1) == "topic:a", "legacy topic anchor A should be preserved")
assert(store.get_topic_anchor(2) == "topic:b", "legacy topic anchor B should be preserved")
assert(topic_graph.state.topic_edges["topic:a"]["topic:b"].transition > 0, "legacy transition should be imported")
assert(type(topic_graph.state.legacy.adaptive_raw) == "string", "legacy adaptive raw should be archived")

local result = topic_graph.retrieve(vec_b, "topic:a", 3, {})
local found_b = false
for _, anchor in ipairs(result.predicted_topics or {}) do
    if tostring(anchor) == "topic:b" then
        found_b = true
        break
    end
end
assert(found_b, "migrated bridge should make topic:b retrievable")

local state_file = io.open(new_root .. "/state.lua", "r")
assert(state_file ~= nil, "new topic_graph state should be persisted")
state_file:close()

print("test_topic_graph_migration: ok")
