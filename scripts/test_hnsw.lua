#!/usr/bin/env luajit

package.path = package.path .. ";./?.lua;./module/?.lua"

local hnsw = require("module.hnsw")

local function assert_ok(ok, err)
    assert(ok, err)
    return ok
end

local index, err = hnsw.new({
    dim = 4,
    max_elements = 32,
    ef_search = 32,
})
assert(index, err)

assert_ok(index:add(201, {1.0, 0.0, 0.0, 0.0}))
assert_ok(index:add(202, {0.0, 1.0, 0.0, 0.0}))
assert_ok(index:add(203, {0.0, 0.0, 1.0, 0.0}))

local hits = index:search({0.995, 0.005, 0.0, 0.0}, 2)
assert(#hits >= 1, "expected search hits")
assert(hits[1].label == 201, "best hit should be label 201")

local tmp_path = string.format("/tmp/mori_hnsw_lua_%d.bin", os.time())
assert_ok(index:save(tmp_path))
index:close()

local restored, load_err = hnsw.load(tmp_path, {
    dim = 4,
    ef_search = 32,
})
assert(restored, load_err)

local hits2 = restored:search({0.99, 0.01, 0.0, 0.0}, 2)
assert(#hits2 >= 1, "expected reload search hits")
assert(hits2[1].label == 201, "reloaded best hit should be label 201")

local ip_index, ip_err = hnsw.new({
    dim = 4,
    max_elements = 16,
    space = hnsw.SPACE_INNER_PRODUCT,
    ef_search = 16,
})
assert(ip_index, ip_err)

assert_ok(ip_index:add(301, {2.0, 0.0, 0.0, 0.0}))
assert_ok(ip_index:add(302, {1.0, 0.0, 0.0, 0.0}))

local ip_hits = ip_index:search({3.0, 0.0, 0.0, 0.0}, 2)
assert(#ip_hits >= 1, "expected ip search hits")
assert(ip_hits[1].label == 301, "ip best hit should be label 301")
assert(math.abs(ip_hits[1].similarity - 6.0) < 1e-5, "ip similarity should equal dot product")

local ip_tmp_path = string.format("/tmp/mori_hnsw_lua_ip_%d.bin", os.time())
assert_ok(ip_index:save(ip_tmp_path))
ip_index:close()

local ip_restored, ip_load_err = hnsw.load(ip_tmp_path, {
    dim = 4,
    space = hnsw.SPACE_INNER_PRODUCT,
    ef_search = 16,
})
assert(ip_restored, ip_load_err)

local ip_hits2 = ip_restored:search({3.0, 0.0, 0.0, 0.0}, 2)
assert(#ip_hits2 >= 1, "expected reloaded ip search hits")
assert(ip_hits2[1].label == 301, "reloaded ip best hit should be label 301")
assert(math.abs(ip_hits2[1].similarity - 6.0) < 1e-5, "reloaded ip similarity should equal dot product")
ip_restored:close()

print("lua-hnsw-ok")
