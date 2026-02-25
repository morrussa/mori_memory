-- tool.lua （已移除内部类型检查，全面改为 float 计算）
-- 合并新功能：新增 average_vectors，改进 cosine_similarity/vector_to_bin/bin_to_vector，
-- 更新 Topic 记录格式以包含质心
local M = {}

local ffi = require("ffi")
local simdc
local status, err = pcall(function()
    simdc = ffi.load("./module/simdc_math.so")
end)
if not status then
    print("[ERROR] 加载 AVX 库失败: " .. tostring(err))
    os.exit(1)
else 
    print("[OK] AVX cosine 库加载成功")
end

ffi.cdef[[
    float cosine_similarity_avx(const float* v1, const float* v2, size_t n);
]]

-- ==================== 核心转换函数 ====================

--- 将任意输入（Lua table 或 Python userdata）安全转换为 Lua table 向量
--- 如果已经是 table，则直接返回；如果是 Python list，则转为 Lua table；否则返回空表。
--- 此函数应在所有从外部（Python）获得向量的入口处调用，确保后续处理均为 Lua table。
function M.get_embedding(text)
    local py_vec = py_pipeline:get_embedding(text)
    return M._force_to_table(py_vec)
end

function M._force_to_table(py_obj)
    local t = {}
    local len = python.builtins.len(py_obj)
    for i = 0, len - 1 do
        t[i + 1] = tonumber(py_obj[i])
    end
    return t
end

function M.py_results_to_lua(obj)
    if type(obj) == "table" then return obj end
    local results = {}
    local success, n = pcall(function() return python.builtins.len(obj) end)
    if success and n then
        for i = 0, n - 1 do
            local item = obj[i]
            if item then
                table.insert(results, {
                    index      = tonumber(item.index or 0),
                    similarity = tonumber(item.similarity or 0.0)
                })
            end
        end
    end
    return results
end

-- ==================== 字符串工具 ====================

function M.replace(str, from, to)
    local parts = {}
    local pos = 1
    while true do
        local start, finish = string.find(str, from, pos, true)
        if not start then break end
        table.insert(parts, string.sub(str, pos, start - 1))
        table.insert(parts, to)
        pos = finish + 1
    end
    table.insert(parts, string.sub(str, pos))
    return table.concat(parts)
end

function M.remove_cot(str)
    local marker = "final<|message|>"
    local start = str:find(marker, 1, true)
    local result = ""
    if start then
        result = str:sub(start + #marker)
    end
    result = result:gsub("\n\n", "")
    return result
end

-- ==================== 向量数学工具 (新增) ====================

--- 计算向量的平均值（质心）
function M.average_vectors(vectors)
    if not vectors or #vectors == 0 then return nil end
    local dim = #vectors[1]
    local sum = {}
    for i = 1, dim do sum[i] = 0.0 end
    
    for _, vec in ipairs(vectors) do
        for i = 1, dim do
            sum[i] = sum[i] + (vec[i] or 0.0)
        end
    end
    
    local avg = {}
    local count = #vectors
    for i = 1, dim do
        avg[i] = sum[i] / count
    end
    return avg
end

-- ==================== 向量序列化（字符串形式，保留 double 精度，不影响计算） ====================

function M.vector_to_str(vec)
    if not vec then
        print("[ERROR] vector_to_str: received nil!")
        return "0.0"
    end
    local parts = {}
    for i = 1, #vec do
        table.insert(parts, string.format("%.10f", vec[i]))
    end
    return table.concat(parts, ",")
end

function M.str_to_vector(str)
    if not str or str == "" then return {} end
    local vec = {}
    for num in string.gmatch(str, "[^,]+") do
        table.insert(vec, tonumber(num) or 0.0)
    end
    return vec
end

-- ==================== 基础相似度计算（使用 AVX2 float 加速） ====================

function M.cosine_similarity(vec1, vec2)
    if not vec1 or not vec2 then return 0 end
    local n = #vec1
    if n == 0 or n ~= #vec2 then return 0 end
    -- 创建 float 数组传递给 C
    local p1 = ffi.new("float[?]", n)
    local p2 = ffi.new("float[?]", n)
    for i = 1, n do
        p1[i-1] = vec1[i]
        p2[i-1] = vec2[i]
    end
    return simdc.cosine_similarity_avx(p1, p2, n)
end

-- ==================== 二进制编解码（改为 float，每个元素 4 字节） ====================

function M.vector_to_bin(vec)
    if not vec then return "" end
    local n = #vec
    local elem_size = ffi.sizeof("float")
    local buf_size = 4 + n * elem_size
    local buf = ffi.new("uint8_t[?]", buf_size)

    ffi.cast("uint32_t*", buf)[0] = n

    local flt_arr = ffi.cast("float*", buf + 4)
    for i = 1, n do
        flt_arr[i - 1] = vec[i]
    end

    return ffi.string(buf, buf_size)
end

function M.bin_to_vector(bin, offset)
    offset = offset or 0
    if not bin or #bin < offset + 4 then return nil, 0 end
    
    local p = ffi.cast("const uint8_t*", bin) + offset
    local n = ffi.cast("const uint32_t*", p)[0]
    
    local elem_size = ffi.sizeof("float")
    local expected_len = 4 + n * elem_size
    
    if #bin < offset + expected_len then return nil, 0 end

    local flt_arr = ffi.cast("const float*", p + 4)
    local vec = {}
    for i = 0, n - 1 do
        vec[i + 1] = flt_arr[i]
    end
    return vec, expected_len
end

-- ==================== 全量热记忆搜索（旧版，仍保留 Lua 循环，使用 double） ====================

function M.find_sim_all_heat(vec)
    local memory = require("module.memory")
    local results = {}
    local v1 = vec

    local iter = memory.iterate_all()
    local lineno = 0

    for mem in iter do
        lineno = lineno + 1
        if mem and mem.vec then
            local v2 = mem.vec
            local dot, norm1_sq, norm2_sq = 0, 0, 0
            for j = 1, #v1 do
                local a = v1[j]
                local b = v2[j]
                dot = dot + a * b
                norm1_sq = norm1_sq + a * a
                norm2_sq = norm2_sq + b * b
            end

            local sim = 0
            if norm1_sq > 0 and norm2_sq > 0 then
                sim = dot / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq))
            end

            table.insert(results, {
                index = lineno,
                similarity = sim
            })
        end
    end

    table.sort(results, function(a, b) return a.similarity > b.similarity end)
    return results
end

-- ==================== 文件工具 ====================

function M.file_exists(name)
    local f = io.open(name, "r")
    if f then f:close(); return true end
    return false
end

-- ==================== 二进制记录构造/解析（全面改为 float 存储向量） ====================

function M.create_memory_record(heat, turns, vec)
    local dim = #vec
    local num_turns = #(turns or 0)

    local turns_size = num_turns * 4          -- uint32_t 数组
    local vector_size = dim * 4                -- float 数组（每个 4 字节）
    local record_size = 4 + 4 + 2 + turns_size + 2 + vector_size

    local buf = ffi.new("uint8_t[?]", record_size)
    local offset = 0

    ffi.cast("uint32_t*", buf + offset)[0] = record_size
    offset = offset + 4

    ffi.cast("uint32_t*", buf + offset)[0] = math.floor(heat or 0)
    offset = offset + 4

    ffi.cast("uint16_t*", buf + offset)[0] = num_turns
    offset = offset + 2

    if num_turns > 0 then
        local tptr = ffi.cast("uint32_t*", buf + offset)
        for i = 1, num_turns do
            tptr[i - 1] = turns[i]
        end
        offset = offset + turns_size
    end

    ffi.cast("uint16_t*", buf + offset)[0] = dim
    offset = offset + 2

    local vptr = ffi.cast("float*", buf + offset)
    for i = 1, dim do
        vptr[i - 1] = vec[i] or 0.0
    end

    return ffi.string(buf, record_size)
end

function M.parse_memory_record(data, offset)
    if not data or offset >= #data then return nil, 0 end
    local p = ffi.cast("const uint8_t*", data) + offset

    local record_len = ffi.cast("const uint32_t*", p)[0]
    if record_len < 12 or offset + record_len > #data then
        return nil, 0
    end

    local base = 4
    local heat = ffi.cast("const uint32_t*", p + base)[0]
    base = base + 4

    local num_turns = ffi.cast("const uint16_t*", p + base)[0]
    base = base + 2

    local turns = {}
    for i = 0, num_turns - 1 do
        table.insert(turns, ffi.cast("const uint32_t*", p + base)[i])
    end
    base = base + num_turns * 4

    local dim = ffi.cast("const uint16_t*", p + base)[0]
    base = base + 2

    local vec = {}
    for i = 0, dim - 1 do
        vec[i + 1] = ffi.cast("const float*", p + base)[i]
    end

    return { heat = heat, turns = turns, vec = vec }, record_len
end

function M.create_cluster_record(id, centroid, members, heat, hot_count, cold_count, is_hot)
    local dim = #centroid
    local num_members = #(members or {})

    local members_size = num_members * 4      -- uint32_t 数组
    local vector_size = dim * 4                -- float 数组
    local record_size = 4 + 4 + 4 + 4 + 4 + 1 + 2 + members_size + 2 + vector_size

    local buf = ffi.new("uint8_t[?]", record_size)
    local offset = 0

    ffi.cast("uint32_t*", buf + offset)[0] = record_size
    offset = offset + 4

    ffi.cast("uint32_t*", buf + offset)[0] = id
    offset = offset + 4

    ffi.cast("uint32_t*", buf + offset)[0] = math.floor(heat or 0)
    offset = offset + 4

    ffi.cast("uint32_t*", buf + offset)[0] = math.floor(hot_count or 0)
    offset = offset + 4

    ffi.cast("uint32_t*", buf + offset)[0] = math.floor(cold_count or 0)
    offset = offset + 4

    ffi.cast("uint8_t*", buf + offset)[0] = is_hot and 1 or 0
    offset = offset + 1

    ffi.cast("uint16_t*", buf + offset)[0] = num_members
    offset = offset + 2

    if num_members > 0 then
        local mptr = ffi.cast("uint32_t*", buf + offset)
        for i = 1, num_members do
            mptr[i - 1] = members[i]
        end
        offset = offset + members_size
    end

    ffi.cast("uint16_t*", buf + offset)[0] = dim
    offset = offset + 2

    local vptr = ffi.cast("float*", buf + offset)
    for i = 1, dim do
        vptr[i - 1] = centroid[i] or 0.0
    end

    return ffi.string(buf, record_size)
end

function M.parse_cluster_record(data, offset)
    if not data or offset >= #data then return nil, 0 end
    local p = ffi.cast("const uint8_t*", data) + offset

    local record_len = ffi.cast("const uint32_t*", p)[0]
    if record_len < 20 or offset + record_len > #data then
        return nil, 0
    end

    local base = 4
    local id = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local heat = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local hot_count = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local cold_count = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local is_hot = ffi.cast("const uint8_t*", p + base)[0] == 1; base = base + 1

    local num_members = ffi.cast("const uint16_t*", p + base)[0]; base = base + 2
    local members = {}
    for i = 0, num_members - 1 do
        table.insert(members, ffi.cast("const uint32_t*", p + base)[i])
    end
    base = base + num_members * 4

    local dim = ffi.cast("const uint16_t*", p + base)[0]; base = base + 2
    local centroid = {}
    for i = 0, dim - 1 do
        centroid[i + 1] = ffi.cast("const float*", p + base)[i]
    end

    return {
        id = id,
        centroid = centroid,
        members = members,
        heat = heat,
        hot_count = hot_count,
        cold_count = cold_count,
        is_hot_cluster = is_hot
    }, record_len
end

-- ==================== Topic 记录构造/解析 (新增 Centroid 字段) ====================

function M.create_topic_record(start, end_, summary, centroid)
    summary = summary or ""
    local summary_bytes = tostring(summary)
    local summary_len = #summary_bytes
    local end_val = end_ or 0xFFFFFFFF
    
    local vec_bin = ""
    if centroid then
        vec_bin = M.vector_to_bin(centroid)
    end
    
    -- record_size = size(4) + start(4) + end(4) + sum_len(2) + summary + vec_bin
    local record_size = 4 + 4 + 4 + 2 + summary_len + #vec_bin

    local buf = ffi.new("uint8_t[?]", record_size)
    local offset = 0

    ffi.cast("uint32_t*", buf + offset)[0] = record_size; offset = offset + 4
    ffi.cast("uint32_t*", buf + offset)[0] = start;       offset = offset + 4
    ffi.cast("uint32_t*", buf + offset)[0] = end_val;     offset = offset + 4
    ffi.cast("uint16_t*", buf + offset)[0] = summary_len; offset = offset + 2

    if summary_len > 0 then
        ffi.copy(buf + offset, summary_bytes, summary_len)
        offset = offset + summary_len
    end
    
    if #vec_bin > 0 then
        ffi.copy(buf + offset, vec_bin, #vec_bin)
    end

    return ffi.string(buf, record_size)
end

function M.parse_topic_record(data, offset)
    if not data or offset >= #data then return nil, 0 end
    local p = ffi.cast("const uint8_t*", data) + offset

    local record_len = ffi.cast("const uint32_t*", p)[0]
    if record_len < 14 or offset + record_len > #data then return nil, 0 end

    local base = 4
    local start = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local end_val = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local summary_len = ffi.cast("const uint16_t*", p + base)[0]; base = base + 2

    local summary = summary_len > 0 and ffi.string(p + base, summary_len) or ""
    base = base + summary_len
    
    -- 解析剩余部分为向量
    local centroid = nil
    if base < record_len then
        centroid = M.bin_to_vector(data, offset + base)
    end

    local end_ = (end_val == 0xFFFFFFFF) and nil or end_val

    return { start = start, end_ = end_, summary = summary, centroid = centroid }, record_len
end

return M