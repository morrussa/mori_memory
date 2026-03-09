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
    print("[WARN] AVX 库未加载，启用 LuaJIT fallback: " .. tostring(err))
    simdc = nil
else 
    print("[OK] AVX cosine 库加载成功")
end

ffi.cdef[[
    float cosine_similarity_avx(const float* v1, const float* v2, size_t n);
]]

local simdc_scratch_a = {}
local simdc_scratch_b = {}

local function get_scratch_buf(pool, n)
    local k = tonumber(n) or 0
    if k <= 0 then return nil end
    local buf = pool[k]
    if not buf then
        buf = ffi.new("float[?]", k)
        pool[k] = buf
    end
    return buf
end

local function utf8_sanitize_lossy(s)
    s = tostring(s or "")
    local n = #s
    if n <= 0 then return s end

    local out = {}
    local i = 1
    while i <= n do
        local b1 = s:byte(i)
        if not b1 then break end

        if b1 < 0x80 then
            out[#out + 1] = string.char(b1)
            i = i + 1
        elseif b1 >= 0xC2 and b1 <= 0xDF then
            local b2 = s:byte(i + 1)
            if b2 and b2 >= 0x80 and b2 <= 0xBF then
                out[#out + 1] = s:sub(i, i + 1)
                i = i + 2
            else
                out[#out + 1] = "?"
                i = i + 1
            end
        elseif b1 >= 0xE0 and b1 <= 0xEF then
            local b2 = s:byte(i + 1)
            local b3 = s:byte(i + 2)
            local ok = false
            if b2 and b3 and b2 >= 0x80 and b2 <= 0xBF and b3 >= 0x80 and b3 <= 0xBF then
                if b1 == 0xE0 then
                    ok = (b2 >= 0xA0 and b2 <= 0xBF)
                elseif b1 == 0xED then
                    ok = (b2 >= 0x80 and b2 <= 0x9F)
                else
                    ok = true
                end
            end
            if ok then
                out[#out + 1] = s:sub(i, i + 2)
                i = i + 3
            else
                out[#out + 1] = "?"
                i = i + 1
            end
        elseif b1 >= 0xF0 and b1 <= 0xF4 then
            local b2 = s:byte(i + 1)
            local b3 = s:byte(i + 2)
            local b4 = s:byte(i + 3)
            local ok = false
            if b2 and b3 and b4 and b2 >= 0x80 and b2 <= 0xBF and b3 >= 0x80 and b3 <= 0xBF and b4 >= 0x80 and b4 <= 0xBF then
                if b1 == 0xF0 then
                    ok = (b2 >= 0x90 and b2 <= 0xBF)
                elseif b1 == 0xF4 then
                    ok = (b2 >= 0x80 and b2 <= 0x8F)
                else
                    ok = true
                end
            end
            if ok then
                out[#out + 1] = s:sub(i, i + 3)
                i = i + 4
            else
                out[#out + 1] = "?"
                i = i + 1
            end
        else
            out[#out + 1] = "?"
            i = i + 1
        end
    end
    return table.concat(out)
end

function M.utf8_sanitize_lossy(s)
    return utf8_sanitize_lossy(s)
end

-- ==================== 核心转换函数 ====================

function M.get_embedding(text, mode)
    mode = mode or "query" -- "query" | "passage"
    text = utf8_sanitize_lossy(text)
    local py_vec = py_pipeline:get_embedding(text, mode)
    return M._force_to_table(py_vec)
end

function M.get_embedding_passage(text)
    return M.get_embedding(text, "passage")
end

function M.get_embedding_query(text)
    return M.get_embedding(text, "query")
end

function M._force_nested_to_table(py_obj)
    local out = {}
    if py_obj == nil then return out end

    if type(py_obj) == "table" and #py_obj > 0 and type(py_obj[1]) == "table" then
        for i = 1, #py_obj do
            out[#out + 1] = M._force_to_table(py_obj[i])
        end
        return out
    end

    local has_zero = false
    local ok0, v0 = pcall(function() return py_obj[0] end)
    if ok0 and v0 ~= nil then
        has_zero = true
    end
    if has_zero then
        local i = 0
        while true do
            local ok_idx, item = pcall(function() return py_obj[i] end)
            if not ok_idx or item == nil then break end
            out[#out + 1] = M._force_to_table(item)
            i = i + 1
        end
        return out
    end

    local ok_len1, len1 = pcall(function() return #py_obj end)
    if ok_len1 and tonumber(len1) and len1 > 0 then
        for i = 1, len1 do
            out[#out + 1] = M._force_to_table(py_obj[i])
        end
    end
    return out
end

function M.get_embeddings(texts, mode)
    mode = mode or "query"
    local py_vecs = py_pipeline:get_embeddings(texts, mode)
    return M._force_nested_to_table(py_vecs)
end

function M.get_embeddings_passage(texts)
    return M.get_embeddings(texts, "passage")
end

function M.get_embeddings_query(texts)
    return M.get_embeddings(texts, "query")
end

function M._force_to_table(py_obj)
    local t = {}
    if py_obj == nil then return t end

    local has_zero = false
    local ok0, v0 = pcall(function() return py_obj[0] end)
    if ok0 and v0 ~= nil then
        has_zero = true
    end

    if has_zero then
        -- Python 风格 0-based 索引读取，直到取不到值
        local i = 0
        while true do
            local ok_idx, v = pcall(function() return py_obj[i] end)
            if not ok_idx or v == nil then break end
            t[#t + 1] = tonumber(v) or 0.0
            i = i + 1
        end
        return t
    end

    -- Lua 风格 1-based 连续数组读取（兼容已是 Lua table 的情况）
    local ok_len1, len1 = pcall(function() return #py_obj end)
    if ok_len1 and tonumber(len1) and len1 > 0 then
        for i = 1, len1 do
            t[i] = tonumber(py_obj[i]) or 0.0
        end
    end
    return t
end

function M.py_results_to_lua(obj)
    if type(obj) == "table" then return obj end
    local results = {}
    if obj == nil then return results end

    local i = 0
    while true do
        local ok_idx, item = pcall(function() return obj[i] end)
        if not ok_idx or item == nil then break end
        table.insert(results, {
            index      = tonumber(item.index or 0),
            similarity = tonumber(item.similarity or 0.0)
        })
        i = i + 1
    end

    if #results == 0 then
        local success, n = pcall(function() return #obj end)
        if success and tonumber(n) and n > 0 then
            for j = 1, n do
                local item = obj[j]
                if item then
                    table.insert(results, {
                        index      = tonumber(item.index or 0),
                        similarity = tonumber(item.similarity or 0.0)
                    })
                end
            end
        end
    end
    return results
end

function M.to_ptr_vec(vec)
    if type(vec) == "table" and vec.__ptr then
        return vec
    end
    if type(vec) ~= "table" then
        return nil
    end
    local n = #vec
    if n <= 0 then
        return nil
    end
    local buf = ffi.new("float[?]", n)
    for i = 1, n do
        buf[i - 1] = tonumber(vec[i]) or 0.0
    end
    return { __ptr = buf, __dim = n }
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
    str = tostring(str or "")
    local marker = "</think>"
    local start = str:find(marker, 1, true)
    local result = str
    if start then
        result = str:sub(start + #marker)
    end
    result = result:gsub("\n\n", "")
    return result
end

-- 从任意文本中提取第一段平衡的 Lua table（%b{} 的增强版）
-- 规则：忽略引号内的大括号，返回 table 文本及其起止位置
function M.extract_first_lua_table(raw)
    local text = tostring(raw or "")
    local n = #text
    local i = 1
    local depth = 0
    local start_pos = nil
    local quote = nil

    while i <= n do
        local ch = text:sub(i, i)
        if quote then
            if ch == "\\" then
                i = i + 2
            elseif ch == quote then
                quote = nil
                i = i + 1
            else
                i = i + 1
            end
        else
            if ch == "'" or ch == '"' then
                quote = ch
                i = i + 1
            elseif ch == "{" then
                if depth == 0 then
                    start_pos = i
                end
                depth = depth + 1
                i = i + 1
            elseif ch == "}" and depth > 0 then
                depth = depth - 1
                if depth == 0 and start_pos then
                    local end_pos = i
                    return text:sub(start_pos, end_pos), start_pos, end_pos
                end
                i = i + 1
            else
                i = i + 1
            end
        end
    end

    return nil, nil, nil
end

-- 严格解析 Lua 字符串数组：{"a","b"} 或 {'a','b'}
-- 不执行代码；格式不合法直接返回 nil, reason
function M.parse_lua_string_array_strict(raw, opts)
    opts = opts or {}
    local max_items = tonumber(opts.max_items) or 32
    local max_item_chars = tonumber(opts.max_item_chars) or 200
    local must_full = opts.must_full ~= false
    local extract_first_on_fail = opts.extract_first_on_fail == true

    local function is_space(ch)
        return ch == " " or ch == "\t" or ch == "\r" or ch == "\n"
    end

    local function trim(s)
        return (tostring(s or ""):gsub("^%s*(.-)%s*$", "%1"))
    end

    local function parse_quoted(inner, pos)
        local q = inner:sub(pos, pos)
        if q ~= '"' and q ~= "'" then
            return nil, pos, "expect_quote"
        end
        pos = pos + 1
        local out = {}
        while pos <= #inner do
            local ch = inner:sub(pos, pos)
            if ch == "\\" then
                return nil, pos, "escape_not_allowed"
            end
            if ch == q then
                return table.concat(out), pos + 1, nil
            end
            if ch == "\n" or ch == "\r" then
                return nil, pos, "newline_not_allowed"
            end
            out[#out + 1] = ch
            pos = pos + 1
        end
        return nil, pos, "missing_quote_end"
    end

    local text = trim(raw)
    local candidate = text
    if must_full then
        if not candidate:match("^%b{}$") then
            if extract_first_on_fail then
                local first = M.extract_first_lua_table(text)
                if first then
                    candidate = first
                else
                    return nil, "not_full_lua_table"
                end
            else
                return nil, "not_full_lua_table"
            end
        end
    else
        candidate = M.extract_first_lua_table(text)
        if not candidate then
            return nil, "missing_lua_table"
        end
    end

    local inner = candidate:sub(2, -2)
    local pos = 1
    local n = #inner
    local out = {}

    while true do
        while pos <= n and is_space(inner:sub(pos, pos)) do
            pos = pos + 1
        end
        if pos > n then break end

        local value, next_pos, err = parse_quoted(inner, pos)
        if not value then
            return nil, err or "invalid_item"
        end
        value = trim(value)
        if value ~= "" then
            if #value > max_item_chars then
                value = value:sub(1, max_item_chars)
            end
            out[#out + 1] = value
            if #out >= max_items then
                return out, nil
            end
        end
        pos = next_pos

        while pos <= n and is_space(inner:sub(pos, pos)) do
            pos = pos + 1
        end
        if pos > n then break end
        if inner:sub(pos, pos) ~= "," then
            return nil, "missing_comma"
        end
        pos = pos + 1
    end

    return out, nil
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

local function cosine_table_table(vec1, vec2, n)
    local dot, norm1_sq, norm2_sq = 0.0, 0.0, 0.0
    for i = 1, n do
        local a = vec1[i] or 0.0
        local b = vec2[i] or 0.0
        dot = dot + a * b
        norm1_sq = norm1_sq + a * a
        norm2_sq = norm2_sq + b * b
    end
    if norm1_sq <= 0 or norm2_sq <= 0 then return 0 end
    return dot / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq))
end

local function cosine_ptr_table(ptr, vec, n)
    local dot, norm1_sq, norm2_sq = 0.0, 0.0, 0.0
    for i = 0, n - 1 do
        local a = ptr[i]
        local b = vec[i + 1] or 0.0
        dot = dot + a * b
        norm1_sq = norm1_sq + a * a
        norm2_sq = norm2_sq + b * b
    end
    if norm1_sq <= 0 or norm2_sq <= 0 then return 0 end
    return dot / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq))
end

local function cosine_ptr_ptr(ptr1, ptr2, n)
    local dot, norm1_sq, norm2_sq = 0.0, 0.0, 0.0
    for i = 0, n - 1 do
        local a = ptr1[i]
        local b = ptr2[i]
        dot = dot + a * b
        norm1_sq = norm1_sq + a * a
        norm2_sq = norm2_sq + b * b
    end
    if norm1_sq <= 0 or norm2_sq <= 0 then return 0 end
    return dot / (math.sqrt(norm1_sq) * math.sqrt(norm2_sq))
end

-- 在无 SIMD 环境下使用纯 LuaJIT 的点积/余弦 fallback
function M.cosine_similarity(vec1, vec2)
    -- 1. 检查是否是 FFI 指针 (来自 memory.iterate_all)
    if type(vec1) == "table" and vec1.__ptr then
        if type(vec2) == "table" and vec2.__ptr then
            local d1 = tonumber(vec1.__dim) or 0
            local d2 = tonumber(vec2.__dim) or 0
            local n = math.min(d1, d2)
            if n <= 0 then return 0 end
            if simdc then
                return simdc.cosine_similarity_avx(vec1.__ptr, vec2.__ptr, n)
            end
            return cosine_ptr_ptr(vec1.__ptr, vec2.__ptr, n)
        end
        if simdc then
            local n = tonumber(vec1.__dim) or 0
            if n <= 0 then return 0 end
            local p2 = get_scratch_buf(simdc_scratch_b, n)
            for i = 1, n do
                p2[i - 1] = vec2[i] or 0.0
            end
            return simdc.cosine_similarity_avx(vec1.__ptr, p2, n)
        end
        return cosine_ptr_table(vec1.__ptr, vec2, vec1.__dim)
    end
    if type(vec2) == "table" and vec2.__ptr then
        if simdc then
            local n = tonumber(vec2.__dim) or 0
            if n <= 0 then return 0 end
            local p1 = get_scratch_buf(simdc_scratch_a, n)
            for i = 1, n do
                p1[i - 1] = vec1[i] or 0.0
            end
            return simdc.cosine_similarity_avx(p1, vec2.__ptr, n)
        end
        return cosine_ptr_table(vec2.__ptr, vec1, vec2.__dim)
    end

    -- 2. 标准 Lua Table 处理
    if not vec1 or not vec2 then return 0 end
    local n = #vec1
    if n == 0 or n ~= #vec2 then return 0 end

    if simdc then
        local p1 = get_scratch_buf(simdc_scratch_a, n)
        local p2 = get_scratch_buf(simdc_scratch_b, n)
        for i = 1, n do
            p1[i-1] = vec1[i]
            p2[i-1] = vec2[i]
        end
        return simdc.cosine_similarity_avx(p1, p2, n)
    end

    return cosine_table_table(vec1, vec2, n)
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
    local memory = require("module.memory.store")
    local results = {}
    local v1 = vec

    local iter = memory.iterate_all()
    local lineno = 0

    for mem in iter do
        lineno = lineno + 1
        if mem and mem.vec then
            local sim = M.cosine_similarity(v1, mem.vec)

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

function M.create_memory_record(_legacy_heat, turns, vec)
    turns = (type(turns) == "table") and turns or {}
    vec = (type(vec) == "table") and vec or {}
    local dim = #vec
    local num_turns = #turns

    local turns_size = num_turns * 4          -- uint32_t 数组
    local vector_size = dim * 4                -- float 数组（每个 4 字节）
    local record_size = 4 + 2 + turns_size + 2 + vector_size

    local buf = ffi.new("uint8_t[?]", record_size)
    local offset = 0

    ffi.cast("uint32_t*", buf + offset)[0] = record_size
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
    if not data then return nil, 0 end
    offset = tonumber(offset) or 0
    if offset < 0 then return nil, 0 end
    if offset + 4 > #data then return nil, 0 end
    if offset >= #data then return nil, 0 end
    local p = ffi.cast("const uint8_t*", data) + offset

    local record_len = ffi.cast("const uint32_t*", p)[0]
    if record_len < 12 or offset + record_len > #data then
        return nil, 0
    end

    local base = 4
    if base + 2 > record_len then return nil, 0 end

    local num_turns = ffi.cast("const uint16_t*", p + base)[0]
    base = base + 2

    local turns_bytes = num_turns * 4
    if base + turns_bytes + 2 > record_len then
        return nil, 0
    end

    local turns = {}
    for i = 0, num_turns - 1 do
        table.insert(turns, ffi.cast("const uint32_t*", p + base)[i])
    end
    base = base + turns_bytes

    local dim = ffi.cast("const uint16_t*", p + base)[0]
    base = base + 2

    local vec_bytes = dim * 4
    if base + vec_bytes ~= record_len then
        return nil, 0
    end

    local vec = {}
    for i = 0, dim - 1 do
        vec[i + 1] = ffi.cast("const float*", p + base)[i]
    end

    return { turns = turns, vec = vec }, record_len
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
    if not data then return nil, 0 end
    offset = tonumber(offset) or 0
    if offset < 0 then return nil, 0 end
    if offset + 4 > #data then return nil, 0 end
    if offset >= #data then return nil, 0 end
    local p = ffi.cast("const uint8_t*", data) + offset

    local record_len = ffi.cast("const uint32_t*", p)[0]
    if record_len < 25 or offset + record_len > #data then
        return nil, 0
    end

    local base = 4
    if base + 4 + 4 + 4 + 4 + 1 + 2 > record_len then
        return nil, 0
    end
    local id = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local heat = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local hot_count = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local cold_count = ffi.cast("const uint32_t*", p + base)[0]; base = base + 4
    local is_hot = ffi.cast("const uint8_t*", p + base)[0] == 1; base = base + 1

    local num_members = ffi.cast("const uint16_t*", p + base)[0]; base = base + 2
    local members_bytes = num_members * 4
    if base + members_bytes + 2 > record_len then
        return nil, 0
    end

    local members = {}
    for i = 0, num_members - 1 do
        table.insert(members, ffi.cast("const uint32_t*", p + base)[i])
    end
    base = base + members_bytes

    local dim = ffi.cast("const uint16_t*", p + base)[0]; base = base + 2
    local vec_bytes = dim * 4
    if base + vec_bytes ~= record_len then
        return nil, 0
    end

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
