-- memory.lua (混合优化版)
-- 结合了 FFI 连续内存的高性能检索和 Lua Table 的灵活性

local M = {}

local tool = require("module.tool")
local config = require("module.config")
local cluster = require("module.cluster")
local adaptive = require("module.adaptive")
local ffi = require("ffi")
local saver = require("module.saver")
local persistence = require("module.persistence")

-- 尝试加载 C 加速库
local simdc_lib = nil
local ok, lib = pcall(ffi.load, "./module/simdc_math.so")
if ok then
    print("[Memory] 检测到 simdc_math.so，AVX 加速已启用")
    ffi.cdef[[
        float cosine_similarity_avx(const float* v1, const float* v2, size_t n);
    ]]
    simdc_lib = lib
else
    print("[Memory] 未检测到 C 库，使用 Lua 备用计算（速度较慢）")
end

M.memories = {}        -- 元数据存储 (heat, turns)
M.vec_pool = nil       -- FFI 连续向量池
M.heat_pool = nil      -- FFI 热力池

local next_line = 1
local file_path = "memory/memory.bin"
local VECTOR_DIM = config.settings.dim
local MAX_MEMORY = 200000 -- 预分配内存上限 (约 600MB)
local FAST_SCAN_TOPK = math.max(64, (tonumber((((config.settings or {}).ai_query or {}).max_memory)) or 5) * 8)

local function normalize_line(line)
    local n = tonumber(line)
    if not n then return nil end
    n = math.floor(n)
    if n < 1 or n >= next_line then return nil end
    return n
end

local function init_empty_pools(dim)
    dim = tonumber(dim) or VECTOR_DIM
    if dim < 1 then dim = 1 end
    VECTOR_DIM = dim
    M.vec_pool = ffi.new("float[?]", MAX_MEMORY * VECTOR_DIM)
    M.heat_pool = ffi.new("uint32_t[?]", MAX_MEMORY)
end

-- ====================== 内部辅助：向量拷贝 ======================
local function copy_vec_to_pool(line_idx, vec_table)
    if not M.vec_pool then return end
    local offset = (line_idx - 1) * VECTOR_DIM
    -- 简单的循环拷贝，LuaJIT 会自动优化为高效的机器码
    for i = 0, VECTOR_DIM - 1 do
        M.vec_pool[offset + i] = vec_table[i+1]
    end
end

-- ====================== 核心检索：极速模式 ======================
function M.find_similar_all_fast(query_vec_table, max_results)
    if not M.vec_pool then
        -- 极端降级：没有向量池时，走旧版兼容路径
        return tool.find_sim_all_heat(query_vec_table) 
    end

    local use_full_sort = false
    if max_results == nil then
        max_results = FAST_SCAN_TOPK
    end
    max_results = tonumber(max_results) or FAST_SCAN_TOPK
    if max_results <= 0 then
        use_full_sort = true
    end

    local results = {}
    local topk = {}
    local total = next_line - 1

    -- 1. 预计算 query 范数（纯 LuaJIT fallback 会用到）
    local q_norm_sq = 0.0
    for i = 1, VECTOR_DIM do
        local q = query_vec_table[i] or 0.0
        q_norm_sq = q_norm_sq + q * q
    end
    if q_norm_sq <= 0 then return results end
    local q_norm = math.sqrt(q_norm_sq)

    -- SIMD 可用时准备查询向量指针
    local q_ptr = nil
    if simdc_lib then
        q_ptr = ffi.new("float[?]", VECTOR_DIM)
        for i = 0, VECTOR_DIM - 1 do
            q_ptr[i] = query_vec_table[i + 1] or 0.0
        end
    end

    -- 2. 遍历 FFI 内存池
    for i = 1, total do
        -- 直接从 FFI 热力池读取，比读取 Lua Table 快得多
        local h = M.heat_pool[i-1] -- FFI 是 0-based
        
        if h > 0 then -- 只检索热记忆
            local mem_ptr = M.vec_pool + (i - 1) * VECTOR_DIM

            local score = 0.0
            if simdc_lib then
                -- C 函数计算相似度
                score = simdc_lib.cosine_similarity_avx(q_ptr, mem_ptr, VECTOR_DIM)
            else
                -- 纯 LuaJIT fallback：循环点积+范数
                local dot = 0.0
                local m_norm_sq = 0.0
                for j = 0, VECTOR_DIM - 1 do
                    local qv = query_vec_table[j + 1] or 0.0
                    local mv = mem_ptr[j]
                    dot = dot + qv * mv
                    m_norm_sq = m_norm_sq + mv * mv
                end
                if m_norm_sq > 0 then
                    score = dot / (q_norm * math.sqrt(m_norm_sq))
                end
            end
            
            -- 简单的预过滤，减少 Lua Table 插入开销
            if score > 0.5 then 
                if use_full_sort then
                    table.insert(results, {index = i, similarity = score})
                else
                    if #topk < max_results then
                        table.insert(topk, {index = i, similarity = score})
                        local j = #topk
                        while j > 1 and topk[j].similarity < topk[j - 1].similarity do
                            topk[j], topk[j - 1] = topk[j - 1], topk[j]
                            j = j - 1
                        end
                    elseif score > topk[1].similarity then
                        topk[1] = {index = i, similarity = score}
                        local j = 1
                        while j < #topk and topk[j].similarity > topk[j + 1].similarity do
                            topk[j], topk[j + 1] = topk[j + 1], topk[j]
                            j = j + 1
                        end
                    end
                end
            end
        end
    end

    if not use_full_sort then
        for i = #topk, 1, -1 do
            results[#results + 1] = topk[i]
        end
        return results
    end

    -- 3. 全量排序（仅在显式 max_results<=0 时）
    table.sort(results, function(a, b) return a.similarity > b.similarity end)
    return results
end

-- ====================== LOAD：从二进制加载 ======================
function M.load()
    M.memories = {}
    next_line = 1
    -- 先初始化默认空池，保证任何提前 return 都不会留下 nil 池
    init_empty_pools(VECTOR_DIM)

    local f = io.open(file_path, "rb")
    if not f then
        print("[Memory] memory.bin 不存在，初始化新记忆池")
        return
    end

    local data = f:read("*a")
    f:close()

    if #data < 20 or data:sub(1,4) ~= "MEMB" then
        print("[Memory] 文件头无效，已回退到空记忆池")
        return
    end

    -- 解析 Header
    local header_count = 0
    if #data >= 24 then
        local p = ffi.cast("const uint32_t*", ffi.cast("const uint8_t*", data) + 8)
        header_count = p[0]
    end

    -- 无记录时保留默认维度的空池
    if header_count <= 0 then
        print(string.format("[Memory] 空记忆文件，使用默认池 Dim=%d", VECTOR_DIM))
        return
    end

    -- 解析第一条记录以确定维度并重新初始化 Pool
    local first_mem, _ = tool.parse_memory_record(data, 20)
    if first_mem and first_mem.vec and #first_mem.vec > 0 then
        init_empty_pools(#first_mem.vec)
        print(string.format("[Memory] 初始化 FFI 池: Dim=%d, MaxCount=%d", VECTOR_DIM, MAX_MEMORY))
    else
        print("[Memory] 解析首条记录失败，已回退到空记忆池")
        return
    end

    -- 遍历解析并填充
    local offset = 20
    while offset < #data do
        if next_line > MAX_MEMORY then
            print(string.format("[Memory][WARN] memory.bin 超过预分配上限，已截断到 %d 条", MAX_MEMORY))
            break
        end
        local mem, record_size = tool.parse_memory_record(data, offset)
        if mem then
            -- 存入 Lua Table (元数据)
            M.memories[next_line] = { turns = mem.turns }
            
            -- 存入 FFI Pool (向量与热力)
            copy_vec_to_pool(next_line, mem.vec)
            M.heat_pool[next_line - 1] = mem.heat
            
            next_line = next_line + 1
            offset = offset + record_size
        else
            break
        end
    end

    print(string.format("[Memory] 加载完成: %d 条记忆，FFI加速就绪", next_line - 1))
end

-- ====================== ADD_MEMORY（修改版，增加热记忆过滤） ======================
function M.add_memory(vec, turn)
    local heat_mod = require("module.heat")

    local new_heat = config.settings.heat.new_memory_heat
    local merge_limit = adaptive.get_merge_limit(config.settings.merge_limit or 0.95)
    local cluster_sim_th = config.settings.cluster.cluster_sim or 0.75

    -- 1. 优先在最佳簇内搜索相似记忆（跳桶）
    local sim_results = {}
    local best_id, best_sim = cluster.find_best_cluster(vec, {
        super_topn = config.settings.cluster.supercluster_topn_add,
    })
    if best_id and best_sim >= cluster_sim_th then
        -- 簇命中：获取簇内所有成员（已按相似度排序）
        sim_results = cluster.find_sim_in_cluster(vec, best_id, {
            only_hot = true,
            max_results = FAST_SCAN_TOPK,
        })
        -- 如果过滤后没有热记忆，则回退全热区扫描
        if #sim_results == 0 then
            sim_results = M.find_similar_all_fast(vec, FAST_SCAN_TOPK)
        end
    else
        -- 无簇或相似度不足：回退全热区扫描
        sim_results = M.find_similar_all_fast(vec, FAST_SCAN_TOPK)
    end

    -- 2. 合并逻辑（保持不变，且此时 sim_results 中的记忆均为热记忆）
    if #sim_results > 0 then
        local top = sim_results[1]
        if top and top.similarity >= merge_limit then
            local target = top.index
            if M.get_heat_by_index(target) > 0 then -- 二次确认（安全起见）
                local mem = M.memories[target]
                if mem and mem.turns then
                    local merged_heat = M.get_heat_by_index(target) + new_heat
                    M.set_heat(target, merged_heat)
                    heat_mod.sync_line(target)
                    heat_mod.normalize_heat()
                    table.insert(mem.turns, 1, turn)
                    saver.mark_dirty()
                    print(string.format("[Memory] 合并 -> 行 %d (sim=%.4f)", target, top.similarity))
                    return target
                end
            end
        end
    end

    -- 3. 新建记忆行
    if next_line > MAX_MEMORY then
        print(string.format("[Memory][WARN] 记忆池已满（MAX_MEMORY=%d），跳过写入", MAX_MEMORY))
        return nil, "capacity_reached"
    end

    local new_line = next_line
    M.memories[new_line] = { turns = {turn} }
    
    -- 存入 FFI（热力由 heat 模块统一下发，避免重复预写）
    copy_vec_to_pool(new_line, vec)
    
    next_line = next_line + 1
    saver.mark_dirty()

    -- 关联热力与簇
    -- 注意：这里需要创建一个临时 table 传给旧接口，会有微小开销，但为了兼容性值得
    local vec_table = {}
    for i=1, VECTOR_DIM do vec_table[i] = vec[i] end

    -- 先入簇，再按簇做热力 cap；否则 add_new_with_cluster_cap 拿不到 cluster id
    cluster.add(vec_table, new_line)
    heat_mod.add_new_with_cluster_cap(new_line, vec_table)
    if M.get_heat_by_index(new_line) <= 0 then
        cluster.mark_cold(new_line)
    end

    print(string.format("[Memory] 新建 -> 行 %d", new_line))
    return new_line
end

-- ====================== 兼容性接口 ======================

function M.set_heat(line, new_heat)
    local idx = normalize_line(line)
    if not idx then return false end

    local h = math.max(0, tonumber(new_heat) or 0)
    if M.heat_pool then
        M.heat_pool[idx - 1] = h
    elseif M.memories[idx] then
        -- 仅在无 FFI 热力池时使用 Lua 元数据兜底
        M.memories[idx].heat = h
    end
    saver.mark_dirty()
    return true
end

function M.get_total_lines() return next_line - 1 end

function M.get_heat_by_index(line)
    local idx = normalize_line(line)
    if not idx then return 0 end
    if M.heat_pool then return M.heat_pool[idx - 1] end
    return (M.memories[idx] and M.memories[idx].heat) or 0
end

-- 获取向量 (返回 Lua Table 以兼容 tool.lua 的其他函数)
function M.return_mem_vec(line)
    local idx = normalize_line(line)
    if M.vec_pool and idx then
        local vec = {}
        local offset = (idx - 1) * VECTOR_DIM
        for i=0, VECTOR_DIM-1 do
            vec[i+1] = M.vec_pool[offset + i]
        end
        return vec
    end
    return nil
end

-- 迭代器 (兼容 tool.find_sim_all_heat 的旧版兜底逻辑)
function M.iterate_all()
    local i = 0
    return function()
        i = i + 1
        if i >= next_line then return nil end
        -- 返回一个包含 FFI 引用的 table，避免大量内存拷贝
        local vec_ref
        if M.vec_pool then
            vec_ref = { __ptr = M.vec_pool + (i-1)*VECTOR_DIM, __dim = VECTOR_DIM } -- 特殊标记
        end
        return {
            heat = (M.heat_pool and M.heat_pool[i-1]) or (M.memories[i] and M.memories[i].heat) or 0,
            turns = M.memories[i] and M.memories[i].turns,
            vec = vec_ref
        }
    end
end

-- ====================== SAVE ======================
function M.save_to_disk()
    local ok, err = persistence.write_atomic(file_path, "wb", function(f)
        local function write_or_fail(chunk)
            local w_ok, w_err = f:write(chunk)
            if not w_ok then
                return false, w_err
            end
            return true
        end

        local w1_ok, w1_err = write_or_fail("MEMB")
        if not w1_ok then return false, w1_err end

        local header = ffi.new("uint32_t[4]", 1, next_line-1, 0, 0)
        local w2_ok, w2_err = write_or_fail(ffi.string(header, 16))
        if not w2_ok then return false, w2_err end

        for i = 1, next_line-1 do
            local m = M.memories[i]
            if m then
                -- 从 FFI 读取向量生成二进制
                local vec_table = M.return_mem_vec(i)
                local bin = tool.create_memory_record(M.heat_pool[i-1], m.turns, vec_table)
                local wb_ok, wb_err = write_or_fail(bin)
                if not wb_ok then return false, wb_err end
            end
        end
        return true
    end)

    if not ok then
        return false, err
    end

    print(string.format("[Memory] 保存完成（%d 条）", next_line-1))
    return true
end

return M
