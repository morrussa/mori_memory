-- memory.lua (混合优化版)
-- 结合了 FFI 连续内存的高性能检索和 Lua Table 的灵活性

local M = {}

local tool = require("module.tool")
local config = require("module.config")
local cluster = require("module.cluster")
local ffi = require("ffi")
local saver = require("module.saver")

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
local VECTOR_DIM = 768 -- 默认维度，加载时会自动校准
local MAX_MEMORY = 200000 -- 预分配内存上限 (约 600MB)

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
function M.find_similar_all_fast(query_vec_table)
    if not M.vec_pool or not simdc_lib then
        -- 降级：调用旧版 tool 中的 Lua 实现
        return tool.find_sim_all_heat(query_vec_table) 
    end

    local results = {}
    local total = next_line - 1
    
    -- 1. 准备查询向量 (FFI 临时空间)
    local q_ptr = ffi.new("float[?]", VECTOR_DIM)
    for i=0, VECTOR_DIM-1 do q_ptr[i] = query_vec_table[i+1] end

    -- 2. 遍历 FFI 内存池
    for i = 1, total do
        -- 直接从 FFI 热力池读取，比读取 Lua Table 快得多
        local h = M.heat_pool[i-1] -- FFI 是 0-based
        
        if h > 0 then -- 只检索热记忆
            local mem_ptr = M.vec_pool + (i - 1) * VECTOR_DIM
            
            -- C 函数计算相似度
            local score = simdc_lib.cosine_similarity_avx(q_ptr, mem_ptr, VECTOR_DIM)
            
            -- 简单的预过滤，减少 Lua Table 插入开销
            if score > 0.5 then 
                table.insert(results, {index = i, similarity = score})
            end
        end
    end

    -- 3. 排序
    table.sort(results, function(a, b) return a.similarity > b.similarity end)
    return results
end

-- ====================== LOAD：从二进制加载 ======================
function M.load()
    M.memories = {}
    next_line = 1

    local f = io.open(file_path, "rb")
    if not f then
        print("[Memory] memory.bin 不存在，初始化新记忆池")
        -- 初始化空池
        M.vec_pool = ffi.new("float[?]", MAX_MEMORY * VECTOR_DIM)
        M.heat_pool = ffi.new("uint32_t[?]", MAX_MEMORY)
        return
    end

    local data = f:read("*a")
    f:close()

    if #data < 20 or data:sub(1,4) ~= "MEMB" then
        print("[Memory] 文件头无效")
        return
    end

    -- 解析 Header
    local header_count = 0
    if #data >= 24 then
        local p = ffi.cast("const uint32_t*", ffi.cast("const uint8_t*", data) + 8)
        header_count = p[0]
    end

    -- 解析第一条记录以确定维度并初始化 Pool
    local first_mem, _ = tool.parse_memory_record(data, 20)
    if first_mem and first_mem.vec then
        VECTOR_DIM = #first_mem.vec
        M.vec_pool = ffi.new("float[?]", MAX_MEMORY * VECTOR_DIM)
        M.heat_pool = ffi.new("uint32_t[?]", MAX_MEMORY)
        print(string.format("[Memory] 初始化 FFI 池: Dim=%d, MaxCount=%d", VECTOR_DIM, MAX_MEMORY))
    else
        print("[Memory] 解析首条记录失败，无法初始化")
        return
    end

    -- 遍历解析并填充
    local offset = 20
    while offset < #data do
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
    local merge_limit = config.settings.merge_limit or 0.95
    local cluster_sim_th = config.settings.cluster.cluster_sim or 0.75

    -- 1. 优先在最佳簇内搜索相似记忆（跳桶）
    local sim_results = {}
    local best_id, best_sim = cluster.find_best_cluster(vec)
    if best_id and best_sim >= cluster_sim_th then
        -- 簇命中：获取簇内所有成员（已按相似度排序）
        local all_in_cluster = cluster.find_sim_in_cluster(vec, best_id)
        -- 过滤出热记忆（heat_pool > 0）
        for _, item in ipairs(all_in_cluster) do
            if M.heat_pool[item.index - 1] > 0 then
                table.insert(sim_results, item)
            end
        end
        -- 如果过滤后没有热记忆，则回退全热区扫描
        if #sim_results == 0 then
            sim_results = M.find_similar_all_fast(vec)
        end
    else
        -- 无簇或相似度不足：回退全热区扫描
        sim_results = M.find_similar_all_fast(vec)
    end

    -- 2. 合并逻辑（保持不变，且此时 sim_results 中的记忆均为热记忆）
    if #sim_results > 0 then
        local top = sim_results[1]
        if top and top.similarity >= merge_limit then
            local target = top.index
            if M.heat_pool[target-1] > 0 then -- 二次确认（安全起见）
                -- 更新热力 (FFI)
                M.heat_pool[target-1] = M.heat_pool[target-1] + new_heat
                -- 更新元数据
                local mem = M.memories[target]
                if mem then
                    table.insert(mem.turns, 1, turn)
                    saver.mark_dirty()
                    print(string.format("[Memory] 合并 -> 行 %d (sim=%.4f)", target, top.similarity))
                    return target
                end
            end
        end
    end

    -- 3. 新建记忆行
    local new_line = next_line
    M.memories[new_line] = { turns = {turn} }
    
    -- 存入 FFI
    copy_vec_to_pool(new_line, vec)
    M.heat_pool[new_line - 1] = new_heat
    
    next_line = next_line + 1
    saver.mark_dirty()

    -- 关联热力与簇
    -- 注意：这里需要创建一个临时 table 传给旧接口，会有微小开销，但为了兼容性值得
    local vec_table = {}
    for i=1, VECTOR_DIM do vec_table[i] = vec[i] end
    
    heat_mod.add_new_heat(new_line)
    cluster.add(vec_table, new_line)

    print(string.format("[Memory] 新建 -> 行 %d", new_line))
    return new_line
end

-- ====================== 兼容性接口 ======================

function M.set_heat(line, new_heat)
    if M.heat_pool then
        M.heat_pool[line - 1] = math.max(0, new_heat)
    end
    if M.memories[line] then M.memories[line].heat = math.max(0, new_heat) end -- 保持元数据同步(可选)
    saver.mark_dirty()
end

function M.get_total_lines() return next_line - 1 end

function M.get_heat_by_index(line)
    if M.heat_pool then return M.heat_pool[line - 1] end
    return (M.memories[line] and M.memories[line].heat) or 0
end

-- 获取向量 (返回 Lua Table 以兼容 tool.lua 的其他函数)
function M.return_mem_vec(line)
    if M.vec_pool and line < next_line then
        local vec = {}
        local offset = (line - 1) * VECTOR_DIM
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
            heat = M.heat_pool and M.heat_pool[i-1] or 0,
            turns = M.memories[i] and M.memories[i].turns,
            vec = vec_ref
        }
    end
end

-- ====================== SAVE ======================
function M.save_to_disk()
    local temp = file_path .. ".tmp"
    local f = io.open(temp, "wb")
    if not f then return end

    f:write("MEMB")
    local header = ffi.new("uint32_t[4]", 1, next_line-1, 0, 0)
    f:write(ffi.string(header, 16))

    for i = 1, next_line-1 do
        local m = M.memories[i]
        if m then
            -- 从 FFI 读取向量生成二进制
            local vec_table = M.return_mem_vec(i) 
            local bin = tool.create_memory_record(M.heat_pool[i-1], m.turns, vec_table)
            f:write(bin)
        end
    end
    f:close()

    os.remove(file_path)
    os.rename(temp, file_path)
    print(string.format("[Memory] 保存完成（%d 条）", next_line-1))
end

return M