
local M = {}

local config = require("module.config")
local tool = require("module.tool")
local persistence = require("module.persistence")

local HEAT_CFG = config.settings.heat
local SOFTMAX_ENABLED = HEAT_CFG.softmax == true
local TARGET_HEAT = HEAT_CFG.total_heat or 10000000
local TOLERANCE = HEAT_CFG.tolerance or 500

local COLD_CFG = (config.settings.heat and config.settings.heat.cold_cluster)
    or config.settings.cold_cluster
    or {}
local COLD_NB_MULT = COLD_CFG.neighbor_multiplier or 2.2
local COLD_WAKE_MULT = COLD_CFG.wake_multiplier or 1.8
local COLD_EXTRA_NB = COLD_CFG.extra_neighbor_heat or 18000
local OLD_FLOOR = config.settings.cluster.cluster_heat_floor

print(string.format("[Heat] 归一化策略 %s（总热力上限 %d）", 
      SOFTMAX_ENABLED and "均匀扣除(O(n log n))" or "已禁用", TARGET_HEAT))
      
local NEW_HEAT = HEAT_CFG.new_memory_heat or 43000

M.heat = { indices = {}, values = {} }          -- 热区实时缓存（只存 heat > 0）
M.heat_pos = {}                                 -- line -> indices/values 的位置（O(1) 更新）
M.pending_cold = {}                             -- 冷区邻居任务缓存（仅存储记忆行号）
M.pending_dirty = false                          -- 标记 pending_cold 是否有变动

local pending_file = "memory/pending_cold.txt"   -- 持久化文件

local function normalize_line(line)
    local n = tonumber(line)
    if not n then return nil end
    n = math.floor(n)
    if n < 1 then return nil end
    local memory = require("module.memory")
    if n > memory.get_total_lines() then return nil end
    return n
end

-- ====================== pending_cold 持久化 ======================
function M.save_pending()
    if not M.pending_dirty then return true end
    local ok, err = persistence.write_atomic(pending_file, "w", function(f)
        for _, task in ipairs(M.pending_cold) do
            local w_ok, w_err = f:write(task.line .. "\n")
            if not w_ok then
                return false, w_err
            end
        end
        return true
    end)
    if not ok then
        return false, err
    end
    M.pending_dirty = false
    print(string.format("[Heat] pending_cold 已保存，共 %d 个任务", #M.pending_cold))
    return true
end

function M.load_pending()
    if not tool.file_exists(pending_file) then return end
    local f = io.open(pending_file, "r")
    if not f then return end
    M.pending_cold = {}
    local seen = {}
    for line in f:lines() do
        local l = normalize_line(line)
        if l and not seen[l] then
            seen[l] = true
            table.insert(M.pending_cold, { line = l })
        end
    end
    f:close()
    print(string.format("[Heat] pending_cold 已加载，共 %d 个任务", #M.pending_cold))
    M.pending_dirty = false
end

-- ====================== 均匀扣除归一化（O(n log n) 优化版） ======================
function M.normalize_heat()
    if not SOFTMAX_ENABLED then return end

    local current_total = M.get_total_heat()
    
    -- 容差内不处理
    if current_total <= TARGET_HEAT + TOLERANCE then return end

    local memory = require("module.memory")
    local cluster = require("module.cluster")
    
    local excess = current_total - TARGET_HEAT
    
    -- 1. 提取活跃成员并按热力升序排序（小到大）
    local active = {}
    for i, line in ipairs(M.heat.indices) do
        local h = M.heat.values[i]
        if h > 0 then
            table.insert(active, {line = line, heat = h})
        end
    end
    
    table.sort(active, function(a, b) return a.heat < b.heat end)
    
    -- 2. 单次遍历确定截断点
    local remaining_debt = excess
    local n = #active
    local deduction = 0      -- 最终每人扣除额度
    local cutoff_idx = 0     -- 截断点：索引 <= cutoff_idx 的成员将归零
    
    -- 从最穷的人开始遍历
    for i = 1, n do
        local h_i = active[i].heat
        local remaining_people = n - (i - 1)
        
        -- 假设当前所有人（包括自己）都能承担均摊的债务
        local share = remaining_debt / remaining_people
        
        -- 核心判断：
        -- 如果当前最穷的人 h_i 连平均份额都付不起 (h_i < share)，
        -- 说明他必须归零（破产），扣除他所有的热力，债务转嫁给剩下的人。
        if h_i < share then
            remaining_debt = remaining_debt - h_i
            -- 这个人归零，继续循环看下一个人
        else
            -- 如果当前最穷的人能付得起 share，说明剩下所有人都能付得起。
            -- 此时的 share 就是最终扣除额度 K。
            deduction = share
            cutoff_idx = i - 1
            break -- 找到解，退出循环
        end
    end
    
    -- 3. 应用扣除
    M.heat.indices = {}
    M.heat.values = {}
    M.heat_pos = {}
    
    for i = 1, n do
        local item = active[i]
        local new_h
        
        if i <= cutoff_idx then
            -- 这部分人归零
            new_h = 0
        else
            -- 这部分人扣除 deduction
            -- 使用 floor 确保热力向下取整，防止微小浮点误差导致不合规
            new_h = math.floor(item.heat - deduction)
            -- 理论上 new_h > 0，但防止计算误差
            if new_h < 0 then new_h = 0 end
        end
        
        if new_h > 0 then
            memory.set_heat(item.line, new_h)
            local pos = #M.heat.indices + 1
            M.heat.indices[pos] = item.line
            M.heat.values[pos] = new_h
            M.heat_pos[item.line] = pos
        else
            memory.set_heat(item.line, 0)
            cluster.mark_cold(item.line)
            -- print(string.format("[Heat Deduct] 记忆 %d 热力归零", item.line))
        end
    end
    
    -- 4. 修正舍入误差（可选的微调，确保总量尽可能精准）
    -- 因为 math.floor 可能导致最终总量略小于 TARGET_HEAT，这通常是可以接受的（总量略微缩水）。
    
    local final_total = M.get_total_heat()
    print(string.format(
        "[Heat Deduct] 归一化完成 | 原始 %d | 扣除额度 %.2f | 最终 %d | 热区条目 %d",
        current_total, deduction, final_total, #M.heat.indices
    ))
end

-- ====================== 内存同步辅助 ======================
local function sync_heat_to_table(line, new_heat)
    local i = M.heat_pos[line]
    if i then
        if new_heat > 0 then
            M.heat.values[i] = new_heat
            return
        end

        local last = #M.heat.indices
        if i ~= last then
            local last_line = M.heat.indices[last]
            local last_val = M.heat.values[last]
            M.heat.indices[i] = last_line
            M.heat.values[i] = last_val
            M.heat_pos[last_line] = i
        end
        M.heat.indices[last] = nil
        M.heat.values[last] = nil
        M.heat_pos[line] = nil
        return
    end

    if new_heat > 0 then
        local pos = #M.heat.indices + 1
        M.heat.indices[pos] = line
        M.heat.values[pos] = new_heat
        M.heat_pos[line] = pos
    end
end

function M.sync_line(line)
    local memory = require("module.memory")
    local idx = normalize_line(line)
    if not idx then return false end
    local h = memory.get_heat_by_index(idx) or 0
    sync_heat_to_table(idx, h)
    return true
end

function M.get_total_heat()
    local sum = 0
    for _, v in ipairs(M.heat.values) do sum = sum + v end
    return sum
end

function M.is_hot(idx)
    return M.heat_pos[idx] ~= nil
end

-- ====================== LOAD：从 memory.txt 重建热区 ======================
function M.load()
    local memory = require("module.memory")
    M.heat = { indices = {}, values = {} }
    M.heat_pos = {}
    local iter = memory.iterate_all()
    local lineno = 0
    for mem in iter do
        lineno = lineno + 1
        if mem and mem.heat and mem.heat > 0 then
            sync_heat_to_table(lineno, mem.heat)
        end
    end
    print(string.format("[Heat] 已从 memory.txt 加载热区，共 %d 条热记忆", #M.heat.indices))
    M.load_pending()
end

-- ====================== 邻居加热 ======================
function M.neighbors_add_heat(vec, total_turn, target_mem_line)
    local memory = require("module.memory")
    local cluster = require("module.cluster")

    target_mem_line = target_mem_line or memory.get_total_lines()
    target_mem_line = normalize_line(target_mem_line)
    if not target_mem_line then return end

    -- 获取目标记忆所在的簇
    local cid = cluster.get_cluster_id_for_line(target_mem_line)
    local is_cold_cluster = false
    if cid then
        local clu = cluster.clusters[cid]
        is_cold_cluster = clu and not clu.is_hot_cluster
    end

    -- [修复点] 只有目标冷且在冷簇中，才缓存任务
    if is_cold_cluster and not M.is_hot(target_mem_line) then
        local already_pending = false
        for _, task in ipairs(M.pending_cold) do
            if task.line == target_mem_line then
                already_pending = true
                break
            end
        end
        if not already_pending then
            table.insert(M.pending_cold, { line = target_mem_line })
            M.pending_dirty = true
            print(string.format("[Heat] 目标 %d 在冷簇 %d 且为冷记忆，邻居加热任务已缓存", target_mem_line, cid))
        end
        return
    end

    -- ========== 立即执行邻居加热（目标热或簇热） ==========
    -- 查找相似记忆（簇内优先）
    local sim_results
    if cid then
        sim_results = cluster.find_sim_in_cluster(vec, cid, { only_hot = true })
    else
        sim_results = tool.find_sim_all_heat(vec) or {}
        if #sim_results > 30 then
            for i = 31, #sim_results do table.remove(sim_results, 31) end
        end
    end

    -- 时间加权
    local weighted = {}
    local init_w = config.settings.time.time_boost or 0.2
    local decay = config.settings.time.loss_turn or 50
    for _, item in ipairs(sim_results) do
        local age = math.max(0, total_turn - item.index)
        local w = (age < decay) and init_w * (1 - age / decay) or 0
        table.insert(weighted, {index = item.index, weighted = item.similarity + w})
    end
    table.sort(weighted, function(a,b) return a.weighted > b.weighted end)

    -- 选出邻居（排除目标自己）
    local nb_indices = {}
    local max_nb = config.settings.heat.max_neighbors or 5
    local count = 0
    for _, item in ipairs(weighted) do
        if item.index ~= target_mem_line then
            table.insert(nb_indices, item.index)
            count = count + 1
            if count >= max_nb then break end
        end
    end

    if #nb_indices == 0 then return end

    local nbheat = config.settings.heat.neighbors_heat or 20000
    -- 如果簇是冷的但目标是热的（例如刚唤醒），仍可使用冷簇倍数（可选）
    if is_cold_cluster then
        nbheat = math.floor(nbheat * COLD_NB_MULT)
        print(string.format("[ColdBoost] 冷簇%d 邻居热力提升至 %d", cid, nbheat))
    end

    local per = math.floor(nbheat / #nb_indices)
    local extra = nbheat - #nb_indices * per

    for _, idx in ipairs(nb_indices) do
        local old_h = memory.get_heat_by_index(idx) or 0
        local new_h = old_h + per
        memory.set_heat(idx, new_h)
        sync_heat_to_table(idx, new_h)
    end

    if nb_indices[1] then
        local first_h = memory.get_heat_by_index(nb_indices[1]) or 0
        local new_first_h = first_h + extra
        memory.set_heat(nb_indices[1], new_first_h)
        sync_heat_to_table(nb_indices[1], new_first_h)
    end

    M.normalize_heat()
end

-- ====================== 冷热交换 ======================
function M.perform_cold_exchange()
    local memory = require("module.memory")
    local cluster = require("module.cluster")
    local saver = require("module.saver")
    local history = require("module.history")
    if #M.pending_cold == 0 then return end

    print(string.format("[Cold Awakening] 执行激进冷热交换，%d 个任务", #M.pending_cold))

    for _, task in ipairs(M.pending_cold) do
        local line = normalize_line(task.line)
        if not line then
            print(string.format("[Cold Awakening] 跳过非法任务行号: %s", tostring(task.line)))
            goto continue
        end

        local cid = cluster.get_cluster_id_for_line(line)
        local clu = cid and cluster.clusters[cid]
        local is_still_cold = clu and not clu.is_hot_cluster

        local wake_heat = NEW_HEAT / 2
        if is_still_cold then
            wake_heat = math.floor(NEW_HEAT * COLD_WAKE_MULT)
            print(string.format("[ColdBoost] 冷簇%d 记忆 %d 激进唤醒 (热力=%d)", cid, line, wake_heat))
        end

        local old_heat = memory.get_heat_by_index(line) or 0
        memory.set_heat(line, wake_heat)
        sync_heat_to_table(line, wake_heat)
        if old_heat <= 0 then
            cluster.mark_hot(line)
        end

        if is_still_cold then
            local vec = memory.return_mem_vec(line)
            if vec then
                local old_nb = config.settings.heat.neighbors_heat
                config.settings.heat.neighbors_heat = COLD_EXTRA_NB
                M.neighbors_add_heat(vec, history.get_turn() + 1, line)
                config.settings.heat.neighbors_heat = old_nb
            end
        end
        ::continue::
    end

    M.pending_cold = {}
    M.pending_dirty = true
    cluster.update_hot_status()
    M.normalize_heat()
    saver.flush_all()
end

-- ====================== 新记忆加入热区 ======================
function M.add_new_heat(line, heat_value)
    local memory = require("module.memory")
    heat_value = heat_value or NEW_HEAT
    memory.set_heat(line, heat_value)
    sync_heat_to_table(line, heat_value)
    print(string.format("[Heat] 新记忆 %d 加入热区 (热力=%d)", line, heat_value))
    M.normalize_heat()
end

-- ====================== 零 Floor 最终版 ======================
function M.add_new_with_cluster_cap(new_line, vec)
    local cluster = require("module.cluster")
    local memory = require("module.memory")
    local cfg = config.settings.cluster
    local CAP = cfg.cluster_heat_cap or 520000

    local cid = cluster.get_cluster_id_for_line(new_line)
    if not cid then
        M.add_new_heat(new_line, NEW_HEAT)
        return
    end

    local clu = cluster.clusters[cid]
    local members = clu.members or {}
    local num_old = #members - 1

    -- 首条记忆不做簇内压缩分配，直接走统一的新记忆热力。
    if num_old <= 0 then
        M.add_new_heat(new_line, NEW_HEAT)
        print(string.format("[ZeroFloorCap] 簇%d 首条记忆行%d 使用基准热力 %d", cid, new_line, NEW_HEAT))
        return
    end

    local base_share = 0.46
    local decay = 0.0092
    local new_share = base_share * math.exp(-decay * num_old)
    new_share = math.max(0.085, math.min(0.46, new_share))

    local target_new = math.floor(CAP * new_share)
    target_new = math.min(target_new, NEW_HEAT)

    local old_total = 0
    for _, line in ipairs(members) do
        if line ~= new_line then
            local h = memory.get_heat_by_index(line)
            if h > 0 then
                local compressed = math.floor((CAP / 6.2) * math.tanh(h / (CAP / 6.2)))
                old_total = old_total + compressed
            end
        end
    end

    local total_after = old_total + target_new
    local final_scale = (total_after > CAP) and (CAP / total_after) or 1.0

    for _, line in ipairs(members) do
        if line ~= new_line then
            local h = memory.get_heat_by_index(line)
            if h > 0 then
                local compressed = math.floor((CAP / 6.2) * math.tanh(h / (CAP / 6.2)))
                local final_h = math.floor(compressed * final_scale)
                memory.set_heat(line, final_h)
                sync_heat_to_table(line, final_h)
            end
        end
    end

    local final_new = math.floor(target_new * final_scale)
    memory.set_heat(new_line, final_new)
    sync_heat_to_table(new_line, final_new)

    print(string.format("[ZeroFloorCap] 簇%d | 新记忆行%d 获得 %d", cid, new_line, final_new))
    M.normalize_heat()
end

return M
