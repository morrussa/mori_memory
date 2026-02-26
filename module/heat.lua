-- heat.lua
local M = {}

local config = require("module.config")
local tool = require("module.tool")

local HEAT_CFG = config.settings.heat
local SOFTMAX_ENABLED = HEAT_CFG.softmax == true
local TARGET_HEAT = HEAT_CFG.total_heat or 10000000
local TOLERANCE = HEAT_CFG.tolerance or 500

print(string.format("[Heat] Softmax模式 %s（总热力永远固定为 %d）", 
      SOFTMAX_ENABLED and "已启用" or "已禁用", TARGET_HEAT))
      
local NEW_HEAT = HEAT_CFG.new_memory_heat or 43000

M.heat = { indices = {}, values = {} }          -- 热区实时缓存（只存 heat > 0）
M.pending_cold = {}                             -- 冷区邻居任务缓存（仅存储记忆行号）
M.pending_dirty = false                          -- 标记 pending_cold 是否有变动

local pending_file = "memory/pending_cold.txt"   -- 持久化文件

-- ====================== pending_cold 持久化 ======================
function M.save_pending()
    if not M.pending_dirty then return end        -- 无变动则跳过
    local temp = pending_file .. ".tmp"
    local f = io.open(temp, "w")
    if not f then return end
    for _, task in ipairs(M.pending_cold) do
        f:write(task.line .. "\n")
    end
    f:close()
    os.remove(pending_file)
    os.rename(temp, pending_file)
    M.pending_dirty = false
    print(string.format("[Heat] pending_cold 已保存，共 %d 个任务", #M.pending_cold))
end

function M.load_pending()
    if not tool.file_exists(pending_file) then return end
    local f = io.open(pending_file, "r")
    if not f then return end
    M.pending_cold = {}
    for line in f:lines() do
        local l = tonumber(line)
        if l then
            table.insert(M.pending_cold, { line = l })
        end
    end
    f:close()
    print(string.format("[Heat] pending_cold 已加载，共 %d 个任务", #M.pending_cold))
    M.pending_dirty = false   -- 刚加载时视为未变动
end

-- ====================== Softmax 全局归一化（核心） ======================
function M.normalize_heat()
    if not SOFTMAX_ENABLED then return end

    local current_total = M.get_total_heat()
    if current_total < 100 then return end

    -- 如果已经很接近目标，就不折腾了（避免无谓计算）
    if math.abs(current_total - TARGET_HEAT) <= TOLERANCE then
        return
    end

    local scale = TARGET_HEAT / current_total
    local memory = require("module.memory")
    local cluster = require("module.cluster")

    local new_indices = {}
    local new_values = {}
    local final_total = 0

    -- 第一步：四舍五入缩放
    for i = 1, #M.heat.indices do
        local line = M.heat.indices[i]
        local old_h = M.heat.values[i]
        local scaled = old_h * scale
        local new_h = math.floor(scaled + 0.5)  -- 四舍五入

        if new_h > 0 then
            table.insert(new_indices, line)
            table.insert(new_values, new_h)
            memory.set_heat(line, new_h)
            final_total = final_total + new_h
        else
            -- 缩放到0或负 → 直接冷掉
            memory.set_heat(line, 0)
            cluster.mark_cold(line)
            print(string.format("[Heat] 记忆 %d 四舍五入后 ≤0，已移入冷区", line))
        end
    end

    -- 第二步：微调补差（±1 轮询方式，移除了 goto 语法）
    local diff = TARGET_HEAT - final_total
    if diff ~= 0 and #new_indices > 0 then
        local step = diff > 0 and 1 or -1
        local abs_diff = math.abs(diff)
        local applied = 0
        local i = 1
        local loop_count = 0
        local max_loop = #new_indices * 2 -- 安全防护，防止死循环

        while applied < abs_diff do
            loop_count = loop_count + 1
            if loop_count > max_loop then break end

            -- 扣减时保护最小值
            if step == -1 and new_values[i] <= 1 then
                -- 跳过，寻找下一个目标
            else
                new_values[i] = new_values[i] + step
                memory.set_heat(new_indices[i], new_values[i])
                applied = applied + 1
            end
            
            i = i + 1
            if i > #new_indices then i = 1 end
        end

        final_total = M.get_total_heat()  -- 重新计算一次
    end

    -- 更新热区列表（已经过滤掉0的了）
    M.heat.indices = new_indices
    M.heat.values  = new_values

    print(string.format(
        "[Heat Softmax] 归一化完成 | 目标 %d | 缩放后 %d | 微调后 %d | 热区条目 %d",
        TARGET_HEAT, current_total, final_total, #M.heat.indices
    ))
end

-- ====================== 内存同步辅助 ======================
local function sync_heat_to_table(line, new_heat)
    for i, idx in ipairs(M.heat.indices) do
        if idx == line then
            if new_heat > 0 then
                M.heat.values[i] = new_heat
            else
                table.remove(M.heat.indices, i)
                table.remove(M.heat.values, i)
            end
            return
        end
    end
    if new_heat > 0 then
        table.insert(M.heat.indices, line)
        table.insert(M.heat.values, new_heat)
    end
end

function M.get_total_heat()
    local sum = 0
    for _, v in ipairs(M.heat.values) do sum = sum + v end
    return sum
end

function M.is_hot(idx)
    for _, i in ipairs(M.heat.indices) do
        if i == idx then return true end
    end
    return false
end

-- ====================== LOAD：从 memory.txt 重建热区 ======================
function M.load()
    local memory = require("module.memory")
    M.heat = { indices = {}, values = {} }
    local iter = memory.iterate_all()
    local lineno = 0
    for mem in iter do
        lineno = lineno + 1
        if mem and mem.heat and mem.heat > 0 then
            table.insert(M.heat.indices, lineno)
            table.insert(M.heat.values, mem.heat)
        end
    end
    print(string.format("[Heat] 已从 memory.txt 加载热区，共 %d 条热记忆", #M.heat.indices))

    -- 加载未完成的 pending_cold 任务
    M.load_pending()
end

-- ====================== 邻居加热 ======================
function M.neighbors_add_heat(vec, total_turn, target_mem_line)
    local memory = require("module.memory")
    local cluster = require("module.cluster")
    target_mem_line = target_mem_line or memory.get_total_lines()

    -- 冷区缓存：仅存储行号
    if not M.is_hot(target_mem_line) then
        table.insert(M.pending_cold, { line = target_mem_line })
        M.pending_dirty = true
        print(string.format("[Heat] 目标 %d 在冷区，任务已缓存", target_mem_line))
    end

    -- 查找相似记忆
    local cid = cluster.get_cluster_id_for_line(target_mem_line)
    local sim_results
    if cid then
        sim_results = cluster.find_sim_in_cluster(vec, cid)
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

    -- 选取邻居（排除自身）
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

    -- 统一由 normalize_heat 处理热力平衡
    M.normalize_heat()
end

-- ====================== 冷热交换 ======================
function M.perform_cold_exchange()
    local memory = require("module.memory")
    local cluster = require("module.cluster")
    local saver = require("module.saver")
    if #M.pending_cold == 0 then return end
    print(string.format("[Cold Awakening] 执行冷热交换，%d 个任务", #M.pending_cold))

    for _, task in ipairs(M.pending_cold) do
        local cid = cluster.get_cluster_id_for_line(task.line)
        if cid and cluster.clusters and cluster.clusters[cid] and cluster.clusters[cid].is_hot_cluster then
            local new_h = NEW_HEAT / 2
            memory.set_heat(task.line, new_h)
            sync_heat_to_table(task.line, new_h)
            cluster.mark_hot(task.line)
            print(string.format("[Cold Awakening] 记忆 %d 从冷区唤醒", task.line))
        end
    end

    M.pending_cold = {}
    M.pending_dirty = true   -- 清空后需保存空文件
    cluster.update_hot_status()
    M.normalize_heat()
    saver.flush_all()        -- 触发完整保存（包括 pending）
end

-- ====================== 新记忆加入热区 ======================
function M.add_new_heat(line, heat_value)
    local memory = require("module.memory")
    heat_value = heat_value or NEW_HEAT
    memory.set_heat(line, heat_value)
    sync_heat_to_table(line, heat_value)
    print(string.format("[Heat] 新记忆 %d 加入热区 (热力=%d)", line, heat_value))
    
    -- 统一由 normalize_heat 处理热力平衡
    M.normalize_heat()
end

return M