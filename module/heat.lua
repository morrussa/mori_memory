local M = {}

local config = require("module.config")
local tool = require("module.tool")

local HEAT_CFG = config.settings.heat
local TOTAL_HEAT = HEAT_CFG.total_heat          -- 10M
local POOL_SIZE = HEAT_CFG.heat_pool_ratio      -- 1M
local NEW_HEAT = HEAT_CFG.new_memory_heat

M.heat = { indices = {}, values = {} }          -- 热区实时缓存（只存 heat > 0）
M.pending_cold = {}                             -- 冷区邻居任务缓存

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

-- ====================== LOAD：从 memory.txt 重建热区（唯一权威） ======================
function M.load()
    local memory = require("module.memory")   -- 延迟加载
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
    print(string.format("[Heat] 已从 memory.txt 加载热区，共 %d 条热记忆（唯一权威）", #M.heat.indices))
end

-- ====================== 热力回收（全局均减） ======================
function M.perform_recovery()
    local memory = require("module.memory")
    local cluster = require("module.cluster")
    local total = M.get_total_heat()
    if total <= POOL_SIZE then return false end

    print(string.format("[Heat Recovery] 热区满 %d > %d，开始全局均减...", total, POOL_SIZE))

    local num_hot = #M.heat.values
    if num_hot == 0 then return false end

    local excess = total - POOL_SIZE
    local delta = math.ceil(excess / num_hot)

    local new_idx, new_val = {}, {}
    local to_cold = 0

    for i = 1, num_hot do
        local new_h = math.max(0, M.heat.values[i] - delta)
        local line = M.heat.indices[i]

        memory.set_heat(line, new_h)          -- 只写 memory.txt
        sync_heat_to_table(line, new_h)

        if new_h == 0 then
            cluster.mark_cold(line)
            to_cold = to_cold + 1
            print(string.format("[Heat] 记忆 %d 热力归零 → 冷区", line))
        end
    end

    cluster.update_hot_status()
    print(string.format("[Heat Recovery] 完成，%d 条进入冷区", to_cold))
    return true
end

-- ====================== 邻居加热（核心修复：不再自加热） ======================
function M.neighbors_add_heat(vec, total_turn, target_mem_line)
    local memory = require("module.memory")
    local cluster = require("module.cluster")
    -- vec 已经是 Lua table，无需转换
    target_mem_line = target_mem_line or memory.get_total_lines()

    -- 冷区缓存
    if not M.is_hot(target_mem_line) then
        table.insert(M.pending_cold, {vec = vec, turn = total_turn, line = target_mem_line})
        print(string.format("[Heat] 目标 %d 在冷区，任务已缓存", target_mem_line))
    end

    -- ==================== 关键修复：区分 Lua table / Python list ====================
    local cid = cluster.get_cluster_id_for_line(target_mem_line)
    local sim_results
    if cid then
        sim_results = cluster.find_sim_in_cluster(vec, cid)          -- 已经是完美 Lua table
    else
        sim_results = tool.find_sim_all_heat(vec) or {}
        -- 只取前30个最相似的
        if #sim_results > 30 then
            for i = 31, #sim_results do
                table.remove(sim_results, 31)
            end
        end
    end
    -- =====================================================================

    -- 加权
    local weighted = {}
    local init_w = config.settings.time.time_boost or 0.2
    local decay = config.settings.time.loss_turn or 50
    for _, item in ipairs(sim_results) do
        local age = math.max(0, total_turn - item.index)
        local w = (age < decay) and init_w * (1 - age / decay) or 0
        table.insert(weighted, {index = item.index, weighted = item.similarity + w})
    end
    table.sort(weighted, function(a,b) return a.weighted > b.weighted end)

    -- 排除自身，避免自加热
    local nb_indices = {}
    local max_nb = config.settings.heat.max_neighbors or 5
    local count = 0
    for _, item in ipairs(weighted) do
        if item.index ~= target_mem_line then
            table.insert(nb_indices, item.index or 0)
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

    -- 回收检查
    if M.get_total_heat() > TOTAL_HEAT then
        M.perform_recovery()
    end
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
    cluster.update_hot_status()
    saver.flush_all()
end

-- ====================== 新记忆加入热区 ======================
function M.add_new_heat(line, heat_value)
    local memory = require("module.memory")
    heat_value = heat_value or NEW_HEAT
    memory.set_heat(line, heat_value)     -- 只写 memory.txt
    sync_heat_to_table(line, heat_value)
    print(string.format("[Heat] 新记忆 %d 加入热区 (热力=%d)", line, heat_value))

    if M.get_total_heat() > TOTAL_HEAT then
        M.perform_recovery()
    end
end

return M