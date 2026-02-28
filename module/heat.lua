
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
M.pending_cold = {}                             -- 延迟冷救援队列：{ due_turn=number, line=number }
M.pending_set = {}                              -- 去重：line -> true
M.pending_dirty = false                         -- 标记 pending_cold 是否有变动

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
        table.sort(M.pending_cold, function(a, b)
            local da = tonumber(a.due_turn) or 0
            local db = tonumber(b.due_turn) or 0
            if da ~= db then return da < db end
            return (tonumber(a.line) or 0) < (tonumber(b.line) or 0)
        end)
        for _, task in ipairs(M.pending_cold) do
            local due_turn = math.max(0, math.floor(tonumber(task.due_turn) or 0))
            local line = math.max(1, math.floor(tonumber(task.line) or 0))
            local w_ok, w_err = f:write(string.format("%d,%d\n", due_turn, line))
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
    M.pending_set = {}
    local seen = {}
    for line in f:lines() do
        local due_s, line_s = tostring(line or ""):match("^%s*(%-?%d+)%s*,%s*(%-?%d+)%s*$")
        local l = nil
        local due_turn = 0
        if due_s and line_s then
            l = normalize_line(line_s)
            due_turn = math.max(0, math.floor(tonumber(due_s) or 0))
        else
            -- 兼容旧格式：单列 line
            l = normalize_line(line)
        end
        if l and not seen[l] then
            seen[l] = true
            M.pending_set[l] = true
            table.insert(M.pending_cold, { due_turn = due_turn, line = l })
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
    local total = memory.get_total_lines()
    for lineno = 1, total do
        local h = memory.get_heat_by_index(lineno)
        if h and h > 0 then
            sync_heat_to_table(lineno, h)
        end
    end
    print(string.format("[Heat] 已从 V3 memory_index 加载热区，共 %d 条热记忆", #M.heat.indices))
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

    -- 新策略：查询驱动冷救援，不在写入阶段自动挂队列
    if is_cold_cluster and not M.is_hot(target_mem_line) then
        return
    end

    -- ========== 立即执行邻居加热（目标热或簇热） ==========
    -- 查找相似记忆（簇内优先）
    local sim_results = {}
    if cid then
        sim_results = cluster.find_sim_in_cluster(vec, cid, { only_hot = true, max_results = 30 })
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

local function topic_relation_to_target(turn, target_info, sim_th)
    local topic = require("module.topic")
    local ti = topic.get_topic_for_turn and topic.get_topic_for_turn(turn) or nil
    if not target_info or not ti then
        return "cross"
    end

    if ti.is_active and target_info.is_active then
        return "same"
    end

    if (not ti.is_active) and (not target_info.is_active)
        and ti.topic_idx and target_info.topic_idx
        and ti.topic_idx == target_info.topic_idx then
        return "same"
    end

    if ti.centroid and target_info.centroid then
        local ts = tool.cosine_similarity(ti.centroid, target_info.centroid)
        if ts >= sim_th then return "near" end
    end

    return "cross"
end

function M.enqueue_cold_rescue(query_vec, current_turn, target_topic_info, min_gate)
    local memory = require("module.memory")
    local cluster = require("module.cluster")
    local adaptive = require("module.adaptive")
    local aq = ((config.settings or {}).ai_query or {})
    if not query_vec or #query_vec == 0 then return end

    local max_q = math.max(1, tonumber(aq.cold_rescue_max_queue) or 50000)
    if #M.pending_cold >= max_q then return end

    local gate = tonumber(min_gate) or tonumber(aq.min_sim_gate) or 0.58
    local topn = math.max(1, tonumber(aq.cold_rescue_topn) or 3)
    local scan_limit = math.max(topn * 6, 18)
    local sim_th = tonumber(aq.topic_sim_threshold) or 0.70
    local cur_turn = math.max(0, math.floor(tonumber(current_turn) or 0))
    local delay_min = math.max(1, math.floor(tonumber(aq.cold_rescue_delay_min) or 24))
    local delay_max = math.max(delay_min, math.floor(tonumber(aq.cold_rescue_delay_max) or 120))

    local cluster_sim_th = (((config.settings or {}).cluster or {}).cluster_sim) or 0.72
    local super_topn = (((config.settings or {}).cluster or {}).supercluster_topn_query) or 4
    local best_id, best_sim = cluster.find_best_cluster(query_vec, { super_topn = super_topn })

    local candidates = {}
    if best_id and best_sim >= cluster_sim_th then
        candidates = cluster.find_sim_in_cluster(query_vec, best_id, {
            only_cold = true,
            max_results = scan_limit,
        })
    else
        -- V3: 禁止全量冷区扫描 fallback
        adaptive.add_counter("full_scan_guard_violations", 1)
        return
    end

    local chosen = 0
    for _, item in ipairs(candidates) do
        local mem_idx = tonumber(item.index)
        local sim = tonumber(item.similarity) or 0
        if sim < gate then
            break
        end
        if mem_idx and sim >= gate and not M.pending_set[mem_idx] then
            local mem_turns = memory.get_turns(mem_idx)
            local has_target_turn = target_topic_info == nil
            if mem_turns and #mem_turns > 0 and (not has_target_turn) then
                for _, t in ipairs(mem_turns) do
                    if topic_relation_to_target(t, target_topic_info, sim_th) == "same" then
                        has_target_turn = true
                        break
                    end
                end
            end
            if has_target_turn then
                local delay = math.random(delay_min, delay_max)
                local due_turn = cur_turn + delay
                M.pending_cold[#M.pending_cold + 1] = { due_turn = due_turn, line = mem_idx }
                M.pending_set[mem_idx] = true
                M.pending_dirty = true
                adaptive.add_counter("cold_rescue_enqueued", 1)
                chosen = chosen + 1
                if chosen >= topn or #M.pending_cold >= max_q then
                    break
                end
            end
        end
    end
end

-- ====================== 冷救援执行（兼容旧函数名） ======================
function M.perform_cold_exchange(current_turn)
    local memory = require("module.memory")
    local cluster = require("module.cluster")
    local saver = require("module.saver")
    local history = require("module.history")
    local adaptive = require("module.adaptive")
    local aq = ((config.settings or {}).ai_query or {})
    if #M.pending_cold == 0 then return end

    local now_turn = tonumber(current_turn)
    if not now_turn then
        now_turn = history.get_turn()
    end
    now_turn = math.max(0, math.floor(now_turn))

    local maintenance = math.max(1, tonumber((config.settings.time or {}).maintenance_task) or 75)
    if (now_turn % maintenance) ~= 0 then
        return
    end

    local batch = math.max(1, tonumber(aq.cold_rescue_batch) or 24)
    table.sort(M.pending_cold, function(a, b)
        local da = tonumber(a.due_turn) or 0
        local db = tonumber(b.due_turn) or 0
        if da ~= db then return da < db end
        return (tonumber(a.line) or 0) < (tonumber(b.line) or 0)
    end)

    local done = 0
    local remain = {}
    for _, task in ipairs(M.pending_cold) do
        local line = normalize_line(task.line)
        local due_turn = math.max(0, math.floor(tonumber(task.due_turn) or 0))
        if (not line) then
            goto continue
        end
        if done >= batch or due_turn > now_turn then
            remain[#remain + 1] = { due_turn = due_turn, line = line }
            goto continue
        end

        if memory.get_heat_by_index(line) > 0 then
            M.pending_set[line] = nil
            goto continue
        end

        local wake_heat = math.max(NEW_HEAT, math.floor(NEW_HEAT * COLD_WAKE_MULT))
        memory.set_heat(line, wake_heat)
        sync_heat_to_table(line, wake_heat)
        cluster.mark_hot(line)

        local old_nb = config.settings.heat.neighbors_heat
        config.settings.heat.neighbors_heat = math.max(tonumber(old_nb) or 0, COLD_EXTRA_NB)
        local vec = memory.return_mem_vec(line)
        if vec then
            M.neighbors_add_heat(vec, now_turn, line)
        end
        config.settings.heat.neighbors_heat = old_nb

        done = done + 1
        M.pending_set[line] = nil
        adaptive.add_counter("cold_rescue_executed", 1)
        ::continue::
    end

    M.pending_cold = remain
    M.pending_set = {}
    for _, task in ipairs(M.pending_cold) do
        M.pending_set[task.line] = true
    end
    M.pending_dirty = true

    if done > 0 then
        print(string.format("[Cold Rescue] 本轮执行 %d 个延迟冷救援任务", done))
        cluster.update_hot_status()
        M.normalize_heat()
        saver.mark_dirty()
        saver.flush_all()
    end
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
