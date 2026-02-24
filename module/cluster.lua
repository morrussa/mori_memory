-- cluster.lua （二进制高效版 - 已全部修复）
local M = {}

local tool = require("module.tool")
local config = require("module.config")
local ffi = require("ffi")

local file_path = "memory/clusters.bin"

M.clusters = {}          
local next_cluster_id = 1

local function deepcopy_vec(vec)
    -- vec 已经是 Lua table
    local copy = {}
    for i = 1, #vec do copy[i] = vec[i] end
    return copy
end

function M.load()
    M.clusters = {}
    next_cluster_id = 1
    local cluster_count = 0

    local f = io.open(file_path, "rb")
    if not f then
        print("[Cluster] clusters.bin 不存在 → 首次运行将自动创建")
        return
    end

    local data = f:read("*a")
    f:close()

    if #data < 20 or data:sub(1,4) ~= "CLUS" then
        print("[Cluster] 文件头无效（可能是旧格式），请删除 clusters.bin 后重启")
        return
    end

    -- 跳过 header: magic(4) + version(4) + count(4) + reserved(8)
    local offset = 20
    while offset < #data do
        local clu, record_size = tool.parse_cluster_record(data, offset)
        if clu then
            M.clusters[clu.id] = {
                centroid       = clu.centroid,
                members        = clu.members,
                heat           = clu.heat,
                hot_count      = clu.hot_count,
                cold_count     = clu.cold_count,
                is_hot_cluster = clu.is_hot_cluster
            }
            next_cluster_id = math.max(next_cluster_id, clu.id + 1)
            cluster_count = cluster_count + 1
            offset = offset + record_size
        else
            print("[Cluster] 解析在 offset", offset, "处中断")
            break
        end
    end

    print(string.format("[Cluster] 已从二进制加载 %d 个语义簇，下一个ID = %d", cluster_count, next_cluster_id))
    M.update_hot_status()
end

function M.save_to_disk()
    local temp = file_path .. ".tmp"
    local f = io.open(temp, "wb")
    if not f then return end

    -- Header: CLUS + version(4) + count(4) + reserved(8)
    f:write("CLUS")
    local count = 0
    for _ in pairs(M.clusters) do count = count + 1 end
    local header = ffi.new("uint32_t[4]", 1, count, 0, 0)
    f:write(ffi.string(header, 16))

    for id, c in pairs(M.clusters) do
        if c and c.centroid then
            local bin = tool.create_cluster_record(
                id,
                c.centroid,
                c.members or {},
                c.heat or 0,
                c.hot_count or 0,
                c.cold_count or 0,
                c.is_hot_cluster or false
            )
            f:write(bin)
        end
    end
    f:close()

    os.remove(file_path)
    os.rename(temp, file_path)
    print(string.format("[Cluster] 二进制原子保存完成（%d 个簇）", count))
end

-- ====================== 以下全部保持你原来的实现（仅移除多余转换） ======================
function M.find_best_cluster(vec)
    -- vec 已经是 Lua table
    local best_id, best_sim = nil, -1
    for id, c in pairs(M.clusters) do
        if c and c.centroid then
            local sim = tool.cosine_similarity(vec, c.centroid)
            if sim > best_sim then
                best_sim = sim
                best_id = id
            end
        end
    end
    return best_id, best_sim
end

function M.add(vec, mem_index)
    local saver = require("module.saver")
    -- vec 已经是 Lua table
    if #vec == 0 then 
        print("[Cluster] Error: 收到空向量")
        return 
    end

    local th = config.settings.cluster.cluster_sim or 0.75
    local best_id, sim = M.find_best_cluster(vec)

    if best_id and sim >= th then
        local c = M.clusters[best_id]
        table.insert(c.members, mem_index)
        print(string.format("【簇匹配】记忆行 %d → 簇 %d (sim=%.4f)", mem_index, best_id, sim))
    else
        local id = next_cluster_id
        M.clusters[id] = {
            centroid = deepcopy_vec(vec),
            members = {mem_index},
            hot_count = 1,
            cold_count = 0,
            heat = 0,                    -- 显式初始化
            is_hot_cluster = false       -- 显式初始化
        }
        next_cluster_id = id + 1
        print(string.format("【新簇创建】记忆行 %d 成为簇 %d 质心（固定）", mem_index, id))
    end
    saver.mark_dirty()
end

function M.mark_cold(mem_line)
    local saver = require("module.saver")
    for id, c in pairs(M.clusters) do
        for _, m in ipairs(c.members or {}) do
            if m == mem_line then
                c.cold_count = (c.cold_count or 0) + 1
                c.hot_count = math.max(0, (c.hot_count or 1) - 1)
                saver.mark_dirty()
                return id
            end
        end
    end
end

function M.mark_hot(mem_line)
    local saver = require("module.saver")
    for id, c in pairs(M.clusters) do
        for _, m in ipairs(c.members or {}) do
            if m == mem_line then
                c.hot_count = (c.hot_count or 0) + 1
                c.cold_count = math.max(0, (c.cold_count or 0) - 1)
                saver.mark_dirty()
                return id
            end
        end
    end
end

function M.get_hot_ratio(id)
    local c = M.clusters[id]
    if not c then return 0 end
    local total = (c.hot_count or 0) + (c.cold_count or 0)
    return total > 0 and (c.hot_count / total) or 0
end

function M.update_hot_status()
    local ratios = {}
    for id, _ in pairs(M.clusters) do
        table.insert(ratios, {id = id, ratio = M.get_hot_ratio(id)})
    end
    table.sort(ratios, function(a, b) return a.ratio > b.ratio end)
    for _, r in ipairs(ratios) do
        local c = M.clusters[r.id]
        c.is_hot_cluster = r.ratio >= (config.settings.cluster.hot_cluster_ratio or 0.65)
    end
end

function M.get_cold_members(cid)
    return {}
end

function M.find_sim_in_cluster(vec, cluster_id)
    local memory = require("module.memory")
    -- vec 已经是 Lua table
    if not M.clusters[cluster_id] then return {} end
    local results = {}
    for _, idx in ipairs(M.clusters[cluster_id].members) do
        local mem_vec = memory.return_mem_vec(idx)   -- 已经是 Lua table
        if mem_vec then
            local sim = tool.cosine_similarity(vec, mem_vec)
            table.insert(results, {index = idx, similarity = sim})
        end
    end
    table.sort(results, function(a, b) return a.similarity > b.similarity end)
    return results
end

function M.get_cluster_id_for_line(mem_line)
    for id, c in pairs(M.clusters) do
        for _, m in ipairs(c.members or {}) do
            if m == mem_line then return id end
        end
    end
    return nil
end

return M