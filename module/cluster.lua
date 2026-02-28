---@diagnostic disable: deprecated
-- cluster.lua （含 supercluster 路由版）
local M = {}

local tool = require("module.tool")
local config = require("module.config")
local ffi = require("ffi")
local persistence = require("module.persistence")

local file_path = "memory/clusters.bin"

M.clusters = {}
M.line_to_cluster = {}
local next_cluster_id = 1

M.super_members = {}
M.super_centroids = {}
M.super_of_cluster = {}
M.super_indexed_clusters = 0
M.super_last_rebuild_clusters = 0
M.super_rebuild_count = 0

local function refresh_hot_flag(id, clu)
    local c = clu or M.clusters[id]
    if not c then return end
    local total = (c.hot_count or 0) + (c.cold_count or 0)
    local ratio = (total > 0) and ((c.hot_count or 0) / total) or 0
    c.is_hot_cluster = ratio >= (config.settings.cluster.hot_cluster_ratio or 0.65)
end

local function deepcopy_vec(vec)
    local copy = {}
    for i = 1, #vec do copy[i] = vec[i] end
    return copy
end

local function normalize_vec(vec)
    local out = {}
    local norm = 0.0
    for i = 1, #vec do
        local v = tonumber(vec[i]) or 0.0
        out[i] = v
        norm = norm + v * v
    end
    norm = math.sqrt(norm)
    if norm > 0 then
        for i = 1, #out do
            out[i] = out[i] / norm
        end
    end
    return out
end

local function blend_centroid(old_vec, old_n, add_vec)
    if not old_vec or #old_vec == 0 then
        return normalize_vec(add_vec or {})
    end
    old_n = math.max(0, tonumber(old_n) or 0)
    local out = {}
    local new_n = old_n + 1
    for i = 1, #old_vec do
        local ov = tonumber(old_vec[i]) or 0.0
        local av = tonumber((add_vec or {})[i]) or 0.0
        out[i] = (ov * old_n + av) / new_n
    end
    return normalize_vec(out)
end

local function cluster_count()
    local n = 0
    for _ in pairs(M.clusters) do n = n + 1 end
    return n
end

local function sorted_cluster_ids()
    local ids = {}
    for id, _ in pairs(M.clusters) do
        ids[#ids + 1] = tonumber(id)
    end
    table.sort(ids)
    return ids
end

local function clear_super_index()
    M.super_members = {}
    M.super_centroids = {}
    M.super_of_cluster = {}
    M.super_indexed_clusters = 0
    M.super_last_rebuild_clusters = 0
end

local function use_superclusters()
    local cfg = (config.settings or {}).cluster or {}
    if cfg.hierarchical_cluster_enabled ~= true then
        return false
    end
    local min_c = math.max(2, tonumber(cfg.supercluster_min_clusters) or 64)
    return cluster_count() >= min_c
end

local function effective_super_topn(base)
    local cfg = (config.settings or {}).cluster or {}
    local cnum = math.max(1, cluster_count())
    local min_c = math.max(2, tonumber(cfg.supercluster_min_clusters) or 64)
    local ratio = cnum / min_c
    local scale = math.max(0.0, tonumber(cfg.supercluster_topn_scale) or 0.20)
    local dyn = 1.0
    if ratio > 1.0 then
        dyn = dyn + scale * (math.log(ratio) / math.log(2))
    end
    local n = math.floor((tonumber(base) or 1) * dyn + 0.5)
    if n < 1 then n = 1 end
    return n
end

local function attach_cluster_to_super(cid)
    local clu = M.clusters[cid]
    if not clu or not clu.centroid then
        return nil
    end

    local cfg = (config.settings or {}).cluster or {}
    local target_size = math.max(8, tonumber(cfg.supercluster_target_size) or 64)
    local max_size = math.max(
        target_size,
        math.floor(target_size * math.max(1.0, tonumber(cfg.supercluster_max_size_mult) or 1.8))
    )
    local sim_th = tonumber(cfg.supercluster_sim) or 0.52

    local best_sid, best_sim = nil, -1
    for sid, svec in ipairs(M.super_centroids) do
        local members = M.super_members[sid] or {}
        if #members < max_size then
            local sim = tool.cosine_similarity(clu.centroid, svec)
            if sim > best_sim then
                best_sim = sim
                best_sid = sid
            end
        end
    end

    if (not best_sid) or best_sim < sim_th then
        local sid = #M.super_members + 1
        M.super_members[sid] = { cid }
        M.super_centroids[sid] = deepcopy_vec(clu.centroid)
        M.super_of_cluster[cid] = sid
        return sid
    end

    local members = M.super_members[best_sid]
    local old_n = #members
    members[#members + 1] = cid
    M.super_of_cluster[cid] = best_sid
    M.super_centroids[best_sid] = blend_centroid(M.super_centroids[best_sid], old_n, clu.centroid)
    return best_sid
end

local function rebuild_superclusters()
    clear_super_index()
    local ids = sorted_cluster_ids()
    if #ids <= 0 then
        return
    end

    for _, cid in ipairs(ids) do
        attach_cluster_to_super(cid)
    end

    M.super_indexed_clusters = cluster_count()
    M.super_last_rebuild_clusters = M.super_indexed_clusters
    M.super_rebuild_count = M.super_rebuild_count + 1
end

local function ensure_supercluster_index()
    if not use_superclusters() then
        if #M.super_members > 0 then
            clear_super_index()
        end
        return false
    end

    local cnum = cluster_count()
    if cnum <= 0 then
        clear_super_index()
        return false
    end

    if M.super_indexed_clusters <= 0 or #M.super_members <= 0 then
        rebuild_superclusters()
        return #M.super_members > 0
    end

    if cnum < M.super_indexed_clusters then
        rebuild_superclusters()
        return #M.super_members > 0
    end

    if cnum > M.super_indexed_clusters then
        local ids = sorted_cluster_ids()
        for _, cid in ipairs(ids) do
            if not M.super_of_cluster[cid] then
                attach_cluster_to_super(cid)
            end
        end
        M.super_indexed_clusters = cnum
    end

    local cfg = (config.settings or {}).cluster or {}
    local every = math.max(0, tonumber(cfg.supercluster_rebuild_every) or 600)
    if every > 0 and (cnum - M.super_last_rebuild_clusters) >= every then
        rebuild_superclusters()
    end

    return #M.super_members > 0
end

function M.load()
    M.clusters = {}
    M.line_to_cluster = {}
    next_cluster_id = 1
    clear_super_index()
    M.super_rebuild_count = 0
    local cluster_count_loaded = 0

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
            for _, m in ipairs(clu.members or {}) do
                M.line_to_cluster[m] = clu.id
            end
            next_cluster_id = math.max(next_cluster_id, clu.id + 1)
            cluster_count_loaded = cluster_count_loaded + 1
            offset = offset + record_size
        else
            print("[Cluster] 解析在 offset", offset, "处中断")
            break
        end
    end

    print(string.format("[Cluster] 已从二进制加载 %d 个语义簇，下一个ID = %d", cluster_count_loaded, next_cluster_id))
    M.update_hot_status()
end

function M.save_to_disk()
    local count = 0
    for _ in pairs(M.clusters) do count = count + 1 end

    local ok, err = persistence.write_atomic(file_path, "wb", function(f)
        local function write_or_fail(chunk)
            local w_ok, w_err = f:write(chunk)
            if not w_ok then
                return false, w_err
            end
            return true
        end

        local w1_ok, w1_err = write_or_fail("CLUS")
        if not w1_ok then return false, w1_err end

        local header = ffi.new("uint32_t[4]", 1, count, 0, 0)
        local w2_ok, w2_err = write_or_fail(ffi.string(header, 16))
        if not w2_ok then return false, w2_err end

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
                local wb_ok, wb_err = write_or_fail(bin)
                if not wb_ok then return false, wb_err end
            end
        end
        return true
    end)

    if not ok then
        return false, err
    end

    print(string.format("[Cluster] 二进制原子保存完成（%d 个簇）", count))
    return true
end

function M.get_cluster_ids()
    return sorted_cluster_ids()
end

function M.cluster_count()
    return cluster_count()
end

function M.super_candidate_clusters(vec, base_topn)
    local ids = sorted_cluster_ids()
    if #ids <= 0 then
        return {}, 0
    end

    if not ensure_supercluster_index() then
        return ids, 0
    end

    local ops = 0
    local scored = {}
    for sid, svec in ipairs(M.super_centroids) do
        local sim = tool.cosine_similarity(vec, svec)
        scored[#scored + 1] = { sid = sid, sim = sim }
        ops = ops + 1
    end

    table.sort(scored, function(a, b) return a.sim > b.sim end)
    local topn = effective_super_topn(base_topn or (((config.settings or {}).cluster or {}).supercluster_topn_query or 4))
    if topn > #scored then topn = #scored end

    local used = {}
    local out = {}
    for i = 1, topn do
        local sid = scored[i].sid
        for _, cid in ipairs(M.super_members[sid] or {}) do
            if not used[cid] then
                used[cid] = true
                out[#out + 1] = cid
            end
        end
    end

    if #out <= 0 then
        return ids, ops
    end

    return out, ops
end

function M.find_best_cluster(vec, opts)
    local cfg = (config.settings or {}).cluster or {}
    local topn = tonumber((opts or {}).super_topn) or tonumber(cfg.supercluster_topn_add) or 3

    local candidates, ops0 = M.super_candidate_clusters(vec, topn)
    if #candidates <= 0 then
        return nil, -1, ops0
    end

    local best_id, best_sim = nil, -1
    local ops = ops0
    for _, id in ipairs(candidates) do
        local c = M.clusters[id]
        if c and c.centroid then
            local sim = tool.cosine_similarity(vec, c.centroid)
            ops = ops + 1
            if sim > best_sim then
                best_sim = sim
                best_id = id
            end
        end
    end
    return best_id, best_sim, ops
end

function M.add(vec, mem_index)
    local saver = require("module.saver")
    if #vec == 0 then
        print("[Cluster] Error: 收到空向量")
        return
    end

    local th = config.settings.cluster.cluster_sim or 0.75
    local best_id, sim = M.find_best_cluster(vec, { super_topn = config.settings.cluster.supercluster_topn_add })

    if best_id and sim >= th then
        local c = M.clusters[best_id]
        table.insert(c.members, mem_index)
        c.hot_count = (c.hot_count or 0) + 1
        refresh_hot_flag(best_id, c)
        M.line_to_cluster[mem_index] = best_id
        print(string.format("【簇匹配】记忆行 %d → 簇 %d (sim=%.4f)", mem_index, best_id, sim))
    else
        local id = next_cluster_id
        M.clusters[id] = {
            centroid = deepcopy_vec(vec),
            members = {mem_index},
            hot_count = 1,
            cold_count = 0,
            heat = 0,
            is_hot_cluster = true
        }
        refresh_hot_flag(id, M.clusters[id])
        M.line_to_cluster[mem_index] = id
        next_cluster_id = id + 1
        M.super_indexed_clusters = 0
        print(string.format("【新簇创建】记忆行 %d 成为簇 %d 质心（固定）", mem_index, id))
    end
    saver.mark_dirty()
end

function M.mark_cold(mem_line)
    local saver = require("module.saver")
    local id = M.line_to_cluster[mem_line]
    local c = id and M.clusters[id] or nil
    if c then
        c.cold_count = (c.cold_count or 0) + 1
        c.hot_count = math.max(0, (c.hot_count or 1) - 1)
        refresh_hot_flag(id, c)
        saver.mark_dirty()
        return id
    end
end

function M.mark_hot(mem_line)
    local saver = require("module.saver")
    local id = M.line_to_cluster[mem_line]
    local c = id and M.clusters[id] or nil
    if c then
        c.hot_count = (c.hot_count or 0) + 1
        c.cold_count = math.max(0, (c.cold_count or 0) - 1)
        refresh_hot_flag(id, c)
        saver.mark_dirty()
        return id
    end
end

function M.get_hot_ratio(id)
    local c = M.clusters[id]
    if not c then return 0 end
    local total = (c.hot_count or 0) + (c.cold_count or 0)
    return total > 0 and (c.hot_count / total) or 0
end

function M.update_hot_status()
    for id, c in pairs(M.clusters) do
        refresh_hot_flag(id, c)
    end
end

function M.get_cold_members(cid)
    local memory = require("module.memory")
    local c = M.clusters[cid]
    if not c then return {} end
    local out = {}
    for _, idx in ipairs(c.members or {}) do
        if memory.get_heat_by_index(idx) <= 0 then
            out[#out + 1] = idx
        end
    end
    return out
end

function M.find_sim_in_cluster(vec, cluster_id, opts)
    local memory = require("module.memory")
    opts = opts or {}
    local only_hot = opts.only_hot == true
    local only_cold = opts.only_cold == true
    local max_results = tonumber(opts.max_results)
    if max_results and max_results <= 0 then
        max_results = nil
    end

    if not M.clusters[cluster_id] then return {} end

    local results = {}
    local topk = {}
    local use_topk = max_results ~= nil

    for _, idx in ipairs(M.clusters[cluster_id].members or {}) do
        local heat = memory.get_heat_by_index(idx)
        if only_hot and heat <= 0 then
            goto continue
        end
        if only_cold and heat > 0 then
            goto continue
        end

        local mem_vec = memory.return_mem_vec(idx)
        if mem_vec then
            local sim = tool.cosine_similarity(vec, mem_vec)
            if use_topk then
                if #topk < max_results then
                    topk[#topk + 1] = { index = idx, similarity = sim }
                    local j = #topk
                    while j > 1 and topk[j].similarity < topk[j - 1].similarity do
                        topk[j], topk[j - 1] = topk[j - 1], topk[j]
                        j = j - 1
                    end
                elseif sim > topk[1].similarity then
                    topk[1] = { index = idx, similarity = sim }
                    local j = 1
                    while j < #topk and topk[j].similarity > topk[j + 1].similarity do
                        topk[j], topk[j + 1] = topk[j + 1], topk[j]
                        j = j + 1
                    end
                end
            else
                results[#results + 1] = { index = idx, similarity = sim }
            end
        end
        ::continue::
    end

    if use_topk then
        for i = #topk, 1, -1 do
            results[#results + 1] = topk[i]
        end
        return results
    end

    table.sort(results, function(a, b) return a.similarity > b.similarity end)
    return results
end

function M.get_cluster_id_for_line(mem_line)
    return M.line_to_cluster[mem_line]
end

function M.rebuild_superclusters()
    rebuild_superclusters()
end

return M
