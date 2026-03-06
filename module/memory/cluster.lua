---@diagnostic disable: deprecated
-- cluster.lua (V3 cluster index + shard-aware retrieval)

local M = {}

local tool = require("module.tool")
local config = require("module.config")
local ffi = require("ffi")
local persistence = require("module.persistence")

local STORAGE_CFG = (config.settings or {}).storage_v3 or {}
local ROOT_DIR = STORAGE_CFG.root or "memory/v3"
local file_path = ROOT_DIR .. "/cluster_index.bin"

local CLUSTER_MAGIC = "CID3"
local CLUSTER_VERSION = 1

M.clusters = {}
M.line_to_cluster = {}
local next_cluster_id = 1

M.super_members = {}
M.super_centroids = {}
M.super_of_cluster = {}
M.super_indexed_clusters = 0
M.super_last_rebuild_clusters = 0
M.super_rebuild_count = 0

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function refresh_hot_flag(id, clu)
    local c = clu or M.clusters[id]
    if not c then return end
    local total = (c.hot_count or 0) + (c.cold_count or 0)
    local ratio = (total > 0) and ((c.hot_count or 0) / total) or 0
    c.is_hot_cluster = ratio >= (config.settings.cluster.hot_cluster_ratio or 0.65)
end

local function ensure_member_tables(c)
    if not c then return end
    c.hot_members = c.hot_members or {}
    c.cold_members = c.cold_members or {}
    c._hot_pos = c._hot_pos or {}
    c._cold_pos = c._cold_pos or {}

    if next(c._hot_pos) == nil and #c.hot_members > 0 then
        for i, line in ipairs(c.hot_members) do
            c._hot_pos[line] = i
        end
    end
    if next(c._cold_pos) == nil and #c.cold_members > 0 then
        for i, line in ipairs(c.cold_members) do
            c._cold_pos[line] = i
        end
    end
end

local function remove_member(arr, pos_map, line)
    local pos = pos_map[line]
    if not pos then return false end

    local last = #arr
    if pos ~= last then
        local moved = arr[last]
        arr[pos] = moved
        pos_map[moved] = pos
    end
    arr[last] = nil
    pos_map[line] = nil
    return true
end

local function add_member(arr, pos_map, line)
    if pos_map[line] then return false end
    arr[#arr + 1] = line
    pos_map[line] = #arr
    return true
end

local function sync_counts(c)
    c.hot_count = #(c.hot_members or {})
    c.cold_count = #(c.cold_members or {})
end

local function type_cfg()
    return ((config.settings or {}).memory_types or {})
end

local function cluster_type_bonus()
    return tonumber(type_cfg().cluster_type_bonus) or 0.0
end

local function memory_type_name(line)
    local ok_mem, memory = pcall(require, "module.memory.store")
    if ok_mem and memory and memory.get_type_name then
        return memory.get_type_name(line)
    end
    return tostring(type_cfg().default or "Fact")
end

local function ensure_type_counts(c)
    if not c then return end

    local total_members = #(c.members or {})
    local need_rebuild = (c.type_counts == nil)
    if not need_rebuild then
        local counted = 0
        for _, n in pairs(c.type_counts) do
            counted = counted + math.max(0, math.floor(tonumber(n) or 0))
        end
        need_rebuild = counted ~= total_members
    end
    if not need_rebuild then
        return
    end

    c.type_counts = {}
    for _, line in ipairs(c.members or {}) do
        local type_name = memory_type_name(line)
        if type_name and type_name ~= "" then
            c.type_counts[type_name] = (c.type_counts[type_name] or 0) + 1
        end
    end
end

local function adjust_type_count(c, type_name, delta)
    if not c then return end
    local key = tostring(type_name or "")
    if key == "" then return end

    ensure_type_counts(c)
    local next_value = (tonumber((c.type_counts or {})[key]) or 0) + math.floor(tonumber(delta) or 0)
    if next_value > 0 then
        c.type_counts[key] = next_value
    else
        c.type_counts[key] = nil
    end
end

local function cluster_type_affinity_value(c, preferred_type)
    local type_name = tostring(preferred_type or "")
    if type_name == "" or not c then
        return 0.0
    end

    ensure_type_counts(c)
    local total = #(c.members or {})
    if total <= 0 then
        return 0.0
    end

    local count = tonumber((c.type_counts or {})[type_name]) or 0.0
    if count <= 0 then
        return 0.0
    end
    return count / total
end

local function set_member_hot_state(c, line, to_hot)
    ensure_member_tables(c)
    line = tonumber(line)
    if not line or line <= 0 then return false end

    local changed = false
    if to_hot then
        if remove_member(c.cold_members, c._cold_pos, line) then
            changed = true
        end
        if add_member(c.hot_members, c._hot_pos, line) then
            changed = true
        end
    else
        if remove_member(c.hot_members, c._hot_pos, line) then
            changed = true
        end
        if add_member(c.cold_members, c._cold_pos, line) then
            changed = true
        end
    end

    if changed then
        sync_counts(c)
        refresh_hot_flag(nil, c)
    end
    return changed
end

local function rebuild_cluster_member_views(c)
    if not c then return end

    local ok_mem, memory = pcall(require, "module.memory.store")
    if not ok_mem or not memory then
        c.hot_members = {}
        c.cold_members = {}
        c._hot_pos = {}
        c._cold_pos = {}
        sync_counts(c)
        refresh_hot_flag(nil, c)
        return
    end

    c.hot_members = {}
    c.cold_members = {}
    c._hot_pos = {}
    c._cold_pos = {}
    c.type_counts = {}

    local dedup_members = {}
    local seen = {}
    for _, raw in ipairs(c.members or {}) do
        local line = tonumber(raw)
        if line and line > 0 and (not seen[line]) then
            seen[line] = true
            dedup_members[#dedup_members + 1] = line
            if (memory.get_heat_by_index(line) or 0) > 0 then
                add_member(c.hot_members, c._hot_pos, line)
            else
                add_member(c.cold_members, c._cold_pos, line)
            end
            adjust_type_count(c, memory.get_type_name(line), 1)
        end
    end
    c.members = dedup_members

    sync_counts(c)
    refresh_hot_flag(nil, c)
end

local function rebuild_all_member_views()
    for _, c in pairs(M.clusters) do
        rebuild_cluster_member_views(c)
    end
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
    ensure_dir(ROOT_DIR)

    M.clusters = {}
    M.line_to_cluster = {}
    next_cluster_id = 1
    clear_super_index()
    M.super_rebuild_count = 0
    local cluster_count_loaded = 0

    local f = io.open(file_path, "rb")
    if not f then
        print("[Cluster] cluster_index.bin 不存在 → 首次运行将自动创建")
        return
    end

    local data = f:read("*a")
    f:close()

    if #data < 20 or data:sub(1, 4) ~= CLUSTER_MAGIC then
        print("[Cluster] V3 cluster_index.bin 头无效，已忽略")
        return
    end

    local p = ffi.cast("const uint8_t*", data)
    local header = ffi.cast("const uint32_t*", p + 4)
    local version = tonumber(header[0]) or 0
    if version ~= CLUSTER_VERSION then
        print(string.format("[Cluster] V3 版本不匹配，got=%d expect=%d", version, CLUSTER_VERSION))
        return
    end

    local offset = 20
    while offset < #data do
        local clu, record_size = tool.parse_cluster_record(data, offset)
        if clu then
            M.clusters[clu.id] = {
                centroid = clu.centroid,
                members = clu.members,
                heat = clu.heat,
                hot_count = clu.hot_count,
                cold_count = clu.cold_count,
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

    local ok_mem, memory = pcall(require, "module.memory.store")
    if ok_mem and memory and memory.set_cluster_id then
        for line, cid in pairs(M.line_to_cluster) do
            memory.set_cluster_id(line, cid)
        end
    end

    rebuild_all_member_views()

    print(string.format("[Cluster] V3 加载完成: %d 个语义簇，下一个ID = %d", cluster_count_loaded, next_cluster_id))
    M.update_hot_status()
end

function M.save_to_disk()
    ensure_dir(ROOT_DIR)

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

        local w1_ok, w1_err = write_or_fail(CLUSTER_MAGIC)
        if not w1_ok then return false, w1_err end

        local header = ffi.new("uint32_t[4]", CLUSTER_VERSION, count, 0, 0)
        local w2_ok, w2_err = write_or_fail(ffi.string(header, 16))
        if not w2_ok then return false, w2_err end

        local ids = sorted_cluster_ids()
        for _, id in ipairs(ids) do
            local c = M.clusters[id]
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

    print(string.format("[Cluster] V3 索引保存完成（%d 个簇）", count))
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
    opts = opts or {}
    local cfg = (config.settings or {}).cluster or {}
    local topn = tonumber(opts.super_topn) or tonumber(cfg.supercluster_topn_add) or 3
    local strict_type = opts.strict_type == true
    local preferred_type = tostring(opts.preferred_type or "")
    if preferred_type ~= "" then
        local ok_mem, memory = pcall(require, "module.memory.store")
        if ok_mem and memory and memory.match_type_name then
            preferred_type = memory.match_type_name(preferred_type) or preferred_type
        end
    end

    local candidates, ops0 = M.super_candidate_clusters(vec, topn)
    if #candidates <= 0 then
        return nil, -1, ops0
    end

    local best_id, best_sim = nil, -1
    local best_rank = -math.huge
    local ops = ops0
    for _, id in ipairs(candidates) do
        local c = M.clusters[id]
        if c and c.centroid then
            local sim = tool.cosine_similarity(vec, c.centroid)
            ops = ops + 1
            local affinity = cluster_type_affinity_value(c, preferred_type)
            if (not strict_type) or preferred_type == "" or affinity > 0.0 then
                local rank = sim + affinity * cluster_type_bonus()
                if rank > best_rank or (rank == best_rank and sim > best_sim) then
                    best_rank = rank
                    best_sim = sim
                    best_id = id
                end
            end
        end
    end
    return best_id, best_sim, ops
end

function M.add(vec, mem_index, opts)
    local saver = require("module.memory.saver")
    local memory = require("module.memory.store")
    opts = opts or {}
    if #vec == 0 then
        print("[Cluster] Error: 收到空向量")
        return nil
    end

    local th = config.settings.cluster.cluster_sim or 0.75
    local preferred_type = memory.match_type_name(opts.preferred_type) or memory.get_type_name(mem_index)
    local strict_type = opts.strict_type == true
    local best_id, sim = M.find_best_cluster(vec, {
        super_topn = config.settings.cluster.supercluster_topn_add,
        preferred_type = preferred_type,
        strict_type = strict_type,
    })

    local assigned
    if best_id and sim >= th then
        local c = M.clusters[best_id]
        ensure_member_tables(c)
        table.insert(c.members, mem_index)
        adjust_type_count(c, preferred_type, 1)
        set_member_hot_state(c, mem_index, true)
        M.line_to_cluster[mem_index] = best_id
        assigned = best_id
        print(string.format("【簇匹配】记忆行 %d → 簇 %d (sim=%.4f)", mem_index, best_id, sim))
    else
        local id = next_cluster_id
        M.clusters[id] = {
            centroid = deepcopy_vec(vec),
            members = {mem_index},
            hot_members = {mem_index},
            cold_members = {},
            _hot_pos = { [mem_index] = 1 },
            _cold_pos = {},
            hot_count = 1,
            cold_count = 0,
            heat = 0,
            is_hot_cluster = true,
            type_counts = preferred_type ~= "" and { [preferred_type] = 1 } or {}
        }
        refresh_hot_flag(id, M.clusters[id])
        M.line_to_cluster[mem_index] = id
        next_cluster_id = id + 1
        M.super_indexed_clusters = 0
        assigned = id
        print(string.format("【新簇创建】记忆行 %d 成为簇 %d 质心（固定）", mem_index, id))
    end

    memory.set_cluster_id(mem_index, assigned)
    saver.mark_dirty()
    return assigned
end

function M.mark_cold(mem_line)
    local saver = require("module.memory.saver")
    local id = M.line_to_cluster[mem_line]
    local c = id and M.clusters[id] or nil
    if c then
        if set_member_hot_state(c, mem_line, false) then
            saver.mark_dirty()
        end
        return id
    end
end

function M.mark_hot(mem_line)
    local saver = require("module.memory.saver")
    local id = M.line_to_cluster[mem_line]
    local c = id and M.clusters[id] or nil
    if c then
        if set_member_hot_state(c, mem_line, true) then
            saver.mark_dirty()
        end
        return id
    end
end

function M.on_memory_heat_change(mem_line, old_heat, new_heat)
    local saver = require("module.memory.saver")
    local id = M.line_to_cluster[mem_line]
    local c = id and M.clusters[id] or nil
    if not c then
        return nil, false
    end

    local was_hot = (tonumber(old_heat) or 0) > 0
    local now_hot = (tonumber(new_heat) or 0) > 0
    if was_hot == now_hot then
        return id, false
    end

    local changed = set_member_hot_state(c, mem_line, now_hot)
    if changed then
        saver.mark_dirty()
    end
    return id, changed
end

function M.on_memory_type_change(mem_line, old_type, new_type)
    local saver = require("module.memory.saver")
    local id = M.line_to_cluster[mem_line]
    local c = id and M.clusters[id] or nil
    if not c then
        return nil, false
    end

    local from = tostring(old_type or "")
    local to = tostring(new_type or "")
    if from == to then
        return id, false
    end

    adjust_type_count(c, from, -1)
    adjust_type_count(c, to, 1)
    saver.mark_dirty()
    return id, true
end

function M.get_hot_ratio(id)
    local c = M.clusters[id]
    if not c then return 0 end
    local total = (c.hot_count or 0) + (c.cold_count or 0)
    return total > 0 and (c.hot_count / total) or 0
end

function M.update_hot_status()
    for id, c in pairs(M.clusters) do
        ensure_member_tables(c)
        sync_counts(c)
        refresh_hot_flag(id, c)
    end
end

function M.get_cold_members(cid)
    local c = M.clusters[cid]
    if not c then return {} end
    local out = {}
    ensure_member_tables(c)
    for i = 1, #(c.cold_members or {}) do
        out[#out + 1] = c.cold_members[i]
    end
    return out
end

function M.find_sim_in_cluster(vec, cluster_id, opts)
    local memory = require("module.memory.store")
    opts = opts or {}
    local only_hot = opts.only_hot == true
    local only_cold = opts.only_cold == true
    local allowed_types = opts.allowed_types
    local blocked_types = opts.blocked_types
    local max_results = tonumber(opts.max_results)
    if max_results and max_results <= 0 then
        max_results = nil
    end

    if not M.clusters[cluster_id] then return {} end

    local qv = tool.to_ptr_vec(vec)
    if not qv then return {} end

    local shard = memory.get_cluster_shard(cluster_id)
    if not shard then return {} end

    if only_hot and only_cold then
        return {}
    end

    local c = M.clusters[cluster_id]
    ensure_member_tables(c)
    local scan_members = c.members or {}
    if only_hot then
        scan_members = c.hot_members or {}
    elseif only_cold then
        scan_members = c.cold_members or {}
    end

    local results = {}
    local topk = {}
    local use_topk = max_results ~= nil

    for _, idx in ipairs(scan_members) do
        local type_ok = true
        if allowed_types or blocked_types then
            type_ok = select(1, memory.type_matches(idx, allowed_types, blocked_types))
        end

        if type_ok then
        local mem_vec = shard.ptr_by_line and shard.ptr_by_line[idx] or nil
            if mem_vec then
                local sim = tool.cosine_similarity(qv, mem_vec)
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
        end
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

function M.get_cluster_type_affinity(cluster_id, preferred_type)
    local c = M.clusters[cluster_id]
    if not c then
        return 0.0
    end

    local type_name = tostring(preferred_type or "")
    if type_name == "" then
        return 0.0
    end

    local ok_mem, memory = pcall(require, "module.memory.store")
    if ok_mem and memory and memory.match_type_name then
        type_name = memory.match_type_name(type_name) or type_name
    end
    return cluster_type_affinity_value(c, type_name)
end

function M.rebuild_superclusters()
    rebuild_superclusters()
end

return M
