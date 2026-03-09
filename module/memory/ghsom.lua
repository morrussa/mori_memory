local M = {}

local config = require("module.config")
local ffi = require("ffi")
local persistence = require("module.persistence")
local tool = require("module.tool")

local STORAGE_CFG = (config.settings or {}).storage_v3 or {}
local ROOT_DIR = STORAGE_CFG.root or "memory/v3"
local GHSOM_PATH = ROOT_DIR .. "/ghsom_index.bin"

local MAGIC = "GHSM"
local VERSION = 2

local HEAT_CFG = (config.settings or {}).heat or {}
local GHSOM_CFG = (config.settings or {}).ghsom or {}

local MAP_WIDTH = math.max(2, math.floor(tonumber(GHSOM_CFG.map_width) or 2))
local MAP_HEIGHT = math.max(2, math.floor(tonumber(GHSOM_CFG.map_height) or 2))
local MAX_LEVEL = math.max(1, math.floor(tonumber(GHSOM_CFG.max_level) or 4))
local MIN_SPLIT_MEMBERS = math.max(4, math.floor(tonumber(GHSOM_CFG.min_split_members) or 12))
local SPLIT_SIM_THRESHOLD = tonumber(GHSOM_CFG.split_similarity_threshold) or 0.78
local LEARNING_RATE = tonumber(GHSOM_CFG.learning_rate) or 0.18
local NEIGHBOR_RADIUS = math.max(0, tonumber(GHSOM_CFG.neighbor_radius) or 1)

local NEW_HEAT = math.max(1, math.floor(tonumber(HEAT_CFG.new_memory_heat) or 43000))
local MAX_NEIGHBORS = math.max(1, math.floor(tonumber(HEAT_CFG.max_neighbors) or 5))
local NEIGHBOR_HEAT = math.max(1, math.floor(tonumber(HEAT_CFG.neighbors_heat) or 26000))
local ACTIVITY_DECAY = tonumber(GHSOM_CFG.activity_decay_rate)
    or tonumber(HEAT_CFG.activity_decay_rate)
    or 0.985
local SUPPRESSION_THRESHOLD = math.max(
    1,
    math.floor(
        tonumber(GHSOM_CFG.activity_threshold)
            or tonumber(HEAT_CFG.suppression_threshold)
            or math.max(2000, math.floor(NEW_HEAT * 0.18))
    )
)
local REACTIVATE_TOPN = math.max(1, math.floor(tonumber(GHSOM_CFG.reactivate_topn) or 2))
local REACTIVATE_BOOST = math.max(
    NEW_HEAT,
    math.floor(tonumber(GHSOM_CFG.reactivate_boost) or (NEW_HEAT + NEIGHBOR_HEAT))
)
local REACTIVATE_SCAN_NODES = math.max(2, math.floor(tonumber(GHSOM_CFG.reactivate_scan_nodes) or 6))
local REACTIVATE_SCAN_LIMIT = math.max(REACTIVATE_TOPN * 4, math.floor(tonumber(GHSOM_CFG.reactivate_scan_limit) or 24))

local function get_memory()
    return require("module.memory.store")
end

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function shallow_copy(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function vector_to_table(vec)
    if type(vec) ~= "table" then
        return {}
    end

    if vec.__ptr and vec.__dim then
        local dim = math.max(0, tonumber(vec.__dim) or 0)
        local out = {}
        for i = 0, dim - 1 do
            out[i + 1] = tonumber(vec.__ptr[i]) or 0.0
        end
        return out
    end

    return shallow_copy(vec)
end

local function normalize_vector(vec)
    local out = {}
    local norm = 0.0
    for i = 1, #(vec or {}) do
        local v = tonumber(vec[i]) or 0.0
        out[i] = v
        norm = norm + v * v
    end
    norm = math.sqrt(norm)
    if norm <= 0 then
        return out
    end
    for i = 1, #out do
        out[i] = out[i] / norm
    end
    return out
end

local function perturb_vector(base_vec, magnitude)
    local out = {}
    magnitude = tonumber(magnitude) or 0.035
    for i = 1, #(base_vec or {}) do
        out[i] = (tonumber(base_vec[i]) or 0.0) + ((math.random() * 2.0 - 1.0) * magnitude)
    end
    return normalize_vector(out)
end

local function safe_similarity(a, b)
    return tonumber(tool.cosine_similarity(a, b)) or 0.0
end

local function add_member(arr, pos_map, line)
    if pos_map[line] then return false end
    arr[#arr + 1] = line
    pos_map[line] = #arr
    return true
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

local function topk_push(topk, item, limit)
    limit = tonumber(limit)
    if not limit or limit <= 0 then
        topk[#topk + 1] = item
        return
    end

    if #topk < limit then
        topk[#topk + 1] = item
        local j = #topk
        while j > 1 and (topk[j].similarity or -1) < (topk[j - 1].similarity or -1) do
            topk[j], topk[j - 1] = topk[j - 1], topk[j]
            j = j - 1
        end
        return
    end

    if (item.similarity or -1) <= (topk[1].similarity or -1) then
        return
    end

    topk[1] = item
    local j = 1
    while j < #topk and (topk[j].similarity or -1) > (topk[j + 1].similarity or -1) do
        topk[j], topk[j + 1] = topk[j + 1], topk[j]
        j = j + 1
    end
end

local function topk_to_desc(topk)
    local out = {}
    for i = #topk, 1, -1 do
        out[#out + 1] = topk[i]
    end
    return out
end

local function node_cell_count(node)
    return (node.width or MAP_WIDTH) * (node.height or MAP_HEIGHT)
end

local function slot_to_xy(node, slot)
    local width = node.width or MAP_WIDTH
    local x = ((slot - 1) % width) + 1
    local y = math.floor((slot - 1) / width) + 1
    return x, y
end

local function cell_distance(node, a, b)
    local ax, ay = slot_to_xy(node, a)
    local bx, by = slot_to_xy(node, b)
    local dx = ax - bx
    local dy = ay - by
    return math.sqrt(dx * dx + dy * dy)
end

local function average_vectors(vectors)
    local avg = tool.average_vectors(vectors)
    if not avg then return {} end
    return normalize_vector(avg)
end

local function new_node(id, parent_id, parent_slot, level, seed_vec)
    local node = {
        id = id,
        parent_id = parent_id,
        parent_slot = parent_slot or 0,
        level = level or 1,
        width = MAP_WIDTH,
        height = MAP_HEIGHT,
        neurons = {},
        children = {},
        members = {},
        member_pos = {},
        hot_members = {},
        hot_pos = {},
        cold_members = {},
        cold_pos = {},
        slot_members = {},
        slot_pos = {},
        line_to_slot = {},
        line_similarity = {},
        slot_count = {},
        slot_sim_sum = {},
        centroid = {},
        member_subtree_count = 0,
        hot_subtree_count = 0,
        initialized = false,
    }

    local cells = node_cell_count(node)
    for slot = 1, cells do
        node.neurons[slot] = {}
        node.children[slot] = 0
        node.slot_members[slot] = {}
        node.slot_pos[slot] = {}
        node.slot_count[slot] = 0
        node.slot_sim_sum[slot] = 0.0
    end

    if seed_vec and #seed_vec > 0 then
        local base = normalize_vector(seed_vec)
        for slot = 1, cells do
            node.neurons[slot] = perturb_vector(base, 0.045 + (slot - 1) * 0.004)
        end
        node.centroid = average_vectors(node.neurons)
        node.initialized = true
    end

    return node
end

local function refresh_node_centroid(node)
    local vectors = {}
    for slot = 1, node_cell_count(node) do
        local neuron = node.neurons[slot]
        if neuron and #neuron > 0 then
            vectors[#vectors + 1] = neuron
        end
    end
    node.centroid = average_vectors(vectors)
end

local function ensure_seed(node, seed_vec)
    if node.initialized or (not seed_vec) or #seed_vec <= 0 then
        return
    end
    local base = normalize_vector(seed_vec)
    for slot = 1, node_cell_count(node) do
        node.neurons[slot] = perturb_vector(base, 0.045 + (slot - 1) * 0.004)
    end
    refresh_node_centroid(node)
    node.initialized = true
end

local function find_bmu(node, vec)
    ensure_seed(node, vec)

    local best_slot = 1
    local best_sim = -1.0
    for slot = 1, node_cell_count(node) do
        local sim = safe_similarity(vec, node.neurons[slot])
        if sim > best_sim then
            best_sim = sim
            best_slot = slot
        end
    end
    return best_slot, best_sim
end

local function update_neurons(node, vec, bmu_slot)
    ensure_seed(node, vec)

    local alpha = LEARNING_RATE / math.sqrt(1.0 + math.max(0, #node.members) / 24.0 + (node.level - 1) * 0.35)
    local sigma = math.max(0.7, NEIGHBOR_RADIUS)
    for slot = 1, node_cell_count(node) do
        local dist = cell_distance(node, slot, bmu_slot)
        if dist <= NEIGHBOR_RADIUS then
            local influence = math.exp(-(dist * dist) / (2 * sigma * sigma))
            local lr = alpha * influence
            local neuron = node.neurons[slot]
            for i = 1, #vec do
                neuron[i] = (tonumber(neuron[i]) or 0.0) + lr * ((tonumber(vec[i]) or 0.0) - (tonumber(neuron[i]) or 0.0))
            end
            node.neurons[slot] = normalize_vector(neuron)
        end
    end
    refresh_node_centroid(node)
end

local function add_line_to_activity_set(line)
    if M.active_pos[line] then return end
    M.active_lines[#M.active_lines + 1] = line
    M.active_pos[line] = #M.active_lines
end

local function remove_line_from_activity_set(line)
    local pos = M.active_pos[line]
    if not pos then return end

    local last = #M.active_lines
    if pos ~= last then
        local moved = M.active_lines[last]
        M.active_lines[pos] = moved
        M.active_pos[moved] = pos
    end
    M.active_lines[last] = nil
    M.active_pos[line] = nil
end

local function sync_node_hot_state(node, line, is_hot)
    if is_hot then
        remove_member(node.cold_members, node.cold_pos, line)
        add_member(node.hot_members, node.hot_pos, line)
        add_line_to_activity_set(line)
    else
        remove_member(node.hot_members, node.hot_pos, line)
        add_member(node.cold_members, node.cold_pos, line)
        remove_line_from_activity_set(line)
    end
end

local function recompute_subtree_counts(node)
    local total_members = #(node.members or {})
    local total_hot = #(node.hot_members or {})
    for slot = 1, node_cell_count(node) do
        local child = M.nodes[tonumber(node.children[slot]) or 0]
        if child then
            total_members = total_members + (tonumber(child.member_subtree_count) or 0)
            total_hot = total_hot + (tonumber(child.hot_subtree_count) or 0)
        end
    end
    node.member_subtree_count = total_members
    node.hot_subtree_count = total_hot
end

local function bubble_subtree_counts(node)
    local current = node
    while current do
        recompute_subtree_counts(current)
        current = M.nodes[tonumber(current.parent_id) or 0]
    end
end

local function detach_line(node, line)
    local slot = tonumber(node.line_to_slot[line])
    if not slot then
        return false, nil, 0.0
    end

    remove_member(node.members, node.member_pos, line)
    remove_member(node.slot_members[slot], node.slot_pos[slot], line)
    remove_member(node.hot_members, node.hot_pos, line)
    remove_member(node.cold_members, node.cold_pos, line)

    local sim = tonumber(node.line_similarity[line]) or 0.0
    node.slot_count[slot] = math.max(0, (tonumber(node.slot_count[slot]) or 0) - 1)
    node.slot_sim_sum[slot] = (tonumber(node.slot_sim_sum[slot]) or 0.0) - sim
    if node.slot_count[slot] <= 0 then
        node.slot_count[slot] = 0
        node.slot_sim_sum[slot] = 0.0
    end

    node.line_to_slot[line] = nil
    node.line_similarity[line] = nil
    M.line_to_node[line] = nil
    return true, slot, sim
end

local function attach_line(node, line, slot, similarity)
    add_member(node.members, node.member_pos, line)
    add_member(node.slot_members[slot], node.slot_pos[slot], line)
    node.line_to_slot[line] = slot
    node.line_similarity[line] = tonumber(similarity) or 0.0
    node.slot_count[slot] = (node.slot_count[slot] or 0) + 1
    node.slot_sim_sum[slot] = (node.slot_sim_sum[slot] or 0.0) + (tonumber(similarity) or 0.0)
    M.line_to_node[line] = node.id

    sync_node_hot_state(node, line, M.active_pos[line] ~= nil)
end

local function sorted_node_ids()
    local ids = {}
    for id in pairs(M.nodes) do
        ids[#ids + 1] = tonumber(id)
    end
    table.sort(ids)
    return ids
end

local function scannable_node_ids()
    local ids = {}
    for id, node in pairs(M.nodes) do
        if #(node.members or {}) > 0 then
            ids[#ids + 1] = tonumber(id)
        end
    end
    table.sort(ids)
    return ids
end

local function reset_state()
    M.root_id = nil
    M.nodes = {}
    M.line_to_node = {}
    M.active_lines = {}
    M.active_pos = {}
    M.next_node_id = 1
    M.last_decay_turn = 0
    M.state_dirty = false
end

local function create_root()
    local root = new_node(M.next_node_id, 0, 0, 1, nil)
    M.nodes[root.id] = root
    M.root_id = root.id
    M.next_node_id = root.id + 1
    M.state_dirty = true
    return root
end

local function ensure_root()
    local root = M.root_id and M.nodes[M.root_id] or nil
    if root then
        return root
    end
    return create_root()
end

local function route_path_from(start_node, vec)
    local path = {}
    local current = start_node

    while current do
        local slot, sim = find_bmu(current, vec)
        path[#path + 1] = {
            node = current,
            slot = slot,
            similarity = sim,
        }

        local child_id = tonumber(current.children[slot]) or 0
        if child_id > 0 and M.nodes[child_id] then
            current = M.nodes[child_id]
        else
            break
        end
    end

    return path
end

local function insert_into_subtree(start_node, vec, mem_index)
    local v = normalize_vector(vector_to_table(vec))
    if #v <= 0 then
        return nil, {}
    end

    local path = route_path_from(start_node, v)
    if #path <= 0 then
        return nil, {}
    end

    for _, step in ipairs(path) do
        update_neurons(step.node, v, step.slot)
    end

    local leaf = path[#path]
    attach_line(leaf.node, mem_index, leaf.slot, leaf.similarity)
    return leaf.node.id, path
end

local function rehome_slot_members(node, slot, child)
    local memory = get_memory()
    local moving = shallow_copy(node.slot_members[slot] or {})
    for _, line in ipairs(moving) do
        local ok_detach, old_slot, old_sim = detach_line(node, line)
        if ok_detach then
            local vec = memory.return_mem_vec(line)
            if vec then
                insert_into_subtree(child, vec, line)
            else
                attach_line(node, line, old_slot or slot, old_sim)
            end
        end
    end
    bubble_subtree_counts(node)
    bubble_subtree_counts(child)
end

local function maybe_split(node, slot)
    if not node then return nil end
    if node.level >= MAX_LEVEL then return nil end
    if (node.children[slot] or 0) > 0 then return node.children[slot] end

    local count = tonumber(node.slot_count[slot]) or 0
    if count < MIN_SPLIT_MEMBERS then
        return nil
    end

    local avg_sim = (tonumber(node.slot_sim_sum[slot]) or 0.0) / math.max(1, count)
    if avg_sim >= SPLIT_SIM_THRESHOLD then
        return nil
    end

    local seed = node.neurons[slot]
    local child = new_node(M.next_node_id, node.id, slot, node.level + 1, seed)
    M.nodes[child.id] = child
    M.next_node_id = child.id + 1
    node.children[slot] = child.id
    M.state_dirty = true

    rehome_slot_members(node, slot, child)
    for child_slot = 1, node_cell_count(child) do
        maybe_split(child, child_slot)
    end

    print(string.format(
        "[GHSOM] split node %d slot %d -> child %d (count=%d avg_sim=%.4f)",
        node.id, slot, child.id, count, avg_sim
    ))
    return child.id
end

local function route_path(vec)
    return route_path_from(ensure_root(), vec)
end

local function insert_vector(vec, mem_index)
    local v = normalize_vector(vector_to_table(vec))
    if #v <= 0 then
        return nil
    end

    local path = route_path(v)
    if #path <= 0 then
        return nil
    end

    for _, step in ipairs(path) do
        update_neurons(step.node, v, step.slot)
    end

    local leaf = path[#path]
    attach_line(leaf.node, mem_index, leaf.slot, leaf.similarity)
    maybe_split(leaf.node, leaf.slot)
    local final_node = M.nodes[tonumber(M.line_to_node[mem_index]) or leaf.node.id] or leaf.node
    bubble_subtree_counts(final_node)
    M.state_dirty = true
    return tonumber(M.line_to_node[mem_index]) or leaf.node.id, path
end

local function rebuild_runtime_views()
    M.line_to_node = {}
    M.active_lines = {}
    M.active_pos = {}

    local memory = get_memory()
    for _, node_id in ipairs(sorted_node_ids()) do
        local node = M.nodes[node_id]
        node.member_pos = {}
        node.hot_members = {}
        node.hot_pos = {}
        node.cold_members = {}
        node.cold_pos = {}
        node.line_similarity = {}

        local cells = node_cell_count(node)
        for slot = 1, cells do
            node.slot_members[slot] = {}
            node.slot_pos[slot] = {}
            node.slot_count[slot] = 0
            node.slot_sim_sum[slot] = 0.0
        end

        for i, line in ipairs(node.members or {}) do
            local slot = tonumber(node.line_to_slot[line]) or 1
            if slot < 1 or slot > cells then
                slot = 1
                node.line_to_slot[line] = slot
            end
            add_member(node.slot_members[slot], node.slot_pos[slot], line)
            node.member_pos[line] = i
            M.line_to_node[line] = node.id

            local mem_vec = memory.return_mem_vec(line)
            local sim = mem_vec and safe_similarity(mem_vec, node.neurons[slot]) or 0.0
            node.line_similarity[line] = sim
            node.slot_count[slot] = (node.slot_count[slot] or 0) + 1
            node.slot_sim_sum[slot] = (node.slot_sim_sum[slot] or 0.0) + sim

            if M.active_pos[line] then
                add_member(node.hot_members, node.hot_pos, line)
                add_line_to_activity_set(line)
            else
                add_member(node.cold_members, node.cold_pos, line)
            end
        end

        refresh_node_centroid(node)
        node.initialized = node.initialized or (#(node.centroid or {}) > 0)
    end

    local ids = sorted_node_ids()
    table.sort(ids, function(a, b)
        local na = M.nodes[a]
        local nb = M.nodes[b]
        if (na.level or 0) ~= (nb.level or 0) then
            return (na.level or 0) > (nb.level or 0)
        end
        return a > b
    end)
    for _, node_id in ipairs(ids) do
        recompute_subtree_counts(M.nodes[node_id])
    end
end

local function rebuild_from_memory()
    reset_state()
    create_root()

    local memory = get_memory()
    local total = memory.get_total_lines()
    for line = 1, total do
        local vec = memory.return_mem_vec(line)
        if vec then
            insert_vector(vec, line)
        end
    end

    rebuild_runtime_views()
    print(string.format("[GHSOM] rebuilt from memory vectors: nodes=%d memories=%d", #sorted_node_ids(), total))
end

local function save_record(node)
    local children_n = node_cell_count(node)
    local neuron_bins = {}
    local member_count = #(node.members or {})
    local record_size = 4 + 4 + 4 + 2 + 2 + 2 + 2 + 4 + (children_n * 4) + (member_count * 6)

    for slot = 1, children_n do
        local neuron = normalize_vector(node.neurons[slot] or {})
        node.neurons[slot] = neuron
        local bin = tool.vector_to_bin(neuron)
        neuron_bins[slot] = bin
        record_size = record_size + #bin
    end

    local buf = ffi.new("uint8_t[?]", record_size)
    local base = 0

    ffi.cast("uint32_t*", buf + base)[0] = record_size
    base = base + 4

    ffi.cast("uint32_t*", buf + base)[0] = node.id
    base = base + 4

    ffi.cast("int32_t*", buf + base)[0] = math.floor(tonumber(node.parent_id) or 0)
    base = base + 4

    ffi.cast("uint16_t*", buf + base)[0] = math.floor(tonumber(node.parent_slot) or 0)
    base = base + 2

    ffi.cast("uint16_t*", buf + base)[0] = math.floor(tonumber(node.level) or 1)
    base = base + 2

    ffi.cast("uint16_t*", buf + base)[0] = math.floor(tonumber(node.width) or MAP_WIDTH)
    base = base + 2

    ffi.cast("uint16_t*", buf + base)[0] = math.floor(tonumber(node.height) or MAP_HEIGHT)
    base = base + 2

    ffi.cast("uint32_t*", buf + base)[0] = member_count
    base = base + 4

    local child_ptr = ffi.cast("int32_t*", buf + base)
    for slot = 1, children_n do
        child_ptr[slot - 1] = math.floor(tonumber(node.children[slot]) or 0)
    end
    base = base + children_n * 4

    for slot = 1, children_n do
        local chunk = neuron_bins[slot] or ""
        if #chunk > 0 then
            ffi.copy(buf + base, chunk, #chunk)
            base = base + #chunk
        end
    end

    for _, line in ipairs(node.members or {}) do
        ffi.cast("uint32_t*", buf + base)[0] = math.floor(tonumber(line) or 0)
        base = base + 4
        ffi.cast("uint16_t*", buf + base)[0] = math.floor(tonumber(node.line_to_slot[line]) or 1)
        base = base + 2
    end

    return ffi.string(buf, record_size)
end

local function load_state_from_disk()
    local f = io.open(GHSOM_PATH, "rb")
    if not f then
        return false, "missing"
    end

    local data = f:read("*a")
    f:close()

    if (not data) or #data < 24 or data:sub(1, 4) ~= MAGIC then
        return false, "header"
    end

    local p = ffi.cast("const uint8_t*", data)
    local header = ffi.cast("const uint32_t*", p + 4)
    local version = tonumber(header[0]) or 0
    if version ~= VERSION then
        return false, "version"
    end

    reset_state()
    M.next_node_id = math.max(1, tonumber(header[1]) or 1)
    M.root_id = tonumber(header[2]) or 0
    local node_count = tonumber(header[3]) or 0
    M.last_decay_turn = math.max(0, tonumber(header[4]) or 0)

    local offset = 24
    for _ = 1, node_count do
        if offset + 4 > #data then
            return false, "record_truncated"
        end

        local rec_len = tonumber(ffi.cast("const uint32_t*", p + offset)[0]) or 0
        if rec_len < 24 or (offset + rec_len) > #data then
            return false, "record_size"
        end

        local base = offset + 4
        local id = tonumber(ffi.cast("const uint32_t*", p + base)[0]) or 0
        base = base + 4

        local parent_id = tonumber(ffi.cast("const int32_t*", p + base)[0]) or 0
        base = base + 4

        local parent_slot = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 0
        base = base + 2

        local level = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 1
        base = base + 2

        local width = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or MAP_WIDTH
        base = base + 2

        local height = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or MAP_HEIGHT
        base = base + 2

        local member_count = tonumber(ffi.cast("const uint32_t*", p + base)[0]) or 0
        base = base + 4

        local node = new_node(id, parent_id, parent_slot, level, nil)
        node.width = math.max(2, width)
        node.height = math.max(2, height)

        local cells = node_cell_count(node)
        node.children = {}
        node.slot_members = {}
        node.slot_pos = {}
        node.slot_count = {}
        node.slot_sim_sum = {}
        for slot = 1, cells do
            node.slot_members[slot] = {}
            node.slot_pos[slot] = {}
            node.slot_count[slot] = 0
            node.slot_sim_sum[slot] = 0.0
        end

        local child_ptr = ffi.cast("const int32_t*", p + base)
        for slot = 1, cells do
            node.children[slot] = tonumber(child_ptr[slot - 1]) or 0
        end
        base = base + cells * 4

        for slot = 1, cells do
            local neuron, used = tool.bin_to_vector(data, base)
            if not neuron or used <= 0 then
                return false, "neuron_vector"
            end
            node.neurons[slot] = normalize_vector(neuron)
            base = base + used
        end

        node.members = {}
        node.line_to_slot = {}
        for i = 1, member_count do
            if base + 6 > offset + rec_len then
                return false, "member_record"
            end
            local line = tonumber(ffi.cast("const uint32_t*", p + base)[0]) or 0
            base = base + 4
            local slot = tonumber(ffi.cast("const uint16_t*", p + base)[0]) or 1
            base = base + 2
            if line > 0 then
                node.members[#node.members + 1] = line
                node.line_to_slot[line] = clamp(slot, 1, cells)
                node.slot_count[node.line_to_slot[line]] = (node.slot_count[node.line_to_slot[line]] or 0) + 1
            end
        end

        refresh_node_centroid(node)
        node.initialized = true
        M.nodes[id] = node
        offset = offset + rec_len
    end

    if not M.root_id or not M.nodes[M.root_id] then
        return false, "root_missing"
    end

    rebuild_runtime_views()
    M.state_dirty = false
    return true
end

function M.load()
    ensure_dir(ROOT_DIR)

    local ok, err = load_state_from_disk()
    if not ok then
        print(string.format("[GHSOM] load fallback (%s), rebuilding from memory", tostring(err)))
        rebuild_from_memory()
    end

    ensure_root()
    print(string.format(
        "[GHSOM] ready: nodes=%d active=%d last_decay_turn=%d",
        #scannable_node_ids(),
        #M.active_lines,
        M.last_decay_turn
    ))
end

function M.save_to_disk()
    ensure_dir(ROOT_DIR)

    local ids = sorted_node_ids()
    local ok, err = persistence.write_atomic(GHSOM_PATH, "wb", function(f)
        local header = ffi.new("uint32_t[5]", VERSION, M.next_node_id, M.root_id or 0, #ids, M.last_decay_turn or 0)

        local ok0, err0 = f:write(MAGIC)
        if not ok0 then return false, err0 end

        local ok1, err1 = f:write(ffi.string(header, 20))
        if not ok1 then return false, err1 end

        for _, id in ipairs(ids) do
            local node = M.nodes[id]
            local blob = save_record(node)
            local okw, errw = f:write(blob)
            if not okw then
                return false, errw
            end
        end

        return true
    end)

    if ok then
        M.state_dirty = false
    end
    return ok, err
end

function M.find_best_node(vec)
    local v = normalize_vector(vector_to_table(vec))
    if #v <= 0 then
        return nil, {}, 0.0
    end

    local path = route_path(v)
    if #path <= 0 then
        return nil, {}, 0.0
    end

    local ids = {}
    for _, step in ipairs(path) do
        ids[#ids + 1] = step.node.id
    end
    return path[#path].node.id, ids, tonumber(path[#path].similarity) or 0.0
end

function M.add(vec, mem_index)
    if type(vec) ~= "table" or #vec <= 0 then
        return nil
    end
    return insert_vector(vec, mem_index)
end

function M.get_node_for_line(mem_line)
    return tonumber(M.line_to_node[mem_line])
end

function M.get_hierarchical_path(mem_line)
    local out = {}
    local current = tonumber(M.line_to_node[mem_line]) or 0
    while current > 0 and M.nodes[current] do
        table.insert(out, 1, current)
        current = tonumber(M.nodes[current].parent_id) or 0
    end
    return out
end

function M.get_node_ids()
    return scannable_node_ids()
end

function M.node_count()
    return #scannable_node_ids()
end

function M.probe_nodes(vec, max_results, opts)
    opts = opts or {}
    local limit = math.max(1, tonumber(max_results) or 4)
    local only_hot = opts.only_hot == true
    local include_empty = opts.include_empty == true
    local beam_width = math.max(limit, tonumber(opts.beam_width) or math.max(3, limit * 2))

    local v = normalize_vector(vector_to_table(vec))
    if #v <= 0 then
        return {}
    end

    local collected = {}
    local seen = {}
    local root = ensure_root()
    local frontier = {
        {
            node = root,
            similarity = safe_similarity(v, root.centroid),
        }
    }

    while #frontier > 0 do
        local next_frontier = {}

        for _, state in ipairs(frontier) do
            local node = state.node
            local members_n = #(node.members or {})
            local hot_n = #(node.hot_members or {})
            local subtree_members = tonumber(node.member_subtree_count) or members_n
            local subtree_hot = tonumber(node.hot_subtree_count) or hot_n

            if members_n > 0 and (include_empty or members_n > 0) and ((not only_hot) or hot_n > 0) and not seen[node.id] then
                seen[node.id] = true
                topk_push(collected, {
                    id = node.id,
                    similarity = tonumber(state.similarity) or safe_similarity(v, node.centroid),
                    hot = hot_n,
                    size = members_n,
                }, limit)
            end

            for slot = 1, node_cell_count(node) do
                local child_id = tonumber(node.children[slot]) or 0
                local child = child_id > 0 and M.nodes[child_id] or nil
                if child then
                    local child_members = tonumber(child.member_subtree_count) or #(child.members or {})
                    local child_hot = tonumber(child.hot_subtree_count) or #(child.hot_members or {})
                    if child_members > 0 and (include_empty or child_members > 0) and ((not only_hot) or child_hot > 0) then
                        topk_push(next_frontier, {
                            node = child,
                            similarity = safe_similarity(v, child.centroid),
                        }, beam_width)
                    end
                end
            end
        end

        frontier = topk_to_desc(next_frontier)
    end

    return topk_to_desc(collected)
end

function M.find_sim_in_node(vec, node_id, opts)
    opts = opts or {}
    local memory = get_memory()
    local node = M.nodes[tonumber(node_id)]
    if not node then return {} end

    local v = normalize_vector(vector_to_table(vec))
    if #v <= 0 then return {} end

    local only_hot = opts.only_hot == true
    local only_cold = opts.only_cold == true
    if only_hot and only_cold then
        return {}
    end

    local members = node.members or {}
    if only_hot then
        members = node.hot_members or {}
    elseif only_cold then
        members = node.cold_members or {}
    end

    local topk = {}
    local limit = tonumber(opts.max_results)
    if limit then
        limit = math.max(1, math.floor(limit))
    end

    for _, line in ipairs(members) do
        local mem_vec = memory.return_mem_vec(line)
        if mem_vec then
            local sim = safe_similarity(v, mem_vec)
            if limit then
                topk_push(topk, { index = line, similarity = sim }, limit)
            else
                topk[#topk + 1] = { index = line, similarity = sim }
            end
        end
    end

    if limit then
        return topk_to_desc(topk)
    end

    table.sort(topk, function(a, b)
        return (a.similarity or -1) > (b.similarity or -1)
    end)
    return topk
end

M.find_similar_in_node = M.find_sim_in_node

function M.on_memory_heat_change(mem_line, old_heat, new_heat)
    return tonumber(M.line_to_node[mem_line]), false
end

function M.is_hot(line)
    return M.active_pos[line] ~= nil
end

local function set_line_activation(mem_line, is_active)
    local idx = tonumber(mem_line)
    if not idx or idx <= 0 then
        return false
    end

    local node_id = M.line_to_node[idx]
    if not node_id or not M.nodes[node_id] then
        if is_active then
            add_line_to_activity_set(idx)
        else
            remove_line_from_activity_set(idx)
        end
        return true
    end

    local node = M.nodes[node_id]
    local was_active = node.hot_pos[idx] ~= nil
    if was_active == is_active then
        return false
    end

    sync_node_hot_state(node, idx, is_active)
    bubble_subtree_counts(node)
    return true
end

function M.get_active_lines()
    return shallow_copy(M.active_lines)
end

function M.activate_lines(lines, opts)
    opts = opts or {}
    local mode = tostring(opts.mode or "replace")
    local target = {}
    for _, mem_line in ipairs(lines or {}) do
        local idx = tonumber(mem_line)
        if idx and idx > 0 then
            target[idx] = true
        end
    end

    local changed = 0
    if mode == "replace" then
        local snapshot = shallow_copy(M.active_lines)
        for _, mem_line in ipairs(snapshot) do
            if not target[mem_line] and set_line_activation(mem_line, false) then
                changed = changed + 1
            end
        end
    elseif mode == "remove" then
        for mem_line in pairs(target) do
            if set_line_activation(mem_line, false) then
                changed = changed + 1
            end
        end
        if changed > 0 then
            M.state_dirty = true
        end
        return changed
    elseif mode ~= "append" then
        mode = "replace"
    end

    for mem_line in pairs(target) do
        if set_line_activation(mem_line, true) then
            changed = changed + 1
        end
    end

    if changed > 0 then
        M.state_dirty = true
    end
    return changed
end

local function normalize_score_map(src)
    local out = {}
    local max_score = 0.0
    for key, score in pairs(src or {}) do
        local s = math.max(0.0, tonumber(score) or 0.0)
        if s > 0.0 then
            out[key] = s
            if s > max_score then
                max_score = s
            end
        end
    end
    if max_score > 0.0 then
        for key, score in pairs(out) do
            out[key] = score / max_score
        end
    end
    return out
end

function M.plan_probe_budget(vec, opts)
    opts = opts or {}
    local max_nodes = math.max(1, math.floor(tonumber(opts.max_nodes) or 3))
    local total_scan_budget = math.max(max_nodes, math.floor(tonumber(opts.total_scan_budget) or max_nodes * 8))
    local per_node_floor = math.max(1, math.floor(tonumber(opts.per_node_floor) or 2))
    local prior_scale = math.max(0.0, tonumber(opts.prior_scale) or 0.75)
    local activation_bonus = math.max(0.0, tonumber(opts.activation_bonus) or 0.18)
    local semantic_limit = math.max(max_nodes, math.floor(tonumber(opts.semantic_limit) or (max_nodes * 4)))
    local predicted_nodes = normalize_score_map(opts.predicted_nodes or {})

    local semantic_nodes = M.probe_nodes(vec, semantic_limit, { only_hot = false, include_empty = false })
    local scored = {}
    local seen = {}

    local function ensure_node_entry(node_id)
        local cid = tonumber(node_id)
        if not cid or cid <= 0 then
            return nil
        end
        local entry = scored[cid]
        if not entry then
            local node = M.nodes[cid]
            if not node or #(node.members or {}) <= 0 then
                return nil
            end
            entry = {
                id = cid,
                similarity = 0.0,
                prior = 0.0,
                active_members = #(node.hot_members or {}),
                member_count = #(node.members or {}),
            }
            scored[cid] = entry
            seen[#seen + 1] = cid
        end
        return entry
    end

    for _, item in ipairs(semantic_nodes) do
        local entry = ensure_node_entry(item.id)
        if entry then
            entry.similarity = math.max(entry.similarity, tonumber(item.similarity) or 0.0)
        end
    end
    for cid, prior in pairs(predicted_nodes) do
        local entry = ensure_node_entry(cid)
        if entry then
            entry.prior = math.max(entry.prior, tonumber(prior) or 0.0)
        end
    end

    local ranked = {}
    for _, cid in ipairs(seen) do
        local entry = scored[cid]
        if entry then
            local combined = math.max(0.0, entry.similarity)
                + prior_scale * math.max(0.0, entry.prior)
                + ((entry.active_members or 0) > 0 and activation_bonus or 0.0)
            entry.score = combined
            ranked[#ranked + 1] = entry
        end
    end

    table.sort(ranked, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return (a.id or 0) < (b.id or 0)
    end)

    while #ranked > max_nodes do
        table.remove(ranked)
    end
    if #ranked <= 0 then
        return {}
    end

    local base_need = per_node_floor * #ranked
    local remaining = math.max(0, total_scan_budget - base_need)
    local total_score = 0.0
    for _, entry in ipairs(ranked) do
        total_score = total_score + math.max(1e-6, tonumber(entry.score) or 0.0)
    end

    for _, entry in ipairs(ranked) do
        local extra = 0
        if remaining > 0 and total_score > 0 then
            extra = math.floor(remaining * (math.max(1e-6, tonumber(entry.score) or 0.0) / total_score) + 0.5)
        end
        entry.scan_limit = per_node_floor + extra
        entry.prefer_active = (tonumber(entry.prior) or 0.0) > 0.0 or (tonumber(entry.active_members) or 0) > 0
    end

    return ranked
end

function M.maintain_activity(current_turn)
    local turn = math.max(0, math.floor(tonumber(current_turn) or M.last_decay_turn or 0))
    if turn > M.last_decay_turn then
        M.last_decay_turn = turn
        M.state_dirty = true
    end
    return 0
end

function M.touch_line(mem_line, boost, current_turn)
    M.maintain_activity(current_turn)
    local idx = tonumber(mem_line)
    if not idx or idx <= 0 then
        return false
    end
    M.activate_lines({ idx }, { mode = "append" })
    return true
end

function M.add_new_memory(mem_line, vec, current_turn)
    return M.touch_line(mem_line, NEW_HEAT, current_turn)
end

local function collect_neighbor_candidates(node, slot, target_line)
    local candidates = {}
    local seen = {}
    if not node then return candidates end

    for other_slot = 1, node_cell_count(node) do
        if cell_distance(node, slot, other_slot) <= math.max(1, NEIGHBOR_RADIUS) then
            for _, line in ipairs(node.slot_members[other_slot] or {}) do
                if line ~= target_line and not seen[line] then
                    seen[line] = true
                    candidates[#candidates + 1] = line
                end
            end
        end
    end

    if #candidates <= 0 and (tonumber(node.parent_id) or 0) > 0 then
        local parent = M.nodes[node.parent_id]
        local parent_slot = tonumber(node.parent_slot) or 1
        if parent then
            for other_slot = 1, node_cell_count(parent) do
                if cell_distance(parent, parent_slot, other_slot) <= math.max(1, NEIGHBOR_RADIUS) then
                    for _, line in ipairs(parent.slot_members[other_slot] or {}) do
                        if line ~= target_line and not seen[line] then
                            seen[line] = true
                            candidates[#candidates + 1] = line
                        end
                    end
                end
            end
        end
    end

    return candidates
end

function M.neighbors_add_heat(vec, current_turn, target_mem_line)
    return 0
end

function M.find_cold_candidates(vec, opts)
    opts = opts or {}
    local probe_limit = math.max(1, tonumber(opts.max_nodes) or REACTIVATE_SCAN_NODES)
    local result_limit = math.max(1, tonumber(opts.max_results) or REACTIVATE_SCAN_LIMIT)
    local probe = M.probe_nodes(vec, probe_limit, { only_hot = false, include_empty = false })

    local topk = {}
    for _, node_item in ipairs(probe) do
        local results = M.find_sim_in_node(vec, node_item.id, {
            only_cold = true,
            max_results = result_limit,
        })
        for _, item in ipairs(results) do
            topk_push(topk, item, result_limit)
        end
    end

    return topk_to_desc(topk)
end

function M.reactivate_cold_candidates(vec, current_turn, opts)
    return 0, {}
end

function M.print_tree()
    local function walk(node_id, indent)
        local node = M.nodes[node_id]
        if not node then return end

        print(string.format(
            "%snode=%d level=%d members=%d hot=%d",
            string.rep("  ", indent),
            node.id,
            node.level,
            #(node.members or {}),
            #(node.hot_members or {})
        ))

        for slot = 1, node_cell_count(node) do
            local child_id = tonumber(node.children[slot]) or 0
            if child_id > 0 then
                walk(child_id, indent + 1)
            end
        end
    end

    if M.root_id and M.nodes[M.root_id] then
        walk(M.root_id, 0)
    else
        print("[GHSOM] empty tree")
    end
end

return M
