local hnsw = require("module.hnsw")
local util = require("mori_memory.util")
local persistence = require("module.persistence")

local M = {}

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function trim(text)
    return tostring(text or ""):match("^%s*(.-)%s*$")
end

local function copy_vec(vec)
    local out = {}
    for i = 1, #(vec or {}) do
        out[i] = tonumber(vec[i]) or 0.0
    end
    return out
end

local function any_dim(centroids)
    for _, vec in pairs(centroids or {}) do
        if type(vec) == "table" and #vec > 0 then
            return #vec
        end
    end
    return 0
end

local Index = {}
Index.__index = Index

function M.new(opts)
    opts = type(opts) == "table" and opts or {}
    return setmetatable({
        opts = opts,
        centroids = {},
        label_by_anchor = {},
        anchor_by_label = {},
        next_label = 1,
        dirty = true,
        index = nil,
    }, Index)
end

function Index:close()
    if self.index and self.index.close then
        self.index:close()
    end
    self.index = nil
end

function Index:_ensure_label(anchor)
    anchor = trim(anchor)
    if anchor == "" then
        return nil
    end
    local label = tonumber(self.label_by_anchor[anchor])
    if label and label > 0 then
        return label
    end
    label = tonumber(self.next_label) or 1
    self.next_label = label + 1
    self.label_by_anchor[anchor] = label
    self.anchor_by_label[label] = anchor
    return label
end

function Index:set_centroids(centroids)
    self.centroids = {}
    for anchor, vec in pairs(centroids or {}) do
        local key = trim(anchor)
        if key ~= "" and type(vec) == "table" and #vec > 0 then
            self.centroids[key] = copy_vec(vec)
            self:_ensure_label(key)
        end
    end
    self.dirty = true
end

function Index:update(anchor, centroid)
    anchor = trim(anchor)
    if anchor == "" or type(centroid) ~= "table" or #centroid <= 0 then
        return
    end
    self:_ensure_label(anchor)
    self.centroids[anchor] = copy_vec(centroid)
    self.dirty = true
end

function Index:remove(anchor)
    anchor = trim(anchor)
    if anchor == "" then
        return
    end
    self.centroids[anchor] = nil
    self.dirty = true
end

function Index:_build()
    self:close()
    local enabled = (((self.opts or {}).enabled) ~= false)
    if not enabled then
        self.dirty = false
        return true
    end

    local dim = any_dim(self.centroids)
    local count = 0
    for _ in pairs(self.centroids or {}) do
        count = count + 1
    end
    if dim <= 0 or count <= 0 then
        self.dirty = false
        return true
    end

    local idx, err = hnsw.new({
        dim = dim,
        max_elements = math.max(count * 2, tonumber((self.opts or {}).max_elements) or 128),
        m = tonumber((self.opts or {}).m) or 16,
        ef_construction = tonumber((self.opts or {}).ef_construction) or 80,
        ef_search = tonumber((self.opts or {}).ef_search) or 32,
        space = hnsw.SPACE_COSINE,
    })
    if not idx then
        return nil, err
    end

    for anchor, vec in pairs(self.centroids or {}) do
        local label = self:_ensure_label(anchor)
        local ok, add_err = idx:add(label, vec)
        if not ok then
            idx:close()
            return nil, add_err
        end
    end

    self.index = idx
    self.dirty = false
    return true
end

function Index:search(query_vec, k)
    if self.dirty then
        local ok = self:_build()
        if not ok then
            return {}
        end
    end
    if not self.index then
        return {}
    end
    local hits = self.index:search(query_vec or {}, math.max(1, math.floor(tonumber(k) or 1)))
    local out = {}
    for _, hit in ipairs(hits or {}) do
        local label = tonumber(hit.label)
        local anchor = self.anchor_by_label[label]
        if anchor and anchor ~= "" then
            out[#out + 1] = {
                anchor = anchor,
                label = label,
                similarity = tonumber(hit.similarity) or 0.0,
            }
        end
    end
    return out
end

function Index:save(root)
    root = trim(root)
    if root == "" then
        return true
    end
    ensure_dir(root)
    if self.dirty then
        local ok, err = self:_build()
        if not ok then
            return false, err
        end
    end

    local meta_path = root .. "/meta.lua"
    local ok_meta, err_meta = persistence.write_atomic(meta_path, "w", function(f)
        return f:write(util.encode_lua_value({
            label_by_anchor = self.label_by_anchor,
            anchor_by_label = self.anchor_by_label,
            next_label = tonumber(self.next_label) or 1,
        }, 0))
    end)
    if not ok_meta then
        return false, err_meta
    end

    if self.index then
        local ok_idx, err_idx = self.index:save(root .. "/index.bin")
        if not ok_idx then
            return false, err_idx
        end
    end
    return true
end

function Index:load(root)
    root = trim(root)
    if root == "" then
        return true
    end
    local meta_path = root .. "/meta.lua"
    local f = io.open(meta_path, "r")
    if f then
        local raw = f:read("*a")
        f:close()
        local parsed = util.parse_lua_table_literal(raw or "")
        if type(parsed) == "table" then
            self.label_by_anchor = type(parsed.label_by_anchor) == "table" and parsed.label_by_anchor or {}
            self.anchor_by_label = type(parsed.anchor_by_label) == "table" and parsed.anchor_by_label or {}
            self.next_label = math.max(1, math.floor(tonumber(parsed.next_label) or 1))
        end
    end

    local dim = any_dim(self.centroids)
    local idx_path = root .. "/index.bin"
    local exists = io.open(idx_path, "rb")
    if exists then
        exists:close()
    else
        self.dirty = true
        return true
    end
    if dim <= 0 then
        self.dirty = true
        return true
    end

    local idx, err = hnsw.load(idx_path, {
        dim = dim,
        ef_search = tonumber((self.opts or {}).ef_search) or 32,
        space = hnsw.SPACE_COSINE,
        max_elements = math.max(tonumber((self.opts or {}).max_elements) or 128, 16),
    })
    if not idx then
        self.dirty = true
        return nil, err
    end
    self:close()
    self.index = idx
    self.dirty = false
    return true
end

return M
