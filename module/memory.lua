-- memory.lua （二进制高效版）
local M = {}

local tool = require("module.tool")
local config = require("module.config")
local ffi = require("ffi")

M.memories = {}        
local next_line = 1
local file_path = "memory/memory.bin"   -- 新二进制文件

-- ====================== LOAD ======================
function M.load()
    M.memories = {}
    next_line = 1

    local f = io.open(file_path, "rb")
    if not f then
        print("[Memory] memory.bin 不存在，内存从空开始")
        return
    end

    local data = f:read("*a")
    f:close()

    if #data < 20 or data:sub(1,4) ~= "MEMB" then
        print("[Memory] 文件头无效，无法加载，内存从空开始")
        return
    end

    -- Header: "MEMB"(4) + uint32_t[4](16) = 20 字节
    local offset = 20
    
    -- 额外输出 header 中的 count（用于调试）
    local header_count = 0
    if #data >= 24 then
        local p = ffi.cast("const uint32_t*", ffi.cast("const uint8_t*", data) + 8)
        header_count = p[0]   -- 第二个 uint32 是 count
        print(string.format("[Memory] 文件头声明共 %d 条记忆", header_count))
    end

    while offset < #data do
        local mem, record_size = tool.parse_memory_record(data, offset)
        if mem then
            M.memories[next_line] = mem
            next_line = next_line + 1
            offset = offset + record_size
        else
            print(string.format("[Memory] 解析在 offset %d 处失败（文件可能损坏或旧版本）", offset))
            print(string.format("[Memory] 已成功加载 %d 条，文件总大小 %d 字节", next_line-1, #data))
            break
        end
    end

    print(string.format("[Memory] 已从二进制加载 %d 条记忆（高效格式）", next_line-1))
end

-- ====================== 其他函数（完全保持不变） ======================
function M.add_memory(vec, turn)
    local saver = require("module.saver")
    local cluster = require("module.cluster")
    local heat_mod = require("module.heat")

    local new_heat = config.settings.heat.new_memory_heat
    local merge_limit = config.settings.merge_limit or 0.95

    -- 【纯Lua线性搜索】彻底取代ANN
    local sim_results = tool.find_sim_all_heat(vec)
    local sim_table = sim_results or {}

    if #sim_table > 0 then
        local top = sim_table[1]
        if top and top.similarity and top.similarity >= merge_limit then
            local target = top.index
            
            -- 只允许热记忆合并
            if heat_mod.is_hot(target) then
                local mem = M.memories[target]
                if mem then
                    mem.heat = mem.heat + new_heat
                    table.insert(mem.turns, 1, turn)
                    saver.mark_dirty()
                    print(string.format("[Memory] 原子合并 → 热记忆行 %d (sim=%.4f)", target, top.similarity))
                    return target
                end
            else
                print(string.format("[Memory] 高相似但冷记忆 %d (sim=%.4f) → 不合并", target, top.similarity))
            end
        end
    end

    -- 新建记忆行
    local new_line = next_line
    M.memories[new_line] = {heat = new_heat, turns = {turn}, vec = vec}
    next_line = next_line + 1
    saver.mark_dirty()

    heat_mod.add_new_heat(new_line)
    cluster.add(vec, new_line)

    print(string.format("[Memory] 新记忆创建 → 行 %d", new_line))
    return new_line
end

function M.set_heat(line, new_heat)
    local mem = M.memories[line]
    if mem then
        mem.heat = math.max(0, new_heat)
        local saver = require("module.saver")
        saver.mark_dirty()
    end
end

function M.get_total_lines() return next_line - 1 end
function M.get_heat_by_index(line)
    local mem = M.memories[line]
    return mem and mem.heat or 0
end
function M.return_mem_vec(line)
    local mem = M.memories[line]
    return mem and mem.vec
end

function M.save_to_disk()
    local temp = file_path .. ".tmp"
    local f = io.open(temp, "wb")
    if not f then return end

    -- Header: magic + version + count + reserved
    f:write("MEMB")
    local header = ffi.new("uint32_t[4]", 1, next_line-1, 0, 0)
    f:write(ffi.string(header, 16))

    for i = 1, next_line-1 do
        local m = M.memories[i]
        if m then
            local bin = tool.create_memory_record(m.heat, m.turns, m.vec)
            f:write(bin)
        end
    end
    f:close()

    os.remove(file_path)
    os.rename(temp, file_path)
    print(string.format("[Memory] 二进制原子保存完成（%d 条）", next_line-1))
end

function M.iterate_all()
    local i = 0
    return function()
        i = i + 1
        return M.memories[i]
    end
end

return M