-- history.lua （带内存 turn 计数器版）
local M = {}

local tool = require("module.tool")
-- 移除顶层 require saver，改为函数内延迟加载

local file_path = "memory/history.txt"

M.entries = {}
M.turn_counter = 0

-- ====================== LOAD：启动时全扫描一次，顺便算出 turn ======================
function M.load()
    M.entries = {}
    M.turn_counter = 0
    local f = io.open(file_path, "r")
    if f then
        for line in f:lines() do
            if line and line ~= "" then
                table.insert(M.entries, line)
                M.turn_counter = M.turn_counter + 1
            end
        end
        f:close()
    end
    print(string.format("[History] 已加载 %d 条对话记录到内存，当前 turn_counter = %d", #M.entries, M.turn_counter))
end

function M.add_history(user, ai)
    local saver = require("module.saver")   -- 延迟加载
    local str = tool.replace(ai, "\n", "\x1F")
    local str2 = tool.remove_cot(str)
    table.insert(M.entries, "user:" .. user .. "ai:" .. str2)
    M.turn_counter = M.turn_counter + 1   -- 原子自增
    saver.mark_dirty()
end

function M.get_turn()
    return M.turn_counter   -- 纯内存返回，O(1)
end

function M.save_to_disk()
    local temp = file_path .. ".tmp"
    local f = io.open(temp, "w")
    if not f then return end
    for _, line in ipairs(M.entries) do
        f:write(line .. "\n")
    end
    f:close()
    os.remove(file_path)
    os.rename(temp, file_path)
end

return M