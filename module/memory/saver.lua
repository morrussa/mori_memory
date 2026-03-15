-- module/memory/saver.lua
local M = {}

M.dirty = false

function M.mark_dirty()
    M.dirty = true
end

function M.flush_all(force)
    if not M.dirty and not force then return true end

    local memory = require("module.memory.store")
    local history = require("module.memory.history")

    print("[Saver] === 开始原子保存 memory 状态 ===")

    local function run_save(name, fn)
        local ok, err = fn()
        if not ok then
            M.dirty = true
            print(string.format("[Saver][ERROR] %s 保存失败: %s", name, tostring(err)))
            return false
        end
        return true
    end

    if not run_save("topic_graph", memory.save_to_disk) then return false end
    if not run_save("history.txt", history.save_to_disk) then return false end

    M.dirty = false
    print("[Saver] topic_graph + history.txt 已更新")
    return true
end

-- 程序正常退出时自动调用
function M.on_exit()
    print("[Saver] 正在原子保存并归档...")
    local topic = require("module.memory.topic")
    topic.finalize()
    local ok = M.flush_all(true)
    if not ok then
        print("[Saver][ERROR] 退出保存失败，跳过 raw 清理以避免数据丢失")
        return false
    end
    return true
end

return M
