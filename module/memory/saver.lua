-- module/memory/saver.lua
local M = {}

M.dirty = false

function M.mark_dirty()
    M.dirty = true
end

function M.flush_all(force)
    if not M.dirty and not force then return true end

    local memory = require("module.memory.store")
    local cluster = require("module.memory.cluster")
    local history = require("module.memory.history")
    local heat = require("module.memory.heat")   -- 延迟加载 heat 模块
    local notebook = require("module.agent.notebook")
    local adaptive = require("module.memory.adaptive")

    print("[Saver] === 开始原子保存 raw 文件 ===")

    local function run_save(name, fn)
        local ok, err = fn()
        if not ok then
            M.dirty = true
            print(string.format("[Saver][ERROR] %s 保存失败: %s", name, tostring(err)))
            return false
        end
        return true
    end

    if not run_save("memory_index.bin", memory.save_index_to_disk) then return false end
    if not run_save("cluster_index.bin", cluster.save_to_disk) then return false end
    if not run_save("cluster_shards", memory.save_dirty_shards) then return false end
    if not run_save("history.txt", history.save_to_disk) then return false end
    if not run_save("notebook.txt", notebook.save_to_disk) then return false end
    if not run_save("adaptive_state.txt", adaptive.save_to_disk) then return false end

    -- 保存 pending_cold 任务队列
    if not run_save("pending_cold.txt", heat.save_pending) then return false end

    local pack_ok, pack_err = pcall(function()
        py_pipeline:pack_state()
    end)
    if not pack_ok then
        M.dirty = true
        print("[Saver][ERROR] pack_state 失败: " .. tostring(pack_err))
        return false
    end

    M.dirty = false
    print("[Saver] raw 文件 + state.zst 已更新")
    return true
end

-- 程序正常退出时自动调用
function M.on_exit()
    print("[Saver] 正在原子保存并最终归档...")
    local topic = require("module.memory.topic")
    topic.finalize()
    local ok = M.flush_all(true)
    if not ok then
        print("[Saver][ERROR] 退出保存失败，跳过 raw 清理以避免数据丢失")
        return false
    end
    py_pipeline:cleanup_raw_files()
    return true
    -- print("[Saver] 程序退出完成，仅保留 state.zst")
end

return M
