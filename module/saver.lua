-- module/saver.lua
local M = {}

M.dirty = false

function M.mark_dirty()
    M.dirty = true
end

function M.flush_all(force)
    if not M.dirty and not force then return end

    local memory = require("module.memory")
    local cluster = require("module.cluster")
    local history = require("module.history")

    print("[Saver] === 开始原子保存 raw 文件 ===")

    memory.save_to_disk()
    cluster.save_to_disk()
    history.save_to_disk()

    py_pipeline:pack_state()

    M.dirty = false
    print("[Saver] raw 文件 + state.zst 已更新")
end

-- 程序正常退出时自动调用
function M.on_exit()
    print("[Saver] 正在原子保存并最终归档...")
    local topic = require("module.topic")
    topic.finalize()
    M.flush_all(true)
    py_pipeline:cleanup_raw_files()
    print("[Saver] 程序退出完成，仅保留 state.zst")
end

return M