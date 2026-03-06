
-- module/experience/init.lua
-- Experience模块入口点

local M = {}

-- 子模块
M.store = require("module.experience.store")
M.retriever = require("module.experience.retriever")
M.adaptive = require("module.experience.adaptive")
M.run_builder = require("module.experience.run_builder")
M.builder = require("module.experience.builder") -- legacy topic-based builder

M._initialized = false

-- ==================== 初始化 ====================

function M.init()
    if M._initialized == true then
        return
    end

    M.adaptive.load()
    M.store.init()
    M._initialized = true

    print("[Experience] Module initialized")
end

-- 保存状态（退出时调用）
function M.finalize()
    if M._initialized ~= true then
        return true
    end
    M.adaptive.save_to_disk()
    M.store.save()
    print("[Experience] Module finalized")
    return true
end

-- ==================== 核心API ====================

-- 添加经验
function M.add_experience(experience)
    return M.store.add(experience)
end

-- 相关性门控检索（核心检索方法）
function M.retrieve(query, options)
    return M.retriever.retrieve_intersection_priority(query, options)
end

-- 带反馈的检索
function M.retrieve_with_feedback(query, options)
    return M.retriever.retrieve_with_feedback(query, options)
end

-- 记录效用反馈
function M.record_utility_feedback(retrieved_ids, effective_ids)
    M.retriever.record_utility_feedback(retrieved_ids, effective_ids)
end

-- ==================== 统计信息 ====================

function M.get_stats()
    local store_stats = M.store.get_stats()
    local adaptive_stats = M.adaptive.get_stats()
    return {
        store = store_stats,
        adaptive = adaptive_stats
    }
end

return M
