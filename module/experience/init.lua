
-- module/experience/init.lua
-- Experience模块入口点

local M = {}

-- 子模块
M.store = require("module.experience.store")
M.builder = require("module.experience.builder")
M.retriever = require("module.experience.retriever")
M.adaptive = require("module.experience.adaptive")  -- [新增]

-- ==================== 初始化 ====================

function M.init()
    -- 初始化自适应学习系统
    M.adaptive.load()

    -- 初始化store
    M.store.init()

    -- 初始化builder（会自动注册topic监听器）
    M.builder.init()

    print("[Experience] Module initialized")
end

-- 保存状态（退出时调用）
function M.finalize()
    M.adaptive.save_to_disk()
    M.store.save()
    print("[Experience] Module finalized")
end

-- ==================== 便捷API ====================

-- 添加经验
function M.add_experience(experience)
    return M.store.add(experience)
end

-- 检索经验
function M.retrieve_experience(options)
    return M.retriever.retrieve_hybrid(options)
end

-- 检查潜在失败
function M.check_failure_risk(proposed_solution, current_context)
    return M.retriever.check_potential_failure(proposed_solution, current_context)
end

-- 检索成功案例
function M.get_success_cases(task_description, current_context)
    return M.retriever.retrieve_success_cases(task_description, current_context)
end

-- [新增] 获取推荐输出策略
function M.get_recommended_output_strategy(context_sig, task_type)
    return M.retriever.get_recommended_output_strategy(context_sig, task_type)
end

-- [新增] 记录输出结果（用于自适应学习）
function M.record_output_result(strategy, context_sig, task_type, success)
    M.retriever.record_output_result(strategy, context_sig, task_type, success)
end

-- [新增] 记录检索反馈
function M.record_retrieval_feedback(results, positive_ids, negative_ids)
    M.retriever.record_retrieval_feedback(results, positive_ids, negative_ids)
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
