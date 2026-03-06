
-- module/experience/bridge.lua
-- Topic和Experience之间的桥接层
-- 设计原则：松散耦合，单向依赖（topic -> experience）

local M = {}

local topic = require("module.memory.topic")
local experience = require("module.experience")

-- ==================== 配置 ====================

-- 是否启用experience构建
local ENABLED = true

-- ==================== Topic事件钩子 ====================

-- 在topic.lua中添加以下钩子调用：
-- 1. 在开启新话题时调用 experience_bridge.on_topic_start(topic_data)
-- 2. 在话题更新时调用 experience_bridge.on_topic_update(topic_data)
-- 3. 在话题结束时调用 experience_bridge.on_topic_end(topic_data)

-- ==================== 事件处理 ====================

-- Topic开始事件
function M.on_topic_start(topic_data)
    if not ENABLED then return end

    local ok, err = pcall(function()
        experience.builder.on_topic_start({
            topic_idx = topic_data.topic_idx,
            start = topic_data.start,
            context = topic_data.context or {}
        })
    end)

    if not ok then
        print(string.format("[ExperienceBridge] on_topic_start error: %s", err or "unknown"))
    end
end

-- Topic更新事件
function M.on_topic_update(topic_data)
    if not ENABLED then return end

    local ok, err = pcall(function()
        experience.builder.on_topic_update({
            topic_idx = topic_data.topic_idx,
            tool_used = topic_data.tool_used,
            tool_args = topic_data.tool_args,
            tool_result = topic_data.tool_result,
            intermediate_state = topic_data.intermediate_state,
            error = topic_data.error
        })
    end)

    if not ok then
        print(string.format("[ExperienceBridge] on_topic_update error: %s", err or "unknown"))
    end
end

-- Topic结束事件
function M.on_topic_end(topic_data)
    if not ENABLED then return end

    local ok, err = pcall(function()
        experience.builder.on_topic_end({
            topic_idx = topic_data.topic_idx,
            end_turn = topic_data.end_turn,
            anchor = topic_data.anchor,
            summary = topic_data.summary,
            outcome = topic_data.outcome,
            errors = topic_data.errors
        })
    end)

    if not ok then
        print(string.format("[ExperienceBridge] on_topic_end error: %s", err or "unknown"))
    end
end

-- ==================== 辅助函数 ====================

-- 启用/禁用experience构建
function M.set_enabled(enabled)
    ENABLED = enabled
    print(string.format("[ExperienceBridge] Experience构建已%s", 
        enabled and "启用" or "禁用"))
end

-- 检查是否启用
function M.is_enabled()
    return ENABLED
end

return M
