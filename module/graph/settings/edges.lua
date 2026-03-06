-- ============================================================================
-- 声明式边定义
-- 
-- 格式：
--   静态边: { to = "target_node" }
--   条件边: { conditional = true, branches = { [target] = condition_fn, ... } }
--
-- 条件函数签名：function(state, cfg) -> boolean
-- 条件边按定义顺序检查，第一个返回 true 的分支被选中
-- ============================================================================

local command = require("module.graph.command")

return {
    ingest_node = { to = "recall_node" },
    
    recall_node = { to = "experience_node" },

    experience_node = { to = "context_node" },
    
    context_node = { to = "agent_node" },
    
    agent_node = {
        conditional = true,
        branches = {
            -- 有待处理的工具调用 -> 执行工具
            ["tools_node"] = function(state, cfg)
                local pending_calls = (((state or {}).agent_loop or {}).pending_tool_calls) or {}
                local loop_count = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0
                local tool_loop_max = math.max(1, math.floor(tonumber(cfg.tool_loop_max) or 5))
                local stop_reason = tostring((((state or {}).agent_loop or {}).stop_reason) or "")
                
                if #pending_calls > 0 then
                    -- 超过循环上限，清除待处理调用并结束
                    if loop_count >= tool_loop_max then
                        state.agent_loop = state.agent_loop or {}
                        state.agent_loop.pending_tool_calls = {}
                        state.agent_loop.stop_reason = "tool_loop_max_exceeded"
                        return false -- 走默认分支
                    end
                    -- 模型调用失败，直接结束
                    if stop_reason == "model_call_failed" then
                        return false
                    end
                    return true
                end
                return false
            end,
            -- 默认分支：结束代理循环
            ["finalize_node"] = function(state, cfg)
                return true
            end,
        },
    },
    
    tools_node = {
        conditional = true,
        branches = {
            -- finish_turn 控制信号 -> 直接结束
            ["finalize_node"] = function(state, cfg)
                local stop_reason = tostring((((state or {}).agent_loop or {}).stop_reason) or "")
                return stop_reason == "finish_turn_called"
            end,
            -- 默认：回到 agent 继续循环
            ["agent_node"] = function(state, cfg)
                return true
            end,
        },
    },
    
    finalize_node = { to = "writeback_node" },
    
    writeback_node = { to = "persist_node" },
    
    persist_node = { to = command.END },
}
