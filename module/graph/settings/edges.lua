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
            {
                to = "tools_node",
                when = function(state, cfg)
                    local pending_calls = (((state or {}).agent_loop or {}).pending_tool_calls) or {}
                    local loop_count = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0
                    local tool_loop_max = math.max(1, math.floor(tonumber(cfg.tool_loop_max) or 5))
                    local stop_reason = tostring((((state or {}).agent_loop or {}).stop_reason) or "")

                    if #pending_calls > 0 then
                        if loop_count >= tool_loop_max then
                            state.agent_loop = state.agent_loop or {}
                            state.agent_loop.pending_tool_calls = {}
                            state.agent_loop.stop_reason = "tool_loop_max_exceeded"
                            return false
                        end
                        if stop_reason == "model_call_failed" then
                            return false
                        end
                        return true
                    end
                    return false
                end,
            },
            {
                to = "finalize_node",
                when = function(_state, _cfg)
                    return true
                end,
            },
        },
    },
    
    tools_node = {
        conditional = true,
        branches = {
            {
                to = "finalize_node",
                when = function(state, _cfg)
                    local stop_reason = tostring((((state or {}).agent_loop or {}).stop_reason) or "")
                    return stop_reason == "finish_turn_called"
                        or stop_reason == "remaining_steps_exhausted"
                end,
            },
            {
                to = "agent_node",
                when = function(_state, _cfg)
                    return true
                end,
            },
        },
    },
    
    finalize_node = { to = "writeback_node" },
    
    writeback_node = { to = "persist_node" },
    
    persist_node = { to = command.END },
}
