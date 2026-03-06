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
    ingest_node = { to = "task_node" },
    task_node = { to = "recall_node" },
    recall_node = { to = "context_node" },

    context_node = { to = "planner_node" },

    planner_node = {
        conditional = true,
        branches = {
            {
                to = "responder_node",
                when = function(state, _cfg)
                    return ((((state or {}).termination or {}).finish_requested) == true)
                end,
            },
            {
                to = "tool_exec_node",
                when = function(state, _cfg)
                    local pending_calls = (((state or {}).planner or {}).tool_calls) or {}
                    return type(pending_calls) == "table" and #pending_calls > 0
                end,
            },
            {
                to = "repair_node",
                when = function(_state, _cfg)
                    return true
                end,
            },
        },
    },

    tool_exec_node = {
        conditional = true,
        branches = {
            {
                to = "repair_node",
                when = function(state, _cfg)
                    return ((((state or {}).repair or {}).pending) == true)
                end,
            },
            {
                to = "planner_node",
                when = function(_state, _cfg)
                    return true
                end,
            },
        },
    },

    repair_node = {
        conditional = true,
        branches = {
            {
                to = "responder_node",
                when = function(state, _cfg)
                    return ((((state or {}).termination or {}).finish_requested) == true)
                end,
            },
            {
                to = "planner_node",
                when = function(state, _cfg)
                    return ((((state or {}).repair or {}).retry_requested) == true)
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

    responder_node = { to = "finalize_node" },

    finalize_node = { to = "writeback_node" },
    writeback_node = { to = "persist_node" },
    persist_node = { to = command.END },
}
