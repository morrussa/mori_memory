local command = require("module.graph.command")
local config = require("module.config")
local EDGES = require("module.graph.settings.edges")

local ingest_node = require("module.graph.nodes.ingest_node")
local task_node = require("module.graph.nodes.task_node")
local context_node = require("module.graph.nodes.context_node")
local recall_policy_node = require("module.graph.nodes.recall_policy_node")
local recall_node = require("module.graph.nodes.recall_node")
local recall_reentry_node = require("module.graph.nodes.recall_reentry_node")
local planner_node = require("module.graph.nodes.planner_node")
local tool_exec_node = require("module.graph.nodes.tool_exec_node")
local repair_node = require("module.graph.nodes.repair_node")
local responder_node = require("module.graph.nodes.responder_node")
local finalize_node = require("module.graph.nodes.finalize_node")
local writeback_node = require("module.graph.nodes.writeback_node")
local persist_node = require("module.graph.nodes.persist_node")
local end_node = require("module.graph.nodes.end_node")

local M = {}

---解析下一个节点
---@param current string 当前节点名称
---@param state table 当前状态
---@param node_result any 节点执行结果（可能是 Command）
---@return string|nil next_node 下一个节点名称（nil 表示结束）
---@return table|nil updates 状态更新（来自 Command）
function M.resolve_next(current, state, node_result)
    -- 优先处理 Command 返回值
    if command.is_command(node_result) then
        local target = command.get_target(node_result)
        local updates = node_result.update
        
        -- 特殊处理
        if target == command.END then
            return nil, updates
        end
        if target == command.SELF then
            return current, updates
        end
        
        return target, updates
    end
    
    -- 使用声明式边解析
    local edge = EDGES[current]
    if not edge then
        return nil, nil
    end
    
    -- 条件边
    if edge.conditional then
        local cfg = config.get("graph", {})
        local branches = edge.branches or {}

        if #branches > 0 then
            for _, branch in ipairs(branches) do
                local target = branch.to or branch.target
                local condition_fn = branch.when or branch.condition
                if type(target) == "string" and type(condition_fn) == "function" then
                    local ok, result = pcall(condition_fn, state, cfg)
                    if ok and result then
                        return target, nil
                    end
                end
            end
        else
            for target, condition_fn in pairs(branches) do
                if type(condition_fn) == "function" then
                    local ok, result = pcall(condition_fn, state, cfg)
                    if ok and result then
                        return target, nil
                    end
                end
            end
        end

        return nil, nil
    end
    
    -- 静态边
    if edge.to == command.END then
        return nil, nil
    end
    
    return edge.to, nil
end

---兼容旧版：next_node 函数（保留向后兼容）
---@param current string 当前节点名称
---@param state table 当前状态
---@return string|nil
function M.next_node(current, state)
    local next_target, _ = M.resolve_next(current, state, nil)
    return next_target or "end"
end

function M.build()
    local nodes = {
        ingest_node = ingest_node,
        task_node = task_node,
        recall_policy_node = recall_policy_node,
        recall_node = recall_node,
        recall_reentry_node = recall_reentry_node,
        context_node = context_node,
        planner_node = planner_node,
        tool_exec_node = tool_exec_node,
        repair_node = repair_node,
        responder_node = responder_node,
        finalize_node = finalize_node,
        writeback_node = writeback_node,
        persist_node = persist_node,
        ["end"] = end_node,
    }

    return {
        start_node = "ingest_node",
        nodes = nodes,
        next_node = M.next_node,  -- 兼容旧版
        resolve_next = M.resolve_next,  -- 新版 API
    }
end

return M
