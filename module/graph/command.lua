--[[
Command 模式 - 节点返回此对象来控制流程

官方 LangGraph 的核心思想：节点返回 Command(update=..., goto=...)
而不是直接修改 state 后隐式决定下一个节点。

用法示例：
    local command = require("module.graph.command")
    
    -- 简单跳转
    return command.goto_node("next_node")
    
    -- 带状态更新的跳转
    return command.goto_node("next_node", { some_field = "value" })
    
    -- 结束图执行
    return command.finish({ final_message = "done" })
    
    -- 条件分支（在节点内部决定）
    if condition then
        return command.goto_node("branch_a")
    else
        return command.goto_node("branch_b")
    end
]]

local M = {}

-- Command 标识符
M.MARKER = "__command__"

-- 终止节点名称
M.END = "__end__"

-- 自循环（继续当前节点，通常用于 retry）
M.SELF = "__self__"

---检查对象是否为 Command
---@param obj any
---@return boolean
function M.is_command(obj)
    if type(obj) ~= "table" then
        return false
    end
    return obj[M.MARKER] == true
end

---创建跳转命令
---@param target string 目标节点名称
---@param updates table|nil 可选的状态更新
---@return table
function M.goto_node(target, updates)
    return {
        [M.MARKER] = true,
        target = target,
        update = updates or {},
    }
end

---创建结束命令（跳转到 END）
---@param updates table|nil 可选的状态更新
---@return table
function M.finish(updates)
    return {
        [M.MARKER] = true,
        target = M.END,
        update = updates or {},
    }
end

---创建自循环命令（重试当前节点）
---@param updates table|nil 可选的状态更新
---@return table
function M.retry(updates)
    return {
        [M.MARKER] = true,
        target = M.SELF,
        update = updates or {},
    }
end

---获取 Command 的目标节点
---@param cmd table
---@return string|nil
function M.get_target(cmd)
    if not M.is_command(cmd) then
        return nil
    end
    return cmd.target
end

---应用 Command 的更新到 state
---@param state table 当前状态
---@param cmd table Command 对象
---@return table 更新后的状态
function M.apply_update(state, cmd)
    if not M.is_command(cmd) then
        return state
    end
    
    local updates = cmd.update or {}
    for key, value in pairs(updates) do
        -- 支持嵌套更新（浅合并）
        if type(state[key]) == "table" and type(value) == "table" then
            for k2, v2 in pairs(value) do
                state[key][k2] = v2
            end
        else
            state[key] = value
        end
    end
    
    return state
end

return M
