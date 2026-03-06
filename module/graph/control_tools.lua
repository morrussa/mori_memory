local util = require("module.graph.util")
local config = require("module.config")

local M = {}

-- 控制工具：用于让agent明确表达意图
-- 这是架构层面的改进，避免依赖关键词检测来判断agent意图

function M.supported_tools()
    return {
        finish_turn = true,
        continue_task = true,
    }
end

function M.get_tool_schemas()
    return {
        {
            type = "function",
            ["function"] = {
                name = "finish_turn",
                description = [[完成本轮对话。当你已经完成了用户的请求，或者需要等待用户更多信息时调用。

这是结束agent循环的标准方式。调用此工具后，agent循环会终止，你的message会作为最终回复发送给用户。

重要：
- 如果你需要继续执行任务（如继续读取文件、继续搜索等），不要调用此工具，而是调用其他工具
- 只有在确认任务完成或需要用户输入时才调用此工具]],
                parameters = {
                    type = "object",
                    properties = {
                        message = {
                            type = "string",
                            description = "给用户的最终回复"
                        },
                        status = {
                            type = "string",
                            enum = { "completed", "need_more_info", "partial" },
                            description = "任务状态：completed=已完成，need_more_info=需要更多信息，partial=部分完成但暂时无法继续"
                        },
                    },
                    required = { "message" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "continue_task",
                description = [[表示你想要继续执行当前任务，但需要系统为你提供更多上下文或建议下一步。

当你：
- 不确定下一步应该调用什么工具
- 需要系统帮你分析当前状态并建议下一步
- 遇到问题但想继续尝试解决

调用此工具，系统会为你提供继续任务所需的上下文。]],
                parameters = {
                    type = "object",
                    properties = {
                        reason = {
                            type = "string",
                            description = "说明为什么需要继续，以及你当前的困难"
                        },
                        suggested_tools = {
                            type = "array",
                            items = { type = "string" },
                            description = "你打算接下来尝试的工具（可选）"
                        },
                    },
                    required = { "reason" },
                },
            },
        },
    }
end

-- 执行控制工具调用
-- 返回: ok, result_or_error, control_action
-- control_action: "finish" | "continue" | nil
function M.execute(call)
    local name = util.trim((call or {}).tool)
    local args = (call or {}).args or {}

    if name == "finish_turn" then
        local message = util.trim(args.message or "")
        if message == "" then
            message = "好的，已处理。"
        end
        local status = util.trim(args.status or "completed")
        return true, message, {
            action = "finish",
            message = message,
            status = status,
        }
    end

    if name == "continue_task" then
        local reason = util.trim(args.reason or "")
        local suggested = args.suggested_tools or {}
        return true, "continue_requested", {
            action = "continue",
            reason = reason,
            suggested_tools = suggested,
        }
    end

    return false, "unsupported_control_tool", nil
end

return M
