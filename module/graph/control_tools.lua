local util = require("module.graph.util")

local M = {}

function M.supported_tools()
    return {
        finish_turn = true,
    }
end

function M.get_tool_schemas()
    return {
        {
            type = "function",
            ["function"] = {
                name = "finish_turn",
                description = [[Finish the current turn. Call this only when you are ready to send the final user-visible reply for this turn.]],
                parameters = {
                    type = "object",
                    properties = {
                        message = {
                            type = "string",
                            description = "Final user-visible reply for this turn."
                        },
                        status = {
                            type = "string",
                            enum = { "completed", "need_more_info", "partial", "failed" },
                            description = "Turn status."
                        },
                    },
                    required = { "message" },
                },
            },
        },
    }
end

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

    return false, "unsupported_control_tool", nil
end

return M
