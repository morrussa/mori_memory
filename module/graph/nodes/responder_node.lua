local context_builder = require("module.graph.context_builder")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function append_tool_summary(user_input, tool_results)
    local lines = {
        tostring(user_input or ""),
    }

    local rows = tool_results or {}
    if type(rows) == "table" and #rows > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "[ToolExecutionSummary]"
        for _, row in ipairs(rows) do
            local tool_name = util.trim((row or {}).tool)
            if tool_name == "" then
                tool_name = "unknown"
            end
            if row.ok == true then
                local result = util.utf8_take(util.trim((row or {}).result), 320)
                lines[#lines + 1] = string.format("- %s: ok | %s", tool_name, result)
            else
                local err = util.utf8_take(util.trim((row or {}).error), 200)
                lines[#lines + 1] = string.format("- %s: failed | %s", tool_name, err)
            end
        end
    end

    return table.concat(lines, "\n")
end

function M.run(state, _ctx)
    local cfg = graph_cfg().responder or {}
    local tool_results = (((state or {}).tool_exec or {}).results) or {}
    local original_user = tostring((((state or {}).input or {}).message) or "")
    local merged_user = append_tool_summary(original_user, tool_results)

    local original = (((state or {}).input or {}).message) or ""
    state.input.message = merged_user
    local messages, meta = context_builder.build_chat_messages(state)
    state.input.message = original

    local final_text = py_pipeline:generate_chat_sync(messages, {
        max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 1024)),
        temperature = tonumber(cfg.temperature) or 0.75,
        seed = tonumber(cfg.seed) or math.random(1, 2147483647),
    })

    final_text = util.trim(final_text)
    if final_text == "" then
        final_text = "好的，已处理。"
    end

    state.final_response = {
        message = final_text,
        context_meta = meta,
    }
    return state
end

return M
