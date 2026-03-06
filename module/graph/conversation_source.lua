local util = require("module.graph.util")
local config = require("module.config")
local history = require("module.memory.history")

local M = {}

local DEFAULT_SYSTEM_PROMPT = [[
你叫 Mori，是一名天才AI极客少女，常用颜文字 (´･ω･')ﾉ
你喜欢有趣和有创意的对话，对于用户的提问会尽力给出有帮助的回答。
当遇到你不确定或觉得信息不足的问题时，你会要求用户提供更多信息，而不是直接拒绝。
你尊重每一个认真提问的人。

你有长期记忆系统，系统会在后台按需召回历史，并在回合结束后提取可复用事实写入记忆。

规则：
1. 当需要读取文件、检索信息或调用外部能力时，使用相应的工具调用，不要编造已执行结果。
2. 当你确认本轮任务已完成时，调用 finish_turn 工具，把最终回复放在 message 参数里。
3. 如果任务还没完成但你不确定下一步该做什么，可以调用 continue_task 工具说明情况，系统会帮你继续。
4. 工具调用属于结构化元数据；不要把工具协议、路由信号或调试信息写在可见回复正文里。
5. 信息不足时，先说明缺失信息再继续，不要编造已完成的步骤。

重要提示：
- finish_turn 是结束本轮对话的唯一正确方式。如果你不调用任何工具，系统也会结束本轮对话。
- 如果你想继续执行任务（比如继续读取更多文件），直接调用相应的工具，不要调用 finish_turn。
- 永远不要在文本中说"再试一次"或"继续"而不实际调用工具——这是无效的，因为文本内容不会触发任何行动。
]]

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function safe_call(fn, ...)
    if type(fn) ~= "function" then
        return false, nil
    end
    local ok, out = pcall(fn, ...)
    if not ok then
        return false, nil
    end
    return true, out
end

local function resolve_system_prompt(override)
    local text = util.trim(override)
    if text ~= "" then
        return text
    end

    local cfg = graph_cfg()
    text = util.trim(((cfg.prompts or {}).system) or cfg.system_prompt)
    if text ~= "" then
        return text
    end
    return DEFAULT_SYSTEM_PROMPT
end

local function normalize_history_rows(rows)
    local out = {}
    if type(rows) ~= "table" then
        return out
    end
    for _, row in ipairs(rows) do
        if type(row) == "table" then
            local role = util.trim(row.role)
            local content = tostring(row.content or "")
            if role ~= "" and util.trim(content) ~= "" then
                out[#out + 1] = {
                    role = role,
                    content = content,
                }
            end
        end
    end
    return out
end

local function ensure_system_row(rows, prompt)
    local out = normalize_history_rows(rows)
    if #out <= 0 then
        return {
            { role = "system", content = prompt },
        }
    end
    if tostring((out[1] or {}).role or "") ~= "system" then
        table.insert(out, 1, { role = "system", content = prompt })
    elseif util.trim((out[1] or {}).content or "") == "" then
        out[1].content = prompt
    end
    return out
end

local function parse_history_entry(entry)
    if entry == nil then
        return "", ""
    end

    local ok_parse, user_text, assistant_text = pcall(function()
        if type(history.parse_entry) == "function" then
            return history.parse_entry(entry)
        end
        return nil, nil
    end)
    if ok_parse and (user_text ~= nil or assistant_text ~= nil) then
        return tostring(user_text or ""), tostring(assistant_text or "")
    end

    local text = tostring(entry or "")
    if text == "" then
        return "", ""
    end
    local left, right = text:match("^(.-)\x1F(.*)$")
    if left ~= nil then
        return tostring(left or ""), tostring(right or "")
    end
    return text, ""
end

local function build_rows_from_history(prompt)
    local cfg = graph_cfg()
    local history_cfg = cfg.history or {}
    local bootstrap_turns = math.max(0, math.floor(tonumber(history_cfg.bootstrap_turns) or 200))

    local rows = {
        { role = "system", content = prompt },
    }

    local ok_turn, turn_value = safe_call(history.get_turn)
    local total_turns = ok_turn and (tonumber(turn_value) or 0) or 0
    if total_turns <= 0 then
        return rows
    end

    local start_turn = 1
    if bootstrap_turns > 0 and total_turns > bootstrap_turns then
        start_turn = total_turns - bootstrap_turns + 1
    end

    for turn = start_turn, total_turns do
        local entry = nil
        local ok_entry, raw = safe_call(history.get_by_turn, turn)
        if ok_entry then
            entry = raw
        elseif type(history.entries) == "table" then
            entry = history.entries[turn]
        end

        local user_text, assistant_text = parse_history_entry(entry)
        user_text = util.trim(user_text)
        assistant_text = util.trim(assistant_text)

        if user_text ~= "" then
            rows[#rows + 1] = { role = "user", content = user_text }
        end
        if assistant_text ~= "" then
            rows[#rows + 1] = { role = "assistant", content = assistant_text }
        end
    end

    return rows
end

function M.resolve_conversation(override_rows, override_prompt)
    local prompt = resolve_system_prompt(override_prompt)
    local rows = ensure_system_row(override_rows, prompt)
    if #rows > 1 then
        return rows, prompt
    end
    rows = build_rows_from_history(prompt)
    return rows, prompt
end

return M
