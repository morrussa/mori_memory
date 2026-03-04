package.loaded["module.tool"] = nil
package.preload["module.tool"] = function()
    local M = {}
    function M.extract_first_lua_table(s)
        return tostring(s or ""):match("%b{}")
    end
    return M
end

local parser = require("module.agent.tool_parser")

local function assert_eq(actual, expected, msg)
    if actual ~= expected then
        error(string.format("%s | expected=%s actual=%s", tostring(msg), tostring(expected), tostring(actual)))
    end
end

local function assert_contains(text, needle, msg)
    if not tostring(text or ""):find(tostring(needle), 1, true) then
        error(string.format("%s | missing=%s text=%s", tostring(msg), tostring(needle), tostring(text)))
    end
end

local react_text = [[
Question: 用户的长期目标是什么？
Thought: 先查一下长期计划
Action: query_record
Action Input: {"query":"长期目标","types":"long_term_plan"}
Observation: (pending)
Thought: 再看偏好
Action: query_record
Action Input: {
  "query": "用户偏好",
  "types": "preference"
}
Observation: (pending)
Final Answer: 已完成检索
]]

local react_calls, react_visible = parser.split_tool_calls_and_text(react_text)
assert_eq(#react_calls, 2, "ReAct blocks should parse two calls")
assert_eq(react_calls[1].act, "query_record", "ReAct call#1 act")
assert_eq(react_calls[1].query, "长期目标", "ReAct call#1 query")
assert_eq(react_calls[2].act, "query_record", "ReAct call#2 act")
assert_eq(react_calls[2].query, "用户偏好", "ReAct call#2 query")
assert_contains(react_visible, "Final Answer", "Visible text should keep non-tool reply")
if react_visible:find("Action:", 1, true) then
    error("Visible text should strip Action blocks")
end

local qwen_symbol = [[
✿FUNCTION✿: query_record
✿ARGS✿: {"query":"主题A","types":"long_term_plan"}
]]
local qwen_calls = parser.collect_tool_calls_only(qwen_symbol)
assert_eq(#qwen_calls, 1, "Qwen symbol should parse")
assert_eq(qwen_calls[1].query, "主题A", "Qwen symbol query")

local line_call = parser.parse_tool_call_line([[Action: query_record
Action Input: {"query":"单行测试","types":"identity"}]])
assert(line_call ~= nil, "parse_tool_call_line should support ReAct snippet")
assert_eq(line_call.act, "query_record", "parse_tool_call_line act")
assert_eq(line_call.query, "单行测试", "parse_tool_call_line query")

print("TOOL_PARSER_QWEN_COMPAT_TESTS_PASS")
