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
Action Input: {query="长期目标",types="long_term_plan"}
Observation: (pending)
Thought: 再看偏好
Action: query_record
Action Input: {
  query = "用户偏好",
  types = "preference"
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
✿ARGS✿: {query="主题A",types="long_term_plan"}
]]
local qwen_calls = parser.collect_tool_calls_only(qwen_symbol)
assert_eq(#qwen_calls, 1, "Qwen symbol should parse")
assert_eq(qwen_calls[1].query, "主题A", "Qwen symbol query")

local line_call = parser.parse_tool_call_line([[Action: query_record
Action Input: {query="单行测试",types="identity"}]])
assert(line_call ~= nil, "parse_tool_call_line should support ReAct snippet")
assert_eq(line_call.act, "query_record", "parse_tool_call_line act")
assert_eq(line_call.query, "单行测试", "parse_tool_call_line query")

local file_call = parser.parse_tool_call_line([[{act="read_agent_file",path="./agent_files/a.txt",start_char=11,max_chars=120}]])
assert(file_call ~= nil, "read_agent_file lua table call should parse")
assert_eq(file_call.act, "read_agent_file", "read_agent_file act")
assert_eq(file_call.path, "./agent_files/a.txt", "read_agent_file path")
assert_eq(tostring(file_call.start_char), "11", "read_agent_file start_char")
assert_eq(tostring(file_call.max_chars), "120", "read_agent_file max_chars")

local alias_call = parser.parse_tool_call_line([[Action: read_file
Action Input: {path="./agent_files/a.txt"}]])
assert(alias_call ~= nil, "read_file alias should parse")
assert_eq(alias_call.act, "read_agent_file", "read_file alias -> read_agent_file")

local lines_call = parser.parse_tool_call_line([[{act="read_agent_file_lines",path="./agent_files/a.txt",start_line=3,max_lines=40}]])
assert(lines_call ~= nil, "read_agent_file_lines lua table call should parse")
assert_eq(lines_call.act, "read_agent_file_lines", "read_agent_file_lines act")
assert_eq(lines_call.path, "./agent_files/a.txt", "read_agent_file_lines path")
assert_eq(tostring(lines_call.start_line), "3", "read_agent_file_lines start_line")
assert_eq(tostring(lines_call.max_lines), "40", "read_agent_file_lines max_lines")

local search_alias = parser.parse_tool_call_line([[Action: find_in_file
Action Input: {path="./agent_files/a.txt",pattern="todo",context_lines=1}]])
assert(search_alias ~= nil, "find_in_file alias should parse")
assert_eq(search_alias.act, "search_agent_file", "find_in_file alias -> search_agent_file")
assert_eq(search_alias.pattern, "todo", "search_agent_file pattern")
assert_eq(tostring(search_alias.context_lines), "1", "search_agent_file context_lines")

local multi_search = parser.parse_tool_call_line([[{act="search_agent_files",prefix="session_1",pattern="TODO",max_files=10,per_file_hits=2}]])
assert(multi_search ~= nil, "search_agent_files lua table call should parse")
assert_eq(multi_search.act, "search_agent_files", "search_agent_files act")
assert_eq(multi_search.prefix, "session_1", "search_agent_files prefix")
assert_eq(tostring(multi_search.max_files), "10", "search_agent_files max_files")
assert_eq(tostring(multi_search.per_file_hits), "2", "search_agent_files per_file_hits")

local multi_alias = parser.parse_tool_call_line([[Action: find_in_files
Action Input: {pattern="deadline",prefix="session_1"}]])
assert(multi_alias ~= nil, "find_in_files alias should parse")
assert_eq(multi_alias.act, "search_agent_files", "find_in_files alias -> search_agent_files")

local json_protocol = parser.parse_tool_call_line([[{"act":"query_record","query":"json_should_not_parse"}]])
assert(json_protocol == nil, "json protocol should be disabled")

local react_json_input = parser.parse_tool_call_line([[Action: query_record
Action Input: {"query":"json_input_should_not_parse","types":"identity"}]])
assert(react_json_input == nil, "react Action Input should require lua table")

local xml_protocol = parser.parse_tool_call_line([[<tool_call>{"name":"query_record","arguments":{"query":"xml_should_not_parse"}}</tool_call>]])
assert(xml_protocol == nil, "xml protocol should be disabled")

print("TOOL_PARSER_QWEN_COMPAT_TESTS_PASS")
