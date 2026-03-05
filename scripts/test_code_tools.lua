#!/usr/bin/env luajit
-- test_code_tools.lua
-- 测试代码分析工具和项目知识注入

local code_tools = require("module.graph.code_tools")
local project_knowledge = require("module.graph.project_knowledge")
local config = require("module.config")

print("=== Testing Code Tools and Project Knowledge ===")
print("")

-- 测试1：检测语言
print("1. Testing language detection...")
local test_paths = {
    "test.lua",
    "test.py",
    "test.js",
    "test.ts",
    "test.c",
    "test.unknown",
}
for _, path in ipairs(test_paths) do
    local lang = code_tools.detect_language(path)
    print(string.format("   %s -> %s", path, lang))
end
print("")

-- 测试2：Lua代码结构提取
print("2. Testing Lua code structure extraction...")
local lua_code = [[
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function helper()
    return "hello"
end

function M.run(state)
    local result = helper()
    return state
end

function M.process(data, options)
    return data
end

M.VERSION = "1.0.0"

return M
]]

local outline = code_tools.generate_outline(lua_code, "test.lua", 10)
local formatted = code_tools.format_outline(outline)
print(formatted)
print("")

-- 测试3：项目知识生成
print("3. Testing project knowledge generation...")
local knowledge = project_knowledge.get_project_knowledge(true)
if knowledge then
    print(knowledge)
else
    print("   [Requires runtime - skipped in standalone test]")
end
print("")

-- 测试4：工具schema
print("4. Testing tool schemas...")
local schemas = code_tools.get_tool_schemas()
for _, schema in ipairs(schemas) do
    local fn = schema["function"]
    if fn then
        print(string.format("   Tool: %s", fn.name))
        print(string.format("   Description: %s", fn.description:sub(1, 80) .. "..."))
        print("")
    end
end

-- 测试5：模块详情
print("5. Testing module detail extraction...")
local detail = project_knowledge.get_module_detail("module.graph.util")
if detail then
    print(project_knowledge.format_module_detail(detail))
else
    print("   [Requires runtime - skipped in standalone test]")
end
print("")

print("=== Tests Complete ===")
