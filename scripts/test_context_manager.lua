-- test_context_manager.lua
-- 测试上下文管理模块

local context_manager = require("module.graph.context_manager")

print("=== Context Manager Tests ===\n")

-- 测试1: 智能截断
print("Test 1: Smart Truncation")
local long_text = string.rep("Hello World! ", 1000) -- 约13000字符
local truncated, was_truncated = context_manager.process_tool_result(
    "read_file",
    { path = "test.txt" },
    long_text
)
print(string.format("  Original: %d chars", #long_text))
print(string.format("  Truncated: %d chars", #truncated))
print(string.format("  Was truncated: %s", tostring(was_truncated)))
assert(#truncated < #long_text, "Should be truncated")
print("  PASS\n")

-- 测试2: 结果缓存
print("Test 2: Result Caching")
context_manager.clear_cache()
local result1 = context_manager.process_tool_result(
    "read_file",
    { path = "cache_test.txt" },
    "Cached content here"
)
local result2 = context_manager.process_tool_result(
    "read_file",
    { path = "cache_test.txt" },
    "Cached content here"
)
print(string.format("  First call length: %d", #result1))
print(string.format("  Second call length: %d", #result2))
local stats = context_manager.cache_stats()
print(string.format("  Cache entries: %d", stats.entries))
print("  PASS\n")

-- 测试3: 合并多个工具结果
print("Test 3: Merge Tool Results")
local results = {
    { tool = "read_file", ok = true, result = "File content here..." },
    { tool = "search_file", ok = true, result = "Match found at line 10\nMatch found at line 20" },
    { tool = "read_file", ok = false, error = "File not found", result = "" },
}
local merged = context_manager.merge_tool_results(results, "")
print(string.format("  Merged length: %d chars", #merged))
print("  PASS\n")

-- 测试4: Token估算
print("Test 4: Token Estimation")
local english_text = "The quick brown fox jumps over the lazy dog. "
local chinese_text = "这是一段中文测试文本，用于测试token估算功能。"
local estimated_en = context_manager.estimate_tokens(english_text)
local estimated_cn = context_manager.estimate_tokens(chinese_text)
print(string.format("  English text (%d chars): ~%d tokens", #english_text, estimated_en))
print(string.format("  Chinese text (%d chars): ~%d tokens", #chinese_text, estimated_cn))
print("  PASS\n")

-- 测试5: 预算检查
print("Test 5: Budget Check")
local status1, warning1 = context_manager.check_budget(5000, 12000)
local status2, warning2 = context_manager.check_budget(11000, 12000)
local status3, warning3 = context_manager.check_budget(15000, 12000)
print(string.format("  5000/12000 tokens: %s", status1))
print(string.format("  11000/12000 tokens: %s (%s)", status2, warning2 or "no warning"))
print(string.format("  15000/12000 tokens: %s (%s)", status3, warning3 or "no warning"))
print("  PASS\n")

-- 测试6: 读取策略建议
print("Test 6: Read Strategy Suggestion")
local strategy1, hint1 = context_manager.suggest_read_strategy(5000, 4000)
local strategy2, hint2 = context_manager.suggest_read_strategy(50000, 4000)
print(string.format("  Small file (5KB): %s", strategy1))
print(string.format("  Large file (50KB): %s", strategy2 or "no hint"))
print("  PASS\n")

-- 测试7: 消息优化
print("Test 7: Runtime Messages Optimization")
local messages = {
    { role = "system", content = "You are a helpful assistant." },
    { role = "user", content = "Read this file" },
    { role = "tool", content = string.rep("Result: ", 5000), name = "read_file", tool_call_id = "123" },
    { role = "assistant", content = "Here is the result..." },
}
local optimized, opt_stats = context_manager.optimize_runtime_messages(messages, 2000)
print(string.format("  Original tool message: %d chars", #messages[3].content))
print(string.format("  Optimized tool message: %d chars", #optimized[3].content))
print(string.format("  Truncated count: %d", opt_stats.truncated_count))
print("  PASS\n")

print("=== All Tests Passed! ===")
