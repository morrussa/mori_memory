local function clear_mods()
    local mods = {
        "module.config",
        "module.agent.notebook",
        "module.memory.topic",
        "module.agent.tool_registry",
    }
    for _, m in ipairs(mods) do
        package.loaded[m] = nil
        package.preload[m] = nil
    end
end

clear_mods()

package.preload["module.config"] = function()
    return {
        settings = {
            keyring = {
                tool_calling = {
                    upsert_min_confidence = 0.82,
                    upsert_max_per_turn = 1,
                    query_max_per_turn = 2,
                    delete_enabled = false,
                    query_max_types = 3,
                    query_fetch_limit = 18,
                    query_inject_top = 3,
                    query_inject_max_chars = 800,
                    tool_pass_temperature = 0.15,
                    tool_pass_max_tokens = 128,
                    tool_pass_seed = 42,
                    parallel_execute_enabled = true,
                    parallel_query_batch_size = 4,
                    retry_transient_max = 1,
                    retry_unknown_max = 0,
                    retry_validation_max = 0,
                    retry_budget_max = 0,
                    retry_total_cap = 2,
                }
            }
        }
    }
end

package.preload["module.memory.topic"] = function()
    local M = {}
    function M.get_topic_anchor(turn)
        return "anchor-" .. tostring(turn or 0)
    end
    return M
end

package.preload["module.agent.notebook"] = function()
    local M = {}
    local retry_once_failed = false
    function M.query_records(query, _)
        if tostring(query):find("重试一次", 1, true) and (not retry_once_failed) then
            retry_once_failed = true
            error("simulated transient failure")
        end
        if tostring(query):find("长期目标", 1, true) then
            return {
                { id = 2, type = "long_term_plan", entity = "Mori 长期目标", value = "构建闭环", confidence = 0.95, score = 1.0, updated_at = 10 },
            }
        end
        return {}
    end
    function M.render_results(results)
        local out = {}
        for _, r in ipairs(results or {}) do
            out[#out + 1] = string.format("[%s] %s", tostring(r.type), tostring(r.entity))
        end
        return table.concat(out, "\n")
    end
    function M.build_long_term_plan_bom()
        return ""
    end
    function M.upsert_record()
        return 1, "inserted"
    end
    function M.delete_record()
        return nil, "disabled"
    end
    return M
end

local registry = require("module.agent.tool_registry")

local calls_dup = {
    { act = "query_record", query = "长期目标", types = "long_term_plan" },
    { act = "query_record", query = "长期目标", types = "long_term_plan" },
}
local r1 = registry.execute_calls(calls_dup, { current_turn = 20, read_only = false })
assert(r1.executed == 1, "r1 executed should be 1")
assert(r1.skipped == 1, "r1 skipped should be 1 (duplicate query)")
assert(r1.context_updated == true, "r1 context_updated should be true")
assert(r1.context_novel == true, "r1 context_novel should be true")
local c1 = registry.consume_pending_system_context_for_turn(20)
assert(c1 ~= "", "r1 should produce pending context")

local calls_same = {
    { act = "query_record", query = "长期目标", types = "long_term_plan" },
}
local r2 = registry.execute_calls(calls_same, { current_turn = 20, read_only = false })
assert(r2.executed == 1, "r2 executed should be 1")
assert(r2.context_novel == false, "r2 context_novel should be false (same turn same signature)")
assert(r2.context_updated == false, "r2 context_updated should be false (suppressed inject)")
local c2 = registry.consume_pending_system_context_for_turn(20)
assert(c2 == "", "r2 should suppress pending context injection")

local r_budget = registry.execute_calls(calls_same, { current_turn = 20, read_only = false })
assert(r_budget.failed == 1, "r_budget failed should be 1 (per-turn query budget reached)")
assert(r_budget.executed == 0, "r_budget executed should be 0 when budget exceeded")
assert(r_budget.context_updated == false, "r_budget should not update context")

local r3 = registry.execute_calls(calls_same, { current_turn = 21, read_only = false })
assert(r3.executed == 1, "r3 executed should be 1")
assert(r3.context_novel == true, "r3 context_novel should be true (new turn)")
assert(r3.context_updated == true, "r3 context_updated should be true (new turn allows inject)")

local upsert_call = {
    { act = "upsert_record", type = "identity", entity = "Mori", value = "agent", confidence = 0.99 },
}
local up1 = registry.execute_calls(upsert_call, { current_turn = 30, read_only = false })
assert(up1.executed == 1, "up1 executed should be 1")
assert(up1.failed == 0, "up1 failed should be 0")
local up2 = registry.execute_calls(upsert_call, { current_turn = 30, read_only = false })
assert(up2.failed == 1, "up2 failed should be 1 (per-turn upsert budget reached)")
assert(up2.executed == 0, "up2 executed should be 0 when budget exceeded")

local retry_call = {
    { act = "query_record", query = "重试一次 长期目标", types = "long_term_plan,identity" },
}
local rr = registry.execute_calls(retry_call, { current_turn = 40, read_only = false })
assert(rr.executed == 1, "rr executed should be 1 after retry success")
assert(rr.failed == 0, "rr failed should be 0 after retry success")
assert((tonumber(rr.retry_total) or 0) >= 1, "rr retry_total should be >= 1")
assert((tonumber(rr.retry_success) or 0) >= 1, "rr retry_success should be >= 1")

local parallel_calls = {
    { act = "query_record", query = "长期目标 A", types = "long_term_plan" },
    { act = "query_record", query = "长期目标 B", types = "long_term_plan" },
}
local rp = registry.execute_calls(parallel_calls, { current_turn = 41, read_only = false })
assert(rp.executed == 2, "rp executed should be 2")
assert(rp.failed == 0, "rp failed should be 0")
assert((tonumber(rp.parallel_batches) or 0) >= 1, "rp parallel_batches should be >= 1")
assert((tonumber(rp.parallel_calls) or 0) >= 2, "rp parallel_calls should be >= 2")

_G.py_pipeline = {
    list_agent_files = function(_, _args_lua, _default_limit, _hard_limit)
        return "[agent_files] showing 1/1 files\n1) ./agent_files/t/a.txt | bytes=12"
    end,
    read_agent_file = function(_, _args_lua, _default_max_chars, _hard_max_chars)
        return "[read_agent_file] path=./agent_files/t/a.txt returned_chars=5\nhello"
    end,
    read_agent_file_lines = function(_, _args_lua, _default_max_lines, _hard_max_lines)
        return "[read_agent_file_lines] path=./agent_files/t/a.txt returned_lines=2\n  1 | hello\n  2 | world"
    end,
    search_agent_file = function(_, _args_lua, _default_max_hits, _hard_max_hits)
        return "[search_agent_file] path=./agent_files/t/a.txt hits=1 shown=1\n#1 line=2\n> 2 | world"
    end,
    search_agent_files = function(_, _args_lua, _default_max_hits, _hard_max_hits, _default_max_files, _hard_max_files, _default_per_file_hits, _hard_per_file_hits)
        return "[search_agent_files] files_total=2 files_scanned=2 hits=2 shown=2\n#1 file=./agent_files/t/a.txt line=2\n> 2 | world"
    end,
}
local file_calls = {
    { act = "list_agent_files", prefix = "t", limit = 5 },
    { act = "read_agent_file", path = "./agent_files/t/a.txt", max_chars = 5 },
    { act = "read_agent_file_lines", path = "./agent_files/t/a.txt", start_line = 1, max_lines = 2 },
    { act = "search_agent_file", path = "./agent_files/t/a.txt", pattern = "world", max_hits = 3 },
    { act = "search_agent_files", prefix = "t", pattern = "world", max_files = 5, per_file_hits = 2, max_hits = 8 },
}
local rf = registry.execute_calls(file_calls, { current_turn = 50, read_only = false })
assert(rf.executed == 5, "rf executed should be 5")
assert(rf.failed == 0, "rf failed should be 0")
local c_file = registry.consume_pending_system_context_for_turn(50)
assert(c_file:find("Tool:list_agent_files", 1, true), "file context should contain list_agent_files")
assert(c_file:find("Tool:read_agent_file", 1, true), "file context should contain read_agent_file")
assert(c_file:find("Tool:read_agent_file_lines", 1, true), "file context should contain read_agent_file_lines")
assert(c_file:find("Tool:search_agent_file", 1, true), "file context should contain search_agent_file")
assert(c_file:find("Tool:search_agent_files", 1, true), "file context should contain search_agent_files")

print("TOOL_REGISTRY_INCREMENTAL_TESTS_PASS")
