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
                },
                external_tools = {
                    enabled = true,
                    include_memory_tools = true,
                    context_inject = true,
                    context_max_chars = 1200,
                    names = { "web_search" },
                },
            },
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
    function M.query_records()
        return {}
    end
    function M.render_results()
        return ""
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

_G.py_pipeline = {
    get_qwen_tool_schemas = function(_, entries)
        local has_web = false
        for _, x in ipairs(entries or {}) do
            if tostring(x) == "web_search" then
                has_web = true
            end
        end
        if not has_web then
            return {}
        end
        return {
            {
                type = "function",
                ["function"] = {
                    name = "web_search",
                    description = "search web",
                    parameters = {
                        type = "object",
                        properties = {
                            query = { type = "string" },
                        },
                        required = { "query" },
                    },
                },
            },
        }
    end,
    call_qwen_tool = function(_, name, args_lua)
        return string.format("tool=%s args=%s", tostring(name), tostring(args_lua))
    end,
}

local registry = require("module.agent.tool_registry")

local tools = registry.get_openai_tools()
local seen_web = false
local seen_query = false
for _, item in ipairs(tools or {}) do
    local fn = (item or {})["function"] or {}
    local name = tostring(fn.name or "")
    if name == "web_search" then seen_web = true end
    if name == "query_record" then seen_query = true end
end
assert(seen_web, "web_search should be exposed in openai tools")
assert(seen_query, "query_record should still exist when include_memory_tools=true")

local acts = registry.get_supported_acts({ query_record = true, upsert_record = true, delete_record = false })
assert(acts.web_search == true, "supported acts should include external tool")
assert(acts.query_record == true, "supported acts should include memory tool")

local calls = {
    {
        act = "web_search",
        tool_call_id = "tc_web_1",
        arguments = '{query="llama.cpp"}',
    }
}
local r1 = registry.execute_calls(calls, { current_turn = 88, read_only = false })
assert(r1.executed == 1, "external call should be executed")
assert(r1.failed == 0, "external call should not fail")
assert(r1.context_updated == true, "external call should update pending context")
assert(type(r1.call_results) == "table" and #r1.call_results >= 1, "call_results should be populated")
assert(r1.call_results[1].tool_call_id == "tc_web_1", "tool_call_id should be preserved")
assert(tostring(r1.call_results[1].arguments or ""):find("{query=", 1, true), "call_results should store lua table arguments")

local ctx = registry.consume_pending_system_context_for_turn(88)
assert(type(ctx) == "string" and ctx:find("Tool:web_search", 1, true), "pending context should contain web_search result")

local r2 = registry.execute_calls(calls, { current_turn = 88, read_only = false })
assert(r2.context_novel == false, "same turn same external signature should be non-novel")
assert(r2.context_updated == false, "non-novel signature should suppress repeated inject")

print("TOOL_REGISTRY_EXTERNAL_TOOLS_TESTS_PASS")
