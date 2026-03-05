local util = require("module.graph.util")
local config = require("module.config")
local file_tools = require("module.graph.file_tools")
local provider_registry = require("module.graph.providers.registry")

local M = {}

local NO_SIDE_EFFECT_TOOLS = {
    list_files = true,
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function external_cfg()
    return (graph_cfg().providers or {}).external or {}
end

local function tool_cfg()
    return (graph_cfg().tools or {})
end

local function normalize_allowlist(raw)
    local out = {}
    local set = {}
    if type(raw) == "string" then
        for name in raw:gmatch("[^,]+") do
            local n = util.trim(name)
            if n ~= "" and (not set[n]) then
                set[n] = true
                out[#out + 1] = n
            end
        end
    elseif type(raw) == "table" then
        for _, item in ipairs(raw) do
            local n = util.trim(item)
            if n ~= "" and (not set[n]) then
                set[n] = true
                out[#out + 1] = n
            end
        end
    end
    return out, set
end

local function normalize_call(call, idx)
    if type(call) ~= "table" then
        return nil, "tool_call_not_table"
    end

    local tool_name = util.trim(call.tool or call.name)
    if tool_name == "" then
        return nil, "tool_call_missing_tool"
    end

    local args = call.args
    if args == nil then
        args = {}
    end
    if type(args) ~= "table" then
        return nil, "tool_call_args_not_table"
    end

    local call_id = util.trim(call.call_id)
    if call_id == "" then
        call_id = string.format("tool_call_%d", tonumber(idx) or 0)
    end

    return {
        tool = tool_name,
        args = args,
        call_id = call_id,
    }
end

local function create_provider_state()
    local ext = external_cfg()
    local enabled = util.to_bool(ext.enabled, false)
    local allowlist, allowset = normalize_allowlist(ext.allowlist or ext.names)

    local provider = nil
    local provider_error = ""
    if enabled and #allowlist > 0 then
        provider, provider_error = provider_registry.create(
            ext.provider or "qwen",
            { allowlist = allowlist }
        )
    elseif enabled and #allowlist == 0 then
        provider_error = "external_allowlist_required"
    end

    return {
        enabled = enabled,
        allowlist = allowlist,
        allowset = allowset,
        provider = provider,
        provider_error = provider_error,
    }
end

function M.get_supported_tools()
    local tools = file_tools.supported_tools()
    local provider_state = create_provider_state()
    if provider_state.enabled and provider_state.provider then
        for _, schema in ipairs(provider_state.provider:list_tools() or {}) do
            local fn = schema["function"] or {}
            local name = util.trim(fn.name)
            if name ~= "" then
                tools[name] = true
            end
        end
    end
    return tools
end

function M.get_tool_schemas()
    local out = {}
    local seen = {}

    for _, schema in ipairs(file_tools.get_tool_schemas() or {}) do
        if type(schema) == "table" then
            local fn = schema["function"] or {}
            local name = util.trim(fn.name)
            if name ~= "" and (not seen[name]) then
                seen[name] = true
                out[#out + 1] = schema
            end
        end
    end

    local provider_state = create_provider_state()
    if provider_state.enabled and provider_state.provider then
        for _, schema in ipairs(provider_state.provider:list_tools() or {}) do
            if type(schema) == "table" then
                local fn = schema["function"] or {}
                local name = util.trim(fn.name)
                if name ~= "" and (not seen[name]) then
                    seen[name] = true
                    out[#out + 1] = schema
                end
            end
        end
    end
    return out
end

function M.execute_calls(calls)
    local out = {
        executed = 0,
        failed = 0,
        skipped = 0,
        call_results = {},
        context_fragments = {},
        parallel_groups = 0,
    }

    if type(calls) ~= "table" or #calls == 0 then
        return out
    end

    local supported = M.get_supported_tools()
    local provider_state = create_provider_state()
    local ext_cfg = external_cfg()
    local inject_external_context = util.to_bool(ext_cfg.context_inject, true)
    local external_context_max_chars = math.max(120, math.floor(tonumber(ext_cfg.context_max_chars) or 1200))
    local file_context_max_chars = math.max(120, math.floor(tonumber((tool_cfg().file_context_max_chars) or 1600)))

    local serial_calls = {}
    local no_side_effect_count = 0

    for i, raw in ipairs(calls) do
        local call, err = normalize_call(raw, i)
        if not call then
            out.failed = out.failed + 1
            out.call_results[#out.call_results + 1] = {
                call_id = string.format("tool_call_%d", i),
                tool = tostring((raw or {}).tool or ""),
                args = type(raw) == "table" and (raw.args or {}) or {},
                ok = false,
                error = err,
                result = "",
            }
        else
            if not supported[call.tool] then
                out.failed = out.failed + 1
                out.call_results[#out.call_results + 1] = {
                    call_id = call.call_id,
                    tool = call.tool,
                    args = call.args,
                    ok = false,
                    error = "tool_not_supported",
                    result = "",
                }
            else
                if NO_SIDE_EFFECT_TOOLS[call.tool] then
                    no_side_effect_count = no_side_effect_count + 1
                end
                serial_calls[#serial_calls + 1] = call
            end
        end
    end

    if no_side_effect_count > 1 then
        out.parallel_groups = 1
    end

    for _, call in ipairs(serial_calls) do
        local ok = false
        local result_or_err = ""

        if NO_SIDE_EFFECT_TOOLS[call.tool] then
            ok, result_or_err = file_tools.execute(call)
        else
            if provider_state.enabled and provider_state.provider then
                ok, result_or_err = provider_state.provider:call(call.tool, call.args)
            else
                ok = false
                result_or_err = provider_state.provider_error ~= "" and provider_state.provider_error or "external_provider_disabled"
            end
        end

        local result_text = util.trim(result_or_err)
        if not ok then
            out.failed = out.failed + 1
            out.call_results[#out.call_results + 1] = {
                call_id = call.call_id,
                tool = call.tool,
                args = call.args,
                ok = false,
                error = result_text ~= "" and result_text or "tool_exec_failed",
                result = "",
            }
        else
            out.executed = out.executed + 1
            out.call_results[#out.call_results + 1] = {
                call_id = call.call_id,
                tool = call.tool,
                args = call.args,
                ok = true,
                error = "",
                result = result_text,
            }

            if NO_SIDE_EFFECT_TOOLS[call.tool] then
                if result_text ~= "" then
                    local clipped = util.utf8_take(result_text, file_context_max_chars)
                    out.context_fragments[#out.context_fragments + 1] = string.format(
                        "[Tool:%s]\n%s",
                        call.tool,
                        clipped
                    )
                end
            else
                if inject_external_context and result_text ~= "" then
                    local clipped = util.utf8_take(result_text, external_context_max_chars)
                    out.context_fragments[#out.context_fragments + 1] = string.format(
                        "[External:%s]\n%s",
                        call.tool,
                        clipped
                    )
                end
            end
        end
    end

    return out
end

return M
