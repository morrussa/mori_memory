local util = require("module.graph.util")
local config = require("module.config")
local file_tools = require("module.graph.file_tools")
local code_tools = require("module.graph.code_tools")
local control_tools = require("module.graph.control_tools")
local provider_registry = require("module.graph.providers.registry")

local M = {}

local READ_ONLY_TOOLS = {
    list_files = true,
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
    code_outline = true,
    project_structure = true,
    code_symbols = true,
}

local SIDE_EFFECT_TOOLS = {
    write_file = true,
    apply_patch = true,
    exec_command = true,
}

local LARGE_RESULT_HINT_THRESHOLD = 10000

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
    local seen = {}
    if type(raw) == "string" then
        for item in raw:gmatch("[^,]+") do
            local name = util.trim(item)
            if name ~= "" and not seen[name] then
                seen[name] = true
                out[#out + 1] = name
            end
        end
    elseif type(raw) == "table" then
        for _, item in ipairs(raw) do
            local name = util.trim(item)
            if name ~= "" and not seen[name] then
                seen[name] = true
                out[#out + 1] = name
            end
        end
    end
    return out, seen
end

local function create_provider_state()
    local ext = external_cfg()
    local enabled = util.to_bool(ext.enabled, false)
    local allowlist, allowset = normalize_allowlist(ext.allowlist or ext.names)

    local provider = nil
    local provider_error = ""
    if enabled and #allowlist > 0 then
        provider, provider_error = provider_registry.create(ext.provider or "qwen", {
            allowlist = allowlist,
        })
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

local function get_task_profile(state)
    local profile = util.trim((((state or {}).context or {}).task_profile) or "")
    if profile ~= "" then
        return profile
    end
    return util.trim((((((state or {}).session or {}).active_task) or {}).profile) or "")
end

local function build_capabilities(state)
    local profile = get_task_profile(state)
    local uploads = (((state or {}).uploads) or {})
    local has_uploads = type(uploads) == "table" and #uploads > 0

    return {
        profile = profile ~= "" and profile or "general",
        workspace = profile == "workspace" or profile == "code" or has_uploads,
        code = profile == "code",
    }
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

    local call_id = util.trim(call.call_id or call.id)
    if call_id == "" then
        call_id = string.format("tool_call_%d", tonumber(idx) or 0)
    end

    return {
        tool = tool_name,
        args = args,
        call_id = call_id,
    }
end

local function append_schema_list(out, seen, schemas)
    for _, schema in ipairs(schemas or {}) do
        if type(schema) == "table" then
            local fn = schema["function"] or {}
            local name = util.trim(fn.name)
            if name ~= "" and not seen[name] then
                seen[name] = true
                out[#out + 1] = schema
            end
        end
    end
end

local function safe_trim_text(value)
    return util.trim(tostring(value or ""))
end

local function build_result_summary_rows(call_results)
    local lines = {}
    for _, row in ipairs(call_results or {}) do
        local tool_name = safe_trim_text((row or {}).tool)
        if tool_name == "" then
            tool_name = "unknown"
        end
        if row.ok == true then
            local detail = util.utf8_take(safe_trim_text((row or {}).result), 240)
            lines[#lines + 1] = string.format("- %s: ok%s", tool_name, detail ~= "" and (" | " .. detail) or "")
        else
            local err = util.utf8_take(safe_trim_text((row or {}).error), 180)
            lines[#lines + 1] = string.format("- %s: failed%s", tool_name, err ~= "" and (" | " .. err) or "")
        end
    end
    return table.concat(lines, "\n")
end

function M.get_supported_tools(state)
    local capabilities = build_capabilities(state)
    local out = {}

    for name, enabled in pairs(control_tools.supported_tools() or {}) do
        if enabled then
            out[name] = true
        end
    end

    if capabilities.workspace then
        for name, enabled in pairs(file_tools.supported_tools() or {}) do
            if enabled then
                out[name] = true
            end
        end
    end

    if capabilities.code then
        for name, enabled in pairs(code_tools.supported_tools() or {}) do
            if enabled then
                out[name] = true
            end
        end
    end

    local provider_state = create_provider_state()
    if provider_state.enabled and provider_state.provider then
        for _, schema in ipairs(provider_state.provider:list_tools() or {}) do
            local fn = schema["function"] or {}
            local name = util.trim(fn.name)
            if name ~= "" then
                out[name] = true
            end
        end
    end

    return out
end

function M.get_tool_schemas(state)
    local out = {}
    local seen = {}
    local capabilities = build_capabilities(state)

    append_schema_list(out, seen, control_tools.get_tool_schemas())
    if capabilities.workspace then
        append_schema_list(out, seen, file_tools.get_tool_schemas())
    end
    if capabilities.code then
        append_schema_list(out, seen, code_tools.get_tool_schemas())
    end

    local provider_state = create_provider_state()
    if provider_state.enabled and provider_state.provider then
        append_schema_list(out, seen, provider_state.provider:list_tools())
    end

    return out
end

function M.execute_calls(calls, state)
    local out = {
        executed = 0,
        failed = 0,
        skipped = 0,
        call_results = {},
        context_fragments = {},
        total_result_chars = 0,
        large_results = {},
        control_action = nil,
        control_data = nil,
        protocol_error = "",
        last_error = "",
        summary = "",
    }

    if type(calls) ~= "table" or #calls == 0 then
        return out
    end

    local supported = M.get_supported_tools(state)
    local provider_state = create_provider_state()
    local ext_cfg = external_cfg()
    local inject_external_context = util.to_bool(ext_cfg.context_inject, true)
    local external_context_max_chars = math.max(120, math.floor(tonumber(ext_cfg.context_max_chars) or 1200))
    local local_context_max_chars = math.max(120, math.floor(tonumber((tool_cfg().file_context_max_chars) or 4000)))
    local read_only = (((state or {}).input or {}).read_only) == true

    local normalized = {}
    local finish_calls = {}
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
        elseif not supported[call.tool] then
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
            normalized[#normalized + 1] = call
            if call.tool == "finish_turn" then
                finish_calls[#finish_calls + 1] = call
            end
        end
    end

    if #finish_calls > 1 then
        out.protocol_error = "multiple_finish_turn_calls"
        out.last_error = out.protocol_error
        out.summary = build_result_summary_rows(out.call_results)
        return out
    end

    if #finish_calls == 1 and #normalized > 1 then
        out.protocol_error = "invalid_mixed_terminal_batch"
        out.last_error = out.protocol_error
        out.summary = build_result_summary_rows(out.call_results)
        return out
    end

    if #finish_calls == 1 then
        local ok, result_or_err, control_info = control_tools.execute(finish_calls[1])
        if ok and control_info then
            out.executed = 1
            out.control_action = control_info.action
            out.control_data = control_info
            out.call_results[#out.call_results + 1] = {
                call_id = finish_calls[1].call_id,
                tool = finish_calls[1].tool,
                args = finish_calls[1].args,
                ok = true,
                error = "",
                result = tostring(result_or_err or ""),
                is_control = true,
            }
        else
            out.failed = out.failed + 1
            out.last_error = tostring(result_or_err or "finish_turn_failed")
            out.call_results[#out.call_results + 1] = {
                call_id = finish_calls[1].call_id,
                tool = finish_calls[1].tool,
                args = finish_calls[1].args,
                ok = false,
                error = out.last_error,
                result = "",
                is_control = true,
            }
        end
        out.summary = build_result_summary_rows(out.call_results)
        return out
    end

    for _, call in ipairs(normalized) do
        local ok = false
        local result_or_err = ""

        if read_only and SIDE_EFFECT_TOOLS[call.tool] then
            ok = false
            result_or_err = "read_only_mode_blocks_side_effect_tool"
        elseif file_tools.supported_tools()[call.tool] == true then
            ok, result_or_err = file_tools.execute(call)
        elseif code_tools.supported_tools()[call.tool] == true then
            ok, result_or_err = code_tools.execute(call)
        elseif provider_state.enabled and provider_state.provider then
            ok, result_or_err = provider_state.provider:call(call.tool, call.args)
        else
            ok = false
            result_or_err = provider_state.provider_error ~= "" and provider_state.provider_error or "external_provider_disabled"
        end

        local result_text = safe_trim_text(result_or_err)
        out.total_result_chars = out.total_result_chars + #result_text
        if #result_text > LARGE_RESULT_HINT_THRESHOLD then
            out.large_results[#out.large_results + 1] = {
                tool = call.tool,
                chars = #result_text,
                call_id = call.call_id,
            }
        end

        if ok then
            out.executed = out.executed + 1
            out.call_results[#out.call_results + 1] = {
                call_id = call.call_id,
                tool = call.tool,
                args = call.args,
                ok = true,
                error = "",
                result = result_text,
            }

            local clip_limit = READ_ONLY_TOOLS[call.tool] and local_context_max_chars or external_context_max_chars
            local prefix = (READ_ONLY_TOOLS[call.tool] or SIDE_EFFECT_TOOLS[call.tool]) and "[Tool:%s]\n%s" or "[External:%s]\n%s"
            local should_inject = READ_ONLY_TOOLS[call.tool] or SIDE_EFFECT_TOOLS[call.tool] or inject_external_context
            if should_inject and result_text ~= "" then
                out.context_fragments[#out.context_fragments + 1] = string.format(
                    prefix,
                    call.tool,
                    util.utf8_take(result_text, clip_limit)
                )
            end
        else
            out.failed = out.failed + 1
            out.last_error = result_text ~= "" and result_text or "tool_exec_failed"
            out.call_results[#out.call_results + 1] = {
                call_id = call.call_id,
                tool = call.tool,
                args = call.args,
                ok = false,
                error = out.last_error,
                result = "",
            }
        end
    end

    out.summary = build_result_summary_rows(out.call_results)
    return out
end

return M
