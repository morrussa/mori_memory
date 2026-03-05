local util = require("module.graph.util")

local M = {}

local function has_method(runtime, method_name)
    if runtime == nil then
        return false
    end
    local ok, attr = pcall(function()
        return runtime[method_name]
    end)
    return ok and attr ~= nil
end

local function get_runtime()
    return _G.py_pipeline
end

local function normalize_allowlist(raw)
    local out = {}
    local seen = {}
    if type(raw) == "string" then
        for name in raw:gmatch("[^,]+") do
            local n = util.trim(name)
            if n ~= "" and (not seen[n]) then
                seen[n] = true
                out[#out + 1] = n
            end
        end
        return out
    end
    if type(raw) ~= "table" then
        return out
    end
    for _, item in ipairs(raw) do
        local n = util.trim(item)
        if n ~= "" and (not seen[n]) then
            seen[n] = true
            out[#out + 1] = n
        end
    end
    return out
end

function M.new(opts)
    opts = opts or {}
    local allowlist = normalize_allowlist(opts.allowlist)
    local allowset = {}
    for _, name in ipairs(allowlist) do
        allowset[name] = true
    end

    local obj = {}

    function obj:list_tools()
        local runtime = get_runtime()
        if not has_method(runtime, "get_qwen_tool_schemas") then
            return {}
        end
        local ok, schemas = pcall(function()
            if #allowlist > 0 then
                return runtime:get_qwen_tool_schemas(allowlist)
            end
            return runtime:get_qwen_tool_schemas(nil)
        end)
        if not ok or type(schemas) ~= "table" then
            return {}
        end
        local out = {}
        for _, item in ipairs(schemas) do
            if type(item) == "table" and type(item["function"]) == "table" then
                local name = util.trim((item["function"] or {}).name)
                if name ~= "" then
                    if (#allowlist == 0) or allowset[name] then
                        out[#out + 1] = item
                    end
                end
            end
        end
        return out
    end

    function obj:call(tool_name, tool_args)
        local name = util.trim(tool_name)
        if name == "" then
            return false, "empty_external_tool_name"
        end
        if #allowlist > 0 and (not allowset[name]) then
            return false, "external_tool_not_allowlisted"
        end

        local runtime = get_runtime()
        if not has_method(runtime, "call_qwen_tool") then
            return false, "external_tool_runtime_unavailable"
        end

        local args_lua = util.encode_lua_value(tool_args or {}, 0)
        local ok, result_or_err = pcall(function()
            return runtime:call_qwen_tool(name, args_lua)
        end)
        if not ok then
            return false, tostring(result_or_err or "external_tool_call_failed")
        end
        return true, tostring(result_or_err or "")
    end

    return obj
end

return M
