local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function file_cfg()
    return (graph_cfg().file_tools or {})
end

local function get_runtime()
    return _G.py_pipeline
end

local function has_method(runtime, method_name)
    if runtime == nil then
        return false
    end
    local ok, attr = pcall(function()
        return runtime[method_name]
    end)
    return ok and attr ~= nil
end

local function call_runtime(method_names, args_lua, defaults)
    local runtime = get_runtime()
    if runtime == nil then
        return false, "python_runtime_unavailable"
    end

    local selected = nil
    for _, name in ipairs(method_names or {}) do
        if has_method(runtime, name) then
            selected = name
            break
        end
    end
    if not selected then
        return false, "python_method_unavailable"
    end

    local ok, result_or_err = pcall(function()
        local unpack_fn = (table and table.unpack) or unpack
        if defaults and #defaults > 0 then
            return runtime[selected](runtime, args_lua, unpack_fn(defaults))
        end
        return runtime[selected](runtime, args_lua)
    end)
    if not ok then
        return false, tostring(result_or_err or "runtime_call_failed")
    end
    return true, tostring(result_or_err or "")
end

local function default_numbers(...)
    local out = {}
    for i = 1, select("#", ...) do
        local value = select(i, ...)
        out[i] = math.max(1, math.floor(tonumber(value) or 1))
    end
    return out
end

function M.supported_tools()
    return {
        list_files = true,
        read_file = true,
        read_lines = true,
        search_file = true,
        search_files = true,
    }
end

function M.execute(call)
    local name = util.trim((call or {}).tool)
    local args = (call or {}).args or {}
    local args_lua = util.encode_lua_value(args, 0)
    local cfg = file_cfg()

    if name == "list_files" then
        return call_runtime(
            { "list_files", "list_agent_files" },
            args_lua,
            default_numbers(cfg.list_default_limit or 12, cfg.list_hard_limit or 64)
        )
    end

    if name == "read_file" then
        return call_runtime(
            { "read_file", "read_agent_file" },
            args_lua,
            default_numbers(cfg.read_default_max_chars or 3000, cfg.read_hard_max_chars or 12000)
        )
    end

    if name == "read_lines" then
        return call_runtime(
            { "read_lines", "read_agent_file_lines" },
            args_lua,
            default_numbers(cfg.read_lines_default_max_lines or 220, cfg.read_lines_hard_max_lines or 1200)
        )
    end

    if name == "search_file" then
        return call_runtime(
            { "search_file", "search_agent_file" },
            args_lua,
            default_numbers(cfg.search_default_max_hits or 20, cfg.search_hard_max_hits or 200)
        )
    end

    if name == "search_files" then
        return call_runtime(
            { "search_files", "search_agent_files" },
            args_lua,
            default_numbers(
                cfg.search_files_default_max_hits or 30,
                cfg.search_files_hard_max_hits or 400,
                cfg.search_files_default_max_files or 24,
                cfg.search_files_hard_max_files or 200,
                cfg.search_files_default_per_file_hits or 5,
                cfg.search_files_hard_per_file_hits or 20
            )
        )
    end

    return false, "unsupported_file_tool"
end

return M
