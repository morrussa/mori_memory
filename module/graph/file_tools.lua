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

local function normalize_workspace_path(raw)
    return util.normalize_tool_path(raw)
end

local function pick_first_nonempty(tbl, keys)
    if type(tbl) ~= "table" then
        return ""
    end
    for _, key in ipairs(keys or {}) do
        local v = util.trim(tbl[key])
        if v ~= "" then
            return v
        end
    end
    return ""
end

local function normalize_file_tool_args(name, args)
    if type(args) ~= "table" then
        args = {}
    end

    if name == "list_files" then
        local prefix = pick_first_nonempty(args, { "prefix", "path_prefix", "dir", "directory", "path" })
        if prefix ~= "" then
            args.prefix = normalize_workspace_path(prefix)
        end
        return args
    end

    if name == "read_file" or name == "read_lines" or name == "search_file" or name == "write_file" then
        local path = pick_first_nonempty(args, {
            "path",
            "file",
            "target",
            "filename",
            "file_name",
            "filepath",
            "file_path",
            "name",
        })
        if path ~= "" then
            args.path = normalize_workspace_path(path)
        end
        return args
    end

    if name == "search_files" then
        local prefix = pick_first_nonempty(args, {
            "prefix",
            "path_prefix",
            "dir",
            "directory",
            "path",
        })
        if prefix ~= "" then
            args.prefix = normalize_workspace_path(prefix)
        end
        return args
    end

    if name == "exec_command" then
        local workdir = pick_first_nonempty(args, { "workdir", "cwd", "dir", "directory", "path" })
        if workdir ~= "" then
            args.workdir = normalize_workspace_path(workdir)
        end
        return args
    end

    if name == "apply_patch" then
        local patch = util.trim(args.patch or args.diff)
        if patch ~= "" then
            args.patch = patch
        end
        return args
    end

    return args
end

function M.supported_tools()
    return {
        list_files = true,
        read_file = true,
        read_lines = true,
        search_file = true,
        search_files = true,
        write_file = true,
        apply_patch = true,
        exec_command = true,
    }
end

function M.get_tool_schemas()
    return {
        {
            type = "function",
            ["function"] = {
                name = "list_files",
                description = "List files from the agent workspace with optional prefix filtering.",
                parameters = {
                    type = "object",
                    properties = {
                        prefix = { type = "string", description = "Workspace prefix, e.g. /mori/workspace/download" },
                        limit = { type = "integer", description = "Maximum number of files to return" },
                    },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "read_file",
                description = "Read file content from /mori/workspace/*.",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "Workspace path, e.g. /mori/workspace/download/a.txt" },
                        max_chars = { type = "integer", description = "Maximum characters to read" },
                    },
                    required = { "path" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "read_lines",
                description = "Read a range of lines from a file in /mori/workspace/*.",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "Workspace path, e.g. /mori/workspace/app.lua" },
                        start_line = { type = "integer", description = "1-based start line" },
                        max_lines = { type = "integer", description = "Maximum number of lines" },
                    },
                    required = { "path" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "search_file",
                description = "Search a /mori/workspace/* file for a text pattern.",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "Workspace path" },
                        pattern = { type = "string", description = "Pattern to search for" },
                        max_hits = { type = "integer", description = "Maximum number of matches" },
                    },
                    required = { "path", "pattern" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "search_files",
                description = "Search files under /mori/workspace/* for a text pattern. Returns matching lines with optional context.",
                parameters = {
                    type = "object",
                    properties = {
                        prefix = { type = "string", description = "Workspace prefix, e.g. /mori/workspace/module/graph" },
                        pattern = { type = "string", description = "Pattern to search for (text or regex)" },
                        context_lines = { type = "integer", description = "Lines of context before/after match (0-30, default 0)" },
                        max_hits = { type = "integer", description = "Maximum total hits (default 30)" },
                        max_files = { type = "integer", description = "Maximum files to scan (default 24)" },
                        per_file_hits = { type = "integer", description = "Maximum hits per file (default 5)" },
                        regex = { type = "boolean", description = "Enable regex mode (default false)" },
                        case_sensitive = { type = "boolean", description = "Case sensitive search (default false)" },
                    },
                    required = { "pattern" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "write_file",
                description = "Write text content to a file under /mori/workspace/*. Use mode=overwrite or mode=append.",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "Workspace path" },
                        content = { type = "string", description = "Text content to write" },
                        mode = { type = "string", description = "overwrite or append" },
                    },
                    required = { "path", "content" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "apply_patch",
                description = "Apply a unified diff patch that only touches files under /mori/workspace/*.",
                parameters = {
                    type = "object",
                    properties = {
                        patch = { type = "string", description = "Unified diff patch text" },
                    },
                    required = { "patch" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "exec_command",
                description = "Run a command inside /mori/workspace/* using argv plus workdir. Shell strings are not allowed.",
                parameters = {
                    type = "object",
                    properties = {
                        argv = {
                            type = "array",
                            items = { type = "string" },
                            description = "Command and arguments"
                        },
                        workdir = { type = "string", description = "Workspace directory path" },
                        timeout_ms = { type = "integer", description = "Timeout in milliseconds" },
                        max_output_chars = { type = "integer", description = "Maximum output size" },
                    },
                    required = { "argv" },
                },
            },
        },
    }
end

function M.execute(call)
    local name = util.trim((call or {}).tool)
    local args = normalize_file_tool_args(name, (call or {}).args or {})
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

    if name == "write_file" then
        return call_runtime({ "write_file", "write_agent_file" }, args_lua)
    end

    if name == "apply_patch" then
        return call_runtime({ "apply_patch", "apply_agent_patch" }, args_lua)
    end

    if name == "exec_command" then
        return call_runtime({ "exec_command", "exec_agent_command" }, args_lua)
    end

    return false, "unsupported_file_tool"
end

return M
