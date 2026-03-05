-- code_tools.lua
-- 代码结构分析工具：提取代码结构摘要，而非完整内容
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
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

-- ==================== Lua 代码结构提取 ====================

local LUA_KEYWORDS = {
    ["and"] = true, ["break"] = true, ["do"] = true, ["else"] = true,
    ["elseif"] = true, ["end"] = true, ["false"] = true, ["for"] = true,
    ["function"] = true, ["goto"] = true, ["if"] = true, ["in"] = true,
    ["local"] = true, ["nil"] = true, ["not"] = true, ["or"] = true,
    ["repeat"] = true, ["return"] = true, ["then"] = true, ["true"] = true,
    ["until"] = true, ["while"] = true,
}

-- 提取Lua函数定义
local function extract_lua_functions(content, max_funcs)
    max_funcs = max_funcs or 30
    local funcs = {}
    
    -- 匹配 function name(args) 和 local function name(args)
    for func_def in content:gmatch("[^\n]*function[^\n]*") do
        if #funcs >= max_funcs then
            break
        end
        
        -- 提取函数名和参数
        local name = func_def:match("function%s+([%w%._:]+)%s*%(")
        if not name then
            name = func_def:match("local%s+function%s+([%w_]+)%s*%(")
        end
        
        if name then
            -- 提取参数
            local params = func_def:match("%(([^)]*)%)") or ""
            
            -- 尝试提取行号
            local line_num = 0
            local pos = content:find(func_def, 1, true)
            if pos then
                for _ in content:sub(1, pos):gmatch("\n") do
                    line_num = line_num + 1
                end
            end
            
            funcs[#funcs + 1] = {
                name = name,
                params = params,
                line = line_num,
                type = name:find("%.") and "method" or "function",
            }
        end
    end
    
    return funcs
end

-- 提取Lua模块依赖（require调用）
local function extract_lua_requires(content)
    local requires = {}
    
    for module_path in content:gmatch('require%s*%(%s*["\']([^"\']+)["\']%s*%)') do
        requires[#requires + 1] = module_path
    end
    
    for module_path in content:gmatch('require%s*"([^"]+)"') do
        requires[#requires + 1] = module_path
    end
    
    for module_path in content:gmatch("require%s*'([^']+)'") do
        requires[#requires + 1] = module_path
    end
    
    -- 去重
    local seen = {}
    local unique = {}
    for _, m in ipairs(requires) do
        if not seen[m] then
            seen[m] = true
            unique[#unique + 1] = m
        end
    end
    
    return unique
end

-- 提取Lua全局变量和模块导出
local function extract_lua_exports(content)
    local exports = {}
    
    -- 提取 M.xxx = 形式的导出
    for name in content:gmatch("M%.([%w_]+)%s*=") do
        exports[#exports + 1] = name
    end
    
    -- 提取 return M 或 return { ... }
    local return_stmt = content:match("return%s+([^\n]+)")
    if return_stmt then
        exports[#exports + 1] = "return: " .. util.trim(return_stmt):sub(1, 50)
    end
    
    return exports
end

-- ==================== Python 代码结构提取 ====================

-- 提取Python函数/类定义
local function extract_python_structure(content, max_items)
    max_items = max_items or 30
    local items = {}
    
    -- 匹配 class 和 function 定义
    for line in content:gmatch("[^\n]+") do
        if #items >= max_items then
            break
        end
        
        -- class定义
        local class_name, bases = line:match("^class%s+([%w_]+)%s*%(([^)]*)%)")
        if class_name then
            items[#items + 1] = {
                type = "class",
                name = class_name,
                bases = bases,
                line = #items + 1,
            }
        elseif line:match("^class%s+[%w_]+") then
            class_name = line:match("^class%s+([%w_]+)")
            if class_name then
                items[#items + 1] = {
                    type = "class",
                    name = class_name,
                    bases = "",
                    line = #items + 1,
                }
            end
        end
        
        -- function定义
        local func_name, params = line:match("^[df][eo][fn]%s+([%w_]+)%s*%(([^)]*)%)")
        if func_name then
            items[#items + 1] = {
                type = "function",
                name = func_name,
                params = params,
                line = #items + 1,
            }
        end
    end
    
    return items
end

-- 提取Python导入
local function extract_python_imports(content)
    local imports = {}
    
    for mod in content:gmatch("from%s+([%w_.]+)%s+import") do
        imports[#imports + 1] = mod
    end
    
    for mod in content:gmatch("import%s+([%w_.]+)") do
        imports[#imports + 1] = mod
    end
    
    -- 去重
    local seen = {}
    local unique = {}
    for _, m in ipairs(imports) do
        if not seen[m] then
            seen[m] = true
            unique[#unique + 1] = m
        end
    end
    
    return unique
end

-- ==================== JavaScript 代码结构提取 ====================

local function extract_js_structure(content, max_items)
    max_items = max_items or 30
    local items = {}
    
    -- function name(...)
    for name, params in content:gmatch("function%s+([%w_]+)%s*%(([^)]*)%)") do
        if #items >= max_items then break end
        items[#items + 1] = { type = "function", name = name, params = params }
    end
    
    -- const name = (...) => 或 function
    for name, params in content:gmatch("const%s+([%w_]+)%s*=%s*%(?([^)]*)%)?%s*=>") do
        if #items >= max_items then break end
        items[#items + 1] = { type = "arrow_function", name = name, params = params }
    end
    
    -- class Name
    for name in content:gmatch("class%s+([%w_]+)") do
        if #items >= max_items then break end
        items[#items + 1] = { type = "class", name = name }
    end
    
    -- export
    for name in content:gmatch("export%s+([%w_]+)") do
        if #items >= max_items then break end
        items[#items + 1] = { type = "export", name = name }
    end
    
    return items
end

-- ==================== 通用结构提取 ====================

local function detect_language(path)
    local ext = path:match("%.(%w+)$")
    if not ext then return "unknown" end
    
    local lang_map = {
        lua = "lua",
        py = "python",
        js = "javascript",
        ts = "typescript",
        tsx = "typescript",
        jsx = "javascript",
        c = "c",
        h = "c",
        cpp = "cpp",
        hpp = "cpp",
        rs = "rust",
        go = "go",
        java = "java",
        kt = "kotlin",
        sh = "shell",
        json = "json",
        yaml = "yaml",
        yml = "yaml",
        md = "markdown",
        txt = "text",
    }
    
    return lang_map[ext:lower()] or "unknown"
end

-- 生成代码结构摘要
local function generate_outline(content, path, max_items)
    max_items = max_items or 30
    local lang = detect_language(path)
    local outline = {
        path = path,
        language = lang,
        line_count = 0,
        char_count = #content,
        functions = {},
        imports = {},
        exports = {},
    }
    
    -- 统计行数
    for _ in content:gmatch("\n") do
        outline.line_count = outline.line_count + 1
    end
    outline.line_count = outline.line_count + 1
    
    -- 根据语言提取结构
    if lang == "lua" then
        outline.functions = extract_lua_functions(content, max_items)
        outline.imports = extract_lua_requires(content)
        outline.exports = extract_lua_exports(content)
    elseif lang == "python" then
        outline.functions = extract_python_structure(content, max_items)
        outline.imports = extract_python_imports(content)
    elseif lang == "javascript" or lang == "typescript" then
        outline.functions = extract_js_structure(content, max_items)
    else
        -- 对于未知语言，提取可能的函数定义
        outline.functions = {}
        for line in content:gmatch("[^\n]+") do
            if #outline.functions >= max_items then break end
            local func_name = line:match("^[ \t]*([%w_]+)%s*%(")
            if func_name and not LUA_KEYWORDS[func_name] then
                outline.functions[#outline.functions + 1] = { name = func_name }
            end
        end
    end
    
    return outline
end

-- 格式化大纲为可读文本
local function format_outline(outline, max_width)
    max_width = max_width or 80
    local lines = {}
    
    lines[#lines + 1] = string.format("=== Code Outline: %s ===", outline.path)
    lines[#lines + 1] = string.format("Language: %s | Lines: %d | Chars: %d",
        outline.language, outline.line_count, outline.char_count)
    
    if #outline.imports > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "Imports/Requires:"
        for _, imp in ipairs(outline.imports) do
            lines[#lines + 1] = "  - " .. imp
        end
    end
    
    if #outline.exports > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "Exports:"
        for _, exp in ipairs(outline.exports) do
            lines[#lines + 1] = "  - " .. exp
        end
    end
    
    if #outline.functions > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "Functions/Classes:"
        for _, fn in ipairs(outline.functions) do
            local sig = fn.name
            if fn.params then
                sig = sig .. "(" .. fn.params .. ")"
            end
            if fn.type and fn.type ~= "function" then
                sig = sig .. " [" .. fn.type .. "]"
            end
            if fn.line then
                sig = sig .. " :L" .. fn.line
            end
            lines[#lines + 1] = "  - " .. sig
        end
    end
    
    return table.concat(lines, "\n")
end

-- ==================== 项目结构分析 ====================

-- 分析目录结构，生成摘要
local function analyze_project_structure(prefix, max_depth, max_files)
    max_depth = max_depth or 2
    max_files = max_files or 50
    
    local runtime = get_runtime()
    if not runtime or not has_method(runtime, "list_files") then
        return nil, "runtime_unavailable"
    end
    
    -- 获取文件列表
    local ok, result = pcall(function()
        return runtime:list_files(prefix or "", 1000)
    end)
    if not ok then
        return nil, tostring(result)
    end
    
    local files = {}
    local dirs = {}
    local ext_stats = {}
    
    -- 解析结果
    for line in tostring(result):gmatch("[^\n]+") do
        if #files >= max_files then break end
        
        local is_dir = line:find("/", #line) ~= nil or line:find("%[dir%]") ~= nil
        local path = util.trim(line:gsub("%[dir%]", ""))
        
        if is_dir then
            dirs[#dirs + 1] = path
        else
            files[#files + 1] = path
            local ext = path:match("%.(%w+)$") or "unknown"
            ext_stats[ext] = (ext_stats[ext] or 0) + 1
        end
    end
    
    return {
        prefix = prefix,
        file_count = #files,
        dir_count = #dirs,
        files = files,
        dirs = dirs,
        ext_stats = ext_stats,
    }
end

local function format_project_structure(structure, show_files)
    local lines = {}
    
    lines[#lines + 1] = string.format("=== Project Structure: %s ===", structure.prefix or "root")
    lines[#lines + 1] = string.format("Files: %d | Directories: %d",
        structure.file_count, structure.dir_count)
    
    -- 文件类型统计
    if structure.ext_stats then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "File Types:"
        local exts = {}
        for ext, count in pairs(structure.ext_stats) do
            exts[#exts + 1] = { ext = ext, count = count }
        end
        table.sort(exts, function(a, b) return a.count > b.count end)
        for _, item in ipairs(exts) do
            lines[#lines + 1] = string.format("  .%s: %d", item.ext, item.count)
        end
    end
    
    -- 目录列表
    if structure.dirs and #structure.dirs > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "Directories:"
        for _, dir in ipairs(structure.dirs) do
            lines[#lines + 1] = "  " .. dir
        end
    end
    
    -- 文件列表（可选）
    if show_files and structure.files and #structure.files > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "Files:"
        for _, file in ipairs(structure.files) do
            lines[#lines + 1] = "  " .. file
        end
    end
    
    return table.concat(lines, "\n")
end

-- ==================== 工具接口 ====================

function M.supported_tools()
    return {
        code_outline = true,
        project_structure = true,
        code_symbols = true,
    }
end

function M.get_tool_schemas()
    return {
        {
            type = "function",
            ["function"] = {
                name = "code_outline",
                description = "Get code structure outline (functions, classes, imports) without reading full file content. Use this for understanding code structure.",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "Relative path to code file" },
                        max_items = { type = "integer", description = "Maximum functions/items to extract (default 30)" },
                    },
                    required = { "path" },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "project_structure",
                description = "Get project directory structure summary with file type statistics. Use this to understand project layout.",
                parameters = {
                    type = "object",
                    properties = {
                        prefix = { type = "string", description = "Relative path prefix to analyze" },
                        max_files = { type = "integer", description = "Maximum files to scan (default 50)" },
                        show_files = { type = "boolean", description = "Include file list in output" },
                    },
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "code_symbols",
                description = "Search for function/class definitions by name across files. Returns file:line locations.",
                parameters = {
                    type = "object",
                    properties = {
                        pattern = { type = "string", description = "Symbol name pattern to search" },
                        prefix = { type = "string", description = "Directory prefix to search in" },
                        max_hits = { type = "integer", description = "Maximum hits (default 20)" },
                    },
                    required = { "pattern" },
                },
            },
        },
    }
end

function M.execute(call)
    local name = util.trim((call or {}).tool)
    local args = (call or {}).args or {}
    
    if name == "code_outline" then
        local path = util.trim(args.path)
        if path == "" then
            return false, "missing path"
        end
        
        local runtime = get_runtime()
        if not runtime or not has_method(runtime, "read_file") then
            return false, "runtime_unavailable"
        end
        
        -- 读取文件
        local ok, content = pcall(function()
            return runtime:read_file(path, 50000) -- 最大读取50000字符用于分析
        end)
        if not ok then
            return false, tostring(content)
        end
        
        content = tostring(content or "")
        if #content == 0 then
            return false, "file_empty"
        end
        
        -- 生成大纲
        local outline = generate_outline(content, path, args.max_items)
        local formatted = format_outline(outline)
        
        return true, formatted
    end
    
    if name == "project_structure" then
        local prefix = util.trim(args.prefix) or ""
        local max_files = math.max(10, math.floor(tonumber(args.max_files) or 50))
        
        local structure, err = analyze_project_structure(prefix, 2, max_files)
        if not structure then
            return false, err or "analysis_failed"
        end
        
        local formatted = format_project_structure(structure, args.show_files)
        return true, formatted
    end
    
    if name == "code_symbols" then
        local pattern = util.trim(args.pattern)
        if pattern == "" then
            return false, "missing pattern"
        end
        
        local runtime = get_runtime()
        if not runtime or not has_method(runtime, "search_files") then
            return false, "runtime_unavailable"
        end
        
        -- 构建搜索模式：匹配函数定义
        local search_patterns = {
            "function%s+" .. pattern,
            "local%s+function%s+" .. pattern,
            pattern .. "%s*%(",
            "class%s+" .. pattern,
            "def%s+" .. pattern,
            "const%s+" .. pattern,
        }
        
        local results = {}
        local seen = {}
        local max_hits = math.max(5, math.floor(tonumber(args.max_hits) or 20))
        
        for _, search_pat in ipairs(search_patterns) do
            if #results >= max_hits then break end
            
            local ok, search_result = pcall(function()
                return runtime:search_files(args.prefix or "", search_pat, max_hits, 10, 2)
            end)
            
            if ok then
                for line in tostring(search_result):gmatch("[^\n]+") do
                    if #results >= max_hits then break end
                    
                    local file, lnum, text = line:match("^([^:]+):(%d+):(.*)$")
                    if file and lnum and not seen[file .. ":" .. lnum] then
                        seen[file .. ":" .. lnum] = true
                        results[#results + 1] = {
                            file = file,
                            line = tonumber(lnum),
                            snippet = util.trim(text):sub(1, 80),
                        }
                    end
                end
            end
        end
        
        if #results == 0 then
            return true, "No symbols found matching: " .. pattern
        end
        
        local lines = { "Symbol search results for: " .. pattern }
        for _, r in ipairs(results) do
            lines[#lines + 1] = string.format("  %s:%d - %s", r.file, r.line, r.snippet)
        end
        
        return true, table.concat(lines, "\n")
    end
    
    return false, "unsupported_code_tool"
end

-- 直接暴露给其他模块使用
M.generate_outline = generate_outline
M.format_outline = format_outline
M.detect_language = detect_language

return M
