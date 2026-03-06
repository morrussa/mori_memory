-- project_knowledge.lua
-- 项目知识注入模块：自动生成项目概览并注入到上下文中
local util = require("module.graph.util")
local config = require("module.config")
local code_tools = require("module.graph.code_tools")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function pk_cfg()
    local cfg = graph_cfg()
    return cfg.project_knowledge or {
        enabled = true,
        -- 项目概览最大字符数
        max_overview_chars = 2000,
        -- 是否自动生成模块列表
        auto_module_list = true,
        -- 是否包含文件类型统计
        include_ext_stats = true,
        -- 缓存时间（秒）
        cache_ttl = 300,
    }
end

local function should_enable_for_state(state)
    local profile = util.trim((((state or {}).context or {}).task_profile) or ((((state or {}).session or {}).active_task or {}).profile) or "")
    return profile == "code"
end

-- 缓存
local knowledge_cache = {
    overview = nil,
    generated_at = 0,
    modules = nil,
}

-- ==================== 模块路径转换 ====================

local function path_to_module(path)
    -- 将文件路径转换为模块名
    -- e.g., "module/graph/util.lua" -> "module.graph.util"
    local mod = path:gsub("%.lua$", "")
    mod = mod:gsub("/", ".")
    return mod
end

local function module_to_path(mod)
    -- 将模块名转换为文件路径
    -- e.g., "module.graph.util" -> "module/graph/util.lua"
    local path = mod:gsub("%.", "/") .. ".lua"
    return path
end

-- ==================== 核心模块识别 ====================

-- 识别项目的核心模块（根据配置或自动检测）
local function identify_core_modules()
    local cfg = pk_cfg()
    local core = {}
    
    -- 从配置中读取核心模块列表
    local configured = cfg.core_modules or {}
    for _, mod in ipairs(configured) do
        core[#core + 1] = mod
    end
    
    -- 如果配置为空，自动检测
    if #core == 0 and cfg.auto_module_list then
        -- 基于项目结构的智能检测
        local runtime = _G.py_pipeline
        if runtime and runtime.list_files then
            local ok, result = pcall(function()
                return runtime:list_files(util.workspace_virtual_root(), 100)
            end)
            if ok then
                local seen = {}
                for line in tostring(result):gmatch("[^\n]+") do
                    local path = util.trim(line)
                    if path:match("%.lua$") and not path:match("_test%.lua$") then
                        local mod = path_to_module(path)
                        if not seen[mod] and #core < 20 then
                            seen[mod] = true
                            core[#core + 1] = mod
                        end
                    end
                end
            end
        end
    end
    
    return core
end

-- ==================== 项目概览生成 ====================

local function generate_module_summary(mod_path)
    local runtime = _G.py_pipeline
    if not runtime or not runtime.read_file then
        return nil
    end
    
    local path = module_to_path(mod_path)
    local ok, content = pcall(function()
        return runtime:read_file(path, 8000) -- 读取前8000字符用于分析
    end)
    
    if not ok then
        return nil
    end
    
    content = tostring(content or "")
    if #content == 0 then
        return nil
    end
    
    -- 使用 code_tools 生成大纲
    local outline = code_tools.generate_outline(content, path, 15)
    
    -- 生成简洁摘要
    local summary = {
        module = mod_path,
        path = path,
        line_count = outline.line_count,
        functions = {},
        exports = {},
    }
    
    -- 提取主要函数名
    for _, fn in ipairs(outline.functions or {}) do
        if #summary.functions < 8 then
            summary.functions[#summary.functions + 1] = fn.name
        end
    end
    
    -- 提取导出
    for _, exp in ipairs(outline.exports or {}) do
        if #summary.exports < 8 then
            summary.exports[#summary.exports + 1] = exp
        end
    end
    
    return summary
end

local function generate_project_overview()
    local cfg = pk_cfg()
    local lines = {}
    
    lines[#lines + 1] = "=== Project Overview ==="
    lines[#lines + 1] = ""
    
    -- 项目基本信息
    lines[#lines + 1] = "Project: workspace code/task context"
    lines[#lines + 1] = ""
    
    -- 核心模块列表
    local core_modules = identify_core_modules()
    if #core_modules > 0 then
        lines[#lines + 1] = "Core Modules:"
        for _, mod in ipairs(core_modules) do
            lines[#lines + 1] = "  - " .. mod
        end
        lines[#lines + 1] = ""
    end
    
    -- 模块功能说明（硬编码 + 动态补充）
    local module_descriptions = {
        ["module.graph.graph_runtime"] = "Main graph execution runtime",
        ["module.graph.state_schema"] = "State definition and validation",
        ["module.graph.context_builder"] = "Builds planner/responder chat context",
        ["module.graph.tool_registry_v2"] = "Tool registry and execution policy",
        ["module.graph.file_tools"] = "Workspace file tools",
        ["module.graph.code_tools"] = "Workspace code analysis tools",
        ["module.graph.nodes.planner_node"] = "Planner with strict finish_turn contract",
        ["module.graph.nodes.tool_exec_node"] = "Tool execution and working-memory updates",
        ["module.graph.nodes.repair_node"] = "Explicit repair stage",
        ["module.graph.nodes.responder_node"] = "Final response generation stage",
        ["module.graph.util"] = "Utility functions (UTF-8, JSON, paths)",
        ["module.config"] = "Configuration management",
    }
    
    lines[#lines + 1] = "Module Functions:"
    for _, mod in ipairs(core_modules) do
        local desc = module_descriptions[mod]
        if desc then
            lines[#lines + 1] = string.format("  %s - %s", mod, desc)
        end
    end
    lines[#lines + 1] = ""
    
    -- 工具列表
    lines[#lines + 1] = "Available Tools:"
    lines[#lines + 1] = "  File: list_files, read_file, read_lines, search_file, search_files, write_file, apply_patch, exec_command"
    lines[#lines + 1] = "  Code: code_outline, project_structure, code_symbols"
    lines[#lines + 1] = "  Control: finish_turn"
    lines[#lines + 1] = ""
    
    -- 工作流说明
    lines[#lines + 1] = "Agent Workflow:"
    lines[#lines + 1] = "  1. Planner emits tools or finish_turn"
    lines[#lines + 1] = "  2. Tools run serially inside /mori/workspace"
    lines[#lines + 1] = "  3. Results update working memory"
    lines[#lines + 1] = "  4. Repair or continue planning until finish_turn"
    lines[#lines + 1] = "  5. Responder/finalize produce the user-visible reply"
    
    return table.concat(lines, "\n")
end

-- ==================== 知识注入 ====================

-- 获取项目知识（带缓存）
function M.get_project_knowledge(state, force_refresh)
    if type(state) ~= "table" then
        force_refresh = state
        state = nil
    end

    local cfg = pk_cfg()
    if not cfg.enabled then
        return nil
    end
    if not should_enable_for_state(state) then
        return nil
    end
    
    local now = os.time()
    local ttl = math.max(60, math.floor(tonumber(cfg.cache_ttl) or 300))
    
    if not force_refresh and knowledge_cache.overview and (now - knowledge_cache.generated_at) < ttl then
        return knowledge_cache.overview
    end
    
    local overview = generate_project_overview()
    
    -- 限制大小
    local max_chars = math.max(500, math.floor(tonumber(cfg.max_overview_chars) or 2000))
    if #overview > max_chars then
        overview = util.utf8_take(overview, max_chars) .. "\n[...truncated...]"
    end
    
    knowledge_cache.overview = overview
    knowledge_cache.generated_at = now
    
    return overview
end

-- 获取特定模块的详细信息
function M.get_module_detail(mod_path)
    return generate_module_summary(mod_path)
end

-- 格式化模块详情为文本
function M.format_module_detail(detail)
    if not detail then
        return "Module not found or could not be analyzed."
    end
    
    local lines = {}
    lines[#lines + 1] = string.format("Module: %s", detail.module)
    lines[#lines + 1] = string.format("Path: %s (%d lines)", detail.path, detail.line_count or 0)
    
    if detail.functions and #detail.functions > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "Functions:"
        for _, fn in ipairs(detail.functions) do
            lines[#lines + 1] = "  - " .. fn
        end
    end
    
    if detail.exports and #detail.exports > 0 then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "Exports:"
        for _, exp in ipairs(detail.exports) do
            lines[#lines + 1] = "  - " .. exp
        end
    end
    
    return table.concat(lines, "\n")
end

-- 清除缓存
function M.clear_cache()
    knowledge_cache = {
        overview = nil,
        generated_at = 0,
        modules = nil,
    }
end

-- 注入到系统提示
function M.inject_to_system_prompt(base_prompt)
    local overview = M.get_project_knowledge()
    if not overview or overview == "" then
        return base_prompt
    end
    
    return tostring(base_prompt or "") .. "\n\n" .. overview
end

-- 获取代码理解提示
function M.get_code_understanding_hints()
    local hints = {
        "When analyzing code:",
        "1. Use code_outline to get structure without reading full file",
        "2. Use project_structure to understand directory layout",
        "3. Use code_symbols to find function/class definitions by name",
        "4. Then use read_lines to read specific sections",
        "5. For large files, read in chunks using start_line and max_lines",
    }
    return table.concat(hints, "\n")
end

return M
