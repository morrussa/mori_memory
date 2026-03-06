local context_builder = require("module.graph.context_builder")
local util = require("module.graph.util")

local M = {}

local CODE_HINT_PATTERNS = {
    "code",
    "bug",
    "debug",
    "patch",
    "repo",
    "function",
    "class",
    "test",
    "compile",
    "build",
    "代码",
    "函数",
    "类",
    "修复",
    "测试",
    "编译",
    "模块",
}

local WORKSPACE_HINT_PATTERNS = {
    "file",
    "files",
    "read",
    "write",
    "search",
    "folder",
    "directory",
    "command",
    "upload",
    "workspace",
    "文件",
    "目录",
    "搜索",
    "命令",
    "上传",
}

local function has_code_extension(text)
    local s = tostring(text or ""):lower()
    return s:find("%.lua", 1, false) ~= nil
        or s:find("%.py", 1, false) ~= nil
        or s:find("%.js", 1, false) ~= nil
        or s:find("%.ts", 1, false) ~= nil
        or s:find("%.tsx", 1, false) ~= nil
        or s:find("%.jsx", 1, false) ~= nil
        or s:find("%.rs", 1, false) ~= nil
        or s:find("%.go", 1, false) ~= nil
        or s:find("%.java", 1, false) ~= nil
        or s:find("%.c", 1, false) ~= nil
        or s:find("%.cpp", 1, false) ~= nil
        or s:find("%.h", 1, false) ~= nil
end

local function detect_task_profile(state)
    local session = ((state or {}).session) or {}
    local active_task = session.active_task or {}
    local task_decision = ((((state or {}).task or {}).decision) or {})
    local existing = util.trim(active_task.profile or "")
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local lower = user_input:lower()
    local working_memory = ((state or {}).working_memory) or {}

    local decision_kind = util.trim(task_decision.kind or "")
    if (decision_kind == "same_task_step" or decision_kind == "same_task_refine" or decision_kind == "meta_turn")
        and existing ~= "" then
        return existing
    end

    if type((state or {}).uploads) == "table" and #((state or {}).uploads) > 0 then
        if has_code_extension(user_input) or has_code_extension(active_task.goal or "") then
            return "code"
        end
        return "workspace"
    end

    for _, pattern in ipairs(CODE_HINT_PATTERNS) do
        if lower:find(pattern, 1, true) ~= nil then
            return "code"
        end
    end
    if has_code_extension(user_input) then
        return "code"
    end

    for path, _ in pairs((working_memory.files_read_set or {})) do
        if has_code_extension(path) then
            return "code"
        end
    end
    for path, _ in pairs((working_memory.files_written_set or {})) do
        if has_code_extension(path) then
            return "code"
        end
    end

    for _, pattern in ipairs(WORKSPACE_HINT_PATTERNS) do
        if lower:find(pattern, 1, true) ~= nil then
            return "workspace"
        end
    end
    if lower:find("/mori/workspace", 1, true) ~= nil then
        return "workspace"
    end

    return existing ~= "" and existing or "general"
end

function M.run(state, _ctx)
    state.context = state.context or {}
    state.session = state.session or { mode = "single", active_task = {} }
    state.session.active_task = state.session.active_task or {}

    local profile = detect_task_profile(state)
    state.context.task_profile = profile
    state.context.workspace_virtual_root = util.workspace_virtual_root()
    state.session.active_task.profile = profile

    local messages, meta = context_builder.build_chat_messages(state)
    state.messages = state.messages or {}
    state.messages.chat_messages = messages
    state.messages.runtime_messages = messages
    state.agent_loop = state.agent_loop or {}
    state.agent_loop.pending_tool_calls = state.agent_loop.pending_tool_calls or {}
    state.agent_loop.stop_reason = util.trim(state.agent_loop.stop_reason or "")
    state.agent_loop.iteration = tonumber(state.agent_loop.iteration) or 0
    state.metrics = state.metrics or {}
    state.metrics.context = meta
    return state
end

return M
