local util = require("module.graph.util")

local M = {}

local WRITE_INTENT_PATTERNS = {
    "write", "patch", "modify", "edit", "update", "implement", "fix", "refactor",
    "写", "修改", "编辑", "实现", "修复", "重构", "补丁",
}

local ANALYSIS_PATTERNS = {
    "analyze", "analysis", "explain", "review", "inspect", "read", "search",
    "分析", "解释", "审查", "检查", "查看", "读取", "搜索",
}

local CODE_HINT_PATTERNS = {
    "code", "bug", "debug", "patch", "repo", "function", "class", "test", "compile", "build",
    "代码", "函数", "类", "修复", "测试", "编译", "模块",
}

local WORKSPACE_HINT_PATTERNS = {
    "file", "files", "read", "write", "search", "folder", "directory", "command", "upload", "workspace",
    "文件", "目录", "搜索", "命令", "上传",
}

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function trim(s)
    return util.trim(s or "")
end

local function deep_copy(value, seen)
    if type(value) ~= "table" then
        return value
    end

    seen = seen or {}
    if seen[value] then
        return seen[value]
    end

    local out = {}
    seen[value] = out
    for k, v in pairs(value) do
        out[deep_copy(k, seen)] = deep_copy(v, seen)
    end
    return out
end

local function normalize_bool(value, fallback)
    if value == nil then
        return fallback
    end
    return value == true
end

local function contains_any(text, patterns)
    local raw = tostring(text or "")
    local lower = raw:lower()
    for _, pattern in ipairs(patterns or {}) do
        local token = tostring(pattern or "")
        if token ~= "" and lower:find(token:lower(), 1, true) ~= nil then
            return true
        end
    end
    return false
end

function M.default_runtime_policy()
    return {
        recall = {
            mode = "auto",
        },
        episode = {
            mode = "auto",
        },
        planner = {
            mode = "auto",
            preferred_tool_chain = {},
            avoid_tools = {},
            force_read_before_write = nil,
        },
        repair = {
            mode = "normal",
        },
        budget = {
            remaining_steps_delta = 0,
            tool_loop_max_delta = 0,
        },
        context = {
            include_memory = true,
            include_episode = true,
        },
    }
end

function M.normalize_runtime_policy(raw)
    local policy = M.default_runtime_policy()
    raw = type(raw) == "table" and raw or {}

    policy.recall.mode = trim((((raw or {}).recall) or {}).mode)
    if policy.recall.mode == "" then
        policy.recall.mode = "auto"
    end

    policy.episode.mode = trim((((raw or {}).episode) or {}).mode)
    if policy.episode.mode == "" then
        policy.episode.mode = "auto"
    end

    policy.planner.mode = trim((((raw or {}).planner) or {}).mode)
    if policy.planner.mode == "" then
        policy.planner.mode = "auto"
    end

    policy.planner.preferred_tool_chain = {}
    for _, name in ipairs(((((raw or {}).planner) or {}).preferred_tool_chain) or {}) do
        local tool_name = trim(name)
        if tool_name ~= "" then
            policy.planner.preferred_tool_chain[#policy.planner.preferred_tool_chain + 1] = tool_name
        end
    end

    policy.planner.avoid_tools = {}
    for tool_name, enabled in pairs(((((raw or {}).planner) or {}).avoid_tools) or {}) do
        if enabled then
            policy.planner.avoid_tools[tostring(tool_name)] = true
        end
    end

    local force_read = ((((raw or {}).planner) or {}).force_read_before_write)
    if type(force_read) == "boolean" then
        policy.planner.force_read_before_write = force_read
    else
        policy.planner.force_read_before_write = nil
    end

    policy.repair.mode = trim((((raw or {}).repair) or {}).mode)
    if policy.repair.mode == "" then
        policy.repair.mode = "normal"
    end

    policy.budget.remaining_steps_delta = math.floor(tonumber(((((raw or {}).budget) or {}).remaining_steps_delta)) or 0)
    policy.budget.tool_loop_max_delta = math.floor(tonumber(((((raw or {}).budget) or {}).tool_loop_max_delta)) or 0)

    local include_memory = ((((raw or {}).context) or {}).include_memory)
    local include_episode = ((((raw or {}).context) or {}).include_episode)
    policy.context.include_memory = normalize_bool(include_memory, true)
    policy.context.include_episode = normalize_bool(include_episode, true)

    return policy
end

function M.merge_runtime_policies(items)
    local merged = M.default_runtime_policy()
    local defaults = M.default_runtime_policy()

    for _, item in ipairs(items or {}) do
        local candidate = M.normalize_runtime_policy((item or {}).patch)

        if merged.recall.mode == defaults.recall.mode and candidate.recall.mode ~= defaults.recall.mode then
            merged.recall.mode = candidate.recall.mode
        end
        if merged.episode.mode == defaults.episode.mode and candidate.episode.mode ~= defaults.episode.mode then
            merged.episode.mode = candidate.episode.mode
        end
        if merged.planner.mode == defaults.planner.mode and candidate.planner.mode ~= defaults.planner.mode then
            merged.planner.mode = candidate.planner.mode
        end
        if #merged.planner.preferred_tool_chain == 0 and #candidate.planner.preferred_tool_chain > 0 then
            merged.planner.preferred_tool_chain = deep_copy(candidate.planner.preferred_tool_chain)
        end
        for tool_name, enabled in pairs(candidate.planner.avoid_tools or {}) do
            if enabled then
                merged.planner.avoid_tools[tostring(tool_name)] = true
            end
        end
        if merged.planner.force_read_before_write == nil and type(candidate.planner.force_read_before_write) == "boolean" then
            merged.planner.force_read_before_write = candidate.planner.force_read_before_write
        end
        if merged.repair.mode == defaults.repair.mode and candidate.repair.mode ~= defaults.repair.mode then
            merged.repair.mode = candidate.repair.mode
        end
        if merged.budget.remaining_steps_delta == defaults.budget.remaining_steps_delta
            and candidate.budget.remaining_steps_delta ~= defaults.budget.remaining_steps_delta then
            merged.budget.remaining_steps_delta = candidate.budget.remaining_steps_delta
        end
        if merged.budget.tool_loop_max_delta == defaults.budget.tool_loop_max_delta
            and candidate.budget.tool_loop_max_delta ~= defaults.budget.tool_loop_max_delta then
            merged.budget.tool_loop_max_delta = candidate.budget.tool_loop_max_delta
        end
        if merged.context.include_memory == defaults.context.include_memory
            and candidate.context.include_memory ~= defaults.context.include_memory then
            merged.context.include_memory = candidate.context.include_memory
        end
        if merged.context.include_episode == defaults.context.include_episode
            and candidate.context.include_episode ~= defaults.context.include_episode then
            merged.context.include_episode = candidate.context.include_episode
        end
    end

    if merged.episode.mode == "prefer" then
        merged.context.include_episode = true
    elseif merged.episode.mode == "suppress" then
        merged.context.include_episode = false
    end

    if merged.recall.mode == "suppress" then
        merged.context.include_memory = false
    end

    return merged
end

local function append_line(lines, text)
    local row = trim(text)
    if row ~= "" then
        lines[#lines + 1] = row
    end
end

function M.summarize_runtime_policy(policy)
    local normalized = M.normalize_runtime_policy(policy)
    local lines = {}

    if normalized.recall.mode ~= "auto" then
        append_line(lines, "recall.mode=" .. normalized.recall.mode)
    end
    if normalized.episode.mode ~= "auto" then
        append_line(lines, "episode.mode=" .. normalized.episode.mode)
    end
    if normalized.planner.mode ~= "auto" then
        append_line(lines, "planner.mode=" .. normalized.planner.mode)
    end
    if #normalized.planner.preferred_tool_chain > 0 then
        append_line(lines, "planner.preferred_tool_chain=" .. table.concat(normalized.planner.preferred_tool_chain, ">"))
    end

    local avoid = {}
    for tool_name, enabled in pairs(normalized.planner.avoid_tools or {}) do
        if enabled then
            avoid[#avoid + 1] = tostring(tool_name)
        end
    end
    table.sort(avoid)
    if #avoid > 0 then
        append_line(lines, "planner.avoid_tools=" .. table.concat(avoid, ","))
    end

    if type(normalized.planner.force_read_before_write) == "boolean" then
        append_line(lines, "planner.force_read_before_write=" .. tostring(normalized.planner.force_read_before_write))
    end
    if normalized.repair.mode ~= "normal" then
        append_line(lines, "repair.mode=" .. normalized.repair.mode)
    end
    if normalized.budget.remaining_steps_delta ~= 0 then
        append_line(lines, string.format("budget.remaining_steps_delta=%+d", normalized.budget.remaining_steps_delta))
    end
    if normalized.budget.tool_loop_max_delta ~= 0 then
        append_line(lines, string.format("budget.tool_loop_max_delta=%+d", normalized.budget.tool_loop_max_delta))
    end
    if normalized.context.include_memory ~= true then
        append_line(lines, "context.include_memory=false")
    end
    if normalized.context.include_episode ~= true then
        append_line(lines, "context.include_episode=false")
    end

    return table.concat(lines, "\n")
end

function M.build_contract_shape(task_profile, user_input, contract, read_only)
    if read_only == true then
        return "readonly"
    end

    local profile = trim(task_profile)
    local raw = table.concat({
        tostring(user_input or ""),
        tostring((((contract or {}).goal) or "")),
        table.concat((((contract or {}).deliverables) or {}), "\n"),
    }, "\n")

    if contains_any(raw, WRITE_INTENT_PATTERNS) then
        return "mutation"
    end

    if profile == "code" or profile == "workspace" then
        if contains_any(raw, ANALYSIS_PATTERNS) then
            return "analysis"
        end
        return "execution"
    end

    if contains_any(raw, ANALYSIS_PATTERNS) then
        return "analysis"
    end

    return "general"
end

function M.build_policy_key(parts)
    local payload = {
        "profile=" .. trim(parts.task_profile or "general"),
        "task=" .. trim(parts.task_type or "general"),
        "domain=" .. trim(parts.domain or "general"),
        "lang=" .. trim(parts.language or "unknown"),
        "readonly=" .. tostring(parts.read_only == true),
        "uploads=" .. tostring(parts.has_uploads == true),
        "shape=" .. trim(parts.contract_shape or "general"),
    }
    return table.concat(payload, "|")
end

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

function M.detect_task_profile(state)
    local active_task = ((((state or {}).session or {}).active_task) or {})
    local existing = trim((active_task or {}).profile or "")
    local user_input = tostring((((state or {}).input or {}).message) or "")
    local lower = user_input:lower()
    local uploads = (((state or {}).uploads) or {})
    local working_memory = ((state or {}).working_memory) or {}

    if type(uploads) == "table" and #uploads > 0 then
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

    for path, enabled in pairs((working_memory.files_read_set or {})) do
        if enabled and has_code_extension(path) then
            return "code"
        end
    end
    for path, enabled in pairs((working_memory.files_written_set or {})) do
        if enabled and has_code_extension(path) then
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

function M.compute_query_score(policy, query)
    if type(policy) ~= "table" or type(query) ~= "table" then
        return 0.0
    end

    local query_key = trim(query.policy_key)
    local policy_key = trim(policy.policy_key)
    if query_key ~= "" and query_key == policy_key then
        return 1.0
    end

    local score = 0.0
    if trim(policy.task_profile) ~= "" and trim(policy.task_profile) == trim(query.task_profile) then
        score = score + 0.28
    end
    if trim(policy.task_type) ~= "" and trim(policy.task_type) == trim(query.task_type) then
        score = score + 0.20
    end
    if trim(policy.domain) ~= "" and trim(policy.domain) == trim(query.domain) then
        score = score + 0.12
    end
    if trim(policy.language) ~= "" and trim(policy.language) == trim(query.language) then
        score = score + 0.10
    end
    if (policy.read_only == true) == (query.read_only == true) then
        score = score + 0.08
    end
    if (policy.has_uploads == true) == (query.has_uploads == true) then
        score = score + 0.06
    end
    if trim(policy.contract_shape) ~= "" and trim(policy.contract_shape) == trim(query.contract_shape) then
        score = score + 0.16
    end
    return clamp(score, 0.0, 0.99)
end

return M
