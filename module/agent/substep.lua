local M = {}

local DEFAULT_PROFILES = {
    ["general-purpose"] = {
        label = "general-purpose",
        description = "通用任务处理、代码搜索、多步骤任务",
        system_prompt = "优先给出可执行结果；必要时分步推进，并保持每一步可验证。",
        planner = {},
    },
    explore = {
        label = "Explore",
        description = "快速探索代码库、查找文件、搜索关键词",
        system_prompt = "先快速定位范围并收集证据，再基于证据给出结论。",
        planner = {
            planner_gate_mode = "always",
            planner_default_when_missing = true,
        },
    },
    plan = {
        label = "Plan",
        description = "架构规划、实现方案设计",
        system_prompt = "先给出架构与实施路径，再细化模块边界、风险和验收方法。",
        planner = {},
    },
}

local DEFAULT_ROUTE_KEYWORDS = {
    plan = {
        "架构",
        "规划",
        "方案",
        "设计",
        "实现方案",
        "技术方案",
        "plan",
        "roadmap",
        "design",
        "architecture",
    },
    explore = {
        "探索",
        "代码搜索",
        "搜索",
        "检索",
        "查找",
        "定位",
        "文件",
        "目录",
        "关键词",
        "grep",
        "rg",
        "ripgrep",
        "find",
    },
}

local NAME_ALIAS = {
    general = "general-purpose",
    ["general-purpose"] = "general-purpose",
    general_purpose = "general-purpose",
    generalpurpose = "general-purpose",
    default = "general-purpose",

    explore = "explore",
    scan = "explore",
    search = "explore",

    plan = "plan",
    planner = "plan",
    design = "plan",
    architecture = "plan",
}

local function trim(s)
    if not s then return "" end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function to_bool(v, fallback)
    if type(v) == "boolean" then return v end
    if type(v) == "number" then return v ~= 0 end
    if type(v) == "string" then
        local s = v:lower()
        if s == "true" or s == "1" or s == "yes" then return true end
        if s == "false" or s == "0" or s == "no" then return false end
    end
    return fallback == true
end

local function normalize_name(raw_name)
    local s = trim(raw_name):lower()
    if s == "" then
        return nil
    end
    s = s:gsub("[%s_]+", "-")
    s = s:gsub("[^%w%-]", "")
    return NAME_ALIAS[s] or s
end

local function clone_profile(profile, name)
    profile = profile or {}
    local planner = {}
    if type(profile.planner) == "table" then
        for k, v in pairs(profile.planner) do
            planner[k] = v
        end
    end
    local out = {
        name = normalize_name(name or profile.name) or "general-purpose",
        label = trim(profile.label),
        description = trim(profile.description),
        system_prompt = trim(profile.system_prompt),
        planner = planner,
    }
    if out.label == "" then
        out.label = out.name
    end
    return out
end

local function clone_registry(registry)
    local out = {}
    for name, profile in pairs(registry or {}) do
        local normalized = normalize_name(name or (profile or {}).name)
        if normalized then
            out[normalized] = clone_profile(profile, normalized)
        end
    end
    return out
end

local function merge_registry(base, extra)
    local out = clone_registry(base)
    for name, profile in pairs(extra or {}) do
        if type(profile) == "table" then
            local normalized = normalize_name(name or profile.name)
            if normalized then
                local merged = out[normalized] or {}
                merged.name = normalized
                merged.label = trim(profile.label or merged.label)
                merged.description = trim(profile.description or merged.description)
                merged.system_prompt = trim(profile.system_prompt or merged.system_prompt)
                local planner = {}
                if type(merged.planner) == "table" then
                    for k, v in pairs(merged.planner) do
                        planner[k] = v
                    end
                end
                if type(profile.planner) == "table" then
                    for k, v in pairs(profile.planner) do
                        planner[k] = v
                    end
                end
                merged.planner = planner
                if merged.label == "" then
                    merged.label = normalized
                end
                out[normalized] = merged
            end
        end
    end
    return out
end

local function normalize_keywords(raw, fallback)
    local out = {}
    local src = raw
    if type(src) ~= "table" then
        src = fallback or {}
    end
    for _, kw in ipairs(src) do
        local token = trim(kw)
        if token ~= "" then
            out[#out + 1] = token
        end
    end
    return out
end

local function contains_any_keyword(text, keywords)
    local source = trim(text)
    if source == "" then
        return false
    end
    local lowered = source:lower()
    for _, kw in ipairs(keywords or {}) do
        local token = trim(kw)
        if token ~= "" then
            if lowered:find(token:lower(), 1, true) then
                return true
            end
        end
    end
    return false
end

local function pick_existing_profile(registry, preferred_name)
    local normalized = normalize_name(preferred_name)
    if normalized and registry[normalized] then
        return registry[normalized]
    end
    if registry["general-purpose"] then
        return registry["general-purpose"]
    end
    for _, profile in pairs(registry) do
        return profile
    end
    return clone_profile(DEFAULT_PROFILES["general-purpose"], "general-purpose")
end

function M.normalize_name(raw_name)
    return normalize_name(raw_name)
end

function M.resolve_registry(runtime_substeps, default_substeps)
    local registry = clone_registry(DEFAULT_PROFILES)
    if type(default_substeps) == "table" then
        registry = merge_registry(registry, default_substeps)
    end
    if type(runtime_substeps) == "table" then
        registry = merge_registry(registry, runtime_substeps)
    end
    if not registry["general-purpose"] then
        registry["general-purpose"] = clone_profile(DEFAULT_PROFILES["general-purpose"], "general-purpose")
    end
    return registry
end

function M.resolve_default_name(agent_cfg, agent_defaults, registry)
    registry = registry or M.resolve_registry(nil, nil)
    local cfg_name = normalize_name((agent_cfg or {}).substep_default)
    if cfg_name and registry[cfg_name] then
        return cfg_name
    end
    local def_name = normalize_name((agent_defaults or {}).substep_default)
    if def_name and registry[def_name] then
        return def_name
    end
    if registry["general-purpose"] then
        return "general-purpose"
    end
    for name in pairs(registry) do
        return name
    end
    return "general-purpose"
end

function M.resolve_turn_substep(opts)
    opts = opts or {}
    local registry = opts.registry or M.resolve_registry(nil, nil)
    local requested = normalize_name(opts.requested)
    if requested and registry[requested] then
        return registry[requested], "requested"
    end

    local auto_route = to_bool(opts.auto_route, true)
    if auto_route then
        local user_text = tostring(opts.user_input or "")
        local plan_keywords = normalize_keywords(opts.plan_keywords, DEFAULT_ROUTE_KEYWORDS.plan)
        if registry.plan and contains_any_keyword(user_text, plan_keywords) then
            return registry.plan, "auto_plan"
        end
        local explore_keywords = normalize_keywords(opts.explore_keywords, DEFAULT_ROUTE_KEYWORDS.explore)
        if registry.explore and contains_any_keyword(user_text, explore_keywords) then
            return registry.explore, "auto_explore"
        end
    end

    local fallback = pick_existing_profile(registry, opts.default_name)
    return fallback, "default"
end

return M
