local policy = require("module.experience.policy")
local adaptive = require("module.experience.adaptive")

local M = {}

local LANGUAGE_HINTS = {
    { pattern = "%f[%a]lua%f[%A]", language = "lua" },
    { pattern = "%f[%a]python%f[%A]", language = "python" },
    { pattern = "%f[%a]javascript%f[%A]", language = "javascript" },
    { pattern = "%f[%a]typescript%f[%A]", language = "typescript" },
    { pattern = "%f[%a]java%f[%A]", language = "java" },
    { pattern = "%f[%a]rust%f[%A]", language = "rust" },
    { pattern = "%f[%a]go%f[%A]", language = "go" },
    { pattern = "%f[%a]cpp%f[%A]", language = "cpp" },
}

local function trim(s)
    return (tostring(s or ""):gsub("^%s*(.-)%s*$", "%1"))
end

local function detect_language(text)
    local raw = tostring(text or "")
    local lower = raw:lower()

    if lower:find("c++", 1, true) ~= nil then
        return "cpp"
    end

    for _, hint in ipairs(LANGUAGE_HINTS) do
        if lower:match(hint.pattern) then
            return hint.language
        end
    end
    return nil
end

function M.extract_features_from_text(text)
    local raw = tostring(text or "")
    local lower = raw:lower()
    local features = {
        query_text = raw,
    }

    if raw:match("代码") or raw:match("编程") or lower:find("debug", 1, true) ~= nil then
        features.task_type = "coding"
        features.domain = "coding"
    elseif raw:match("分析") or lower:find("analy", 1, true) ~= nil or lower:find("review", 1, true) ~= nil then
        features.task_type = "analysis"
        features.domain = "analysis"
    elseif raw:match("对话") then
        features.task_type = "conversation"
        features.domain = "conversation"
    else
        features.task_type = "general"
        features.domain = "general"
    end

    features.language = detect_language(raw) or "unknown"
    return features
end

function M.extract_query_features(query)
    if type(query) == "string" then
        return M.extract_features_from_text(query)
    end

    local features = {}
    if type(query) == "table" then
        features.query_text = tostring(query.text or query.content or query.description or "")
        features.task_profile = trim(query.task_profile or "")
        features.task_type = trim(query.task_type or "")
        features.domain = trim(query.domain or "")
        features.language = trim(query.language or "")
        features.read_only = query.read_only == true
        features.has_uploads = query.has_uploads == true
        features.contract_shape = trim(query.contract_shape or "")
        features.policy_key = trim(query.policy_key or "")
        features.family_key = trim(query.family_key or query.policy_key or "")
        features.context_signature = type(query.context_signature) == "table" and query.context_signature or {}
    end

    if trim(features.query_text) ~= "" then
        local from_text = M.extract_features_from_text(features.query_text)
        if trim(features.task_type) == "" then
            features.task_type = from_text.task_type
        end
        if trim(features.domain) == "" then
            features.domain = from_text.domain
        end
        if trim(features.language) == "" or trim(features.language) == "unknown" then
            features.language = from_text.language
        end
    end

    if trim(features.task_profile) == "" then
        features.task_profile = "general"
    end
    if trim(features.task_type) == "" then
        features.task_type = "general"
    end
    if trim(features.domain) == "" then
        features.domain = features.task_type
    end
    if trim(features.language) == "" then
        features.language = "unknown"
    end
    if trim(features.contract_shape) == "" then
        features.contract_shape = "general"
    end
    if trim(features.policy_key) == "" then
        features.policy_key = policy.build_policy_key(features)
    end
    if trim(features.family_key) == "" then
        features.family_key = features.policy_key
    end
    return features
end

function M.record_utility_feedback(_retrieved_items, _effective_ids)
    local retrieved_ids = {}
    for _, item in ipairs(_retrieved_items or {}) do
        local id = trim((item or {}).id or item)
        if id ~= "" then
            retrieved_ids[#retrieved_ids + 1] = id
        end
    end
    adaptive.update_utility_from_feedback({
        retrieved_ids = retrieved_ids,
        effective_ids = type(_effective_ids) == "table" and _effective_ids or {},
    })
end

return M
