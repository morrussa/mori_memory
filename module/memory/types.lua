local M = {}

local tool = require("module.tool")

M.ORDER = {
    "identity",
    "preference",
    "state",
    "decision",
    "fact",
    "concept",
}

M.DEFAULT_KIND = "fact"

M.DESCRIPTIONS = {
    identity = "用户本人、他人身份、角色、稳定属性、是谁",
    preference = "偏好、风格、习惯、常用选择、喜欢什么",
    state = "当前进展、阶段、上下文、临时约束、正在做的事",
    decision = "已确定方案、冻结结论、采用或不采用什么",
    fact = "客观结果、指标、数值、观察结论、事实",
    concept = "方法、机制、术语、抽象知识、概念原理",
}

local QUERY_PROTOTYPES = {
    identity = "身份 角色 稳定属性 用户本人 他人身份 是谁 设定",
    preference = "偏好 风格 习惯 常用选择 喜欢 不喜欢 倾向",
    state = "当前进展 阶段 上下文 临时约束 正在做 卡住 进行中",
    decision = "决定 方案 采用 不采用 确认 冻结 结论 定了",
    fact = "结果 指标 数值 数据 观察结论 事实 现象",
    concept = "方法 机制 术语 抽象知识 概念 原理 是什么",
}

local PASSAGE_PROTOTYPES = {
    identity = "关于某人的身份角色和稳定属性的记忆",
    preference = "关于偏好风格习惯和常用选择的记忆",
    state = "关于当前进展阶段上下文和临时约束的记忆",
    decision = "关于已经确定方案和采用结论的记忆",
    fact = "关于客观结果指标数值和观察结论的记忆",
    concept = "关于方法机制术语和抽象知识的记忆",
}

local KEYWORD_HINTS = {
    identity = { "我是", "你是", "身份", "角色", "职业", "年龄", "名字", "人设" },
    preference = { "喜欢", "偏好", "习惯", "常用", "更倾向", "不喜欢", "风格" },
    state = { "现在", "当前", "目前", "进度", "阶段", "正在", "临时", "约束", "卡住", "进行中" },
    decision = { "决定", "确定", "采用", "不采用", "方案", "冻结", "拍板", "选定", "定了" },
    fact = { "结果", "指标", "数值", "数据", "观察", "结论", "事实", "报错", "成功", "失败" },
    concept = { "原理", "机制", "术语", "概念", "方法", "模型", "是什么意思", "为什么" },
}

M.KIND_TO_CODE = {}
M.CODE_TO_KIND = {}
for idx, kind in ipairs(M.ORDER) do
    M.KIND_TO_CODE[kind] = idx
    M.CODE_TO_KIND[idx] = kind
end

local prototype_cache = {
    query = {},
    passage = {},
}

local function normalize_key(kind)
    local key = tostring(kind or ""):lower()
    key = key:gsub("[%s_%-]+", "")
    return key
end

local function keyword_bonus(text, kind)
    local bonus = 0.0
    local hints = KEYWORD_HINTS[kind] or {}
    for _, token in ipairs(hints) do
        if token ~= "" and text:find(token, 1, true) then
            bonus = bonus + 0.025
        end
    end
    if bonus > 0.10 then
        bonus = 0.10
    end
    return bonus
end

local function has_vector(vec)
    if type(vec) ~= "table" then
        return false
    end
    if vec.__ptr ~= nil then
        return (tonumber(vec.__dim) or 0) > 0
    end
    return #vec > 0
end

local function ensure_prototype(kind, mode)
    local cache = prototype_cache[mode]
    if not cache then
        return nil
    end

    local cached = cache[kind]
    if cached ~= nil then
        if cached == false then
            return nil
        end
        return cached
    end

    local text = mode == "query" and QUERY_PROTOTYPES[kind] or PASSAGE_PROTOTYPES[kind]
    if not text or text == "" then
        cache[kind] = false
        return nil
    end

    local ok, vec = pcall(tool.get_embedding, text, mode)
    if not ok or type(vec) ~= "table" or #vec == 0 then
        cache[kind] = false
        return nil
    end

    cache[kind] = vec
    return vec
end

function M.normalize(kind, fallback)
    local key = normalize_key(kind)
    if M.KIND_TO_CODE[key] then
        return key
    end

    local fb = normalize_key(fallback)
    if M.KIND_TO_CODE[fb] then
        return fb
    end
    return M.DEFAULT_KIND
end

function M.code_for(kind)
    return tonumber(M.KIND_TO_CODE[M.normalize(kind)]) or tonumber(M.KIND_TO_CODE[M.DEFAULT_KIND]) or 0
end

function M.kind_for_code(code, fallback)
    return M.CODE_TO_KIND[math.floor(tonumber(code) or -1)] or M.normalize(fallback)
end

function M.rank_text(text, mode)
    local raw_text = tostring(text or "")
    local embed_mode = (mode == "passage") and "passage" or "query"
    local ok, vec = pcall(tool.get_embedding, raw_text, embed_mode)
    if not ok or type(vec) ~= "table" or #vec == 0 then
        return {}
    end
    return M.rank_vector(vec, raw_text, embed_mode)
end

function M.rank_vector(vec, raw_text, mode)
    local embed_mode = (mode == "passage") and "passage" or "query"
    local qptr = tool.to_ptr_vec(vec) or vec
    if not has_vector(qptr) then
        return {}
    end

    local text = tostring(raw_text or "")
    local ranked = {}
    for _, kind in ipairs(M.ORDER) do
        local proto = ensure_prototype(kind, embed_mode)
        if proto then
            local score = tonumber(tool.cosine_similarity(qptr, proto)) or 0.0
            score = score + keyword_bonus(text, kind)
            ranked[#ranked + 1] = { kind = kind, score = score }
        end
    end

    table.sort(ranked, function(a, b)
        if a.score == b.score then
            return a.kind < b.kind
        end
        return a.score > b.score
    end)
    return ranked
end

function M.infer_text_type(text, mode)
    local ranked = M.rank_text(text, mode)
    return ranked[1] and ranked[1].kind or M.DEFAULT_KIND
end

function M.infer_vector_type(vec, raw_text, mode)
    local ranked = M.rank_vector(vec, raw_text, mode)
    return ranked[1] and ranked[1].kind or M.DEFAULT_KIND
end

function M.predict_query_profile(user_input, user_vec, cfg)
    cfg = type(cfg) == "table" and cfg or {}
    if cfg.enabled == false then
        return {
            ranked = {},
            selected = {},
            bonus_by_kind = {},
            primary = nil,
            secondary = nil,
        }
    end

    local vec = user_vec
    if type(vec) ~= "table" or #vec == 0 then
        local ok, embedded = pcall(tool.get_embedding_query, tostring(user_input or ""))
        if ok then
            vec = embedded
        end
    end

    local ranked = M.rank_vector(vec or {}, tostring(user_input or ""), "query")
    local topk = math.max(1, math.floor(tonumber(cfg.topk) or 2))
    local min_query_sim = tonumber(cfg.min_query_sim) or 0.18
    local max_bonus = math.max(0.0, tonumber(cfg.max_bonus) or 0.16)

    local selected = {}
    local total = 0.0
    for _, item in ipairs(ranked) do
        local score = tonumber(item.score) or 0.0
        if score >= min_query_sim then
            selected[#selected + 1] = { kind = item.kind, score = score }
            total = total + math.max(0.0, score)
        end
        if #selected >= topk then
            break
        end
    end

    if #selected == 0 and ranked[1] and (tonumber(ranked[1].score) or 0.0) > 0.0 then
        selected[1] = { kind = ranked[1].kind, score = tonumber(ranked[1].score) or 0.0 }
        total = math.max(0.0, tonumber(ranked[1].score) or 0.0)
    end

    local bonus_by_kind = {}
    for idx, item in ipairs(selected) do
        local basis = math.max(0.0, tonumber(item.score) or 0.0)
        local weight = total > 1e-6 and (basis / total) or ((idx == 1) and 1.0 or 0.0)
        bonus_by_kind[item.kind] = max_bonus * weight
    end

    return {
        ranked = ranked,
        selected = selected,
        bonus_by_kind = bonus_by_kind,
        primary = selected[1] and selected[1].kind or nil,
        secondary = selected[2] and selected[2].kind or nil,
    }
end

function M.match_bonus(kind, profile)
    if type(profile) ~= "table" then
        return 0.0
    end
    local key = M.normalize(kind)
    return tonumber((profile.bonus_by_kind or {})[key]) or 0.0
end

return M
