
-- module/experience/store.lua
-- 经验存储系统：独立于memory core，专注于agent经验管理

local M = {}

local config = require("module.config")
local persistence = require("module.persistence")
local tool = require("module.tool")

-- 存储配置
local STORAGE_ROOT = "memory/experiences"
local INDEX_FILE = STORAGE_ROOT .. "/experience_index.txt"
local EXPERIENCES_DIR = STORAGE_ROOT .. "/experiences"

-- 运行时状态
M.experiences = {}           -- 所有经验 {id -> experience}
M.experience_index = {}     -- 多维索引 {key -> [id1, id2, ...]}
M._loaded = false
M._dirty = false

-- 经验类型定义
M.EXP_TYPES = {
    SUCCESS = "success",
    FAILURE = "failure",
    PATTERN = "pattern",
    LESSON = "lesson"
}

-- 索引类型定义（扩展维度以支持更精确的检索，避免点积对比）
local INDEX_TYPES = {
    TYPE = "type",           -- 按类型索引
    TASK = "task",           -- 按任务类型索引
    CONTEXT = "context",     -- 按上下文签名索引
    TOOL = "tool",          -- 按工具使用索引
    TIME = "time",           -- 按时间索引
    
    -- [新增] 扩展维度
    DOMAIN = "domain",       -- 按领域索引
    LANGUAGE = "language",   -- 按编程语言索引
    OUTPUT_STRATEGY = "output_strategy",  -- 按输出策略索引
    PATTERN_KEY = "pattern", -- 按模式特征索引
    ERROR_TYPE = "error",    -- 按错误类型索引
    SUCCESS_KEY = "success", -- 按成功模式键索引
}

-- 基础分区定义（不可被agent修改/删除）
local IMMUTABLE_INDEX_TYPES = {
    [INDEX_TYPES.TYPE] = true,  -- 经验类型是核心分类，不可动
    [INDEX_TYPES.TIME] = true,  -- 时间是基础属性，不可动
}

-- 将INDEX_TYPES暴露出去
M.INDEX_TYPES = INDEX_TYPES

-- ==================== 工具函数 ====================

local function ensure_dir(path)
    os.execute(string.format('mkdir -p "%s"', tostring(path):gsub('"', '\\"')))
end

local function generate_id()
    return string.format("exp_%d_%d", os.time(), math.random(1000, 9999))
end

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function copy_table(value, seen)
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
        out[copy_table(k, seen)] = copy_table(v, seen)
    end
    return out
end

local function serialize_context_signature(sig)
    if not sig or type(sig) ~= "table" then
        return ""
    end
    local parts = {}
    for k, v in pairs(sig) do
        if type(v) == "table" then
            parts[#parts + 1] = string.format("%s=%s", k, serialize_context_signature(v))
        else
            parts[#parts + 1] = string.format("%s=%s", k, tostring(v))
        end
    end
    table.sort(parts)
    return table.concat(parts, "|")
end

local function append_values(out, value)
    if value == nil then
        return
    end

    if type(value) ~= "table" then
        out[#out + 1] = value
        return
    end

    if #value > 0 then
        for _, item in ipairs(value) do
            if item ~= nil then
                out[#out + 1] = item
            end
        end
        return
    end

    for k, flag in pairs(value) do
        if flag then
            out[#out + 1] = k
        end
    end
end

local function sort_strings(values)
    table.sort(values, function(a, b)
        return tostring(a) < tostring(b)
    end)
    return values
end

-- ==================== 初始化 ====================

function M.init()
    ensure_dir(STORAGE_ROOT)
    ensure_dir(EXPERIENCES_DIR)
    M.load()
end

-- ==================== 经验添加 ====================

function M.add(experience)
    if not experience or type(experience) ~= "table" then
        return false, "invalid_experience"
    end

    -- 分配ID
    if not experience.id or experience.id == "" then
        experience.id = generate_id()
    end

    -- 添加时间戳
    if not experience.created_at then
        experience.created_at = os.time()
    end

    -- 补齐检索向量，避免后续检索退化成纯索引扫描
    M.ensure_embedding(experience)

    -- 添加到存储
    M.experiences[experience.id] = experience

    -- 更新索引
    M.update_index(experience)

    -- 标记为脏
    M._dirty = true

    return true, experience.id
end

-- ==================== 索引管理 ====================

function M.update_index(experience)
    -- 按类型索引
    local exp_type = experience.type or M.EXP_TYPES.PATTERN
    M._add_to_index(INDEX_TYPES.TYPE, exp_type, experience.id)

    -- 按任务类型索引
    local task_type = experience.task_type or "general"
    M._add_to_index(INDEX_TYPES.TASK, task_type, experience.id)

    -- 按上下文签名索引
    if experience.context_signature then
        local sig_key = serialize_context_signature(experience.context_signature)
        M._add_to_index(INDEX_TYPES.CONTEXT, sig_key, experience.id)
    end

    -- 按工具使用索引
    if experience.tools_used then
        for tool_name, _ in pairs(experience.tools_used) do
            M._add_to_index(INDEX_TYPES.TOOL, tool_name, experience.id)
        end
    end

    -- 按时间索引（用于时间范围查询）
    local time_key = string.format("%d", experience.created_at)
    M._add_to_index(INDEX_TYPES.TIME, time_key, experience.id)

    -- [新增] 按领域索引
    if experience.domain then
        M._add_to_index(INDEX_TYPES.DOMAIN, experience.domain, experience.id)
    end

    -- [新增] 按编程语言索引
    if experience.language then
        M._add_to_index(INDEX_TYPES.LANGUAGE, experience.language, experience.id)
    elseif experience.context_signature and experience.context_signature.language then
        M._add_to_index(INDEX_TYPES.LANGUAGE, experience.context_signature.language, experience.id)
    end

    -- [新增] 按输出策略索引
    if experience.output_strategy then
        M._add_to_index(INDEX_TYPES.OUTPUT_STRATEGY, experience.output_strategy, experience.id)
    end

    -- [新增] 按模式特征索引
    if experience.patterns then
        for _, pattern in ipairs(experience.patterns) do
            if pattern.key then
                M._add_to_index(INDEX_TYPES.PATTERN_KEY, pattern.key, experience.id)
            end
            if pattern.type then
                M._add_to_index(INDEX_TYPES.PATTERN_KEY, pattern.type, experience.id)
            end
        end
    end

    -- [新增] 按错误类型索引（失败经验）
    if experience.type == M.EXP_TYPES.FAILURE and experience.error_info then
        local error_type = experience.error_info.type or "unknown"
        M._add_to_index(INDEX_TYPES.ERROR_TYPE, error_type, experience.id)
    end

    -- [新增] 按成功模式键索引
    if experience.type == M.EXP_TYPES.SUCCESS and experience.success_key then
        M._add_to_index(INDEX_TYPES.SUCCESS_KEY, experience.success_key, experience.id)
    end
end

function M._add_to_index(index_type, key, experience_id)
    local index_key = string.format("%s:%s", index_type, key)

    if not M.experience_index[index_key] then
        M.experience_index[index_key] = {}
    end

    -- 避免重复添加
    local ids = M.experience_index[index_key]
    for _, id in ipairs(ids) do
        if id == experience_id then
            return
        end
    end

    table.insert(ids, experience_id)
end

function M.build_retrieval_text(experience)
    if not experience or type(experience) ~= "table" then
        return ""
    end

    local parts = {}

    if experience.description and experience.description ~= "" then
        parts[#parts + 1] = tostring(experience.description)
    end

    if experience.type then
        parts[#parts + 1] = string.format("type:%s", tostring(experience.type))
    end

    if experience.task_type then
        parts[#parts + 1] = string.format("task:%s", tostring(experience.task_type))
    end

    if experience.domain then
        parts[#parts + 1] = string.format("domain:%s", tostring(experience.domain))
    end

    if experience.language then
        parts[#parts + 1] = string.format("language:%s", tostring(experience.language))
    end

    if experience.output_strategy then
        parts[#parts + 1] = string.format("output:%s", tostring(experience.output_strategy))
    end

    if experience.context_signature then
        local ctx_parts = {}
        for k, v in pairs(experience.context_signature) do
            ctx_parts[#ctx_parts + 1] = string.format("%s=%s", tostring(k), tostring(v))
        end
        if #ctx_parts > 0 then
            parts[#parts + 1] = "context:" .. table.concat(sort_strings(ctx_parts), ",")
        end
    end

    if experience.tools_used then
        local tools = {}
        for tool_name, count in pairs(experience.tools_used) do
            tools[#tools + 1] = string.format("%s:%s", tostring(tool_name), tostring(count))
        end
        if #tools > 0 then
            parts[#parts + 1] = "tools:" .. table.concat(sort_strings(tools), ",")
        end
    end

    if experience.patterns then
        for _, pattern in ipairs(experience.patterns) do
            if pattern.key then
                parts[#parts + 1] = "pattern:" .. tostring(pattern.key)
            elseif pattern.pattern then
                parts[#parts + 1] = "pattern:" .. tostring(pattern.pattern)
            elseif pattern.type then
                parts[#parts + 1] = "pattern_type:" .. tostring(pattern.type)
            end
        end
    end

    if experience.lessons then
        for _, lesson in ipairs(experience.lessons) do
            if type(lesson) == "table" and lesson.content ~= nil then
                if type(lesson.content) == "table" then
                    for _, item in ipairs(lesson.content) do
                        parts[#parts + 1] = tostring(item)
                    end
                else
                    parts[#parts + 1] = tostring(lesson.content)
                end
            elseif lesson ~= nil then
                parts[#parts + 1] = tostring(lesson)
            end
        end
    end

    if experience.outcome and experience.outcome.reason then
        parts[#parts + 1] = "outcome:" .. tostring(experience.outcome.reason)
    end

    return table.concat(parts, "\n")
end

function M.ensure_embedding(experience)
    if not experience or type(experience) ~= "table" then
        return nil
    end

    if type(experience.embedding) == "table" and #experience.embedding > 0 then
        return experience.embedding
    end

    local retrieval_text = M.build_retrieval_text(experience)
    if retrieval_text == "" then
        return nil
    end

    local embedding = tool.get_embedding_passage(retrieval_text)
    if type(embedding) == "table" and #embedding > 0 then
        experience.embedding = embedding
        M._dirty = true
        return embedding
    end

    return nil
end

-- ==================== 经验检索 ====================

function M.retrieve(options)
    options = options or {}
    local limit = tonumber(options.limit) or 0
    local context_threshold = tonumber(options.context_threshold) or 0.35
    local semantic_threshold = tonumber(options.semantic_threshold)
    if semantic_threshold == nil then
        semantic_threshold = 0.12
    end
    local semantic_scan_limit = tonumber(options.semantic_scan_limit) or math.max(limit, 24)

    local candidate_map = {}
    local has_explicit_filters = false

    local function get_record(id)
        local exp = M.experiences[id]
        if not exp then
            return nil
        end

        local rec = candidate_map[id]
        if rec then
            return rec
        end

        rec = {
            exp = exp,
            index_score = 0.0,
            query_similarity = nil,
            context_similarity = nil,
            matched_sources = {},
            matched_source_count = 0,
        }
        candidate_map[id] = rec
        return rec
    end

    local function add_candidate(id, source_name, weight)
        local rec = get_record(id)
        if not rec then
            return
        end

        if source_name and not rec.matched_sources[source_name] then
            rec.matched_sources[source_name] = true
            rec.matched_source_count = rec.matched_source_count + 1
        end

        rec.index_score = rec.index_score + (tonumber(weight) or 0.0)
    end

    local function add_index_hits(index_type, option_value, source_name, weight)
        local values = {}
        append_values(values, option_value)
        if #values <= 0 then
            return
        end

        has_explicit_filters = true
        for _, value in ipairs(values) do
            local index_key = string.format("%s:%s", index_type, value)
            local ids = M.experience_index[index_key] or {}
            for _, id in ipairs(ids) do
                add_candidate(id, source_name or index_type, weight or 1.0)
            end
        end
    end

    add_index_hits(INDEX_TYPES.TYPE, options.type, "type", 1.15)
    add_index_hits(INDEX_TYPES.TASK, options.task_type, "task", 1.10)
    add_index_hits(INDEX_TYPES.TOOL, options.tool_used or options.tools_used, "tool", 1.05)
    add_index_hits(INDEX_TYPES.DOMAIN, options.domain, "domain", 1.00)
    add_index_hits(INDEX_TYPES.LANGUAGE, options.language, "language", 1.00)
    add_index_hits(INDEX_TYPES.OUTPUT_STRATEGY, options.output_strategy, "output_strategy", 0.90)
    add_index_hits(INDEX_TYPES.PATTERN_KEY, options.pattern_key, "pattern", 0.95)
    add_index_hits(INDEX_TYPES.ERROR_TYPE, options.error_type, "error", 0.95)
    add_index_hits(INDEX_TYPES.SUCCESS_KEY, options.success_key, "success", 0.95)

    if options.context_signature then
        has_explicit_filters = true
        for id, exp in pairs(M.experiences) do
            local sim = M.compute_context_similarity(options.context_signature, exp.context_signature)
            if sim >= context_threshold then
                local rec = get_record(id)
                rec.context_similarity = sim
                add_candidate(id, "context", 1.20 + sim)
            end
        end
    end

    if type(options.query_embedding) == "table" and #options.query_embedding > 0 then
        local semantic_hits = {}
        for id, exp in pairs(M.experiences) do
            local embedding = M.ensure_embedding(exp)
            if embedding and #embedding > 0 then
                local sim = tool.cosine_similarity(options.query_embedding, embedding)
                semantic_hits[#semantic_hits + 1] = { id = id, sim = sim }
            end
        end

        table.sort(semantic_hits, function(a, b)
            return (a.sim or -1.0) > (b.sim or -1.0)
        end)

        local take_n = math.max(limit, math.max(semantic_scan_limit, 1))
        for i = 1, math.min(take_n, #semantic_hits) do
            local hit = semantic_hits[i]
            if (hit.sim or -1.0) >= semantic_threshold then
                local rec = get_record(hit.id)
                rec.query_similarity = hit.sim
                add_candidate(hit.id, "semantic", 1.35 + math.max(0.0, hit.sim))
            end
        end
    end

    if next(candidate_map) == nil then
        for id, _ in pairs(M.experiences) do
            add_candidate(id, has_explicit_filters and "backfill" or "fallback", 0.10)
        end
    end

    local max_index_score = 0.0
    for _, rec in pairs(candidate_map) do
        max_index_score = math.max(max_index_score, tonumber(rec.index_score) or 0.0)
    end

    local candidates = {}
    for id, rec in pairs(candidate_map) do
        local item = copy_table(rec.exp)

        if options.context_signature and rec.context_similarity == nil then
            rec.context_similarity = M.compute_context_similarity(options.context_signature, rec.exp.context_signature)
        end

        if type(options.query_embedding) == "table" and #options.query_embedding > 0 and rec.query_similarity == nil then
            local embedding = M.ensure_embedding(rec.exp)
            if embedding and #embedding > 0 then
                rec.query_similarity = tool.cosine_similarity(options.query_embedding, embedding)
            else
                rec.query_similarity = 0.0
            end
        end

        item.context_similarity = clamp(tonumber(rec.context_similarity) or 0.0, 0.0, 1.0)
        item.query_similarity = tonumber(rec.query_similarity) or 0.0
        item.vector_similarity = item.query_similarity
        item.index_match_score = (max_index_score > 0.0)
            and clamp((tonumber(rec.index_score) or 0.0) / max_index_score, 0.0, 1.0)
            or 0.0
        item.matched_source_count = tonumber(rec.matched_source_count) or 0
        item.retrieval_sources = {}
        for source_name in pairs(rec.matched_sources) do
            item.retrieval_sources[#item.retrieval_sources + 1] = source_name
        end
        sort_strings(item.retrieval_sources)

        item.preliminary_score =
            0.60 * math.max(0.0, item.query_similarity) +
            0.25 * item.context_similarity +
            0.15 * item.index_match_score

        candidates[#candidates + 1] = item
    end

    table.sort(candidates, function(a, b)
        local score_a = tonumber(a.preliminary_score) or 0.0
        local score_b = tonumber(b.preliminary_score) or 0.0
        if score_a == score_b then
            return (tonumber(a.created_at) or 0) > (tonumber(b.created_at) or 0)
        end
        return score_a > score_b
    end)

    if limit > 0 and #candidates > limit then
        local trimmed = {}
        for i = 1, limit do
            trimmed[i] = candidates[i]
        end
        candidates = trimmed
    end

    return candidates
end

-- ==================== 相似度计算 ====================

-- 上下文相似度计算（精确匹配维度，不使用点积）
function M.compute_context_similarity(sig1, sig2)
    if not sig1 or not sig2 then
        return 0.0
    end

    local score = 0.0
    local total = 0

    -- 比较各个维度（精确匹配）
    for key, value in pairs(sig1) do
        total = total + 1
        if sig2[key] == value then
            score = score + 1
        end
    end

    return total > 0 and (score / total) or 0.0
end

-- ==================== 持久化 ====================

function M.save()
    if not M._dirty then
        return true
    end

    -- 保存索引
    local ok, err = persistence.write_atomic(INDEX_FILE, "w", function(f)
        -- 写入版本号
        f:write("VERSION=1\n")

        -- 写入索引
        for key, ids in pairs(M.experience_index) do
            f:write(string.format("%s=%s\n", key, table.concat(ids, ",")))
        end

        return true
    end)

    if not ok then
        return false, err
    end

    -- 保存每个经验
    for id, exp in pairs(M.experiences) do
        local exp_file = string.format("%s/%s.lua", EXPERIENCES_DIR, id)
        local ok2, err2 = persistence.write_atomic(exp_file, "w", function(f)
            f:write("-- Experience: " .. id .. "\n")
            f:write("return " .. tool.json_encode(exp) .. "\n")
            return true
        end)

        if not ok2 then
            print(string.format("[ExperienceStore] Failed to save experience %s: %s", id, err2))
        end
    end

    M._dirty = false
    return true
end

function M.load()
    -- 加载索引
    local f = io.open(INDEX_FILE, "r")
    if f then
        local version = f:read("*l")
        if version ~= "VERSION=1" then
            f:close()
            print("[ExperienceStore] Index version mismatch, starting fresh")
            return
        end

        for line in f:lines() do
            line = tostring(line or ""):gsub("^%s*(.-)%s*$", "%1")
            if line ~= "" then
                local key, ids_str = line:match("^([^=]+)=(.+)$")
                if key and ids_str then
                    local ids = {}
                    for id in ids_str:gmatch("[^,]+") do
                        table.insert(ids, id)
                    end
                    M.experience_index[key] = ids
                end
            end
        end

        f:close()
    end

    -- 加载所有经验
    local exp_files = tool.list_files(EXPERIENCES_DIR)
    for _, file in ipairs(exp_files) do
        if file:match("%.lua$") then
            local path = string.format("%s/%s", EXPERIENCES_DIR, file)
            local ok, exp = pcall(dofile, path)
            if ok and exp and exp.id then
                M.experiences[exp.id] = exp
            end
        end
    end

    M._loaded = true
    print(string.format("[ExperienceStore] Loaded %d experiences", 
        M.count_experiences()))
end

-- ==================== 统计信息 ====================

function M.count_experiences()
    local count = 0
    for _ in pairs(M.experiences) do
        count = count + 1
    end
    return count
end

function M.get_stats()
    return {
        total_experiences = M.count_experiences(),
        index_keys = M._count_index_keys(),
        is_loaded = M._loaded,
        is_dirty = M._dirty
    }
end

function M._count_index_keys()
    local count = 0
    for _ in pairs(M.experience_index) do
        count = count + 1
    end
    return count
end

-- ==================== 索引树结构（供Agent查看） ====================

-- 获取索引树结构
-- 返回一个树形结构，展示每个索引类型及其已有选项
function M.get_index_tree()
    local tree = {
        -- 基础分区（不可动）
        immutable = {},
        -- 可扩展分区
        extensible = {},
        -- 统计信息
        stats = {
            total_experiences = M.count_experiences(),
            total_index_keys = M._count_index_keys(),
        }
    }
    
    -- 构建每个索引类型的选项列表
    local type_options = {}  -- {index_type -> {key -> count}}
    
    for index_key, ids in pairs(M.experience_index) do
        -- 解析 index_key: "type:success" -> index_type="type", key="success"
        local index_type, key = index_key:match("^([^:]+):(.+)$")
        if index_type and key then
            if not type_options[index_type] then
                type_options[index_type] = {}
            end
            type_options[index_type][key] = #ids
        end
    end
    
    -- 分类到 immutable 和 extensible
    for index_type, options in pairs(type_options) do
        local node = {
            type = index_type,
            immutable = IMMUTABLE_INDEX_TYPES[index_type] or false,
            options = {}
        }
        
        -- 按数量排序选项
        local sorted_options = {}
        for key, count in pairs(options) do
            sorted_options[#sorted_options + 1] = {key = key, count = count}
        end
        table.sort(sorted_options, function(a, b) return a.count > b.count end)
        
        for _, opt in ipairs(sorted_options) do
            node.options[#node.options + 1] = {
                key = opt.key,
                count = opt.count
            }
        end
        
        if IMMUTABLE_INDEX_TYPES[index_type] then
            tree.immutable[#tree.immutable + 1] = node
        else
            tree.extensible[#tree.extensible + 1] = node
        end
    end
    
    -- 排序
    table.sort(tree.immutable, function(a, b) return a.type < b.type end)
    table.sort(tree.extensible, function(a, b) return a.type < b.type end)
    
    return tree
end

-- 格式化索引树为文本（供LLM阅读）
function M.format_index_tree_for_llm(max_options_per_type)
    max_options_per_type = max_options_per_type or 20
    
    local tree = M.get_index_tree()
    local lines = {}
    
    lines[#lines + 1] = "## 经验索引树结构"
    lines[#lines + 1] = string.format("总经验数: %d, 总索引键数: %d", 
        tree.stats.total_experiences, tree.stats.total_index_keys)
    lines[#lines + 1] = ""
    
    -- 基础分区（不可动）
    lines[#lines + 1] = "### 基础分区 [不可修改]"
    for _, node in ipairs(tree.immutable) do
        lines[#lines + 1] = string.format("- %s:", node.type)
        for i, opt in ipairs(node.options) do
            if i <= max_options_per_type then
                lines[#lines + 1] = string.format("  - %s (%d条经验)", opt.key, opt.count)
            end
        end
        if #node.options > max_options_per_type then
            lines[#lines + 1] = string.format("  - ... 还有 %d 个选项", #node.options - max_options_per_type)
        end
    end
    lines[#lines + 1] = ""
    
    -- 可扩展分区
    lines[#lines + 1] = "### 可扩展分区 [可添加新选项]"
    for _, node in ipairs(tree.extensible) do
        lines[#lines + 1] = string.format("- %s:", node.type)
        if #node.options == 0 then
            lines[#lines + 1] = "  - (暂无数据)"
        else
            for i, opt in ipairs(node.options) do
                if i <= max_options_per_type then
                    lines[#lines + 1] = string.format("  - %s (%d条经验)", opt.key, opt.count)
                end
            end
            if #node.options > max_options_per_type then
                lines[#lines + 1] = string.format("  - ... 还有 %d 个选项", #node.options - max_options_per_type)
            end
        end
    end
    
    return table.concat(lines, "\n")
end

-- 获取所有可用的索引类型（供agent参考）
function M.get_available_index_types()
    local result = {
        immutable = {},
        extensible = {}
    }
    
    for name, value in pairs(INDEX_TYPES) do
        local info = {
            name = name,
            value = value,
            immutable = IMMUTABLE_INDEX_TYPES[value] or false,
            description = M._get_index_type_description(value)
        }
        
        if IMMUTABLE_INDEX_TYPES[value] then
            result.immutable[#result.immutable + 1] = info
        else
            result.extensible[#result.extensible + 1] = info
        end
    end
    
    table.sort(result.immutable, function(a, b) return a.name < b.name end)
    table.sort(result.extensible, function(a, b) return a.name < b.name end)
    
    return result
end

-- 索引类型描述
function M._get_index_type_description(index_type)
    local descriptions = {
        [INDEX_TYPES.TYPE] = "经验类型：success/failure/pattern/lesson",
        [INDEX_TYPES.TASK] = "任务类型：coding/analysis/conversation/general等",
        [INDEX_TYPES.CONTEXT] = "上下文签名：语言、领域、任务类别等特征组合",
        [INDEX_TYPES.TOOL] = "工具使用：使用的工具名称",
        [INDEX_TYPES.TIME] = "时间索引：经验创建时间",
        [INDEX_TYPES.DOMAIN] = "领域：coding/analysis/conversation等",
        [INDEX_TYPES.LANGUAGE] = "编程语言：python/lua/javascript等",
        [INDEX_TYPES.OUTPUT_STRATEGY] = "输出策略：structured/narrative/code_first等",
        [INDEX_TYPES.PATTERN_KEY] = "模式特征：从经验中提取的模式键",
        [INDEX_TYPES.ERROR_TYPE] = "错误类型：失败经验的错误分类",
        [INDEX_TYPES.SUCCESS_KEY] = "成功模式键：成功经验的关键特征",
    }
    return descriptions[index_type] or "未知类型"
end

-- ==================== Agent分类接口 ====================

-- 构建供agent分类经验的上下文
-- experience: 待分类的经验对象
-- 返回：供LLM阅读的分类提示
function M.build_classification_context(experience)
    local tree_text = M.format_index_tree_for_llm(15)
    
    local exp_info = {}
    exp_info[#exp_info + 1] = "## 待分类经验"
    exp_info[#exp_info + 1] = string.format("类型: %s", experience.type or "unknown")
    exp_info[#exp_info + 1] = string.format("任务类型: %s", experience.task_type or "unknown")
    exp_info[#exp_info + 1] = string.format("描述: %s", experience.description or "无描述")
    
    if experience.context_signature then
        local ctx_parts = {}
        for k, v in pairs(experience.context_signature) do
            ctx_parts[#ctx_parts + 1] = string.format("%s=%s", k, tostring(v))
        end
        exp_info[#exp_info + 1] = string.format("上下文签名: %s", table.concat(ctx_parts, ", "))
    end
    
    if experience.tools_used then
        local tools = {}
        for tool, count in pairs(experience.tools_used) do
            tools[#tools + 1] = string.format("%s(%d)", tool, count)
        end
        exp_info[#exp_info + 1] = string.format("使用的工具: %s", table.concat(tools, ", "))
    end
    
    if experience.output_strategy then
        exp_info[#exp_info + 1] = string.format("输出策略: %s", experience.output_strategy)
    end
    
    if experience.patterns then
        exp_info[#exp_info + 1] = "模式:"
        for _, p in ipairs(experience.patterns) do
            if p.key then
                exp_info[#exp_info + 1] = string.format("  - %s", p.key)
            end
        end
    end
    
    local prompt = {}
    prompt[#prompt + 1] = tree_text
    prompt[#prompt + 1] = ""
    prompt[#prompt + 1] = table.concat(exp_info, "\n")
    prompt[#prompt + 1] = ""
    prompt[#prompt + 1] = "## 分类任务"
    prompt[#prompt + 1] = "请分析上述经验，决定它应该放入哪些索引分区。"
    prompt[#prompt + 1] = ""
    prompt[#prompt + 1] = "**规则：**"
    prompt[#prompt + 1] = "1. 基础分区(type, time)已自动处理，无需关心"
    prompt[#prompt + 1] = "2. 对于可扩展分区，可以："
    prompt[#prompt + 1] = "  - 选择现有选项（匹配度高时优先选择）"
    prompt[#prompt + 1] = "  - 创建新选项（当没有合适选项时）"
    prompt[#prompt + 1] = "3. 返回JSON格式的分类结果"
    prompt[#prompt + 1] = ""
    prompt[#prompt + 1] = "**返回格式示例：**"
    prompt[#prompt + 1] = [[```json
{
  "classifications": [
    {
      "index_type": "task",
      "value": "coding",
      "action": "use_existing",
      "confidence": 0.9,
      "reason": "经验涉及代码编写任务"
    },
    {
      "index_type": "domain",
      "value": "web_development",
      "action": "create_new",
      "confidence": 0.8,
      "reason": "涉及前端开发，现有domain选项中没有合适的分类"
    }
  ],
  "needs_new_partition": false,
  "suggested_new_partition": null
}
```]]
    
    return table.concat(prompt, "\n")
end

-- 应用agent的分类结果
-- experience: 经验对象（将被修改）
-- classification_result: agent返回的分类结果（解析后的table）
function M.apply_classification(experience, classification_result)
    if not classification_result or not classification_result.classifications then
        return false, "invalid_classification_result"
    end
    
    local applied = {}
    local new_keys = {}
    
    for _, classif in ipairs(classification_result.classifications) do
        local index_type = classif.index_type
        local value = classif.value
        local action = classif.action or "use_existing"
        
        -- 跳过不可变分区
        if IMMUTABLE_INDEX_TYPES[index_type] then
            -- 基础分区已自动处理，跳过
        else
            -- 根据index_type设置对应的经验字段
            if index_type == INDEX_TYPES.TASK then
                experience.task_type = value
            elseif index_type == INDEX_TYPES.DOMAIN then
                experience.domain = value
            elseif index_type == INDEX_TYPES.LANGUAGE then
                experience.language = value
            elseif index_type == INDEX_TYPES.OUTPUT_STRATEGY then
                experience.output_strategy = value
            elseif index_type == INDEX_TYPES.PATTERN_KEY then
                if not experience.patterns then experience.patterns = {} end
                experience.patterns[#experience.patterns + 1] = {key = value, type = "classified"}
            elseif index_type == INDEX_TYPES.ERROR_TYPE then
                if not experience.error_info then experience.error_info = {} end
                experience.error_info.type = value
            elseif index_type == INDEX_TYPES.SUCCESS_KEY then
                experience.success_key = value
            elseif index_type == INDEX_TYPES.TOOL then
                if not experience.tools_used then experience.tools_used = {} end
                experience.tools_used[value] = (experience.tools_used[value] or 0) + 1
            elseif index_type == INDEX_TYPES.CONTEXT then
                -- context需要特殊处理，这里简单记录
                if not experience.context_signature then experience.context_signature = {} end
                experience.context_signature.classified_context = value
            end
            
            -- 记录应用的分类
            applied[#applied + 1] = {
                index_type = index_type,
                value = value,
                action = action
            }
            
            -- 记录新建的选项
            if action == "create_new" then
                new_keys[#new_keys + 1] = string.format("%s:%s", index_type, value)
            end
        end
    end
    
    return true, {
        applied = applied,
        new_keys = new_keys
    }
end

-- 检查是否需要创建新的分区类型（供agent判断）
-- 当现有分区类型不足以描述经验时，可以建议新的分区类型
function M.suggest_new_partition_type(experience, suggested_name, suggested_description)
    -- 新分区类型需要满足：
    -- 1. 名称唯一
    -- 2. 名称有效（字母数字下划线）
    -- 3. 有描述
    
    if not suggested_name or suggested_name == "" then
        return false, "invalid_name"
    end
    
    if not suggested_name:match("^[%w_]+$") then
        return false, "invalid_name_format"
    end
    
    -- 检查是否已存在
    for _, existing_type in pairs(INDEX_TYPES) do
        if existing_type == suggested_name then
            return false, "already_exists"
        end
    end
    
    -- 创建新分区类型（运行时添加）
    INDEX_TYPES[suggested_name:upper()] = suggested_name
    
    return true, {
        name = suggested_name,
        description = suggested_description,
        immutable = false
    }
end

return M
