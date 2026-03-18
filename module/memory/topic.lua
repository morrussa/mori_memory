-- topic.lua
local M = {}
local ffi = require("ffi")
local tool = require("module.tool")
local config = require("module.config")
local persistence = require("module.persistence")
local history_module = require("module.memory.history") -- 依赖 history 进行 rebuild
local py_pipeline = py_pipeline

-- ==================== 配置读取 ====================
local T_CONF = config.settings.topic
local MAKE_CLUSTER1 = T_CONF.make_cluster1 or 3
local MAKE_CLUSTER2 = T_CONF.make_cluster2 or 3
local TOPIC_LIMIT = T_CONF.topic_limit or 0.6      -- 全局漂移阈值
local BREAK_LIMIT = T_CONF.break_limit or 0.45     -- 话语对断裂阈值 (论文核心思想)
local CONFIRM_LIMIT = T_CONF.confirm_limit or 0.55 -- 话题无关确认阈值
local MIN_TOPIC_LENGTH = T_CONF.min_topic_length or 2  -- 最小话题长度
local SUMMARY_MAX_TOKENS = math.max(32, tonumber(T_CONF.summary_max_tokens) or 192)
local FINGERPRINT_TOPK = math.max(1, tonumber(T_CONF.fingerprint_topk) or 3)
local CHAIN_TOPN = math.max(1, tonumber(T_CONF.chain_topn) or 3)
local CHAIN_MIN_SCORE = tonumber(T_CONF.chain_min_score) or 0.20
local CHAIN_CENTROID_WEIGHT = tonumber(T_CONF.chain_centroid_weight) or 0.55
local CHAIN_HIST_WEIGHT = tonumber(T_CONF.chain_hist_weight) or 0.45
local CHAIN_DOMINANT_BONUS = tonumber(T_CONF.chain_dominant_bonus) or 0.05
local ALLOW_LLM_SUMMARY = (T_CONF.allow_llm_summary == true)

-- 话题摘要分级配置
local SUMMARY_VARIANT_WEIGHTS = T_CONF.summary_variant_weights or {
    full = 1.00,
    slight = 0.72,
    heavy = 0.40,
    none = 0.00,
}
local SUMMARY_COMPRESS_RATIO_SLIGHT = tonumber(T_CONF.summary_compress_ratio_slight) or 0.65
local SUMMARY_COMPRESS_RATIO_HEAVY = tonumber(T_CONF.summary_compress_ratio_heavy) or 0.30

local DO_REBUILD = T_CONF.rebuild
local IDX_FILE = "memory/topic.bin"
local TOPIC_VERSION = 3  -- 版本升级，支持分级摘要
local ACTIVE_HEAD_MASK = 1
local ACTIVE_TAIL_MASK = 2
local ACTIVE_LAST_MASK = 4

-- ==================== 核心状态 ====================
M.turn_vectors = {}       -- 缓存近期轮次的向量 (只保留必要的)
M.topics = {}             -- 已完成的话题列表 {{start, end_, summary, centroid, summary_variants}, ...}
M.dialogues = {}          -- turn -> {user=, ai=}

-- 当前活跃话题状态
M.active_topic = {
    start = nil,          -- 起始轮次
    head_centroid = nil,  -- 头质心
    vectors = {},         -- 当前话题累积的向量 (用于最后生成 overall centroid)
    tail_window = {},     -- 尾部滑动窗口
    last_vec = nil,       -- [新增] 记录上一轮的向量，用于计算话语对相似度
    scope_key = "",       -- [新增] 作用域标记（用于多人/多源隔离）
    summary_cache = "",
    summary_turn = 0,
    summary_variants = { full = "", slight = "", heavy = "", none = "" }  -- 分级摘要缓存
}

M._last_processed_turn = 0

-- ==================== 二进制读写 ====================

local function has_flag(mask, flag)
    return (mask % (flag * 2)) >= flag
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function trim(text)
    return tostring(text or ""):match("^%s*(.-)%s*$")
end

-- 压缩文本函数，类似于 context_builder.lua 中的实现
local function compress_text(text, max_chars)
    local raw = trim(text or "")
    max_chars = math.max(0, math.floor(tonumber(max_chars) or 0))
    if raw == "" or max_chars <= 0 then
        return ""
    end
    -- 简单的截断，实际实现中可以使用更智能的压缩
    if #raw <= max_chars then
        return raw
    end
    -- 保留开头和结尾的部分
    local prefix_len = math.floor(max_chars * 0.6)
    local suffix_len = max_chars - prefix_len - 3  -- 减去 "..."
    if suffix_len < 10 then
        -- 如果后缀太短，只保留前缀
        return raw:sub(1, max_chars - 3) .. "..."
    end
    return raw:sub(1, prefix_len) .. "..." .. raw:sub(-suffix_len)
end

-- 构建分级摘要
local function build_topic_summary_variants(start_turn, end_turn)
    start_turn = math.max(0, math.floor(tonumber(start_turn) or 0))
    end_turn = math.max(0, math.floor(tonumber(end_turn) or 0))
    if start_turn <= 0 or end_turn <= 0 or end_turn < start_turn then
        return { full = "", slight = "", heavy = "", none = "" }
    end

    local dialog_texts = {}
    for t = start_turn, math.min(start_turn + 4, end_turn) do
        local d = M.dialogues[t]
        local user_text = d and d.user or history_module.get_turn_text(t, "user")
        local ai_text = d and d.ai or history_module.get_turn_text(t, "ai")
        if user_text or ai_text then
            dialog_texts[#dialog_texts + 1] = string.format(
                "第%d轮\n用户：%s\n助手：%s",
                t,
                tostring(user_text or ""),
                tostring(ai_text or "")
            )
        end
    end
    if #dialog_texts <= 0 then
        return { full = "", slight = "", heavy = "", none = "" }
    end

    local prompt = "请用一句话概括话题：\n" .. table.concat(dialog_texts, "\n")
    prompt = prompt .. "\n/no_think"
    local messages = {{role="user", content=prompt}}
    local ok, summary = pcall(function()
        if not py_pipeline or not py_pipeline.generate_chat_sync then
            return ""
        end
        return py_pipeline:generate_chat_sync(messages, {
            max_tokens = SUMMARY_MAX_TOKENS,
            temperature = 0.1,
        }) or ""
    end)
    if not ok then
        print("[Topic][WARN] 按需生成摘要失败: " .. tostring(summary))
        return { full = "", slight = "", heavy = "", none = "" }
    end
    
    local full_summary = trim(tool.remove_cot(summary))
    if full_summary == "" then
        return { full = "", slight = "", heavy = "", none = "" }
    end
    
    -- 生成分级摘要
    local slight_summary = compress_text(full_summary, math.floor(#full_summary * SUMMARY_COMPRESS_RATIO_SLIGHT))
    local heavy_summary = compress_text(full_summary, math.floor(#full_summary * SUMMARY_COMPRESS_RATIO_HEAVY))
    
    return {
        full = full_summary,
        slight = slight_summary,
        heavy = heavy_summary,
        none = ""
    }
end

local function stable_topic_key_from_record(rec)
    local start = tonumber((rec or {}).start)
    if not start or start <= 0 then
        return nil
    end
    return "S:" .. tostring(math.floor(start))
end

local function split_scoped_topic_key(topic_key)
    local key = tostring(topic_key or ""):match("^%s*(.-)%s*$")
    if key == "" then
        return "", ""
    end

    local scope_key, anchor_key = key:match("^(.-)|([ASC]:%d+)$")
    if scope_key and anchor_key then
        return trim(scope_key), trim(anchor_key)
    end
    return "", key
end

local function normalize_topic_key(topic_key)
    local _, raw_key = split_scoped_topic_key(topic_key)
    local key = tostring(raw_key or ""):match("^%s*(.-)%s*$")
    if key == "" then
        return nil
    end

    local prefix, value = key:match("^([ASC]):(%d+)$")
    if not prefix or not value then
        return key
    end

    local n = tonumber(value)
    if not n or n <= 0 then
        return nil
    end

    if prefix == "S" then
        return "S:" .. tostring(math.floor(n))
    end

    if prefix == "A" then
        return "S:" .. tostring(math.floor(n))
    end

    if prefix == "C" then
        local rec = M.topics[math.floor(n)]
        return stable_topic_key_from_record(rec)
    end

    return key
end

local function resolve_topic_key(topic_key)
    local scope_key, raw_key = split_scoped_topic_key(topic_key)
    local stable_key = normalize_topic_key(raw_key)
    if not stable_key then
        return nil
    end

    local scoped_key = scope_key ~= "" and (scope_key .. "|" .. stable_key) or stable_key

    local start_turn = tonumber(stable_key:match("^S:(%d+)$"))
    if not start_turn then
        return stable_key, nil, scoped_key, scope_key
    end

    if M.active_topic.start and tonumber(M.active_topic.start) == start_turn then
        return stable_key, {
            key = stable_key,
            start = start_turn,
            end_ = nil,
            summary = "",
            centroid = M.active_topic.head_centroid or tool.average_vectors(M.active_topic.vectors),
            is_active = true,
            topic_idx = nil,
        }, scoped_key, scope_key
    end

    for idx, rec in ipairs(M.topics) do
        if tonumber(rec.start) == start_turn then
            return stable_key, {
                key = stable_key,
                start = tonumber(rec.start),
                end_ = tonumber(rec.end_),
                summary = tostring(rec.summary or ""),
                centroid = rec.centroid,
                is_active = false,
                topic_idx = idx,
            }, scoped_key, scope_key
        end
    end

    return stable_key, nil, scoped_key, scope_key
end

local function histogram_overlap(a, b)
    local total = 0.0
    local weights_a = ((a or {}).weights) or {}
    local weights_b = ((b or {}).weights) or {}
    for cid, wa in pairs(weights_a) do
        local wb = tonumber(weights_b[cid]) or 0.0
        total = total + math.min(tonumber(wa) or 0.0, wb)
    end
    return total
end

local function save_to_disk()
    local count = #M.topics
    local active_start = M.active_topic.start or 0
    local active_turn = M._last_processed_turn
    local active_mask = 0

    local head_vec = nil
    local tail_vec = nil
    local last_vec = nil
    if active_start > 0 then
        head_vec = M.active_topic.head_centroid
        if M.active_topic.tail_window and #M.active_topic.tail_window > 0 then
            tail_vec = tool.average_vectors(M.active_topic.tail_window)
        end
        last_vec = M.active_topic.last_vec

        if head_vec and #head_vec > 0 then active_mask = active_mask + ACTIVE_HEAD_MASK end
        if tail_vec and #tail_vec > 0 then active_mask = active_mask + ACTIVE_TAIL_MASK end
        if last_vec and #last_vec > 0 then active_mask = active_mask + ACTIVE_LAST_MASK end
    end

    local ok, err = persistence.write_atomic(IDX_FILE, "wb", function(f)
        local function write_or_fail(chunk)
            local w_ok, w_err = f:write(chunk)
            if not w_ok then
                return false, w_err
            end
            return true
        end

        -- 1. Header
        local w1_ok, w1_err = write_or_fail("TOPC")
        if not w1_ok then return false, w1_err end

        -- 写入基础头部 (24 bytes = magic(4) + 5*uint32)
        local header_base = ffi.new("uint32_t[5]", TOPIC_VERSION, count, active_start, active_turn, active_mask)
        local w2_ok, w2_err = write_or_fail(ffi.string(header_base, 20))
        if not w2_ok then return false, w2_err end

        -- 写入 Active 状态
        if active_start > 0 then
            if has_flag(active_mask, ACTIVE_HEAD_MASK) then
                local w3_ok, w3_err = write_or_fail(tool.vector_to_bin(head_vec))
                if not w3_ok then return false, w3_err end
            end
            if has_flag(active_mask, ACTIVE_TAIL_MASK) then
                local w4_ok, w4_err = write_or_fail(tool.vector_to_bin(tail_vec))
                if not w4_ok then return false, w4_err end
            end
            if has_flag(active_mask, ACTIVE_LAST_MASK) then
                local w5_ok, w5_err = write_or_fail(tool.vector_to_bin(last_vec))
                if not w5_ok then return false, w5_err end
            end
        end

        -- 2. Records
        for _, t in ipairs(M.topics) do
            local bin = tool.create_topic_record(t.start, t.end_, t.summary or "", t.centroid, t.summary_variants)
            local wb_ok, wb_err = write_or_fail(bin)
            if not wb_ok then return false, wb_err end
        end
        return true
    end)

    if not ok then
        return false, err
    end
    return true
end

local function load_from_disk()
    if not tool.file_exists(IDX_FILE) then return end
    local f = io.open(IDX_FILE, "rb")
    if not f then return end
    local data = f:read("*a")
    f:close()
    if #data < 24 or data:sub(1,4) ~= "TOPC" then return end

    -- Parse Header
    local p = ffi.cast("const uint8_t*", data)
    local version = ffi.cast("const uint32_t*", p + 4)[0]
    if version ~= TOPIC_VERSION then
        print(string.format("[Topic] 检测到旧版 topic.bin (version=%d)，已跳过加载（仅支持 v%d）", version, TOPIC_VERSION))
        return
    end
    local count = ffi.cast("const uint32_t*", p + 8)[0]
    local active_start = ffi.cast("const uint32_t*", p + 12)[0]
    local active_turn = ffi.cast("const uint32_t*", p + 16)[0]
    local active_mask = ffi.cast("const uint32_t*", p + 20)[0]
    
    M._last_processed_turn = active_turn

    -- Parse Active State
    local offset = 24
    if active_start > 0 then
        local head_vec, tail_vec, last_vec = nil, nil, nil

        if has_flag(active_mask, ACTIVE_HEAD_MASK) then
            local head_len
            head_vec, head_len = tool.bin_to_vector(data, offset)
            if not head_vec or head_len <= 0 then
                print("[Topic] topic.bin 活跃头质心损坏，已跳过加载")
                return
            end
            offset = offset + head_len
        end
        if has_flag(active_mask, ACTIVE_TAIL_MASK) then
            local tail_len
            tail_vec, tail_len = tool.bin_to_vector(data, offset)
            if not tail_vec or tail_len <= 0 then
                print("[Topic] topic.bin 活跃尾质心损坏，已跳过加载")
                return
            end
            offset = offset + tail_len
        end
        if has_flag(active_mask, ACTIVE_LAST_MASK) then
            local last_len
            last_vec, last_len = tool.bin_to_vector(data, offset)
            if not last_vec or last_len <= 0 then
                print("[Topic] topic.bin 活跃 last_vec 损坏，已跳过加载")
                return
            end
            offset = offset + last_len
        end

        M.active_topic.start = active_start
        M.active_topic.head_centroid = head_vec
        M.active_topic.tail_window = tail_vec and {tail_vec} or {}
        M.active_topic.last_vec = last_vec
        M.active_topic.summary_cache = ""
        M.active_topic.summary_turn = 0
        
        print(string.format("[Topic] 恢复活跃话题: %d - ?, LastVec=%s", active_start, last_vec and "Yes" or "No"))
    else
        M.active_topic = {
            start = nil,
            head_centroid = nil,
            vectors = {},
            tail_window = {},
            last_vec = nil,
            summary_cache = "",
            summary_turn = 0,
        }
    end
    
    -- 清空旧 topics
    M.topics = {}
    local rec_count = 0
    while offset < #data do
        local rec, rec_size = tool.parse_topic_record(data, offset)
        if not rec then break end
        table.insert(M.topics, rec)
        offset = offset + rec_size
        rec_count = rec_count + 1
    end
    print(string.format("[Topic] 加载 %d 个历史话题", rec_count))
    if count ~= rec_count then
        print(string.format("[Topic] 话题计数不一致：header=%d parsed=%d", count, rec_count))
    end
end

-- ==================== 异常恢复与重建 ====================

local function rebuild_active_topic(current_turn, forced_start_turn)
    print(string.format(
        "[Topic][WARN] 已跳过话题重建：memory-core 模式不负责从文本重算 embedding (turn=%s, forced_start=%s)",
        tostring(current_turn),
        tostring(forced_start_turn)
    ))
end

-- 启动后恢复活跃话题的运行时向量缓存（避免仅恢复 head/tail 导致状态漂移）
local function hydrate_active_topic_vectors(start_turn, end_turn)
    return 0
end

-- ==================== 核心逻辑 ====================

local function invalidate_active_topic_summary()
    M.active_topic.summary_cache = ""
    M.active_topic.summary_turn = 0
end

local function build_topic_summary(start_turn, end_turn)
    start_turn = math.max(0, math.floor(tonumber(start_turn) or 0))
    end_turn = math.max(0, math.floor(tonumber(end_turn) or 0))
    if start_turn <= 0 or end_turn <= 0 or end_turn < start_turn then
        return ""
    end

    local dialog_texts = {}
    for t = start_turn, math.min(start_turn + 4, end_turn) do
        local d = M.dialogues[t]
        local user_text = d and d.user or history_module.get_turn_text(t, "user")
        local ai_text = d and d.ai or history_module.get_turn_text(t, "ai")
        if user_text or ai_text then
            dialog_texts[#dialog_texts + 1] = string.format(
                "第%d轮\n用户：%s\n助手：%s",
                t,
                tostring(user_text or ""),
                tostring(ai_text or "")
            )
        end
    end
    if #dialog_texts <= 0 then
        return ""
    end

    local prompt = "请用一句话概括话题：\n" .. table.concat(dialog_texts, "\n")
    prompt = prompt .. "\n/no_think"
    local messages = {{role="user", content=prompt}}
    local ok, summary = pcall(function()
        if not py_pipeline or not py_pipeline.generate_chat_sync then
            return ""
        end
        return py_pipeline:generate_chat_sync(messages, {
            max_tokens = SUMMARY_MAX_TOKENS,
            temperature = 0.1,
        }) or ""
    end)
    if not ok then
        print("[Topic][WARN] 按需生成摘要失败: " .. tostring(summary))
        return ""
    end
    return trim(tool.remove_cot(summary))
end

local function ensure_topic_summary(rec, variant)
    if type(rec) ~= "table" then
        return ""
    end
    
    variant = variant or "full"
    if variant ~= "full" and variant ~= "slight" and variant ~= "heavy" and variant ~= "none" then
        variant = "full"
    end
    if variant == "none" then
        return ""
    end

    if rec.is_active == true then
        local current_turn = math.max(0, tonumber(M._last_processed_turn) or 0)
        
        -- 检查缓存
        if variant == "full" then
            if trim(M.active_topic.summary_cache) ~= "" and tonumber(M.active_topic.summary_turn) == current_turn then
                return trim(M.active_topic.summary_cache)
            end
        else
            local cached_variant = trim(M.active_topic.summary_variants[variant] or "")
            if cached_variant ~= "" and tonumber(M.active_topic.summary_turn) == current_turn then
                return cached_variant
            end
        end
        
        if not ALLOW_LLM_SUMMARY then
            return ""
        end

        -- 生成分级摘要
        local variants = build_topic_summary_variants(rec.start, current_turn)
        
        -- 更新缓存
        M.active_topic.summary_cache = variants.full
        M.active_topic.summary_variants = variants
        M.active_topic.summary_turn = current_turn
        
        return trim(variants[variant] or "")
    end

    -- 对于已完成的话题
    local topic = nil
    if rec.topic_idx and M.topics[tonumber(rec.topic_idx)] then
        topic = M.topics[tonumber(rec.topic_idx)]
    else
        -- 查找对应的话题记录
        for _, t in ipairs(M.topics) do
            if tonumber(t.start) == tonumber(rec.start) then
                topic = t
                break
            end
        end
    end
    
    if not topic then
        return ""
    end
    
    -- 确保有summary_variants字段
    topic.summary_variants = topic.summary_variants or { full = "", slight = "", heavy = "", none = "" }
    
    -- 检查是否已经有该变体的摘要
    local cached_variant = trim(topic.summary_variants[variant] or "")
    if cached_variant ~= "" then
        return cached_variant
    end
    
    -- 如果没有full摘要，先生成full摘要
    if not ALLOW_LLM_SUMMARY then
        return ""
    end
    if trim(topic.summary_variants.full or "") == "" then
        local variants = build_topic_summary_variants(rec.start, rec.end_)
        topic.summary_variants = variants
        topic.summary = variants.full  -- 保持向后兼容
        
        -- 保存到磁盘
        local ok, err = save_to_disk()
        if not ok then
            print("[Topic][WARN] 摘要回写失败: " .. tostring(err))
        end
    end
    
    return trim(topic.summary_variants[variant] or "")
end

local function close_current_topic(end_turn)
    if not M.active_topic.start then return end

    -- 生成整体质心
    local overall_centroid = tool.average_vectors(M.active_topic.vectors)
    
    -- 保存已经生成的摘要（如果有的话）
    local summary = trim(M.active_topic.summary_cache or "")
    local summary_variants = M.active_topic.summary_variants or { full = "", slight = "", heavy = "", none = "" }

    table.insert(M.topics, {
        start = M.active_topic.start,
        end_ = end_turn,
        summary = summary,
        centroid = overall_centroid,
        summary_variants = summary_variants
    })

    print(string.format("[Topic] 话题结束: %d-%d. 摘要已保存（如果已生成）", M.active_topic.start, end_turn))

    -- 重置状态
    M.active_topic = {
        start = nil,
        head_centroid = nil,
        vectors = {},
        tail_window = {},
        last_vec = nil,
        scope_key = "",
        summary_cache = "",
        summary_turn = 0,
        summary_variants = { full = "", slight = "", heavy = "", none = "" }
    }
    local ok, err = save_to_disk()
    if not ok then
        print("[Topic][ERROR] 保存失败: " .. tostring(err))
    end
end

function M.add_turn(turn, user_text, vector, meta)
    meta = type(meta) == "table" and meta or {}
    local scope_key = trim(meta.scope_key or meta.scope or meta.scope_id or "")
    local isolate_by_scope = (config.get("guard.topic_scope_isolation", true) ~= false)

    if isolate_by_scope and scope_key ~= "" and M.active_topic.start and trim(M.active_topic.scope_key or "") ~= "" then
        if trim(M.active_topic.scope_key or "") ~= scope_key then
            -- Force a split on scope boundary to reduce multi-user/topic pollution.
            close_current_topic(turn - 1)
        end
    end

    -- 1. 存储基础数据
    M.turn_vectors[turn] = vector
    M.dialogues[turn] = M.dialogues[turn] or {}
    M.dialogues[turn].user = user_text
    M._last_processed_turn = turn
    invalidate_active_topic_summary()
    
    -- 2. 初始化或延续话题
    if not M.active_topic.start then
        -- 开始新话题
        M.active_topic.start = turn
        M.active_topic.vectors = {vector}
        M.active_topic.tail_window = {vector}
        M.active_topic.last_vec = vector
        M.active_topic.head_centroid = nil
        M.active_topic.scope_key = scope_key
        M.active_topic.summary_cache = ""
        M.active_topic.summary_turn = 0
        print("[Topic] 开启新话题: 第" .. turn .. "轮")
    else
        -- ========== 核心优化：基于论文的话语对建模 ==========
        
        -- 计算“话语对相似度”：当前轮 vs 上一轮
        local sim_local = 0.0
        if M.active_topic.last_vec then
            sim_local = tool.cosine_similarity(M.active_topic.last_vec, vector)
        end
        
        -- 追加到当前话题缓冲区
        table.insert(M.active_topic.vectors, vector)
        table.insert(M.active_topic.tail_window, vector)
        if #M.active_topic.tail_window > MAKE_CLUSTER2 then
            table.remove(M.active_topic.tail_window, 1)
        end
        
        -- 3. 构建头质心 (如果尚未完成)
        if not M.active_topic.head_centroid then
            if #M.active_topic.vectors >= MAKE_CLUSTER1 then
                local head_vecs = {}
                for i = 1, MAKE_CLUSTER1 do table.insert(head_vecs, M.active_topic.vectors[i]) end
                M.active_topic.head_centroid = tool.average_vectors(head_vecs)
                print("[Topic] 头质心建立完成")
            end
        end
        
        -- 4. 分割检测逻辑 (双重验证)
        local should_split = false
        
        -- 情况 A: 话语对断裂检测
        if sim_local < BREAK_LIMIT then
            -- 发生了明显的“跳跃”或“断层”
            if M.active_topic.head_centroid then
                -- 计算当前向量与“话题开头”的距离
                local sim_global = tool.cosine_similarity(M.active_topic.head_centroid, vector)
                
                if sim_global < CONFIRM_LIMIT then
                    print(string.format("[Topic] 检测到话题断裂! (Local Sim: %.3f < %.2f, Global Sim: %.3f < %.2f)", 
                        sim_local, BREAK_LIMIT, sim_global, CONFIRM_LIMIT))
                    should_split = true
                else
                    -- 断层发生，但与开头相关 -> 可能是追问、补充细节，不算新话题
                    print(string.format("[Topic] 检测到短暂偏移但属于当前话题 (Local: %.3f, Global: %.3f)", sim_local, sim_global))
                end
            end
        end
        
        -- 情况 B: 全局漂移兜底
        if not should_split and M.active_topic.head_centroid and #M.active_topic.tail_window >= MAKE_CLUSTER2 then
            local tail_centroid = tool.average_vectors(M.active_topic.tail_window)
            local sim_drift = tool.cosine_similarity(M.active_topic.head_centroid, tail_centroid)
            
            if sim_drift < TOPIC_LIMIT then
                 print(string.format("[Topic] 检测到话题长期漂移! (Drift Sim: %.3f < %.2f)", sim_drift, TOPIC_LIMIT))
                 should_split = true
            end
        end
        
        -- 5. 执行分割，但先检查最小话题长度
        if should_split then
            local current_len = #M.active_topic.vectors          -- 包含本轮向量的长度
            local previous_len = current_len - 1                 -- 话题之前已有的轮数（去掉本轮）
            
            if previous_len < MIN_TOPIC_LENGTH then
                print(string.format("[Topic] 话题之前长度 %d < min_topic_length (%d)，暂不分割",
                                    previous_len, MIN_TOPIC_LENGTH))
                should_split = false   -- 放弃分割，继续当前话题
            else
                -- 分割前，将本轮向量从当前话题中移除（因为它属于新话题）
                table.remove(M.active_topic.vectors)  -- 移除 vectors 中的最后一个（即本轮向量）
                if #M.active_topic.tail_window > 0 and
                   M.active_topic.tail_window[#M.active_topic.tail_window] == vector then
                    table.remove(M.active_topic.tail_window)  -- 从尾窗口中移除
                end
                -- 关闭旧话题（结束于 turn-1）
                close_current_topic(turn - 1)
                
                -- 开启新话题，包含本轮向量
                M.active_topic.start = turn
                M.active_topic.vectors = {vector}
                M.active_topic.tail_window = {vector}
                M.active_topic.last_vec = vector
                M.active_topic.head_centroid = nil
                M.active_topic.summary_cache = ""
                M.active_topic.summary_turn = 0
            end
        end
        
        -- 正常延续：更新 last_vec（如果分割被抑制或未检测到分割）
        if not should_split then
            M.active_topic.last_vec = vector
        end
    end
    
    local ok, err = save_to_disk()
    if not ok then
        print("[Topic][ERROR] 保存失败: " .. tostring(err))
    end
end

function M.update_assistant(turn, text)
    if M.dialogues[turn] then M.dialogues[turn].ai = text end
    if M.active_topic.start and tonumber(turn) and tonumber(turn) >= tonumber(M.active_topic.start) then
        invalidate_active_topic_summary()
    end
end

function M.get_summary(turn)
    local t = turn or M._last_processed_turn
    if M.active_topic.start and t >= M.active_topic.start then
        return ensure_topic_summary({
            start = M.active_topic.start,
            is_active = true,
        })
    end
    for _, topic in ipairs(M.topics) do
        if topic.start <= t and t <= (topic.end_ or 999999) then
            return ensure_topic_summary(topic)
        end
    end
    return ""
end

function M.get_summary_variant(turn, variant)
    local t = turn or M._last_processed_turn
    variant = variant or "full"
    if variant ~= "full" and variant ~= "slight" and variant ~= "heavy" and variant ~= "none" then
        variant = "full"
    end
    
    if M.active_topic.start and t >= M.active_topic.start then
        return ensure_topic_summary({
            start = M.active_topic.start,
            is_active = true,
        }, variant)
    end
    for _, topic in ipairs(M.topics) do
        if topic.start <= t and t <= (topic.end_ or 999999) then
            return ensure_topic_summary(topic, variant)
        end
    end
    return ""
end

function M.finalize()
    -- 退出时不主动闭合话题；保留 active 状态，启动时按未完成话题重建
    if M.active_topic.start then
        print(string.format(
            "[Topic] 退出时标记未完成话题: %d-? (last_turn=%d)，下次启动将重建",
            M.active_topic.start,
            M._last_processed_turn
        ))
    end
    local ok, err = save_to_disk()
    if not ok then
        print("[Topic][ERROR] 保存失败: " .. tostring(err))
    end
end

function M.init()
    load_from_disk()
    
    -- 检查是否需要 Rebuild
    local history_turns = history_module.get_turn() -- 假设 history 模块有该函数
    if DO_REBUILD and M.active_topic.start then
        print(string.format(
            "[Topic] 检测到未完成话题 A:%d，执行重建（Hist=%d）...",
            M.active_topic.start, history_turns
        ))
        rebuild_active_topic(history_turns, M.active_topic.start)
    elseif DO_REBUILD and M._last_processed_turn > 0 and history_turns > M._last_processed_turn then
        print("[Topic] 状态不一致 (Bin=" .. M._last_processed_turn .. ", Hist=" .. history_turns .. ")，执行重建...")
        rebuild_active_topic(history_turns)
    elseif M.active_topic.start then
        print(string.format(
            "[Topic] 活跃话题已恢复: start=%d, last_turn=%d (未进行 embedding hydrate)",
            M.active_topic.start,
            M._last_processed_turn
        ))
    end
    
    print("[Topic] 模块初始化完成")
end

function M.get_topic_for_turn(turn)
    -- 返回 {is_active = bool, centroid = vec_table or nil, topic_idx = nil or number}
    if M.active_topic.start and turn >= M.active_topic.start then
        local cent = M.active_topic.head_centroid
        if not cent and #M.active_topic.vectors > 0 then
            cent = tool.average_vectors(M.active_topic.vectors)
        end
        return { is_active = true, centroid = cent, topic_idx = nil }
    end

    for i, t in ipairs(M.topics) do
        if t.start <= turn and (not t.end_ or turn <= t.end_) then
            return { is_active = false, centroid = t.centroid, topic_idx = i }
        end
    end
    return nil
end

function M.get_topic_anchor(turn)
    local t = turn or M._last_processed_turn
    if M.active_topic.start and t >= M.active_topic.start then
        return "A:" .. tostring(M.active_topic.start)
    end
    for i, rec in ipairs(M.topics) do
        if rec.start <= t and (not rec.end_ or t <= rec.end_) then
            return "C:" .. tostring(i)
        end
    end
    return nil
end

function M.get_stable_anchor(turn)
    local t = turn or M._last_processed_turn
    if M.active_topic.start and t >= M.active_topic.start then
        return "S:" .. tostring(M.active_topic.start)
    end
    for _, rec in ipairs(M.topics) do
        if rec.start <= t and (not rec.end_ or t <= rec.end_) then
            return stable_topic_key_from_record(rec)
        end
    end
    return nil
end

function M.get_topic_fingerprint(topic_key)
    local stable_key, rec, scoped_key, scope_key = resolve_topic_key(topic_key)
    if not stable_key then
        return {
            key = nil,
            base_key = nil,
            scope_key = "",
            memory_count = 0,
            cluster_count = 0,
            dominant_cluster = nil,
            top_clusters = {},
            histogram = {},
            weights = {},
            centroid = nil,
            summary = "",
            start = nil,
            topic_idx = nil,
            is_active = false,
        }
    end

    local inferred_start = tonumber(stable_key:match("^S:(%d+)$"))

    local ok_store, store = pcall(require, "module.memory.store")
    local lines = {}
    if ok_store and store and store.iter_topic_lines then
        lines = store.iter_topic_lines(scoped_key) or {}
        if #lines <= 0 and scoped_key ~= stable_key then
            lines = store.iter_topic_lines(stable_key) or {}
        end
    end

    local histogram = {}
    local memory_count = 0
    if ok_store and store and store.get_cluster_id then
        for _, line in ipairs(lines) do
            local cid = tonumber(store.get_cluster_id(line))
            memory_count = memory_count + 1
            if cid then
                histogram[cid] = (histogram[cid] or 0) + 1
            end
        end
    else
        memory_count = #lines
    end

    local cluster_items = {}
    for cid, hits in pairs(histogram) do
        cluster_items[#cluster_items + 1] = {
            cluster_id = tonumber(cid),
            hits = tonumber(hits) or 0,
            weight = memory_count > 0 and ((tonumber(hits) or 0) / memory_count) or 0.0,
        }
    end
    table.sort(cluster_items, function(a, b)
        if (a.hits or 0) ~= (b.hits or 0) then
            return (a.hits or 0) > (b.hits or 0)
        end
        return (a.cluster_id or 0) < (b.cluster_id or 0)
    end)

    local top_clusters = {}
    local weights = {}
    for cid, hits in pairs(histogram) do
        weights[cid] = memory_count > 0 and ((tonumber(hits) or 0) / memory_count) or 0.0
    end
    for i = 1, math.min(FINGERPRINT_TOPK, #cluster_items) do
        top_clusters[#top_clusters + 1] = {
            cluster_id = cluster_items[i].cluster_id,
            hits = cluster_items[i].hits,
            weight = cluster_items[i].weight,
        }
    end

    local summary = ensure_topic_summary(rec)

    return {
        key = scoped_key,
        base_key = stable_key,
        scope_key = scope_key,
        memory_count = memory_count,
        cluster_count = #cluster_items,
        dominant_cluster = top_clusters[1] and top_clusters[1].cluster_id or nil,
        top_clusters = top_clusters,
        histogram = histogram,
        weights = weights,
        centroid = rec and rec.centroid or nil,
        summary = summary,
        start = rec and rec.start or inferred_start,
        topic_idx = rec and rec.topic_idx or nil,
        is_active = rec and (rec.is_active == true) or false,
        lines = shallow_copy_array(lines),
    }
end

function M.get_topic_chain(topic_key, opts)
    opts = type(opts) == "table" and opts or {}
    local base_fp = M.get_topic_fingerprint(topic_key)
    if not base_fp.key then
        return {}
    end
    local base_key = trim(base_fp.base_key or base_fp.key)
    local scope_key = trim(base_fp.scope_key or "")

    local centroid_weight = tonumber(opts.centroid_weight) or CHAIN_CENTROID_WEIGHT
    local hist_weight = tonumber(opts.hist_weight) or CHAIN_HIST_WEIGHT
    local dominant_bonus = tonumber(opts.dominant_bonus) or CHAIN_DOMINANT_BONUS
    local topn = math.max(1, tonumber(opts.topn) or CHAIN_TOPN)
    local min_score = tonumber(opts.min_score) or CHAIN_MIN_SCORE

    local candidates = {}
    for _, rec in ipairs(M.topics) do
        local bare_key = stable_topic_key_from_record(rec)
        if bare_key and bare_key ~= base_key then
            local key = scope_key ~= "" and (scope_key .. "|" .. bare_key) or bare_key
            candidates[#candidates + 1] = key
        end
    end
    if M.active_topic.start then
        local bare_active_key = "S:" .. tostring(M.active_topic.start)
        if bare_active_key ~= base_key then
            local active_key = scope_key ~= "" and (scope_key .. "|" .. bare_active_key) or bare_active_key
            candidates[#candidates + 1] = active_key
        end
    end

    local seen = {}
    local scored = {}
    for _, key in ipairs(candidates) do
        if not seen[key] then
            seen[key] = true
            local cand_fp = M.get_topic_fingerprint(key)
            local centroid_sim = 0.0
            if base_fp.centroid and cand_fp.centroid then
                centroid_sim = tonumber(tool.cosine_similarity(base_fp.centroid, cand_fp.centroid)) or 0.0
            end
            local overlap = histogram_overlap(base_fp, cand_fp)
            local same_dominant = base_fp.dominant_cluster
                and cand_fp.dominant_cluster
                and base_fp.dominant_cluster == cand_fp.dominant_cluster
            local score = centroid_weight * math.max(0.0, centroid_sim)
                + hist_weight * overlap
                + (same_dominant and dominant_bonus or 0.0)

            if score >= min_score then
                scored[#scored + 1] = {
                    key = cand_fp.key,
                    score = score,
                    centroid_similarity = centroid_sim,
                    fingerprint_overlap = overlap,
                    dominant_cluster = cand_fp.dominant_cluster,
                    memory_count = cand_fp.memory_count,
                    topic_idx = cand_fp.topic_idx,
                    is_active = cand_fp.is_active,
                    summary = cand_fp.summary,
                }
            end
        end
    end

    table.sort(scored, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return tostring(a.key or "") < tostring(b.key or "")
    end)

    while #scored > topn do
        table.remove(scored)
    end
    return scored
end

return M
