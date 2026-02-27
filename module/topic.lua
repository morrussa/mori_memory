-- topic.lua
local M = {}
local ffi = require("ffi")
local tool = require("module.tool")
local config = require("module.config")
local persistence = require("module.persistence")
local history_module = require("module.history") -- 依赖 history 进行 rebuild
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

local DO_REBUILD = T_CONF.rebuild
local IDX_FILE = "memory/topic.bin"
local TOPIC_VERSION = 2
local ACTIVE_HEAD_MASK = 1
local ACTIVE_TAIL_MASK = 2
local ACTIVE_LAST_MASK = 4

-- ==================== 核心状态 ====================
M.turn_vectors = {}       -- 缓存近期轮次的向量 (只保留必要的)
M.topics = {}             -- 已完成的话题列表 {{start, end_, summary, centroid}, ...}
M.dialogues = {}          -- turn -> {user=, ai=}

-- 当前活跃话题状态
M.active_topic = {
    start = nil,          -- 起始轮次
    head_centroid = nil,  -- 头质心
    vectors = {},         -- 当前话题累积的向量 (用于最后生成 overall centroid)
    tail_window = {},     -- 尾部滑动窗口
    last_vec = nil        -- [新增] 记录上一轮的向量，用于计算话语对相似度
}

M._last_processed_turn = 0

-- ==================== 二进制读写 ====================

local function has_flag(mask, flag)
    return (mask % (flag * 2)) >= flag
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
            local bin = tool.create_topic_record(t.start, t.end_, t.summary or "", t.centroid)
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
        
        print(string.format("[Topic] 恢复活跃话题: %d - ?, LastVec=%s", active_start, last_vec and "Yes" or "No"))
    else
        M.active_topic = { start = nil, head_centroid = nil, vectors = {}, tail_window = {}, last_vec = nil }
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
    print("[Topic] 触发话题重建...")
    -- 1. 确定重建起点
    local start_turn = tonumber(forced_start_turn)
    if start_turn and start_turn > 0 then
        start_turn = math.floor(start_turn)
    else
        start_turn = 1
        if #M.topics > 0 then
            start_turn = (M.topics[#M.topics].end_ or 0) + 1
        end
    end
    
    -- 2. 重新嵌入并计算
    M.active_topic = { start = start_turn, vectors = {}, tail_window = {}, last_vec = nil }
    
    print(string.format("[Topic] 正在从 history.txt 重建 Turn %d -> %d 的向量...", start_turn, current_turn))
    
    for t = start_turn, current_turn do
        local user_text = history_module.get_turn_text(t, "user")
        if user_text then
            -- local vec = tool.get_embedding(user_text)
            local vec = tool.get_embedding_passage(user_text)
            M.turn_vectors[t] = vec
            table.insert(M.active_topic.vectors, vec)
            
            -- 维护尾窗口
            table.insert(M.active_topic.tail_window, vec)
            if #M.active_topic.tail_window > MAKE_CLUSTER2 then
                table.remove(M.active_topic.tail_window, 1)
            end
        end
    end
    
    -- 3. 计算头质心
    local head_vectors = {}
    for i = 1, math.min(MAKE_CLUSTER1, #M.active_topic.vectors) do
        table.insert(head_vectors, M.active_topic.vectors[i])
    end
    M.active_topic.head_centroid = tool.average_vectors(head_vectors)
    
    -- [新增] 设置 last_vec 为最后一个向量
    if #M.active_topic.vectors > 0 then
        M.active_topic.last_vec = M.active_topic.vectors[#M.active_topic.vectors]
    end
    
    M._last_processed_turn = current_turn
    print("[Topic] 话题重建完成")
end

-- 启动后恢复活跃话题的运行时向量缓存（避免仅恢复 head/tail 导致状态漂移）
local function hydrate_active_topic_vectors(start_turn, end_turn)
    if not start_turn or not end_turn or end_turn < start_turn then
        M.active_topic.vectors = {}
        M.active_topic.tail_window = {}
        M.active_topic.last_vec = nil
        M.active_topic.head_centroid = nil
        return 0
    end

    local vectors = {}
    local tail_window = {}

    for t = start_turn, end_turn do
        local user_text = history_module.get_turn_text(t, "user")
        if user_text and user_text ~= "" then
            local vec = tool.get_embedding_passage(user_text)
            table.insert(vectors, vec)
            table.insert(tail_window, vec)
            if #tail_window > MAKE_CLUSTER2 then
                table.remove(tail_window, 1)
            end
        end
    end

    M.active_topic.vectors = vectors
    M.active_topic.tail_window = tail_window
    M.active_topic.last_vec = vectors[#vectors]

    if #vectors >= MAKE_CLUSTER1 then
        local head_vecs = {}
        for i = 1, MAKE_CLUSTER1 do
            table.insert(head_vecs, vectors[i])
        end
        M.active_topic.head_centroid = tool.average_vectors(head_vecs)
    else
        M.active_topic.head_centroid = nil
    end

    return #vectors
end

-- ==================== 核心逻辑 ====================

local function close_current_topic(end_turn)
    if not M.active_topic.start then return end
    
    -- 生成整体质心
    local overall_centroid = tool.average_vectors(M.active_topic.vectors)
    
    -- 获取摘要
    local key = M.active_topic.start .. "-" .. end_turn
    local summary = ""
    
    local dialog_texts = {}
    for t = M.active_topic.start, math.min(M.active_topic.start + 4, end_turn) do
         local d = M.dialogues[t]
         if d then table.insert(dialog_texts, string.format("第%d轮\n用户：%s\n助手：%s", t, d.user or "", d.ai or "")) end
    end
    if #dialog_texts > 0 then
        local prompt = "请用一句话概括话题：\n" .. table.concat(dialog_texts, "\n")
        -- /no_think 为 Qwen 家族特有控制符，直接硬编码
        prompt = prompt .. "\n/no_think"
        local messages = {{role="user", content=prompt}}
        summary = py_pipeline:generate_chat_sync(messages, {max_tokens=SUMMARY_MAX_TOKENS, temperature=0.1}) or ""
        summary = tool.remove_cot(summary):match("^%s*(.-)%s*$")
    end

    table.insert(M.topics, {
        start = M.active_topic.start,
        end_ = end_turn,
        summary = summary,
        centroid = overall_centroid
    })
    
    print(string.format("[Topic] 话题结束: %d-%d. 摘要: %s", M.active_topic.start, end_turn, summary))
    
    -- 重置状态
    M.active_topic = { start = nil, head_centroid = nil, vectors = {}, tail_window = {}, last_vec = nil }
    local ok, err = save_to_disk()
    if not ok then
        print("[Topic][ERROR] 保存失败: " .. tostring(err))
    end
end

function M.add_turn(turn, user_text, vector)
    -- 1. 存储基础数据
    M.turn_vectors[turn] = vector
    M.dialogues[turn] = M.dialogues[turn] or {}
    M.dialogues[turn].user = user_text
    M._last_processed_turn = turn
    
    -- 2. 初始化或延续话题
    if not M.active_topic.start then
        -- 开始新话题
        M.active_topic.start = turn
        M.active_topic.vectors = {vector}
        M.active_topic.tail_window = {vector}
        M.active_topic.last_vec = vector
        M.active_topic.head_centroid = nil 
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
end

function M.get_summary(turn)
    local t = turn or M._last_processed_turn
    for _, topic in ipairs(M.topics) do
        if topic.start <= t and t <= (topic.end_ or 999999) then
            return topic.summary or ""
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
        local hydrate_end = math.min(M._last_processed_turn, history_turns)
        local restored = hydrate_active_topic_vectors(M.active_topic.start, hydrate_end)
        print(string.format(
            "[Topic] 活跃话题向量缓存已恢复: start=%d, end=%d, vectors=%d",
            M.active_topic.start, hydrate_end, restored
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

return M
