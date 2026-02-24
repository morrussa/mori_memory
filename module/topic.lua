-- module/topic.lua
-- 二进制全量重写版 TopicSegmenter（与 memory.bin / clusters.bin 100% 一致风格：header + 原子 .tmp 重写）

local M = {}
local ffi = require("ffi")
local tool = require("module.tool")
local py_pipeline = py_pipeline

-- ==================== 配置 ====================
local SIMILARITY_THRESHOLD = 0.40
local CONSECUTIVE_LOW_THRESHOLD = 2
local MIN_TOPIC_LENGTH = 5
local IDX_FILE = "memory/history.idx"

-- ==================== 核心状态 ====================
M.turn_vectors = {}
M.topics = {}                 -- {{start= , end_= , summary= }, ...}
M.topics_summary = {}         -- "start-end" -> summary
M._consecutive_low_count = 0
M._pending_topic_start = nil
M._last_processed_turn = 0

M.dialogues = {}              -- turn -> {user=, ai=}

-- ==================== 二进制读写 ====================
local function load_summaries()
    if not tool.file_exists(IDX_FILE) then
        print("[Topic] history.idx 不存在，将从空开始")
        return
    end

    local f = io.open(IDX_FILE, "rb")
    if not f then return end
    local data = f:read("*a")
    f:close()

    if #data < 20 or data:sub(1,4) ~= "TOPC" then
        print("[Topic] 文件头无效（可能是旧 JSON 格式），请删除 history.idx 后重启")
        return
    end

    local offset = 20   -- magic(4) + version(4) + count(4) + reserved(8)
    local count = 0

    while offset < #data do
        local rec, rec_size = tool.parse_topic_record(data, offset)
        if not rec then break end

        local key = rec.start .. "-" .. (rec.end_ or "nil")
        M.topics_summary[key] = rec.summary
        table.insert(M.topics, {
            start = rec.start,
            end_ = rec.end_,
            summary = rec.summary
        })

        if rec.end_ and rec.end_ > M._last_processed_turn then
            M._last_processed_turn = rec.end_
        end

        count = count + 1
        offset = offset + rec_size
    end

    print(string.format("[Topic] 已从二进制加载 %d 个话题记录（header count = %d）", count, count))
end

-- ==================== 全量原子保存（与 memory.bin / clusters.bin 完全一致） ====================
local function save_to_disk()
    local temp = IDX_FILE .. ".tmp"
    local f = io.open(temp, "wb")
    if not f then 
        print("[Topic] 保存失败：无法创建临时文件")
        return 
    end

    -- Header: "TOPC" + version(4) + count(4) + reserved(8)
    f:write("TOPC")
    local count = #M.topics
    local header = ffi.new("uint32_t[4]", 1, count, 0, 0)
    f:write(ffi.string(header, 16))

    for _, t in ipairs(M.topics) do
        local bin = tool.create_topic_record(t.start, t.end_, t.summary or "")
        f:write(bin)
    end
    f:close()

    os.remove(IDX_FILE)
    os.rename(temp, IDX_FILE)
    print(string.format("[Topic] 二进制全量保存完成（%d 个话题）", count))
end

-- 兼容旧追加接口（内部调用全量保存）
local function save_summary(start, end_, summary)
    -- 找到或创建对应 topic 条目
    for _, t in ipairs(M.topics) do
        if t.start == start and (t.end_ == end_ or (not t.end_ and not end_)) then
            t.summary = summary
            break
        end
    end
    save_to_disk()
end

local function cosine_sim(v1, v2)
    return tool.cosine_similarity(v1, v2)
end

local function detect_boundary(turn)
    if #M.topics == 0 then
        table.insert(M.topics, {start = turn, end_ = nil})
        print("[Topic] 开始第一个话题: 第" .. turn .. "轮")
        return
    end

    local last = M.topics[#M.topics]
    if last.end_ ~= nil then
        table.insert(M.topics, {start = turn, end_ = nil})
        print("[Topic] 开始新话题: 第" .. turn .. "轮")
        return
    end

    if not M.turn_vectors[turn - 1] then
        last.end_ = turn - 1
        M:_process_topic(last.start, last.end_)
        table.insert(M.topics, {start = turn, end_ = nil})
        print("[Topic] 因缺失向量强制切换话题 " .. last.start .. "-" .. (turn-1))
        M:_reset_pending()
        return
    end

    local sim = cosine_sim(M.turn_vectors[turn-1], M.turn_vectors[turn])
    print(string.format("[Topic] 第%d↔%d轮 相似度 %.4f", turn-1, turn, sim))

    if sim < SIMILARITY_THRESHOLD then
        M._consecutive_low_count = M._consecutive_low_count + 1
        if not M._pending_topic_start then
            M._pending_topic_start = turn
        end
        if M._consecutive_low_count >= CONSECUTIVE_LOW_THRESHOLD then
            M:_confirm_topic_switch(turn)
        end
    else
        M:_reset_pending()
    end
end

function M:_confirm_topic_switch(current_turn)
    local last = M.topics[#M.topics]
    local new_start = M._pending_topic_start
    local end_old = new_start - 1

    last.end_ = end_old
    M:_process_topic(last.start, end_old)

    table.insert(M.topics, {start = new_start, end_ = nil})
    print(string.format("[Topic] 话题切换确认！旧 %d-%d → 新从 %d", last.start, end_old, new_start))

    M:_reset_pending()
end

function M:_reset_pending()
    M._consecutive_low_count = 0
    M._pending_topic_start = nil
end

function M:_process_topic(start, end_)
    local key = start .. "-" .. (end_ or "nil")
    if M.topics_summary[key] then return end

    if (end_ or M._last_processed_turn) - start + 1 < MIN_TOPIC_LENGTH then
        return
    end

    local dialog_texts = {}
    for t = start, math.min(start + 4, end_ or M._last_processed_turn) do
        local d = M.dialogues[t]
        if d then
            table.insert(dialog_texts, string.format("第%d轮\n用户：%s\n助手：%s", t, d.user or "", d.ai or ""))
        end
    end

    if #dialog_texts == 0 then return end

    local prompt = "请根据以下对话内容，用一句话概括这个话题的主要内容。\n要求：简洁、准确、不超过30个字。\n\n" ..
                   table.concat(dialog_texts, "\n") .. "\n\n概括："

    local messages = {
        {role = "system", content = "你是一个极简话题概括专家。用一句话（≤30字）概括核心话题，不要解释，不要列表。"},
        {role = "user",   content = prompt}
    }
    local params = {max_tokens = 512, temperature = 0.25}

    local summary = py_pipeline:generate_chat_sync(messages, params) or ""
    summary = tool.remove_cot(summary)
    summary = summary:match("^%s*(.-)%s*$"):gsub('^["\']', ""):gsub('["\']$', ""):gsub("。$", "")

    if #summary > 5 and summary ~= "" then
        M.topics_summary[key] = summary
        save_summary(start, end_, summary)
        print(string.format("[Topic] 话题 %d-%s 摘要：%s", start, end_ or "nil", summary))
    end
end

function M.add_turn(turn, user_text, vector)
    -- 假设 vector 已经是 Lua table（由上层 get_embedding 保证）
    if not vector or #vector == 0 then 
        print("[Topic] 警告：第" .. turn .. "轮向量为空或转换失败，跳过")
        return 
    end

    M.turn_vectors[turn] = vector

    -- 保存用户输入
    M.dialogues[turn] = M.dialogues[turn] or {}
    M.dialogues[turn].user = user_text
    M._last_processed_turn = math.max(M._last_processed_turn, turn)
    detect_boundary(turn)
end

function M.update_assistant(turn, assistant_text)
    if M.dialogues[turn] then
        M.dialogues[turn].ai = assistant_text
    end
end

function M.get_summary(turn)
    local t = turn or M._last_processed_turn
    for _, topic in ipairs(M.topics) do
        if topic.end_ and topic.start <= t and t <= topic.end_ then
            return topic.summary or M.topics_summary[topic.start .. "-" .. topic.end_] or ""
        end
    end
    return ""
end

function M.finalize()
    local last = M.topics[#M.topics]
    if last and not last.end_ then
        last.end_ = M._last_processed_turn
        M:_process_topic(last.start, last.end_)
        save_to_disk()   -- 确保退出时保存最后一个话题
    end
    print("[Topic] 二进制话题持久化完成")
end

function M.init()
    load_summaries()
    print("[Topic] 二进制话题分割模块已启动")
end

return M