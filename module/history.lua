local M = {}

local tool = require("module.tool")

-- 字段分隔符：\x1F（单元分隔符），几乎不会出现在正常文本中
local FIELD_SEP = "\x1F"
-- 用于替换 AI 回复中的换行符：\x1E（记录分隔符），还原时再转回 \n
local NEWLINE_REPLACE = "\x1E"

local file_path = "memory/history.txt"

M.entries = {}
M.turn_counter = 0

-- ====================== LOAD：启动时全扫描一次，顺便算出 turn ======================
function M.load()
    M.entries = {}
    M.turn_counter = 0
    local f = io.open(file_path, "r")
    if f then
        for line in f:lines() do
            if line and line ~= "" then
                table.insert(M.entries, line)
                M.turn_counter = M.turn_counter + 1
            end
        end
        f:close()
    end
    print(string.format("[History] 已加载 %d 条对话记录到内存，当前 turn_counter = %d", #M.entries, M.turn_counter))
end

-- ====================== 添加历史记录（新格式：user + FIELD_SEP + ai） ======================
function M.add_history(user, ai)
    local saver = require("module.saver")   -- 延迟加载

    -- 1. 对 AI 回复进行预处理：移除 cot 标记，替换换行符
    local ai_processed = tool.remove_cot(ai)
    ai_processed = tool.replace(ai_processed, "\n", NEWLINE_REPLACE)

    -- 2. 拼接字段分隔符（用户输入无需替换，因为它不包含 FIELD_SEP）
    local line = user .. FIELD_SEP .. ai_processed

    table.insert(M.entries, line)
    M.turn_counter = M.turn_counter + 1
    saver.mark_dirty()
end

-- ====================== 解析单条历史记录（兼容新旧格式） ======================
-- @param entry: 从 entries 中取出的原始字符串
-- @return user, ai: 两个字符串，若解析失败则返回 nil
function M.parse_entry(entry)
    if not entry or entry == "" then
        return nil, nil
    end

    -- 1. 尝试新格式：user \x1F ai
    local user_part, ai_part = entry:match("^(.-)" .. FIELD_SEP .. "(.*)$")
    if user_part and ai_part then
        -- 还原 AI 中的换行符
        ai_part = ai_part:gsub(NEWLINE_REPLACE, "\n")
        return user_part, ai_part
    end

    -- 2. 尝试旧格式（过渡期兼容）：user:...ai:...
    local old_user, old_ai = entry:match("^user:(.+)ai:(.+)$")
    if old_user and old_ai then
        -- 旧格式中换行被替换为 \x1F，需要还原
        old_ai = old_ai:gsub("\x1F", "\n")
        return old_user, old_ai
    end

    -- 3. 未知格式，原样返回（仅用户部分，AI 为空）
    return entry, ""
end

-- ====================== 获取当前对话轮数 ======================
function M.get_turn()
    return M.turn_counter
end

-- ====================== 按轮次获取原始记录 ======================
function M.get_by_turn(turn)
    if turn < 1 or turn > #M.entries then return nil end
    return M.entries[turn]
end

-- ====================== 【新增】按轮次和角色获取文本（用于 Topic 重建） ======================
-- @param turn: 轮次 (1-based)
-- @param role: "user" 或 "ai"
-- @return string: 对应的文本，若不存在则返回 nil
function M.get_turn_text(turn, role)
    local entry = M.get_by_turn(turn)
    if not entry then return nil end
    
    local user_text, ai_text = M.parse_entry(entry)
    
    if role == "user" then
        return user_text
    elseif role == "ai" then
        return ai_text
    else
        return nil
    end
end

-- ====================== 保存到磁盘 ======================
function M.save_to_disk()
    local temp = file_path .. ".tmp"
    local f = io.open(temp, "w")
    if not f then return end
    for _, line in ipairs(M.entries) do
        f:write(line .. "\n")
    end
    f:close()
    os.remove(file_path)
    os.rename(temp, file_path)
end

return M
