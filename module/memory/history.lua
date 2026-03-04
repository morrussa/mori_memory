local M = {}

local tool = require("module.tool")
local persistence = require("module.persistence")

local FIELD_SEP = "\x1F"
local RECORD_SEP = "\x1E"
local HISTORY_HEADER = "HIST_V2"

local file_path = "memory/history.txt"

M.entries = {}
M.turn_counter = 0

local function escape_field(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\")
    s = s:gsub("\n", "\\n")
    s = s:gsub(FIELD_SEP, "\\x1F")
    s = s:gsub(RECORD_SEP, "\\x1E")
    return s
end

local function unescape_field(s)
    s = tostring(s or "")
    local out = {}
    local i = 1
    while i <= #s do
        local ch = s:sub(i, i)
        if ch ~= "\\" then
            out[#out + 1] = ch
            i = i + 1
        else
            local n1 = s:sub(i + 1, i + 1)
            if n1 == "\\" then
                out[#out + 1] = "\\"
                i = i + 2
            elseif n1 == "n" then
                out[#out + 1] = "\n"
                i = i + 2
            elseif n1 == "x" then
                local hex = s:sub(i + 2, i + 3)
                if hex == "1F" then
                    out[#out + 1] = FIELD_SEP
                    i = i + 4
                elseif hex == "1E" then
                    out[#out + 1] = RECORD_SEP
                    i = i + 4
                else
                    out[#out + 1] = "\\"
                    i = i + 1
                end
            else
                out[#out + 1] = "\\"
                i = i + 1
            end
        end
    end
    return table.concat(out)
end

function M.load()
    M.entries = {}
    M.turn_counter = 0

    local f = io.open(file_path, "r")
    if not f then
        print("[History] history.txt 不存在，使用空历史")
        return
    end

    local header = f:read("*l")
    if header ~= HISTORY_HEADER then
        f:close()
        print("[History] 检测到旧版 history.txt，已跳过加载（仅支持 HIST_V2）")
        return
    end

    for line in f:lines() do
        if line and line ~= "" then
            table.insert(M.entries, line)
            M.turn_counter = M.turn_counter + 1
        end
    end
    f:close()
    print(string.format("[History] 已加载 %d 条对话记录到内存，当前 turn_counter = %d", #M.entries, M.turn_counter))
end

function M.add_history(user, ai)
    local saver = require("module.memory.saver")

    local ai_processed = tool.remove_cot(ai)
    local line = escape_field(user) .. FIELD_SEP .. escape_field(ai_processed)

    table.insert(M.entries, line)
    M.turn_counter = M.turn_counter + 1
    saver.mark_dirty()
end

function M.parse_entry(entry)
    if not entry or entry == "" then
        return nil, nil
    end

    local user_part, ai_part = entry:match("^(.-)" .. FIELD_SEP .. "(.*)$")
    if user_part and ai_part then
        return unescape_field(user_part), unescape_field(ai_part)
    end

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
    local ok, err = persistence.write_atomic(file_path, "w", function(f)
        local h_ok, h_err = f:write(HISTORY_HEADER .. "\n")
        if not h_ok then
            return false, h_err
        end

        for _, line in ipairs(M.entries) do
            local w_ok, w_err = f:write(line .. "\n")
            if not w_ok then
                return false, w_err
            end
        end
        return true
    end)
    if not ok then
        return false, err
    end
    return true
end

return M
