local M = {}

local function file_exists(path)
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

-- 同目录原子替换：
-- 1) 先尝试直接 rename(temp -> target)（POSIX 可覆盖）
-- 2) 若失败，则把 target 挪到 .bak，再迁移 temp；失败时回滚
function M.atomic_replace(temp_path, target_path)
    local ok, err = os.rename(temp_path, target_path)
    if ok then return true end

    local had_target = file_exists(target_path)
    if not had_target then
        return false, "rename_failed: " .. tostring(err)
    end

    local backup_path = target_path .. ".bak"
    os.remove(backup_path)

    local mv_ok, mv_err = os.rename(target_path, backup_path)
    if not mv_ok then
        return false, "backup_failed: " .. tostring(mv_err)
    end

    local replace_ok, replace_err = os.rename(temp_path, target_path)
    if replace_ok then
        os.remove(backup_path)
        return true
    end

    -- 回滚旧文件
    os.rename(backup_path, target_path)
    return false, "replace_failed: " .. tostring(replace_err)
end

-- 原子写文件封装：
-- write_cb(f) 返回 true 表示写入成功；返回 false, err 表示失败
function M.write_atomic(path, mode, write_cb)
    local temp_path = path .. ".tmp"
    local f, open_err = io.open(temp_path, mode)
    if not f then
        return false, "open_temp_failed: " .. tostring(open_err)
    end

    local ok, cb_ok, cb_err = pcall(write_cb, f)
    if not ok then
        pcall(function() f:close() end)
        os.remove(temp_path)
        return false, "write_callback_error: " .. tostring(cb_ok)
    end
    if cb_ok == false then
        pcall(function() f:close() end)
        os.remove(temp_path)
        return false, "write_failed: " .. tostring(cb_err)
    end

    local close_ok, close_err = f:close()
    if close_ok == nil then
        os.remove(temp_path)
        return false, "close_failed: " .. tostring(close_err)
    end

    local rep_ok, rep_err = M.atomic_replace(temp_path, path)
    if not rep_ok then
        os.remove(temp_path)
        return false, rep_err
    end

    return true
end

return M
