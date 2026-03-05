-- context_manager.lua
-- Agent上下文优化模块：处理大文本截断、智能压缩、历史摘要等
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

-- ==================== 配置读取 ====================

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function ctx_cfg()
    local cfg = graph_cfg()
    return cfg.context_manager or {
        -- 工具结果最大字符数（单条）
        tool_result_max_chars = 4000,
        -- 工具结果硬上限（超过则强制截断）
        tool_result_hard_max_chars = 8000,
        -- tool_context总上限
        tool_context_total_max_chars = 6000,
        -- 是否启用智能压缩提示
        enable_smart_truncation = true,
        -- 是否启用重复内容检测
        enable_dedup = true,
        -- 重复检测的最小匹配长度
        dedup_min_match_chars = 100,
        -- 是否启用结果缓存
        enable_cache = true,
        -- 缓存最大条目数
        cache_max_entries = 32,
        -- 历史压缩阈值（对话对数）
        history_compress_threshold = 6,
        -- 是否启用上下文预算监控
        enable_budget_monitor = true,
        -- 预算警告阈值（相对于input_token_budget的比例）
        budget_warning_ratio = 0.85,
    }
end

-- ==================== 工具结果缓存 ====================

local tool_result_cache = {}
local cache_order = {}

local function cache_get(key)
    local cfg = ctx_cfg()
    if not cfg.enable_cache then
        return nil
    end
    if not key or key == "" then
        return nil
    end
    local entry = tool_result_cache[key]
    if not entry then
        return nil
    end
    -- 更新访问时间
    entry.last_access = os.time()
    return entry.result
end

local function cache_set(key, result)
    local cfg = ctx_cfg()
    if not cfg.enable_cache then
        return
    end
    if not key or key == "" then
        return
    end
    -- 清理过期缓存
    local max_entries = math.max(1, math.floor(tonumber(cfg.cache_max_entries) or 32))
    while #cache_order > max_entries do
        local oldest_key = table.remove(cache_order, 1)
        if oldest_key then
            tool_result_cache[oldest_key] = nil
        end
    end
    -- 添加新缓存
    if not tool_result_cache[key] then
        cache_order[#cache_order + 1] = key
    end
    tool_result_cache[key] = {
        result = result,
        cached_at = os.time(),
        last_access = os.time(),
    }
end

function M.clear_cache()
    tool_result_cache = {}
    cache_order = {}
end

function M.cache_stats()
    return {
        entries = #cache_order,
        max_entries = math.floor(tonumber(ctx_cfg().cache_max_entries) or 32),
    }
end

-- ==================== 文本截断与压缩 ====================

local function count_lines(text)
    local count = 0
    for _ in text:gmatch("\n") do
        count = count + 1
    end
    if #text > 0 then
        count = count + 1
    end
    return count
end

local function smart_truncate(text, max_chars, hint)
    text = tostring(text or "")
    max_chars = math.max(1, math.floor(tonumber(max_chars) or 4000))
    hint = hint or "[...truncated...]"

    if #text <= max_chars then
        return text, false
    end

    local cfg = ctx_cfg()
    if not cfg.enable_smart_truncation then
        return util.utf8_take(text, max_chars), true
    end

    -- 智能截断：保留开头和结尾
    local head_ratio = 0.6
    local tail_ratio = 0.3
    local hint_len = #hint + 20 -- 留出提示的空间

    local head_chars = math.floor((max_chars - hint_len) * head_ratio)
    local tail_chars = math.floor((max_chars - hint_len) * tail_ratio)

    local head_part = util.utf8_take(text, head_chars)
    local tail_start = math.max(1, #text - tail_chars + 1)

    -- 尝试从完整字符边界开始
    local tail_text = text:sub(tail_start)
    -- 确保UTF-8边界正确
    while tail_text:sub(1, 1):byte() and (tail_text:sub(1, 1):byte() or 0) >= 0x80 and (tail_text:sub(1, 1):byte() or 0) < 0xC0 do
        tail_start = tail_start - 1
        if tail_start < 1 then
            break
        end
        tail_text = text:sub(tail_start)
    end

    local truncated = head_part .. "\n\n" .. hint .. "\n\n" .. tail_text
    return truncated, true
end

-- 检测并移除重复内容
local function dedup_content(new_text, existing_texts)
    local cfg = ctx_cfg()
    if not cfg.enable_dedup then
        return new_text
    end

    local min_match = math.max(50, math.floor(tonumber(cfg.dedup_min_match_chars) or 100))
    new_text = tostring(new_text or "")

    if #new_text < min_match then
        return new_text
    end

    for _, existing in ipairs(existing_texts or {}) do
        existing = tostring(existing or "")
        if #existing >= min_match then
            -- 检查是否有大段重复
            if new_text:find(existing:sub(1, math.min(#existing, 500)), 1, true) then
                -- 发现重复，标记但不完全移除
                return new_text .. "\n[Note: Similar content detected in previous context]"
            end
        end
    end

    return new_text
end

-- ==================== 工具结果处理 ====================

-- 处理单个工具结果
function M.process_tool_result(tool_name, args, result_text)
    local cfg = ctx_cfg()
    result_text = tostring(result_text or "")

    -- 生成缓存key
    local cache_key = nil
    if cfg.enable_cache then
        local args_str = util.json_encode(args or {})
        cache_key = tool_name .. ":" .. args_str
        -- 检查缓存
        local cached = cache_get(cache_key)
        if cached then
            return cached, true, 0 -- 返回缓存结果
        end
    end

    local original_len = #result_text
    local was_truncated = false

    -- 大小检查
    local soft_max = math.max(100, math.floor(tonumber(cfg.tool_result_max_chars) or 4000))
    local hard_max = math.max(soft_max, math.floor(tonumber(cfg.tool_result_hard_max_chars) or 8000))

    if #result_text > soft_max then
        -- 对于文件读取类工具，给出更友好的提示
        local hint = "[Content truncated due to size. Consider using read_lines with specific ranges for large files.]"
        if tool_name == "search_file" or tool_name == "search_files" then
            hint = "[Results truncated. Use more specific patterns or reduce scope.]"
        end

        result_text, was_truncated = smart_truncate(result_text, soft_max, hint)
    end

    -- 添加文件信息提示
    if was_truncated then
        local line_count = count_lines(result_text)
        result_text = result_text .. string.format(
            "\n\n[File has %d characters, showing first ~%d chars. Original: %d chars, ~%d lines]",
            original_len, #result_text, original_len, line_count
        )
    end

    -- 缓存结果
    if cache_key then
        cache_set(cache_key, result_text)
    end

    return result_text, false, original_len
end

-- 合并多个工具结果到context
function M.merge_tool_results(results, existing_context)
    local cfg = ctx_cfg()
    local fragments = {}
    local total_chars = 0
    local max_total = math.max(100, math.floor(tonumber(cfg.tool_context_total_max_chars) or 6000))

    -- 按重要性排序：错误消息优先，然后是文件内容
    local sorted_results = {}
    for _, row in ipairs(results or {}) do
        sorted_results[#sorted_results + 1] = row
    end
    table.sort(sorted_results, function(a, b)
        -- 错误消息优先
        if a.ok ~= b.ok then
            return not a.ok
        end
        -- 较短的优先（更可能是关键信息）
        return #(a.result or "") < #(b.result or "")
    end)

    local existing_texts = {}
    if existing_context and existing_context ~= "" then
        existing_texts[#existing_texts + 1] = existing_context
    end

    for _, row in ipairs(sorted_results) do
        local tool = tostring(row.tool or "")
        local result_text = tostring(row.result or "")
        local ok = row.ok == true

        -- 去重检测
        result_text = dedup_content(result_text, existing_texts)

        local header = ""
        if not ok then
            header = string.format("[ERROR][%s] ", tool)
        else
            header = string.format("[Tool:%s] ", tool)
        end

        local fragment = header .. result_text

        -- 检查总大小
        if total_chars + #fragment > max_total then
            local remaining = max_total - total_chars
            if remaining > 100 then
                fragment = util.utf8_take(fragment, remaining)
                fragment = fragment .. "\n[...additional results truncated...]"
                fragments[#fragments + 1] = fragment
            end
            break
        end

        fragments[#fragments + 1] = fragment
        total_chars = total_chars + #fragment
        existing_texts[#existing_texts + 1] = result_text
    end

    return table.concat(fragments, "\n\n")
end

-- ==================== 历史对话压缩 ====================

-- 为被丢弃的对话生成摘要提示
function M.generate_history_summary_prompt(dropped_pairs)
    if not dropped_pairs or #dropped_pairs <= 0 then
        return nil
    end

    local parts = {}
    parts[#parts + 1] = "[Previous conversations summary:]\n"

    for i, pair in ipairs(dropped_pairs) do
        local user_text = tostring(pair.user or ""):sub(1, 100)
        local assistant_text = tostring(pair.assistant or ""):sub(1, 100)
        parts[#parts + 1] = string.format(
            "- Turn %d: User asked '%s...' → Assistant responded '%s...'",
            pair.turn or i,
            user_text,
            assistant_text
        )
    end

    parts[#parts + 1] = "\n[End of summary. Continue with current context.]"

    return table.concat(parts, "\n")
end

-- ==================== 上下文预算监控 ====================

function M.estimate_tokens(text)
    -- 粗略估算：英文约4字符/token，中文约1.5字符/token
    -- 这里使用保守估计
    text = tostring(text or "")
    local char_count = #text
    local chinese_count = 0
    for _ in text:gmatch("[\228-\233][\128-\191][\128-\191]") do
        chinese_count = chinese_count + 1
    end

    local english_count = char_count - chinese_count * 3
    local estimated = math.floor(chinese_count / 1.5 + english_count / 4 + 0.5)
    return math.max(1, estimated)
end

function M.check_budget(current_tokens, budget)
    local cfg = ctx_cfg()
    if not cfg.enable_budget_monitor then
        return "ok", nil
    end

    budget = math.max(1, math.floor(tonumber(budget) or 12000))
    local warning_threshold = budget * (tonumber(cfg.budget_warning_ratio) or 0.85)

    if current_tokens > budget then
        return "exceeded", string.format(
            "Token budget exceeded: %d > %d. Consider reducing context.",
            current_tokens, budget
        )
    elseif current_tokens > warning_threshold then
        return "warning", string.format(
            "Token budget warning: %d / %d (%.1f%% used)",
            current_tokens, budget, (current_tokens / budget) * 100
        )
    end

    return "ok", nil
end

-- ==================== 文件大小预估 ====================

function M.should_use_streaming_read(file_size, threshold)
    threshold = math.max(1024, math.floor(tonumber(threshold) or 10000))
    return (tonumber(file_size) or 0) > threshold
end

function M.suggest_read_strategy(file_size, max_chars)
    file_size = tonumber(file_size) or 0
    max_chars = math.max(100, math.floor(tonumber(max_chars) or 4000))

    if file_size <= max_chars then
        return "read_file", nil
    end

    local lines_estimate = math.floor(file_size / 40) -- 假设平均每行40字符
    local suggested_lines = math.floor(max_chars / 40)

    return "read_lines", string.format(
        "Large file detected (~%d chars, ~%d lines). Consider using read_lines with start_line=1, max_lines=%d",
        file_size, lines_estimate, suggested_lines
    )
end

-- ==================== 运行时消息优化 ====================

-- 优化runtime_messages，移除过大的tool消息
function M.optimize_runtime_messages(messages, max_tool_msg_chars)
    max_tool_msg_chars = math.max(100, math.floor(tonumber(max_tool_msg_chars) or 4000))
    messages = messages or {}

    local optimized = {}
    local total_chars = 0
    local truncated_count = 0

    for i, msg in ipairs(messages) do
        local copy = util.shallow_copy(msg)

        if msg.role == "tool" then
            local content = tostring(msg.content or "")
            if #content > max_tool_msg_chars then
                local truncated, _ = smart_truncate(
                    content,
                    max_tool_msg_chars,
                    "[Tool output truncated]"
                )
                copy.content = truncated
                truncated_count = truncated_count + 1
            end
        end

        optimized[#optimized + 1] = copy
        total_chars = total_chars + #(copy.content or "")
    end

    return optimized, {
        total_chars = total_chars,
        truncated_count = truncated_count,
        message_count = #optimized,
    }
end

-- ==================== 状态统计 ====================

function M.get_context_stats(state)
    state = state or {}
    local stats = {
        runtime_messages_count = #(((state.messages or {}).runtime_messages) or {}),
        tool_context_chars = #(tostring((((state.context or {}).tool_context) or ""))),
        memory_context_chars = #(tostring((((state.context or {}).memory_context) or ""))),
        conversation_history_count = #(((state.messages or {}).conversation_history) or {}),
        pending_tool_calls = #(((state.agent_loop or {}).pending_tool_calls) or {}),
        tool_executed_total = tonumber((((state.tool_exec or {}).executed_total) or 0)) or 0,
        cache = M.cache_stats(),
    }

    -- 估算总token
    local total_text = ""
    for _, msg in ipairs(((state.messages or {}).runtime_messages) or {}) do
        total_text = total_text .. tostring(msg.content or "")
    end
    stats.estimated_tokens = M.estimate_tokens(total_text)

    return stats
end

return M
