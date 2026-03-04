local M = {}

local config = require("module.config")

local function trim(s)
    if s == nil then
        return ""
    end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function to_int(v, fallback, min_v)
    local n = tonumber(v)
    if not n then
        n = tonumber(fallback) or 0
    end
    n = math.floor(n)
    if min_v and n < min_v then
        n = min_v
    end
    return n
end

local function to_bool(v, fallback)
    if type(v) == "boolean" then
        return v
    end
    if type(v) == "number" then
        return v ~= 0
    end
    if type(v) == "string" then
        local s = v:lower()
        if s == "true" or s == "1" or s == "yes" then
            return true
        end
        if s == "false" or s == "0" or s == "no" then
            return false
        end
    end
    return fallback == true
end

local function utf8_take(s, max_chars)
    s = tostring(s or "")
    max_chars = tonumber(max_chars) or 0
    if max_chars <= 0 then
        return s
    end

    local out = {}
    local count = 0
    for ch in s:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        count = count + 1
        if count > max_chars then
            break
        end
        out[count] = ch
    end
    return table.concat(out)
end

local function utf8_len(s)
    local text = tostring(s or "")
    local _, n = text:gsub("[^\128-\193]", "")
    return n
end

local function append_unique(arr, value)
    for _, v in ipairs(arr) do
        if v == value then
            return
        end
    end
    arr[#arr + 1] = value
end

local function extract_history_pairs(conversation_history)
    local out = {}
    if type(conversation_history) ~= "table" then
        return out
    end

    local idx = 2
    local turn_no = 0
    while idx <= #conversation_history do
        local user_msg = conversation_history[idx]
        local assistant_msg = conversation_history[idx + 1]

        local pair = nil
        if type(user_msg) == "table" and tostring(user_msg.role or "") == "user" then
            turn_no = turn_no + 1
            pair = {
                turn = turn_no,
                user = tostring(user_msg.content or ""),
                assistant = "",
            }
            if type(assistant_msg) == "table" and tostring(assistant_msg.role or "") == "assistant" then
                pair.assistant = tostring(assistant_msg.content or "")
                idx = idx + 2
            else
                idx = idx + 1
            end
        else
            idx = idx + 1
        end

        if pair then
            out[#out + 1] = pair
        end
    end
    return out
end

local function reverse_pairs(src)
    local out = {}
    for i = #src, 1, -1 do
        out[#out + 1] = src[i]
    end
    return out
end

local function copy_pairs(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function sort_pairs_by_turn(pairs)
    table.sort(pairs, function(a, b)
        local ta = tonumber((a or {}).turn) or 0
        local tb = tonumber((b or {}).turn) or 0
        if ta == tb then
            return tostring((a or {}).user or "") < tostring((b or {}).user or "")
        end
        return ta < tb
    end)
end

local function compact_inline_text(s, max_chars)
    local text = trim(s)
    text = text:gsub("[%c]+", " ")
    text = text:gsub("%s+", " ")
    text = trim(text)
    if text == "" then
        return ""
    end
    local clipped = utf8_take(text, max_chars)
    if clipped ~= text then
        return trim(clipped) .. "..."
    end
    return clipped
end

local function build_history_compression_block(pairs, cfg)
    if type(pairs) ~= "table" or #pairs == 0 then
        return ""
    end

    local max_pairs = to_int(cfg.max_pairs, 24, 1)
    local user_chars = to_int(cfg.user_chars, 64, 8)
    local assistant_chars = to_int(cfg.assistant_chars, 96, 8)
    local start_idx = math.max(1, #pairs - max_pairs + 1)

    local lines = {
        "【历史自动压缩】",
        "以下为早期对话压缩视图（用于保持计划和任务连续性）：",
    }

    for i = start_idx, #pairs do
        local pair = pairs[i] or {}
        local turn = tonumber(pair.turn) or i
        local user_text = compact_inline_text(pair.user or "", user_chars)
        local assistant_text = compact_inline_text(pair.assistant or "", assistant_chars)
        if user_text ~= "" or assistant_text ~= "" then
            lines[#lines + 1] = string.format(
                "T%d U:%s | A:%s",
                turn,
                user_text,
                assistant_text
            )
        end
    end

    return table.concat(lines, "\n")
end

local function count_messages_tokens(messages)
    local n = py_pipeline:count_chat_tokens(messages)
    return to_int(n, 0, 0)
end

local function compose_system_prompt(system_prompt, blocks)
    local out = { tostring(system_prompt or "") }

    -- 为兼容要求“system 只能出现在第一条”的模板，将所有系统注入块并入首条 system。
    -- 注入顺序保持与旧链路一致：plan -> tool -> memory
    if blocks.plan_bom and blocks.plan_bom ~= "" then
        out[#out + 1] = tostring(blocks.plan_bom)
    end
    if blocks.tool_context and blocks.tool_context ~= "" then
        out[#out + 1] = tostring(blocks.tool_context)
    end
    if blocks.memory_context and blocks.memory_context ~= "" then
        out[#out + 1] = tostring(blocks.memory_context)
    end
    if blocks.history_summary and blocks.history_summary ~= "" then
        out[#out + 1] = tostring(blocks.history_summary)
    end

    return table.concat(out, "\n\n")
end

local function build_messages(system_prompt, user_input, history_pairs, blocks)
    local merged_system_prompt = compose_system_prompt(system_prompt, blocks)
    local msgs = {
        { role = "system", content = merged_system_prompt },
    }

    for _, pair in ipairs(history_pairs or {}) do
        if trim(pair.user) ~= "" then
            msgs[#msgs + 1] = { role = "user", content = tostring(pair.user) }
        end
        if trim(pair.assistant) ~= "" then
            msgs[#msgs + 1] = { role = "assistant", content = tostring(pair.assistant) }
        end
    end

    msgs[#msgs + 1] = { role = "user", content = user_input }
    return msgs
end

function M.build_messages(opts)
    opts = opts or {}
    local agent_cfg = (config.settings or {}).agent or {}

    local budget = to_int(
        opts.input_token_budget,
        agent_cfg.input_token_budget or 12000,
        128
    )

    local user_input = trim(opts.user_input)
    if user_input == "" then
        error("[ContextWindow] user_input is empty")
    end

    local system_prompt = trim(opts.system_prompt)
    local history_src = opts.conversation_history
    if system_prompt == "" and type(history_src) == "table" and type(history_src[1]) == "table" then
        if tostring(history_src[1].role or "") == "system" then
            system_prompt = tostring(history_src[1].content or "")
        end
    end
    if system_prompt == "" then
        error("[ContextWindow] system prompt missing")
    end

    local blocks = {
        plan_bom = trim(opts.plan_bom),
        tool_context = trim(opts.tool_context),
        memory_context = trim(opts.memory_context),
        history_summary = trim(opts.history_summary),
    }

    local plan_pinned = to_bool(
        opts.plan_bom_pinned,
        to_bool(agent_cfg.plan_bom_pinned, true)
    )
    local history_auto_compress = to_bool(
        opts.history_auto_compress,
        to_bool(agent_cfg.history_auto_compress, true)
    )
    local history_compress_min_dropped = to_int(
        opts.history_auto_compress_min_dropped_pairs,
        agent_cfg.history_auto_compress_min_dropped_pairs or 1,
        1
    )
    local history_compress_max_pairs = to_int(
        opts.history_auto_compress_max_pairs,
        agent_cfg.history_auto_compress_max_pairs or 24,
        1
    )
    local history_compress_user_chars = to_int(
        opts.history_auto_compress_user_chars,
        agent_cfg.history_auto_compress_user_chars or 64,
        8
    )
    local history_compress_assistant_chars = to_int(
        opts.history_auto_compress_assistant_chars,
        agent_cfg.history_auto_compress_assistant_chars or 96,
        8
    )
    local history_compress_max_chars = to_int(
        opts.history_auto_compress_max_chars,
        agent_cfg.history_auto_compress_max_chars or 1400,
        100
    )
    local history_compress_min_chars = to_int(
        opts.history_auto_compress_min_chars,
        agent_cfg.history_auto_compress_min_chars or 220,
        80
    )
    local plan_compact_min_chars = to_int(
        opts.plan_bom_compact_min_chars,
        agent_cfg.plan_bom_compact_min_chars or 120,
        64
    )

    local dropped_blocks = {}
    local function drop_block(name)
        if blocks[name] and blocks[name] ~= "" then
            blocks[name] = ""
            append_unique(dropped_blocks, name)
            return true
        end
        return false
    end

    local base_messages = build_messages(system_prompt, user_input, {}, blocks)
    local total_tokens = count_messages_tokens(base_messages)

    if total_tokens > budget then
        -- 固定顺序：memory -> tool -> (plan 可选)
        local order = { "memory_context", "tool_context" }
        if not plan_pinned then
            order[#order + 1] = "plan_bom"
        end
        for _, name in ipairs(order) do
            if total_tokens <= budget then
                break
            end
            local changed = drop_block(name)
            if changed then
                base_messages = build_messages(system_prompt, user_input, {}, blocks)
                total_tokens = count_messages_tokens(base_messages)
            end
        end
    end

    -- 若 plan 固定保留且仍超预算，先尝试“压缩 plan 文本”而不是直接丢弃。
    if total_tokens > budget and plan_pinned and blocks.plan_bom ~= "" then
        local raw_plan = blocks.plan_bom
        local plan_chars = utf8_len(raw_plan)
        local cap = plan_chars
        local min_cap = math.min(plan_compact_min_chars, cap)
        local compacted = false

        while cap >= min_cap and total_tokens > budget do
            local clipped = utf8_take(raw_plan, cap)
            if clipped ~= raw_plan then
                clipped = trim(clipped) .. "\n...(plan auto-compressed)"
                compacted = true
            end
            blocks.plan_bom = clipped
            base_messages = build_messages(system_prompt, user_input, {}, blocks)
            total_tokens = count_messages_tokens(base_messages)
            if total_tokens <= budget then
                if compacted then
                    append_unique(dropped_blocks, "plan_bom_compacted")
                end
                break
            end
            cap = math.floor(cap * 0.72)
        end
    end

    if total_tokens > budget then
        error(string.format(
            "[ContextWindow] required context exceeds budget: total=%d budget=%d",
            total_tokens,
            budget
        ))
    end

    local all_pairs = extract_history_pairs(history_src)
    local kept_pairs = {}
    local kept_history_pairs = 0
    local dropped_history_pairs = 0
    local dropped_pairs = {}
    local final_messages = base_messages

    for i = #all_pairs, 1, -1 do
        local pair = all_pairs[i]
        local candidate_pairs = { pair }
        for k = 1, #kept_pairs do
            candidate_pairs[#candidate_pairs + 1] = kept_pairs[k]
        end

        local candidate_messages = build_messages(system_prompt, user_input, candidate_pairs, blocks)
        local candidate_tokens = count_messages_tokens(candidate_messages)
        if candidate_tokens <= budget then
            kept_pairs = candidate_pairs
            kept_history_pairs = kept_history_pairs + 1
            final_messages = candidate_messages
            total_tokens = candidate_tokens
        else
            dropped_history_pairs = dropped_history_pairs + 1
            dropped_pairs[#dropped_pairs + 1] = pair
        end
    end

    local history_summary_used = false
    local history_summary_chars = 0
    local compressed_history_pairs = 0

    if history_auto_compress and #dropped_pairs >= history_compress_min_dropped then
        local summary_pool = reverse_pairs(dropped_pairs)
        local summary_kept_pairs = copy_pairs(kept_pairs)
        local rebalance_removed = 0
        local fitted = false

        while true do
            sort_pairs_by_turn(summary_pool)
            local raw_summary = build_history_compression_block(summary_pool, {
                max_pairs = history_compress_max_pairs,
                user_chars = history_compress_user_chars,
                assistant_chars = history_compress_assistant_chars,
            })
            if raw_summary == "" then
                break
            end

            local cap = math.min(history_compress_max_chars, utf8_len(raw_summary))
            local min_cap = math.min(history_compress_min_chars, cap)

            while cap >= min_cap do
                local clipped = utf8_take(raw_summary, cap)
                if clipped ~= raw_summary then
                    clipped = trim(clipped) .. "\n...(history auto-compressed)"
                end
                blocks.history_summary = clipped
                local candidate_messages = build_messages(system_prompt, user_input, summary_kept_pairs, blocks)
                local candidate_tokens = count_messages_tokens(candidate_messages)
                if candidate_tokens <= budget then
                    final_messages = candidate_messages
                    total_tokens = candidate_tokens
                    history_summary_used = true
                    history_summary_chars = utf8_len(clipped)
                    compressed_history_pairs = #summary_pool
                    append_unique(dropped_blocks, "history_pairs_compressed")
                    fitted = true
                    break
                end
                cap = math.floor(cap * 0.72)
            end

            if fitted then
                break
            end

            if #summary_kept_pairs <= 0 then
                break
            end

            local removed_pair = table.remove(summary_kept_pairs, 1)
            if removed_pair then
                summary_pool[#summary_pool + 1] = removed_pair
                rebalance_removed = rebalance_removed + 1
            end
        end

        if fitted and rebalance_removed > 0 then
            kept_pairs = summary_kept_pairs
            kept_history_pairs = math.max(0, kept_history_pairs - rebalance_removed)
            dropped_history_pairs = dropped_history_pairs + rebalance_removed
            append_unique(dropped_blocks, "history_pairs_rebalanced")
        end

        if not fitted then
            blocks.history_summary = ""
        end
    end

    if total_tokens > budget then
        -- 自动压缩失败时保持 fail-close，避免静默超预算。
        error(string.format(
            "[ContextWindow] final context exceeds budget: total=%d budget=%d",
            total_tokens,
            budget
        ))
    end

    local meta = {
        total_tokens = total_tokens,
        kept_history_pairs = kept_history_pairs,
        dropped_history_pairs = dropped_history_pairs,
        dropped_blocks = dropped_blocks,
        budget = budget,
        history_summary_used = history_summary_used,
        history_summary_chars = history_summary_chars,
        compressed_history_pairs = compressed_history_pairs,
    }

    return final_messages, meta
end

return M
