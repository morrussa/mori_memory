local M = {}

local notebook = require("module.notebook")
local topic = require("module.topic")
local config_mem = require("module.config")

M._pending_system_context = ""
M._pending_topic_anchor = nil
M._pending_created_turn = 0

local TOOL_CFG = ((config_mem.settings or {}).keyring or {}).tool_calling or {}

local function trim(s)
    if not s then return "" end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
end

local function to_bool(v, fallback)
    if type(v) == "boolean" then return v end
    if type(v) == "number" then return v ~= 0 end
    if type(v) == "string" then
        local s = v:lower()
        if s == "true" or s == "1" or s == "yes" then return true end
        if s == "false" or s == "0" or s == "no" then return false end
    end
    return fallback == true
end

local function cfg_number(v, fallback, min_v, max_v)
    local n = tonumber(v)
    if not n then n = tonumber(fallback) or 0 end
    if min_v and n < min_v then n = min_v end
    if max_v and n > max_v then n = max_v end
    return n
end

local function split_csv(s)
    local out = {}
    s = trim(s)
    if s == "" then return out end
    for part in s:gmatch("[^,]+") do
        local p = trim(part)
        if p ~= "" then table.insert(out, p) end
    end
    return out
end

local function clamp01(v, fallback)
    local n = tonumber(v)
    if not n then return fallback or 0.7 end
    if n < 0 then return 0 end
    if n > 1 then return 1 end
    return n
end

local function utf8_take(s, max_chars)
    s = tostring(s or "")
    max_chars = tonumber(max_chars) or 0
    if max_chars <= 0 then return s end

    local out = {}
    local count = 0
    for ch in s:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        count = count + 1
        if count > max_chars then break end
        out[count] = ch
    end
    return table.concat(out)
end

local function dedupe_and_clip_types(types, max_types)
    local out = {}
    local seen = {}
    max_types = tonumber(max_types) or 3
    for _, raw in ipairs(types or {}) do
        local t = trim(raw)
        if t ~= "" then
            t = t:gsub("%-", "_")
            t = t:lower()
            if not seen[t] then
                seen[t] = true
                table.insert(out, t)
                if #out >= max_types then break end
            end
        end
    end
    return out
end

local function deterministic_rerank(results, query)
    local ranked = {}
    local now = os.time()
    local q = trim(query):lower()

    for _, r in ipairs(results or {}) do
        local base = tonumber(r.score) or 0
        local conf = clamp01(r.confidence, 0.7)
        local updated_at = tonumber(r.updated_at) or 0
        local recency = 0
        if updated_at > 0 then
            local age_hours = math.max(0, (now - updated_at) / 3600)
            recency = 1 / (1 + age_hours / 96)
        end

        local lexical = 0
        if q ~= "" then
            local entity_l = tostring(r.entity or ""):lower()
            local value_l = tostring(r.value or ""):lower()
            if entity_l:find(q, 1, true) then lexical = lexical + 0.45 end
            if value_l:find(q, 1, true) then lexical = lexical + 0.70 end
        end

        local final = base + conf * 0.40 + recency * 0.25 + lexical
        table.insert(ranked, {
            rec = r,
            final = final,
            updated_at = updated_at,
            id = tonumber(r.id) or 0,
        })
    end

    table.sort(ranked, function(a, b)
        if a.final == b.final then
            if a.updated_at == b.updated_at then
                return a.id < b.id
            end
            return a.updated_at > b.updated_at
        end
        return a.final > b.final
    end)

    local out = {}
    for _, item in ipairs(ranked) do
        table.insert(out, item.rec)
    end
    return out
end

local function clear_pending_system_context()
    M._pending_system_context = ""
    M._pending_topic_anchor = nil
    M._pending_created_turn = 0
end

local function get_tool_policy()
    return {
        upsert_min_confidence = cfg_number(TOOL_CFG.upsert_min_confidence, 0.82, 0, 1),
        upsert_max_per_turn = cfg_number(TOOL_CFG.upsert_max_per_turn, 1, 0),
        query_max_per_turn = cfg_number(TOOL_CFG.query_max_per_turn, 2, 0),
        delete_enabled = to_bool(TOOL_CFG.delete_enabled, false),
        query_max_types = cfg_number(TOOL_CFG.query_max_types, 3, 1),
        query_fetch_limit = cfg_number(TOOL_CFG.query_fetch_limit, 18, 1),
        query_inject_top = cfg_number(TOOL_CFG.query_inject_top, 3, 1),
        query_inject_max_chars = cfg_number(TOOL_CFG.query_inject_max_chars, 800, 200),
        tool_pass_temperature = cfg_number(TOOL_CFG.tool_pass_temperature, 0.15, 0, 1),
        tool_pass_max_tokens = cfg_number(TOOL_CFG.tool_pass_max_tokens, 128, 32),
        tool_pass_seed = cfg_number(TOOL_CFG.tool_pass_seed, 42),
    }
end

local function apply_upsert_record(call, current_turn, policy, state)
    if state.upsert_count >= policy.upsert_max_per_turn then
        return false, string.format("upsert_record 超出预算（max=%d）", policy.upsert_max_per_turn)
    end

    local rec_type = trim(call.type)
    local entity = trim(call.entity)
    local value = trim(call.value)
    if value == "" then value = trim(call.string) end
    if rec_type == "" then return false, "upsert_record 缺少 type" end
    if entity == "" then return false, "upsert_record 缺少 entity" end
    if value == "" then return false, "upsert_record 缺少 value" end

    local confidence = clamp01(call.confidence, 0.75)
    if confidence < policy.upsert_min_confidence then
        return false, string.format(
            "upsert_record 置信度 %.2f 低于阈值 %.2f",
            confidence,
            policy.upsert_min_confidence
        )
    end

    local id, op = notebook.upsert_record(rec_type, entity, value, {
        turn = current_turn,
        confidence = confidence,
        evidence = trim(call.evidence),
        source = "tool_call_upsert_record",
    })
    if not id then
        return false, "upsert_record 写入失败: " .. tostring(op)
    end
    state.upsert_count = state.upsert_count + 1
    return true, string.format("upsert_record %s (id=%d)", tostring(op), id)
end

local function apply_delete_record(call, current_turn, policy)
    if not policy.delete_enabled then
        return false, "delete_record 已禁用"
    end

    local rec_type = trim(call.type)
    local entity = trim(call.entity)
    if rec_type == "" then return false, "delete_record 缺少 type" end
    if entity == "" then return false, "delete_record 缺少 entity" end

    local id, op = notebook.delete_record(rec_type, entity, {
        turn = current_turn,
        evidence = trim(call.evidence),
        source = "tool_call_delete_record",
    })
    if not id then
        return false, "delete_record 失败: " .. tostring(op)
    end
    return true, string.format("delete_record %s (id=%d)", tostring(op), id)
end

local function apply_query_record(call, current_turn, policy, state)
    if state.query_count >= policy.query_max_per_turn then
        return false, string.format("query_record 超出预算（max=%d）", policy.query_max_per_turn)
    end

    local query = trim(call.query)
    if query == "" then query = trim(call.string) end
    if query == "" then query = trim(call.value) end
    if query == "" then return false, "query_record 缺少 query/string/value" end

    local types = split_csv(call.types or "")
    if #types == 0 and trim(call.type) ~= "" then
        table.insert(types, trim(call.type))
    end
    types = dedupe_and_clip_types(types, policy.query_max_types)

    local results = notebook.query_records(query, {
        types = types,
        limit = policy.query_fetch_limit,
        mark_hit = true,
    })
    results = deterministic_rerank(results, query)
    if #results > policy.query_inject_top then
        for i = #results, policy.query_inject_top + 1, -1 do
            table.remove(results, i)
        end
    end

    state.query_count = state.query_count + 1
    print(string.format("[ToolRegistry] query_record 命中 %d 条", #results))
    if #results > 0 then
        print(notebook.render_results(results))
        local block = {}
        table.insert(block, "【Tool:query_record 上一轮检索结果】")
        table.insert(block, "query: " .. query)
        table.insert(block, "请把以下记录当作可引用事实，若不相关可忽略：")
        for i, r in ipairs(results) do
            table.insert(block, string.format(
                "%d) [%s] entity=%s | value=%s | conf=%.2f",
                i,
                r.type or "identity",
                r.entity or "",
                r.value or "",
                r.confidence or 0
            ))
        end
        local payload_raw = table.concat(block, "\n")
        local payload = utf8_take(payload_raw, policy.query_inject_max_chars)
        if payload ~= payload_raw then
            payload = payload .. "\n...(truncated)"
        end
        local anchor = topic.get_topic_anchor and topic.get_topic_anchor(current_turn) or nil
        if M._pending_system_context ~= "" and M._pending_topic_anchor == anchor then
            M._pending_system_context = M._pending_system_context .. "\n\n" .. payload
        else
            M._pending_system_context = payload
            M._pending_topic_anchor = anchor
            M._pending_created_turn = tonumber(current_turn) or 0
        end
    end
    return true, string.format("query_record ok (%d hits)", #results)
end

function M.get_policy()
    return get_tool_policy()
end

function M.execute_calls(calls, exec_ctx)
    exec_ctx = exec_ctx or {}
    local policy = exec_ctx.policy or get_tool_policy()
    local current_turn = tonumber(exec_ctx.current_turn) or 0
    local read_only = exec_ctx.read_only == true

    local result = {
        executed = 0,
        skipped = 0,
        failed = 0,
        logs = {},
    }

    if type(calls) ~= "table" or #calls == 0 then
        return result
    end

    if read_only then
        result.skipped = #calls
        table.insert(result.logs, "read_only 模式：跳过工具写入")
        print("[ToolRegistry] read_only 模式：跳过工具写入")
        return result
    end

    local state = {
        upsert_count = 0,
        query_count = 0,
    }

    for _, call in ipairs(calls) do
        local ok, msg
        if call.act == "upsert_record" then
            ok, msg = apply_upsert_record(call, current_turn, policy, state)
        elseif call.act == "delete_record" then
            ok, msg = apply_delete_record(call, current_turn, policy)
        elseif call.act == "query_record" then
            ok, msg = apply_query_record(call, current_turn, policy, state)
        else
            ok, msg = false, "跳过未知 act: " .. tostring(call.act)
        end

        if ok then
            result.executed = result.executed + 1
        else
            result.failed = result.failed + 1
        end
        result.logs[#result.logs + 1] = msg
        print(string.format("[ToolRegistry] %s | %s", ok and "OK" or "FAIL", msg))
    end

    return result
end

function M.get_pending_system_context(current_turn)
    if M._pending_system_context == "" then return "" end
    local cur_anchor = topic.get_topic_anchor and topic.get_topic_anchor(current_turn) or nil
    if M._pending_topic_anchor and cur_anchor and M._pending_topic_anchor ~= cur_anchor then
        clear_pending_system_context()
        return ""
    end
    return M._pending_system_context
end

function M.consume_pending_system_context_for_turn(current_turn)
    local ctx = M.get_pending_system_context(current_turn)
    if ctx == "" then return "" end
    clear_pending_system_context()
    return ctx
end

function M.get_long_term_plan_bom()
    return notebook.build_long_term_plan_bom and notebook.build_long_term_plan_bom() or ""
end

return M
