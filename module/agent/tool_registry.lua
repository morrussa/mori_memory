local M = {}

local notebook = require("module.agent.notebook")
local topic = require("module.memory.topic")
local config_mem = require("module.config")

M._pending_system_context = ""
M._pending_topic_anchor = nil
M._pending_created_turn = 0
M._last_context_signature = ""
M._last_context_turn = 0
M._turn_budget_state = {
    turn = 0,
    upsert_count = 0,
    query_count = 0,
    file_list_count = 0,
    file_read_count = 0,
    file_read_lines_count = 0,
    file_search_count = 0,
    file_multi_search_count = 0,
}

local TOOL_CFG = ((config_mem.settings or {}).keyring or {}).tool_calling or {}
local CORE_TOOL_ACTS = {
    query_record = true,
    upsert_record = true,
    delete_record = true,
    list_agent_files = true,
    read_agent_file = true,
    read_agent_file_lines = true,
    search_agent_file = true,
    search_agent_files = true,
}

M._external_tools = {
    cfg_signature = "",
    enabled = false,
    schemas = {},
    names = {},
    name_set = {},
}

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

local function lua_escape_str(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\")
    s = s:gsub('"', '\\"')
    s = s:gsub("\r", "\\r")
    s = s:gsub("\n", "\\n")
    return s
end

local function to_lua_literal(v)
    if type(v) == "boolean" then
        return v and "true" or "false"
    end
    if type(v) == "number" then
        if v ~= v or v == math.huge or v == -math.huge then
            return "nil"
        end
        return tostring(v)
    end
    if v == nil then
        return "nil"
    end
    return '"' .. lua_escape_str(v) .. '"'
end

local function parse_lua_table_literal(raw)
    raw = trim(raw)
    if raw == "" or not raw:match("^%b{}$") then
        return nil
    end

    local chunk, load_err = load("return " .. raw, "tool_args", "t", {})
    if not chunk then
        return nil, load_err
    end

    local ok, parsed = pcall(chunk)
    if not ok or type(parsed) ~= "table" then
        return nil, parsed
    end
    return parsed
end

local function is_array_like_table(tbl)
    if type(tbl) ~= "table" then
        return false, 0
    end
    local count = 0
    local max_idx = 0
    for k, _ in pairs(tbl) do
        if type(k) ~= "number" or k < 1 or k % 1 ~= 0 then
            return false, 0
        end
        count = count + 1
        if k > max_idx then
            max_idx = k
        end
    end
    if max_idx ~= count then
        return false, 0
    end
    return true, max_idx
end

local function is_lua_identifier(s)
    return type(s) == "string" and s:match("^[A-Za-z_][A-Za-z0-9_]*$") ~= nil
end

local function encode_lua_value(v, depth)
    depth = tonumber(depth) or 0
    if depth > 24 then
        return "nil"
    end

    local vt = type(v)
    if vt == "table" then
        local is_arr, arr_len = is_array_like_table(v)
        if is_arr then
            local parts = {}
            for i = 1, arr_len do
                parts[#parts + 1] = encode_lua_value(v[i], depth + 1)
            end
            return "{" .. table.concat(parts, ",") .. "}"
        end

        local entries = {}
        for k, val in pairs(v) do
            entries[#entries + 1] = {
                key = k,
                key_type = type(k),
                key_text = tostring(k),
                value = val,
            }
        end
        table.sort(entries, function(a, b)
            if a.key_type == b.key_type then
                return a.key_text < b.key_text
            end
            if a.key_type == "number" then
                return true
            end
            if b.key_type == "number" then
                return false
            end
            return a.key_type < b.key_type
        end)

        local parts = {}
        for _, item in ipairs(entries) do
            local key = item.key
            local key_expr = ""
            if type(key) == "string" and is_lua_identifier(key) then
                key_expr = key
            elseif type(key) == "string" then
                key_expr = '["' .. lua_escape_str(key) .. '"]'
            elseif type(key) == "number" then
                key_expr = "[" .. tostring(key) .. "]"
            elseif type(key) == "boolean" then
                key_expr = key and "[true]" or "[false]"
            else
                key_expr = '["' .. lua_escape_str(tostring(key)) .. '"]'
            end
            parts[#parts + 1] = key_expr .. "=" .. encode_lua_value(item.value, depth + 1)
        end
        return "{" .. table.concat(parts, ",") .. "}"
    end

    if vt == "string" or vt == "number" or vt == "boolean" or v == nil then
        return to_lua_literal(v)
    end
    return '"' .. lua_escape_str(tostring(v)) .. '"'
end

local function normalize_tool_entry_names(raw, out)
    out = out or {}
    if type(raw) == "string" then
        for _, x in ipairs(split_csv(raw)) do
            out[#out + 1] = x
        end
        return out
    end
    if type(raw) ~= "table" then
        return out
    end
    for _, item in ipairs(raw) do
        if type(item) == "string" then
            local n = trim(item)
            if n ~= "" then
                out[#out + 1] = n
            end
        elseif type(item) == "table" then
            local n = trim(item.name or (((item["function"] or {}).name) or ""))
            if n ~= "" then
                out[#out + 1] = n
            end
        end
    end
    return out
end

local function get_external_cfg()
    local keyring = ((config_mem.settings or {}).keyring or {})
    return keyring.external_tools or {}
end

local function get_external_tool_entries()
    local ext_cfg = get_external_cfg()
    local merged = {}
    normalize_tool_entry_names(ext_cfg.tools, merged)
    normalize_tool_entry_names(ext_cfg.names, merged)
    if #merged == 0 then
        normalize_tool_entry_names(ext_cfg.allowlist, merged)
    end
    return merged
end

local function external_cfg_signature(ext_cfg, entries)
    local parts = {
        tostring(to_bool(ext_cfg.enabled, false) and "1" or "0"),
        tostring(to_bool(ext_cfg.include_memory_tools, true) and "1" or "0"),
    }
    local copy = {}
    for _, x in ipairs(entries or {}) do
        copy[#copy + 1] = tostring(x)
    end
    table.sort(copy)
    parts[#parts + 1] = table.concat(copy, ",")
    return table.concat(parts, "|")
end

local function refresh_external_tools()
    local ext_cfg = get_external_cfg()
    local enabled = to_bool(ext_cfg.enabled, false)
    local entries = get_external_tool_entries()
    local sig = external_cfg_signature(ext_cfg, entries)

    if M._external_tools.cfg_signature == sig then
        return M._external_tools
    end

    local state = {
        cfg_signature = sig,
        enabled = false,
        schemas = {},
        names = {},
        name_set = {},
    }

    if (not enabled) or type(py_pipeline) ~= "table" or py_pipeline.get_qwen_tool_schemas == nil then
        M._external_tools = state
        return state
    end

    local ok_schemas, schemas_or_err = pcall(function()
        if #entries > 0 then
            return py_pipeline:get_qwen_tool_schemas(entries)
        end
        return py_pipeline:get_qwen_tool_schemas(nil)
    end)
    if not ok_schemas or type(schemas_or_err) ~= "table" then
        print(string.format(
            "[ToolRegistry][WARN] 外部工具加载失败: %s",
            tostring(schemas_or_err)
        ))
        M._external_tools = state
        return state
    end

    for _, item in ipairs(schemas_or_err) do
        if type(item) == "table" and type(item["function"]) == "table" then
            local fn = item["function"]
            local name = trim(fn.name or "")
            if name ~= "" and (not CORE_TOOL_ACTS[name]) then
                state.schemas[#state.schemas + 1] = item
                if not state.name_set[name] then
                    state.name_set[name] = true
                    state.names[#state.names + 1] = name
                end
            end
        end
    end

    state.enabled = (#state.schemas > 0)
    if state.enabled then
        table.sort(state.names)
        print(string.format(
            "[ToolRegistry] 外部工具已启用，共 %d 个: %s",
            #state.names,
            table.concat(state.names, ",")
        ))
    end

    M._external_tools = state
    return state
end

local function push_pending_context(current_turn, payload)
    payload = trim(payload or "")
    if payload == "" then
        return
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

local function build_call_arguments_lua(call)
    local c = call or {}
    local raw = trim(c.arguments or "")
    if raw ~= "" then
        local parsed_lua = parse_lua_table_literal(raw)
        if type(parsed_lua) == "table" then
            return encode_lua_value(parsed_lua, 0)
        end
    end

    local obj = {}
    if trim(c.query) ~= "" then obj.query = c.query end
    if trim(c.string) ~= "" then obj.string = c.string end
    if trim(c.path) ~= "" then obj.path = c.path end
    if trim(c.file) ~= "" then obj.file = c.file end
    if trim(c.prefix) ~= "" then obj.prefix = c.prefix end
    if trim(c.pattern) ~= "" then obj.pattern = c.pattern end
    if trim(c.value) ~= "" then obj.value = c.value end
    if trim(c.type) ~= "" then obj.type = c.type end
    if trim(c.types) ~= "" then obj.types = c.types end
    if trim(c.entity) ~= "" then obj.entity = c.entity end
    if trim(c.evidence) ~= "" then obj.evidence = c.evidence end
    if trim(c.start_char) ~= "" then obj.start_char = tonumber(c.start_char) or c.start_char end
    if trim(c.offset_char) ~= "" then obj.offset_char = tonumber(c.offset_char) or c.offset_char end
    if trim(c.max_chars) ~= "" then obj.max_chars = tonumber(c.max_chars) or c.max_chars end
    if trim(c.start_line) ~= "" then obj.start_line = tonumber(c.start_line) or c.start_line end
    if trim(c.end_line) ~= "" then obj.end_line = tonumber(c.end_line) or c.end_line end
    if trim(c.max_lines) ~= "" then obj.max_lines = tonumber(c.max_lines) or c.max_lines end
    if trim(c.max_hits) ~= "" then obj.max_hits = tonumber(c.max_hits) or c.max_hits end
    if trim(c.max_files) ~= "" then obj.max_files = tonumber(c.max_files) or c.max_files end
    if trim(c.per_file_hits) ~= "" then obj.per_file_hits = tonumber(c.per_file_hits) or c.per_file_hits end
    if trim(c.context_lines) ~= "" then obj.context_lines = tonumber(c.context_lines) or c.context_lines end
    if trim(c.regex) ~= "" then obj.regex = c.regex end
    if trim(c.case_sensitive) ~= "" then obj.case_sensitive = c.case_sensitive end
    if trim(c.limit) ~= "" then obj.limit = tonumber(c.limit) or c.limit end
    if trim(c.namespace) ~= "" then obj.namespace = c.namespace end
    if trim(c.key) ~= "" then obj.key = c.key end
    if c.confidence ~= nil and tostring(c.confidence) ~= "" then
        obj.confidence = tonumber(c.confidence) or c.confidence
    end

    return encode_lua_value(obj, 0)
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

local function normalize_query_payload(call, policy)
    local query = trim(call.query)
    if query == "" then query = trim(call.string) end
    if query == "" then query = trim(call.value) end

    local types = split_csv(call.types or "")
    if #types == 0 and trim(call.type) ~= "" then
        table.insert(types, trim(call.type))
    end
    types = dedupe_and_clip_types(types, policy.query_max_types)

    local sorted_types = {}
    for i, t in ipairs(types) do
        sorted_types[i] = t
    end
    table.sort(sorted_types)

    local query_key = ""
    if query ~= "" then
        query_key = query:lower() .. "\x1F" .. table.concat(sorted_types, ",")
    end

    return query, types, query_key
end

local function build_query_result_signature(query_key, results)
    if query_key == "" or type(results) ~= "table" or #results == 0 then
        return ""
    end
    local parts = { query_key }
    for _, r in ipairs(results) do
        local rid = tonumber(r.id)
        if rid and rid > 0 then
            parts[#parts + 1] = tostring(rid)
        else
            parts[#parts + 1] = string.format(
                "%s|%s|%s",
                tostring(r.type or ""),
                tostring(r.entity or ""),
                tostring(r.value or "")
            )
        end
    end
    return table.concat(parts, "|")
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

local function get_turn_budget_state(current_turn)
    local turn_id = tonumber(current_turn) or 0
    local state = M._turn_budget_state
    if type(state) ~= "table" or (tonumber(state.turn) or -1) ~= turn_id then
        state = {
            turn = turn_id,
            upsert_count = 0,
            query_count = 0,
            file_list_count = 0,
            file_read_count = 0,
            file_read_lines_count = 0,
            file_search_count = 0,
            file_multi_search_count = 0,
        }
        M._turn_budget_state = state
    end
    state.upsert_count = tonumber(state.upsert_count) or 0
    state.query_count = tonumber(state.query_count) or 0
    state.file_list_count = tonumber(state.file_list_count) or 0
    state.file_read_count = tonumber(state.file_read_count) or 0
    state.file_read_lines_count = tonumber(state.file_read_lines_count) or 0
    state.file_search_count = tonumber(state.file_search_count) or 0
    state.file_multi_search_count = tonumber(state.file_multi_search_count) or 0
    return state
end

local function get_tool_policy()
    local ext_cfg = get_external_cfg()
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
        parallel_execute_enabled = to_bool(TOOL_CFG.parallel_execute_enabled, true),
        parallel_query_batch_size = cfg_number(TOOL_CFG.parallel_query_batch_size, 4, 1),
        retry_transient_max = cfg_number(TOOL_CFG.retry_transient_max, 1, 0),
        retry_unknown_max = cfg_number(TOOL_CFG.retry_unknown_max, 0, 0),
        retry_validation_max = cfg_number(TOOL_CFG.retry_validation_max, 0, 0),
        retry_budget_max = cfg_number(TOOL_CFG.retry_budget_max, 0, 0),
        retry_total_cap = cfg_number(TOOL_CFG.retry_total_cap, 2, 0),
        agent_file_list_max_per_turn = cfg_number(TOOL_CFG.agent_file_list_max_per_turn, 2, 0),
        agent_file_list_default_limit = cfg_number(TOOL_CFG.agent_file_list_default_limit, 12, 1),
        agent_file_list_hard_limit = cfg_number(TOOL_CFG.agent_file_list_hard_limit, 64, 1),
        agent_file_read_max_per_turn = cfg_number(TOOL_CFG.agent_file_read_max_per_turn, 4, 0),
        agent_file_read_default_max_chars = cfg_number(TOOL_CFG.agent_file_read_default_max_chars, 3000, 128),
        agent_file_read_hard_max_chars = cfg_number(TOOL_CFG.agent_file_read_hard_max_chars, 12000, 256),
        agent_file_read_lines_max_per_turn = cfg_number(TOOL_CFG.agent_file_read_lines_max_per_turn, 4, 0),
        agent_file_read_lines_default_max_lines = cfg_number(TOOL_CFG.agent_file_read_lines_default_max_lines, 220, 16),
        agent_file_read_lines_hard_max_lines = cfg_number(TOOL_CFG.agent_file_read_lines_hard_max_lines, 1200, 32),
        agent_file_search_max_per_turn = cfg_number(TOOL_CFG.agent_file_search_max_per_turn, 4, 0),
        agent_file_search_default_max_hits = cfg_number(TOOL_CFG.agent_file_search_default_max_hits, 20, 1),
        agent_file_search_hard_max_hits = cfg_number(TOOL_CFG.agent_file_search_hard_max_hits, 200, 4),
        agent_file_multi_search_max_per_turn = cfg_number(TOOL_CFG.agent_file_multi_search_max_per_turn, 3, 0),
        agent_file_multi_search_default_max_hits = cfg_number(TOOL_CFG.agent_file_multi_search_default_max_hits, 30, 1),
        agent_file_multi_search_hard_max_hits = cfg_number(TOOL_CFG.agent_file_multi_search_hard_max_hits, 400, 4),
        agent_file_multi_search_default_max_files = cfg_number(TOOL_CFG.agent_file_multi_search_default_max_files, 24, 1),
        agent_file_multi_search_hard_max_files = cfg_number(TOOL_CFG.agent_file_multi_search_hard_max_files, 200, 1),
        agent_file_multi_search_default_per_file_hits = cfg_number(TOOL_CFG.agent_file_multi_search_default_per_file_hits, 5, 1),
        agent_file_multi_search_hard_per_file_hits = cfg_number(TOOL_CFG.agent_file_multi_search_hard_per_file_hits, 20, 1),
        agent_file_context_max_chars = cfg_number(TOOL_CFG.agent_file_context_max_chars, 1600, 120, 20000),
        external_enabled = to_bool(ext_cfg.enabled, false),
        external_include_memory_tools = to_bool(ext_cfg.include_memory_tools, true),
        external_context_inject = to_bool(ext_cfg.context_inject, true),
        external_context_max_chars = cfg_number(ext_cfg.context_max_chars, 1200, 120, 10000),
    }
end

local function apply_upsert_record(call, current_turn, policy, state)
    if state.upsert_count >= policy.upsert_max_per_turn then
        return false, string.format("upsert_record 超出预算（max=%d）", policy.upsert_max_per_turn), {
            error_code = "budget",
        }
    end

    local rec_type = trim(call.type)
    local entity = trim(call.entity)
    local value = trim(call.value)
    if value == "" then value = trim(call.string) end
    if rec_type == "" then
        return false, "upsert_record 缺少 type", { error_code = "validation" }
    end
    if entity == "" then
        return false, "upsert_record 缺少 entity", { error_code = "validation" }
    end
    if value == "" then
        return false, "upsert_record 缺少 value", { error_code = "validation" }
    end

    local confidence = clamp01(call.confidence, 0.75)
    if confidence < policy.upsert_min_confidence then
        return false, string.format(
            "upsert_record 置信度 %.2f 低于阈值 %.2f",
            confidence,
            policy.upsert_min_confidence
        ), {
            error_code = "validation",
        }
    end

    local id, op = notebook.upsert_record(rec_type, entity, value, {
        turn = current_turn,
        confidence = confidence,
        evidence = trim(call.evidence),
        source = "tool_call_upsert_record",
    })
    if not id then
        return false, "upsert_record 写入失败: " .. tostring(op), {
            error_code = "transient",
        }
    end
    state.upsert_count = state.upsert_count + 1
    return true, string.format("upsert_record %s (id=%d)", tostring(op), id), {
        error_code = "ok",
    }
end

local function apply_delete_record(call, current_turn, policy)
    if not policy.delete_enabled then
        return false, "delete_record 已禁用", {
            error_code = "validation",
        }
    end

    local rec_type = trim(call.type)
    local entity = trim(call.entity)
    if rec_type == "" then
        return false, "delete_record 缺少 type", { error_code = "validation" }
    end
    if entity == "" then
        return false, "delete_record 缺少 entity", { error_code = "validation" }
    end

    local id, op = notebook.delete_record(rec_type, entity, {
        turn = current_turn,
        evidence = trim(call.evidence),
        source = "tool_call_delete_record",
    })
    if not id then
        return false, "delete_record 失败: " .. tostring(op), {
            error_code = "transient",
        }
    end
    return true, string.format("delete_record %s (id=%d)", tostring(op), id), {
        error_code = "ok",
    }
end

local function apply_query_record(call, current_turn, policy, state)
    if state.query_count >= policy.query_max_per_turn then
        return false, string.format("query_record 超出预算（max=%d）", policy.query_max_per_turn), {
            error_code = "budget",
        }
    end

    local query, types, query_key = normalize_query_payload(call, policy)
    if query == "" then
        return false, "query_record 缺少 query/string/value", {
            error_code = "validation",
        }
    end

    local ok_query, results_or_err = pcall(notebook.query_records, query, {
        types = types,
        limit = policy.query_fetch_limit,
        mark_hit = true,
    })
    if not ok_query then
        return false, "query_record 查询失败: " .. tostring(results_or_err), {
            error_code = "transient",
        }
    end
    local results = results_or_err or {}
    results = deterministic_rerank(results, query)
    if #results > policy.query_inject_top then
        for i = #results, policy.query_inject_top + 1, -1 do
            table.remove(results, i)
        end
    end

    state.query_count = state.query_count + 1
    print(string.format("[ToolRegistry] query_record 命中 %d 条", #results))
    local query_result_signature = build_query_result_signature(query_key, results)
    if #results > 0 then
        local ok_render, rendered = pcall(notebook.render_results, results)
        if ok_render then
            print(rendered)
        else
            print("[ToolRegistry][WARN] render_results 失败: " .. tostring(rendered))
        end
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
        push_pending_context(current_turn, payload)
    end
    return true, string.format("query_record ok (%d hits)", #results), {
        error_code = "ok",
        query_key = query_key,
        query_signature = query_result_signature,
        context_signature = query_result_signature,
        context_added = (#results > 0),
        tool_result = (#results > 0) and ("query_record hits=" .. tostring(#results)) or "query_record hits=0",
    }
end

local function apply_list_agent_files(call, current_turn, policy, state)
    if state.file_list_count >= policy.agent_file_list_max_per_turn then
        return false, string.format("list_agent_files 超出预算（max=%d）", policy.agent_file_list_max_per_turn), {
            error_code = "budget",
        }
    end

    if type(py_pipeline) ~= "table" or py_pipeline.list_agent_files == nil then
        return false, "list_agent_files runtime 不可用", {
            error_code = "transient",
        }
    end

    local args_lua = build_call_arguments_lua(call)
    local ok_call, result_or_err = pcall(function()
        return py_pipeline:list_agent_files(
            args_lua,
            policy.agent_file_list_default_limit,
            policy.agent_file_list_hard_limit
        )
    end)
    if not ok_call then
        return false, "list_agent_files 调用失败: " .. tostring(result_or_err), {
            error_code = "transient",
        }
    end

    state.file_list_count = state.file_list_count + 1
    local raw_result = trim(result_or_err or "")
    if raw_result == "" then
        raw_result = "[empty tool result]"
    end
    local clipped = utf8_take(raw_result, policy.agent_file_context_max_chars)
    if clipped ~= raw_result then
        clipped = clipped .. "\n...(truncated)"
    end

    local payload = "【Tool:list_agent_files 返回】\n" .. clipped
    push_pending_context(current_turn, payload)
    local context_signature = "list_agent_files" .. "\x1F" .. clipped
    return true, "list_agent_files ok", {
        error_code = "ok",
        context_added = true,
        context_signature = context_signature,
        tool_result = clipped,
    }
end

local function apply_read_agent_file(call, current_turn, policy, state)
    if state.file_read_count >= policy.agent_file_read_max_per_turn then
        return false, string.format("read_agent_file 超出预算（max=%d）", policy.agent_file_read_max_per_turn), {
            error_code = "budget",
        }
    end

    if type(py_pipeline) ~= "table" or py_pipeline.read_agent_file == nil then
        return false, "read_agent_file runtime 不可用", {
            error_code = "transient",
        }
    end

    local args_lua = build_call_arguments_lua(call)
    local ok_call, result_or_err = pcall(function()
        return py_pipeline:read_agent_file(
            args_lua,
            policy.agent_file_read_default_max_chars,
            policy.agent_file_read_hard_max_chars
        )
    end)
    if not ok_call then
        return false, "read_agent_file 调用失败: " .. tostring(result_or_err), {
            error_code = "transient",
        }
    end

    state.file_read_count = state.file_read_count + 1
    local raw_result = trim(result_or_err or "")
    if raw_result == "" then
        raw_result = "[empty tool result]"
    end
    local clipped = utf8_take(raw_result, policy.agent_file_context_max_chars)
    if clipped ~= raw_result then
        clipped = clipped .. "\n...(truncated)"
    end

    local payload = "【Tool:read_agent_file 返回】\n" .. clipped
    push_pending_context(current_turn, payload)
    local context_signature = "read_agent_file" .. "\x1F" .. clipped
    return true, "read_agent_file ok", {
        error_code = "ok",
        context_added = true,
        context_signature = context_signature,
        tool_result = clipped,
    }
end

local function apply_read_agent_file_lines(call, current_turn, policy, state)
    if state.file_read_lines_count >= policy.agent_file_read_lines_max_per_turn then
        return false, string.format(
            "read_agent_file_lines 超出预算（max=%d）",
            policy.agent_file_read_lines_max_per_turn
        ), {
            error_code = "budget",
        }
    end

    if type(py_pipeline) ~= "table" or py_pipeline.read_agent_file_lines == nil then
        return false, "read_agent_file_lines runtime 不可用", {
            error_code = "transient",
        }
    end

    local args_lua = build_call_arguments_lua(call)
    local ok_call, result_or_err = pcall(function()
        return py_pipeline:read_agent_file_lines(
            args_lua,
            policy.agent_file_read_lines_default_max_lines,
            policy.agent_file_read_lines_hard_max_lines
        )
    end)
    if not ok_call then
        return false, "read_agent_file_lines 调用失败: " .. tostring(result_or_err), {
            error_code = "transient",
        }
    end

    state.file_read_lines_count = state.file_read_lines_count + 1
    local raw_result = trim(result_or_err or "")
    if raw_result == "" then
        raw_result = "[empty tool result]"
    end
    local clipped = utf8_take(raw_result, policy.agent_file_context_max_chars)
    if clipped ~= raw_result then
        clipped = clipped .. "\n...(truncated)"
    end

    local payload = "【Tool:read_agent_file_lines 返回】\n" .. clipped
    push_pending_context(current_turn, payload)
    local context_signature = "read_agent_file_lines" .. "\x1F" .. clipped
    return true, "read_agent_file_lines ok", {
        error_code = "ok",
        context_added = true,
        context_signature = context_signature,
        tool_result = clipped,
    }
end

local function apply_search_agent_file(call, current_turn, policy, state)
    if state.file_search_count >= policy.agent_file_search_max_per_turn then
        return false, string.format("search_agent_file 超出预算（max=%d）", policy.agent_file_search_max_per_turn), {
            error_code = "budget",
        }
    end

    if type(py_pipeline) ~= "table" or py_pipeline.search_agent_file == nil then
        return false, "search_agent_file runtime 不可用", {
            error_code = "transient",
        }
    end

    local args_lua = build_call_arguments_lua(call)
    local ok_call, result_or_err = pcall(function()
        return py_pipeline:search_agent_file(
            args_lua,
            policy.agent_file_search_default_max_hits,
            policy.agent_file_search_hard_max_hits
        )
    end)
    if not ok_call then
        return false, "search_agent_file 调用失败: " .. tostring(result_or_err), {
            error_code = "transient",
        }
    end

    state.file_search_count = state.file_search_count + 1
    local raw_result = trim(result_or_err or "")
    if raw_result == "" then
        raw_result = "[empty tool result]"
    end
    local clipped = utf8_take(raw_result, policy.agent_file_context_max_chars)
    if clipped ~= raw_result then
        clipped = clipped .. "\n...(truncated)"
    end

    local payload = "【Tool:search_agent_file 返回】\n" .. clipped
    push_pending_context(current_turn, payload)
    local context_signature = "search_agent_file" .. "\x1F" .. clipped
    return true, "search_agent_file ok", {
        error_code = "ok",
        context_added = true,
        context_signature = context_signature,
        tool_result = clipped,
    }
end

local function apply_search_agent_files(call, current_turn, policy, state)
    if state.file_multi_search_count >= policy.agent_file_multi_search_max_per_turn then
        return false, string.format(
            "search_agent_files 超出预算（max=%d）",
            policy.agent_file_multi_search_max_per_turn
        ), {
            error_code = "budget",
        }
    end

    if type(py_pipeline) ~= "table" or py_pipeline.search_agent_files == nil then
        return false, "search_agent_files runtime 不可用", {
            error_code = "transient",
        }
    end

    local args_lua = build_call_arguments_lua(call)
    local ok_call, result_or_err = pcall(function()
        return py_pipeline:search_agent_files(
            args_lua,
            policy.agent_file_multi_search_default_max_hits,
            policy.agent_file_multi_search_hard_max_hits,
            policy.agent_file_multi_search_default_max_files,
            policy.agent_file_multi_search_hard_max_files,
            policy.agent_file_multi_search_default_per_file_hits,
            policy.agent_file_multi_search_hard_per_file_hits
        )
    end)
    if not ok_call then
        return false, "search_agent_files 调用失败: " .. tostring(result_or_err), {
            error_code = "transient",
        }
    end

    state.file_multi_search_count = state.file_multi_search_count + 1
    local raw_result = trim(result_or_err or "")
    if raw_result == "" then
        raw_result = "[empty tool result]"
    end
    local clipped = utf8_take(raw_result, policy.agent_file_context_max_chars)
    if clipped ~= raw_result then
        clipped = clipped .. "\n...(truncated)"
    end

    local payload = "【Tool:search_agent_files 返回】\n" .. clipped
    push_pending_context(current_turn, payload)
    local context_signature = "search_agent_files" .. "\x1F" .. clipped
    return true, "search_agent_files ok", {
        error_code = "ok",
        context_added = true,
        context_signature = context_signature,
        tool_result = clipped,
    }
end

local function apply_external_tool(call, current_turn, policy)
    local ext_state = refresh_external_tools()
    local act = trim((call or {}).act or "")
    if act == "" then
        return false, "external tool 缺少 act", {
            error_code = "validation",
        }
    end
    if (not ext_state.enabled) or (not ext_state.name_set[act]) then
        return false, "external tool 未启用: " .. tostring(act), {
            error_code = "validation",
        }
    end
    if type(py_pipeline) ~= "table" or py_pipeline.call_qwen_tool == nil then
        return false, "external tool runtime 不可用", {
            error_code = "transient",
        }
    end

    local args_lua = build_call_arguments_lua(call)
    local ok_call, result_or_err = pcall(function()
        return py_pipeline:call_qwen_tool(act, args_lua)
    end)
    if not ok_call then
        return false, "external tool 调用失败: " .. tostring(result_or_err), {
            error_code = "transient",
        }
    end

    local raw_result = trim(result_or_err or "")
    if raw_result == "" then
        raw_result = "[empty tool result]"
    end
    local clipped = utf8_take(raw_result, policy.external_context_max_chars)
    if clipped ~= raw_result then
        clipped = clipped .. "\n...(truncated)"
    end

    local context_added = false
    if policy.external_context_inject then
        local payload = string.format("【Tool:%s 返回】\n%s", act, clipped)
        push_pending_context(current_turn, payload)
        context_added = true
    end

    local context_signature = act .. "\x1F" .. clipped
    return true, string.format("external tool %s ok", act), {
        error_code = "ok",
        context_added = context_added,
        context_signature = context_signature,
        tool_result = clipped,
    }
end

local function shallow_copy(tbl)
    local out = {}
    for k, v in pairs(tbl or {}) do
        out[k] = v
    end
    return out
end

local function classify_error_code(msg, meta)
    local code = trim((meta or {}).error_code or "")
    if code ~= "" then
        return code
    end

    local text = trim(msg)
    if text:find("超出预算", 1, true) then return "budget" end
    if text:find("缺少", 1, true) then return "validation" end
    if text:find("低于阈值", 1, true) then return "validation" end
    if text:find("已禁用", 1, true) then return "validation" end
    if text:find("去重跳过", 1, true) then return "duplicate" end
    if text:find("写入失败", 1, true) then return "transient" end
    if text:find("失败", 1, true) then return "transient" end
    if text:find("未知 act", 1, true) then return "unknown_act" end
    return "unknown"
end

local function get_retry_limit(policy, error_code)
    if error_code == "transient" then
        return math.max(0, math.floor(tonumber(policy.retry_transient_max) or 0))
    end
    if error_code == "unknown" then
        return math.max(0, math.floor(tonumber(policy.retry_unknown_max) or 0))
    end
    if error_code == "validation" then
        return math.max(0, math.floor(tonumber(policy.retry_validation_max) or 0))
    end
    if error_code == "budget" then
        return math.max(0, math.floor(tonumber(policy.retry_budget_max) or 0))
    end
    return 0
end

local function build_retry_variant(call, error_code, retry_idx)
    local next_call = shallow_copy(call)
    local strategy = ""

    if next_call.act == "query_record" then
        if retry_idx == 1 then
            if trim(next_call.types or "") ~= "" or trim(next_call.type or "") ~= "" then
                next_call.types = ""
                next_call.type = ""
                strategy = "drop_types"
            end
        elseif retry_idx == 2 then
            local q = trim(next_call.query)
            if q == "" then q = trim(next_call.string) end
            if q == "" then q = trim(next_call.value) end
            if q ~= "" then
                local clipped = utf8_take(q, 48)
                if clipped ~= q then
                    next_call.query = clipped
                    next_call.string = nil
                    next_call.value = nil
                    strategy = "shrink_query"
                end
            end
        end
    end

    if strategy ~= "" then
        return next_call, strategy
    end

    if error_code == "transient" or error_code == "unknown" then
        return next_call, "same_payload"
    end
    return nil, ""
end

local function build_execution_batches(calls, policy)
    local batches = {}
    local query_batch = {}
    local max_batch = math.max(1, math.floor(tonumber(policy.parallel_query_batch_size) or 1))

    local function flush_query_batch()
        if #query_batch == 0 then return end
        local mode = "serial"
        if policy.parallel_execute_enabled and #query_batch > 1 then
            mode = "parallel"
        end
        batches[#batches + 1] = {
            mode = mode,
            calls = query_batch,
        }
        query_batch = {}
    end

    for _, call in ipairs(calls or {}) do
        if policy.parallel_execute_enabled and tostring((call or {}).act or "") == "query_record" then
            query_batch[#query_batch + 1] = call
            if #query_batch >= max_batch then
                flush_query_batch()
            end
        else
            flush_query_batch()
            batches[#batches + 1] = {
                mode = "serial",
                calls = { call },
            }
        end
    end

    flush_query_batch()
    return batches
end

local function execute_apply_once(call, current_turn, policy, state)
    if call.act == "upsert_record" then
        return apply_upsert_record(call, current_turn, policy, state)
    elseif call.act == "delete_record" then
        return apply_delete_record(call, current_turn, policy)
    elseif call.act == "query_record" then
        return apply_query_record(call, current_turn, policy, state)
    elseif call.act == "list_agent_files" then
        return apply_list_agent_files(call, current_turn, policy, state)
    elseif call.act == "read_agent_file" then
        return apply_read_agent_file(call, current_turn, policy, state)
    elseif call.act == "read_agent_file_lines" then
        return apply_read_agent_file_lines(call, current_turn, policy, state)
    elseif call.act == "search_agent_file" then
        return apply_search_agent_file(call, current_turn, policy, state)
    elseif call.act == "search_agent_files" then
        return apply_search_agent_files(call, current_turn, policy, state)
    end
    local ext_state = refresh_external_tools()
    if ext_state.enabled and ext_state.name_set[tostring((call or {}).act or "")] then
        return apply_external_tool(call, current_turn, policy)
    end
    return false, "跳过未知 act: " .. tostring(call.act), {
        error_code = "unknown_act",
    }
end

local function execute_call_with_retry(call, current_turn, policy, state)
    local retry_logs = {}
    local count_result = true
    local initial_query_key = ""
    if call.act == "query_record" then
        local query, _, query_key = normalize_query_payload(call, policy)
        initial_query_key = query_key
        if query ~= "" and query_key ~= "" and state.query_seen[query_key] then
            return true, "query_record 去重跳过（同轮重复 query+types）", {
                error_code = "duplicate",
            }, false, retry_logs
        end
    end

    local attempts = 0
    local max_total = math.max(0, math.floor(tonumber(policy.retry_total_cap) or 0))
    local working = shallow_copy(call)
    local final_ok, final_msg, final_meta
    local used_retries = 0

    while true do
        attempts = attempts + 1
        if working.act == "query_record" then
            local _, _, key = normalize_query_payload(working, policy)
            if key ~= "" then
                state.query_seen[key] = true
            elseif initial_query_key ~= "" then
                state.query_seen[initial_query_key] = true
            end
        end

        local ok, msg, meta = execute_apply_once(working, current_turn, policy, state)
        final_ok, final_msg, final_meta = ok, msg, meta
        if ok then
            break
        end

        local err_code = classify_error_code(msg, meta)
        local allow_retries = get_retry_limit(policy, err_code)
        used_retries = attempts - 1
        if used_retries >= allow_retries or used_retries >= max_total then
            break
        end

        local next_call, strategy = build_retry_variant(working, err_code, attempts)
        if not next_call then
            break
        end

        local retry_msg = string.format(
            "retry#%d act=%s reason=%s strategy=%s",
            used_retries + 1,
            tostring(working.act or ""),
            tostring(err_code),
            tostring(strategy ~= "" and strategy or "none")
        )
        retry_logs[#retry_logs + 1] = retry_msg
        working = next_call
    end

    return final_ok, final_msg, final_meta, count_result, retry_logs
end

function M.get_policy()
    return get_tool_policy()
end

local function get_memory_openai_tools(policy)
    policy = policy or get_tool_policy()
    local tools = {
        {
            type = "function",
            ["function"] = {
                name = "query_record",
                description = "检索长期记忆记录（preference/constraint/identity/credential_hint/long_term_plan）",
                parameters = {
                    type = "object",
                    properties = {
                        query = { type = "string", description = "检索关键词或短句" },
                        types = {
                            type = "string",
                            description = "可选，逗号分隔类型列表（如 preference,constraint）",
                        },
                        type = { type = "string", description = "可选，单类型（types 的简写）" },
                    },
                    required = { "query" },
                    additionalProperties = false,
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "upsert_record",
                description = "写入或更新长期记忆记录",
                parameters = {
                    type = "object",
                    properties = {
                        type = {
                            type = "string",
                            description = "记录类型：preference|constraint|identity|credential_hint|long_term_plan",
                        },
                        entity = { type = "string", description = "主体对象" },
                        value = { type = "string", description = "事实内容" },
                        evidence = { type = "string", description = "可选，证据片段" },
                        confidence = { type = "number", description = "可选，0~1 置信度" },
                    },
                    required = { "type", "entity", "value" },
                    additionalProperties = false,
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "list_agent_files",
                description = "列出附件目录（MORI_AGENT_FILES_DIR，默认 ./workspace）下可读取的文件，支持 prefix 与 limit。",
                parameters = {
                    type = "object",
                    properties = {
                        prefix = { type = "string", description = "可选，仅列出指定子目录前缀（如 session_x/）" },
                        limit = { type = "number", description = "可选，返回条数上限" },
                    },
                    additionalProperties = false,
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "read_agent_file",
                description = "读取附件目录下指定文件的文本片段，按需分段读取。",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "文件路径（例如 ./workspace/<scope>/<name>）" },
                        start_char = { type = "number", description = "可选，1-based 起始字符位置" },
                        max_chars = { type = "number", description = "可选，本次读取最大字符数" },
                    },
                    required = { "path" },
                    additionalProperties = false,
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "read_agent_file_lines",
                description = "按行范围读取附件目录文本文件，返回带行号结果。",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "文件路径（例如 ./workspace/<scope>/<name>）" },
                        start_line = { type = "number", description = "可选，1-based 起始行号" },
                        end_line = { type = "number", description = "可选，结束行号（包含）" },
                        max_lines = { type = "number", description = "可选，本次最多返回行数" },
                    },
                    required = { "path" },
                    additionalProperties = false,
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "search_agent_file",
                description = "在附件目录文本文件中搜索关键词或正则，返回命中行号与上下文。",
                parameters = {
                    type = "object",
                    properties = {
                        path = { type = "string", description = "文件路径（例如 ./workspace/<scope>/<name>）" },
                        pattern = { type = "string", description = "搜索关键词或正则表达式" },
                        regex = { type = "boolean", description = "可选，true=按正则匹配" },
                        case_sensitive = { type = "boolean", description = "可选，true=区分大小写" },
                        context_lines = { type = "number", description = "可选，命中行前后附加的上下文行数" },
                        start_line = { type = "number", description = "可选，扫描起始行号" },
                        end_line = { type = "number", description = "可选，扫描结束行号" },
                        max_hits = { type = "number", description = "可选，最多展示多少个命中" },
                    },
                    required = { "path", "pattern" },
                    additionalProperties = false,
                },
            },
        },
        {
            type = "function",
            ["function"] = {
                name = "search_agent_files",
                description = "跨文件搜索附件目录下文本内容，返回文件路径+行号+上下文。",
                parameters = {
                    type = "object",
                    properties = {
                        pattern = { type = "string", description = "搜索关键词或正则表达式" },
                        prefix = { type = "string", description = "可选，仅扫描指定子目录前缀（如 session_x/）" },
                        regex = { type = "boolean", description = "可选，true=按正则匹配" },
                        case_sensitive = { type = "boolean", description = "可选，true=区分大小写" },
                        context_lines = { type = "number", description = "可选，命中行前后附加的上下文行数" },
                        start_line = { type = "number", description = "可选，每个文件的扫描起始行号" },
                        end_line = { type = "number", description = "可选，每个文件的扫描结束行号" },
                        max_files = { type = "number", description = "可选，最多扫描多少个文件" },
                        per_file_hits = { type = "number", description = "可选，单文件最多展示多少个命中" },
                        max_hits = { type = "number", description = "可选，总命中展示上限" },
                    },
                    required = { "pattern" },
                    additionalProperties = false,
                },
            },
        },
    }

    if policy.delete_enabled then
        tools[#tools + 1] = {
            type = "function",
            ["function"] = {
                name = "delete_record",
                description = "删除长期记忆记录",
                parameters = {
                    type = "object",
                    properties = {
                        type = { type = "string", description = "记录类型" },
                        entity = { type = "string", description = "主体对象" },
                        evidence = { type = "string", description = "可选，删除依据" },
                    },
                    required = { "type", "entity" },
                    additionalProperties = false,
                },
            },
        }
    end

    return tools
end

function M.get_openai_tools(policy_override)
    local policy = policy_override or get_tool_policy()
    local tools = {}

    if policy.external_include_memory_tools then
        local core = get_memory_openai_tools(policy)
        for _, item in ipairs(core or {}) do
            tools[#tools + 1] = item
        end
    end

    local ext_state = refresh_external_tools()
    if policy.external_enabled and ext_state.enabled then
        for _, item in ipairs(ext_state.schemas or {}) do
            tools[#tools + 1] = item
        end
    end

    return tools
end

function M.get_supported_acts(base_acts)
    local acts = {}
    for k, v in pairs(base_acts or {}) do
        if v then
            acts[k] = true
        end
    end

    local policy = get_tool_policy()
    if policy.external_include_memory_tools then
        acts.query_record = true
        acts.upsert_record = true
        acts.list_agent_files = true
        acts.read_agent_file = true
        acts.read_agent_file_lines = true
        acts.search_agent_file = true
        acts.search_agent_files = true
        if policy.delete_enabled then
            acts.delete_record = true
        end
    end

    local ext_state = refresh_external_tools()
    if policy.external_enabled and ext_state.enabled then
        for name, enabled in pairs(ext_state.name_set or {}) do
            if enabled then
                acts[name] = true
            end
        end
    end

    -- 当 memory 工具显式关闭时，移除内置 act，避免解析误命中。
    if not policy.external_include_memory_tools then
        acts.query_record = nil
        acts.upsert_record = nil
        acts.delete_record = nil
        acts.list_agent_files = nil
        acts.read_agent_file = nil
        acts.read_agent_file_lines = nil
        acts.search_agent_file = nil
        acts.search_agent_files = nil
    end

    return acts
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
        call_results = {},
        context_updated = false,
        context_novel = false,
        context_signature = "",
        parallel_batches = 0,
        parallel_calls = 0,
        retry_total = 0,
        retry_success = 0,
        retry_failed = 0,
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
        file_list_count = 0,
        file_read_count = 0,
        file_read_lines_count = 0,
        file_search_count = 0,
        file_multi_search_count = 0,
        query_seen = {},
    }
    local turn_budget = get_turn_budget_state(current_turn)
    state.upsert_count = turn_budget.upsert_count
    state.query_count = turn_budget.query_count
    state.file_list_count = turn_budget.file_list_count
    state.file_read_count = turn_budget.file_read_count
    state.file_read_lines_count = turn_budget.file_read_lines_count
    state.file_search_count = turn_budget.file_search_count
    state.file_multi_search_count = turn_budget.file_multi_search_count
    local step_context_signatures = {}
    local step_context_sig_seen = {}
    local call_seq = 0

    local batches = build_execution_batches(calls, policy)
    for batch_idx, batch in ipairs(batches or {}) do
        local batch_calls = (batch and batch.calls) or {}
        local is_parallel = (batch and batch.mode == "parallel") and #batch_calls > 1
        if is_parallel then
            result.parallel_batches = result.parallel_batches + 1
            result.parallel_calls = result.parallel_calls + #batch_calls
            print(string.format(
                "[ToolRegistry] PARALLEL batch=%d size=%d mode=query_record",
                batch_idx,
                #batch_calls
            ))
        end

        for _, call in ipairs(batch_calls) do
            call_seq = call_seq + 1
            local tool_call_id = trim((call or {}).tool_call_id or "")
            if tool_call_id == "" then
                tool_call_id = string.format("call_%d_%d", current_turn, call_seq)
                call.tool_call_id = tool_call_id
            end

            local ok, msg, meta, count_result, retry_logs = execute_call_with_retry(
                call,
                current_turn,
                policy,
                state
            )
            for _, retry_msg in ipairs(retry_logs or {}) do
                result.retry_total = result.retry_total + 1
                result.logs[#result.logs + 1] = retry_msg
                print(string.format("[ToolRegistry] RETRY | %s", retry_msg))
            end

            if #(retry_logs or {}) > 0 then
                if ok then
                    result.retry_success = result.retry_success + 1
                else
                    result.retry_failed = result.retry_failed + 1
                end
            end

            if ok and meta then
                local sig = trim(meta.context_signature or meta.query_signature)
                if sig ~= "" and (not step_context_sig_seen[sig]) then
                    step_context_sig_seen[sig] = true
                    step_context_signatures[#step_context_signatures + 1] = sig
                end
            end

            result.call_results[#result.call_results + 1] = {
                act = tostring((call or {}).act or ""),
                tool_call_id = tool_call_id,
                arguments = build_call_arguments_lua(call),
                ok = (ok == true),
                skipped = (count_result == false),
                message = tostring(msg or ""),
                result = tostring((meta or {}).tool_result or ""),
            }

            if count_result then
                if ok then
                    result.executed = result.executed + 1
                else
                    result.failed = result.failed + 1
                end
            else
                result.skipped = result.skipped + 1
            end

            result.logs[#result.logs + 1] = msg
            local level = ok and "OK" or "FAIL"
            if count_result == false then
                level = "SKIP"
            end
            print(string.format("[ToolRegistry] %s | %s", level, msg))
        end
    end
    turn_budget.upsert_count = state.upsert_count
    turn_budget.query_count = state.query_count
    turn_budget.file_list_count = state.file_list_count
    turn_budget.file_read_count = state.file_read_count
    turn_budget.file_read_lines_count = state.file_read_lines_count
    turn_budget.file_search_count = state.file_search_count
    turn_budget.file_multi_search_count = state.file_multi_search_count

    if #step_context_signatures > 0 then
        table.sort(step_context_signatures)
        local step_sig = table.concat(step_context_signatures, "||")
        result.context_signature = step_sig
        local same_turn = (tonumber(M._last_context_turn) or 0) == current_turn
        local is_novel = (not same_turn) or (step_sig ~= tostring(M._last_context_signature or ""))
        result.context_novel = is_novel

        if is_novel then
            M._last_context_signature = step_sig
            M._last_context_turn = current_turn
            result.context_updated = trim(M._pending_system_context) ~= ""
        else
            if M._pending_system_context ~= "" then
                clear_pending_system_context()
            end
            result.context_updated = false
            result.logs[#result.logs + 1] = "tool 命中但无增量上下文，已跳过注入"
            print("[ToolRegistry] SKIP | tool 命中但无增量上下文，已跳过注入")
        end
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
