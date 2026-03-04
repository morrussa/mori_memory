local M = {}

local FILE_PATH = "memory/notebook.txt"
local SEP = "\x1F"
local config = require("module.config")
local persistence = require("module.persistence")

local DEFAULT_K1 = 1.2
local DEFAULT_B = 0.75
local PLAN_CFG = (config.settings.keyring and config.settings.keyring.long_term_plan) or {}
local PLAN_MAX_VALUE = tonumber(PLAN_CFG.max_value_chars) or 200
local PLAN_MAX_EVIDENCE = tonumber(PLAN_CFG.max_evidence_chars) or 160
local PLAN_BOM_ITEMS = tonumber(PLAN_CFG.bom_max_items) or 3
local PLAN_BOM_CHARS = tonumber(PLAN_CFG.bom_max_chars) or 800

local ALLOWED_TYPES = {
    preference = true,
    constraint = true,
    identity = true,
    credential_hint = true,
    long_term_plan = true,
}

local CONSTRAINT_WORDS = {
    "不要", "禁止", "必须", "只要", "永远", "以后都"
}

local PREFERENCE_WORDS = {
    "偏好", "喜欢"
}

local PLAN_WORDS = {
    "计划", "打算", "准备", "路线图", "roadmap", "todo", "后续"
}

local CREDENTIAL_WORDS = {
    "token", "apikey", "api key", "password", "passwd", "secret", "凭证", "密钥", "口令", "密码"
}

M.records = {}          -- id -> record
M.by_type_entity = {}   -- type -> normalized_entity -> id
M.postings = {}         -- type -> token -> { [id] = tf }
M.doclen = {}           -- id -> token count
M.type_stats = {}       -- type -> { count = active_docs, sum_len = total_doclen }
M.next_id = 1

local function trim(s)
    if not s then return "" end
    return (tostring(s):gsub("^%s*(.-)%s*$", "%1"))
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
    if #out == 0 then return "" end
    return table.concat(out)
end

local function limit_text(s, max_len)
    s = tostring(s or "")
    max_len = tonumber(max_len) or 0
    if max_len <= 0 then return s end
    return utf8_take(s, max_len)
end

local function clamp01(v, fallback)
    local n = tonumber(v)
    if not n then return fallback or 0.7 end
    if n < 0 then return 0 end
    if n > 1 then return 1 end
    return n
end

local function esc(s)
    s = tostring(s or "")
    s = s:gsub("\\", "\\\\")
    s = s:gsub("\n", "\\n")
    s = s:gsub(SEP, "\\u001F")
    return s
end

local function unesc(s)
    s = tostring(s or "")
    s = s:gsub("\\u001F", SEP)
    s = s:gsub("\\n", "\n")
    s = s:gsub("\\\\", "\\")
    return s
end

local function split_by_sep(s, sep)
    local out = {}
    local pos = 1
    while true do
        local i, j = string.find(s, sep, pos, true)
        if not i then
            table.insert(out, string.sub(s, pos))
            break
        end
        table.insert(out, string.sub(s, pos, i - 1))
        pos = j + 1
    end
    return out
end

local function parse_turns_csv(s)
    local t = {}
    if not s or s == "" then return t end
    for n in s:gmatch("[^,]+") do
        local v = tonumber(n)
        if v then table.insert(t, v) end
    end
    return t
end

local function turns_to_csv(turns)
    if not turns or #turns == 0 then return "" end
    local p = {}
    for i = 1, #turns do p[i] = tostring(turns[i]) end
    return table.concat(p, ",")
end

local function mark_dirty()
    local ok, saver = pcall(require, "module.memory.saver")
    if ok and saver and saver.mark_dirty then
        saver.mark_dirty()
    end
end

local function normalize_type(type_name)
    local t = trim(type_name):lower()
    t = t:gsub("%-", "_")
    t = t:gsub("%s+", "_")
    if t == "credential" then t = "credential_hint" end
    if t == "credentialhint" then t = "credential_hint" end
    if t == "longtermplan" then t = "long_term_plan" end
    if t == "plan" then t = "long_term_plan" end
    if ALLOWED_TYPES[t] then return t end
    return nil
end

local function normalize_entity(entity)
    local e = trim(entity)
    e = e:gsub("%s+", " ")
    return e
end

local function normalize_entity_key(entity)
    local e = normalize_entity(entity)
    return e:lower()
end

local function ensure_type_tables(type_name)
    if not M.by_type_entity[type_name] then M.by_type_entity[type_name] = {} end
    if not M.postings[type_name] then M.postings[type_name] = {} end
    if not M.type_stats[type_name] then
        M.type_stats[type_name] = { count = 0, sum_len = 0 }
    end
end

local function utf8_codepoint(ch)
    local b1, b2, b3, b4 = ch:byte(1, 4)
    if not b1 then return nil end
    if b1 < 0x80 then return b1 end
    if b1 < 0xE0 then
        if not b2 then return nil end
        return (b1 - 0xC0) * 0x40 + (b2 - 0x80)
    elseif b1 < 0xF0 then
        if not b2 or not b3 then return nil end
        return (b1 - 0xE0) * 0x1000 + (b2 - 0x80) * 0x40 + (b3 - 0x80)
    else
        if not b2 or not b3 or not b4 then return nil end
        return (b1 - 0xF0) * 0x40000 + (b2 - 0x80) * 0x1000 + (b3 - 0x80) * 0x40 + (b4 - 0x80)
    end
end

local function is_cjk_like(cp)
    if not cp then return false end
    if cp >= 0x3400 and cp <= 0x4DBF then return true end
    if cp >= 0x4E00 and cp <= 0x9FFF then return true end
    if cp >= 0xF900 and cp <= 0xFAFF then return true end
    if cp >= 0x20000 and cp <= 0x2EBEF then return true end
    if cp >= 0x3040 and cp <= 0x30FF then return true end
    if cp >= 0xAC00 and cp <= 0xD7AF then return true end
    return false
end

local function tokenize(text)
    local tf = {}
    text = tostring(text or "")

    -- ASCII token（英文/数字）
    for tok in string.lower(text):gmatch("[%a%d_%.%-%/]+") do
        if #tok >= 2 then
            tf[tok] = (tf[tok] or 0) + 1
        end
    end

    -- CJK token（单字 + 双字）
    local run = {}
    local function add_token(tok)
        if tok and tok ~= "" then
            tf[tok] = (tf[tok] or 0) + 1
        end
    end
    local function flush_run()
        local n = #run
        if n == 0 then return end
        for i = 1, n do
            add_token(run[i])
        end
        if n >= 2 then
            for i = 1, n - 1 do
                add_token(run[i] .. run[i + 1])
            end
        end
        run = {}
    end

    for ch in text:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        local cp = utf8_codepoint(ch)
        if is_cjk_like(cp) then
            run[#run + 1] = ch
        else
            flush_run()
        end
    end
    flush_run()

    local tokens = {}
    for token, _ in pairs(tf) do
        table.insert(tokens, token)
    end
    return tokens, tf
end

local function unindex_doc(id, rec)
    if not rec or rec.status ~= "active" then return end
    local t = rec.type
    local type_posting = M.postings[t]
    if not type_posting then return end

    for token, posting in pairs(type_posting) do
        posting[id] = nil
        if next(posting) == nil then
            type_posting[token] = nil
        end
    end

    local dl = M.doclen[id] or 0
    M.doclen[id] = nil
    if M.type_stats[t] then
        M.type_stats[t].count = math.max(0, M.type_stats[t].count - 1)
        M.type_stats[t].sum_len = math.max(0, M.type_stats[t].sum_len - dl)
    end
end

local function index_doc(id, rec)
    if not rec or rec.status ~= "active" then return end
    ensure_type_tables(rec.type)

    local doc_text = table.concat({
        rec.type or "",
        rec.entity or "",
        rec.value or "",
        rec.evidence or "",
    }, " ")
    local tokens, tf = tokenize(doc_text)

    M.doclen[id] = #tokens
    M.type_stats[rec.type].count = M.type_stats[rec.type].count + 1
    M.type_stats[rec.type].sum_len = M.type_stats[rec.type].sum_len + #tokens

    for token, freq in pairs(tf) do
        local posting = M.postings[rec.type][token]
        if not posting then
            posting = {}
            M.postings[rec.type][token] = posting
        end
        posting[id] = freq
    end
end

local function add_turn_if_needed(rec, turn)
    turn = tonumber(turn) or 0
    if turn <= 0 then return end
    if rec.first_turn == 0 then rec.first_turn = turn end
    rec.last_turn = math.max(rec.last_turn or 0, turn)
    for _, t in ipairs(rec.turns) do
        if t == turn then return end
    end
    table.insert(rec.turns, turn)
end

local function as_result(rec, score)
    return {
        id = rec.id,
        type = rec.type,
        entity = rec.entity,
        value = rec.value,
        evidence = rec.evidence,
        confidence = rec.confidence,
        status = rec.status,
        updated_at = rec.updated_at,
        turns = rec.turns,
        score = score or 0,
    }
end

local function ns_to_type(ns)
    local t = normalize_type(ns)
    if t then return t end
    ns = trim(ns):lower()
    if ns == "policy" then return "constraint" end
    if ns == "preference" then return "preference" end
    if ns == "constraint" then return "constraint" end
    if ns == "long_term_plan" then return "long_term_plan" end
    if ns == "credential_hint" then return "credential_hint" end
    return "identity"
end

local function detect_types_from_query(query)
    query = tostring(query or "")
    local picked = {}
    local set = {}

    local function add(t)
        if not set[t] then
            set[t] = true
            table.insert(picked, t)
        end
    end

    for _, kw in ipairs(CONSTRAINT_WORDS) do
        if query:find(kw, 1, true) then add("constraint") end
    end
    for _, kw in ipairs(PREFERENCE_WORDS) do
        if query:find(kw, 1, true) then add("preference") end
    end
    for _, kw in ipairs(PLAN_WORDS) do
        if query:lower():find(kw:lower(), 1, true) then add("long_term_plan") end
    end
    for _, kw in ipairs(CREDENTIAL_WORDS) do
        if query:lower():find(kw:lower(), 1, true) then add("credential_hint") end
    end
    if query:find("我是谁", 1, true) or query:find("身份", 1, true) or query:find("名字", 1, true) then
        add("identity")
    end

    if #picked == 0 then
        for t, _ in pairs(ALLOWED_TYPES) do table.insert(picked, t) end
    end
    return picked
end

function M.reset()
    M.records = {}
    M.by_type_entity = {}
    M.postings = {}
    M.doclen = {}
    M.type_stats = {}
    M.next_id = 1
end

function M.upsert_record(type_name, entity, value, opts)
    opts = opts or {}
    local t = normalize_type(type_name)
    if not t then return nil, "invalid_type" end

    local entity_raw = normalize_entity(entity)
    if entity_raw == "" then return nil, "empty_entity" end
    local entity_key = normalize_entity_key(entity_raw)

    local val = trim(value)
    if val == "" then return nil, "empty_value" end
    local evidence_clean = trim(opts.evidence or "")
    if t == "long_term_plan" then
        val = limit_text(val, PLAN_MAX_VALUE)
        evidence_clean = limit_text(evidence_clean, PLAN_MAX_EVIDENCE)
    end

    ensure_type_tables(t)
    local id = M.by_type_entity[t][entity_key]
    local now = tonumber(opts.updated_at) or os.time()

    if id then
        local rec = M.records[id]
        if not rec then return nil, "broken_index" end

        unindex_doc(id, rec)
        local was_deleted = rec.status == "deleted"

        rec.type = t
        rec.entity = entity_raw
        rec.entity_norm = entity_key
        rec.value = val
        if evidence_clean ~= "" then
            rec.evidence = evidence_clean
        end
        rec.confidence = clamp01(opts.confidence, rec.confidence or 0.7)
        rec.updated_at = now
        rec.status = "active"
        rec.deleted_at = 0
        rec.source = trim(opts.source or rec.source or "")
        rec.version = (rec.version or 1) + 1
        add_turn_if_needed(rec, opts.turn)

        index_doc(id, rec)
        mark_dirty()
        return id, was_deleted and "revived" or "updated"
    end

    local rec = {
        id = M.next_id,
        type = t,
        entity = entity_raw,
        entity_norm = entity_key,
        value = val,
        evidence = evidence_clean,
        confidence = clamp01(opts.confidence, 0.7),
        status = "active",
        created_at = now,
        updated_at = now,
        deleted_at = 0,
        first_turn = 0,
        last_turn = 0,
        turns = {},
        hits = 0,
        source = trim(opts.source or ""),
        version = 1,
    }
    add_turn_if_needed(rec, opts.turn)

    M.records[rec.id] = rec
    M.by_type_entity[t][entity_key] = rec.id
    index_doc(rec.id, rec)
    M.next_id = rec.id + 1
    mark_dirty()
    return rec.id, "inserted"
end

function M.get_active_by_type(type_name, limit)
    local t = normalize_type(type_name)
    if not t then return {} end

    local out = {}
    for _, rec in pairs(M.records) do
        if rec.type == t and rec.status == "active" then
            table.insert(out, as_result(rec, 0))
        end
    end
    table.sort(out, function(a, b)
        local au = a.updated_at or 0
        local bu = b.updated_at or 0
        if au == bu then
            return (a.confidence or 0) > (b.confidence or 0)
        end
        return au > bu
    end)

    if limit and #out > limit then
        for i = #out, limit + 1, -1 do
            table.remove(out, i)
        end
    end
    return out
end

function M.build_long_term_plan_bom()
    local plans = M.get_active_by_type("long_term_plan", PLAN_BOM_ITEMS)
    if #plans == 0 then return "" end

    local lines = {
        "【LongTermPlan BOM】",
        "以下是当前长期计划，请回答时与其保持一致：",
    }
    for i, p in ipairs(plans) do
        lines[#lines + 1] = string.format(
            "%d) [%s] %s (conf=%.2f)",
            i,
            p.entity or "plan",
            p.value or "",
            p.confidence or 0
        )
    end
    local bom = table.concat(lines, "\n")
    if PLAN_BOM_CHARS > 0 then
        local clipped = limit_text(bom, PLAN_BOM_CHARS)
        if clipped ~= bom then
            bom = clipped .. "\n...(truncated)"
        end
    end
    return bom
end

function M.delete_record(type_name, entity, opts)
    opts = opts or {}
    local t = normalize_type(type_name)
    if not t then return nil, "invalid_type" end

    local entity_key = normalize_entity_key(entity)
    if entity_key == "" then return nil, "empty_entity" end

    local id = M.by_type_entity[t] and M.by_type_entity[t][entity_key]
    if not id then return nil, "not_found" end

    local rec = M.records[id]
    if not rec then return nil, "broken_index" end
    if rec.status == "deleted" then return id, "already_deleted" end

    unindex_doc(id, rec)
    rec.status = "deleted"
    rec.deleted_at = os.time()
    rec.updated_at = rec.deleted_at
    if trim(opts.evidence) ~= "" then
        rec.evidence = trim(opts.evidence)
    end
    rec.source = trim(opts.source or rec.source or "")
    rec.version = (rec.version or 1) + 1
    add_turn_if_needed(rec, opts.turn)
    mark_dirty()
    return id, "deleted"
end

function M.get_record(type_name, entity)
    local t = normalize_type(type_name)
    if not t then return nil end
    local key = normalize_entity_key(entity)
    local id = M.by_type_entity[t] and M.by_type_entity[t][key]
    if not id then return nil end
    local rec = M.records[id]
    if not rec then return nil end
    return as_result(rec, 0)
end

function M.search(query, opts)
    opts = opts or {}
    local limit = tonumber(opts.limit) or 8
    local k1 = tonumber(opts.k1) or DEFAULT_K1
    local b = tonumber(opts.b) or DEFAULT_B
    local include_deleted = opts.include_deleted == true
    local mark_hit = opts.mark_hit == true

    local type_filter = {}
    local filter_count = 0

    local src_types = opts.types or opts.namespaces
    if type(src_types) == "table" and #src_types > 0 then
        for _, raw in ipairs(src_types) do
            local t = normalize_type(raw) or ns_to_type(raw)
            if t and not type_filter[t] then
                type_filter[t] = true
                filter_count = filter_count + 1
            end
        end
    end

    local _, qtf = tokenize(query or "")
    local q_tokens = {}
    for token, _ in pairs(qtf) do
        table.insert(q_tokens, token)
    end
    if #q_tokens == 0 then return {} end

    local scores = {}
    for t, token_map in pairs(M.postings) do
        if filter_count == 0 or type_filter[t] then
            local stat = M.type_stats[t] or { count = 0, sum_len = 0 }
            local n = stat.count
            local avgdl = (n > 0) and (stat.sum_len / n) or 1.0
            if n > 0 then
                for _, token in ipairs(q_tokens) do
                    local posting = token_map[token]
                    if posting then
                        local df = 0
                        for _ in pairs(posting) do df = df + 1 end
                        local idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                        for id, tf in pairs(posting) do
                            local dl = M.doclen[id] or 1
                            local denom = tf + k1 * (1 - b + b * dl / avgdl)
                            local bm25 = idf * ((tf * (k1 + 1)) / denom)
                            scores[id] = (scores[id] or 0) + bm25
                        end
                    end
                end
            end
        end
    end

    local out = {}
    for id, score in pairs(scores) do
        local rec = M.records[id]
        if rec then
            if include_deleted or rec.status == "active" then
                local final_score = score * (0.6 + 0.4 * (rec.confidence or 0.7))
                table.insert(out, as_result(rec, final_score))
                if mark_hit then rec.hits = (rec.hits or 0) + 1 end
            end
        end
    end

    table.sort(out, function(a, b)
        if a.score == b.score then
            return (a.updated_at or 0) > (b.updated_at or 0)
        end
        return a.score > b.score
    end)

    if #out > limit then
        for i = #out, limit + 1, -1 do
            table.remove(out, i)
        end
    end
    return out
end

function M.query_records(query, opts)
    opts = opts or {}
    local types = opts.types
    if not types or #types == 0 then
        types = detect_types_from_query(query)
    end
    return M.search(query, {
        types = types,
        limit = opts.limit,
        include_deleted = opts.include_deleted,
        mark_hit = opts.mark_hit,
    })
end

function M.audit(current_turn, opts)
    opts = opts or {}
    local stale_window = tonumber(opts.stale_window) or 200
    local low_conf_th = tonumber(opts.low_confidence) or 0.65
    local turn_now = tonumber(current_turn) or 0

    local by_type = {}
    local total = 0
    local active = 0
    local deleted = 0
    local stale = 0
    local low_conf = 0

    for _, rec in pairs(M.records) do
        total = total + 1
        by_type[rec.type] = by_type[rec.type] or { active = 0, deleted = 0 }
        if rec.status == "active" then
            active = active + 1
            by_type[rec.type].active = by_type[rec.type].active + 1
        else
            deleted = deleted + 1
            by_type[rec.type].deleted = by_type[rec.type].deleted + 1
        end

        if (rec.confidence or 0) < low_conf_th then
            low_conf = low_conf + 1
        end
        if turn_now > 0 and rec.last_turn > 0 and (turn_now - rec.last_turn) >= stale_window then
            stale = stale + 1
        end
    end

    return {
        total = total,
        active = active,
        deleted = deleted,
        stale = stale,
        low_confidence = low_conf,
        by_type = by_type,
    }
end

function M.render_results(results)
    local lines = {}
    for i, r in ipairs(results or {}) do
        lines[i] = string.format(
            "[%s] %s | %.4f | conf=%.2f | %s",
            r.type or "identity",
            r.entity or "",
            r.score or 0,
            r.confidence or 0,
            r.value or ""
        )
    end
    return table.concat(lines, "\n")
end

local function add_fact(out, seen, type_name, entity, value, evidence, confidence, source)
    local t = normalize_type(type_name)
    if not t then return end
    local e = normalize_entity(entity)
    local v = trim(value)
    if e == "" or v == "" then return end

    local uniq = t .. SEP .. normalize_entity_key(e) .. SEP .. v
    if seen[uniq] then return end
    seen[uniq] = true

    table.insert(out, {
        type = t,
        entity = e,
        value = v,
        evidence = trim(evidence or v),
        confidence = clamp01(confidence, 0.7),
        source = source or "rule",
        namespace = t, -- 兼容旧字段
        key = e,       -- 兼容旧字段
    })
end

function M.extract(text)
    text = tostring(text or "")
    local out = {}
    local seen = {}

    -- credential hints
    for sent in text:gmatch("[^\n。！？!?]+") do
        local s = trim(sent)
        if s ~= "" then
            local lower = s:lower()
            for _, kw in ipairs(CREDENTIAL_WORDS) do
                if lower:find(kw:lower(), 1, true) then
                    add_fact(out, seen, "credential_hint", "credentials", s, s, 0.85, "kw_credential")
                    break
                end
            end
        end
    end

    -- preference/constraint
    for sent in text:gmatch("[^\n。！？!?]+") do
        local s = trim(sent)
        if s ~= "" then
            for _, kw in ipairs(PREFERENCE_WORDS) do
                if s:find(kw, 1, true) then
                    local entity = s:match("喜欢([^，。！？!?]+)") or "general"
                    add_fact(out, seen, "preference", entity, s, s, 0.82, "kw_preference")
                end
            end
            for _, kw in ipairs(CONSTRAINT_WORDS) do
                if s:find(kw, 1, true) then
                    local entity = "global"
                    if s:find("python", 1, true) or s:find("Python", 1, true) then entity = "python" end
                    add_fact(out, seen, "constraint", entity, s, s, 0.86, "kw_constraint")
                    break
                end
            end
            for _, kw in ipairs(PLAN_WORDS) do
                if s:lower():find(kw:lower(), 1, true) then
                    add_fact(out, seen, "long_term_plan", "roadmap", s, s, 0.80, "kw_plan")
                    break
                end
            end
        end
    end

    -- identity-like snippets
    for email in text:gmatch("[%w%._%+%-]+@[%w%.%-]+%.[%a]+") do
        add_fact(out, seen, "identity", "email", email, email, 0.95, "regex_email")
    end
    for url in text:gmatch("https?://[%w%-%._~:/%?#%[%]@!$&'()*+,;=%%]+") do
        add_fact(out, seen, "identity", "url", url, url, 0.95, "regex_url")
    end
    for version in text:gmatch("[Vv]?%d+%.%d+%.?%d*") do
        add_fact(out, seen, "identity", "version", version, version, 0.88, "regex_version")
    end
    for idv in text:gmatch("[Ii][Dd][%s:#=_%-]+[%w%-_]+") do
        add_fact(out, seen, "identity", "id", idv, idv, 0.80, "regex_id")
    end

    if #out == 0 and trim(text) ~= "" then
        add_fact(out, seen, "identity", "note", trim(text), trim(text), 0.70, "fallback_note")
    end
    return out
end

function M.observe_turn(turn, user_text, assistant_text)
    local inserted, updated = 0, 0
    local all = {}
    local u = M.extract(user_text or "")
    local a = M.extract(assistant_text or "")
    for i = 1, #u do table.insert(all, u[i]) end
    for i = 1, #a do table.insert(all, a[i]) end

    for _, f in ipairs(all) do
        local _, op = M.upsert_record(f.type, f.entity, f.value, {
            turn = turn,
            confidence = f.confidence,
            evidence = f.evidence,
            source = f.source,
        })
        if op == "inserted" then inserted = inserted + 1 end
        if op == "updated" or op == "revived" then updated = updated + 1 end
    end
    return { inserted = inserted, updated = updated, total = #all }
end

-- 兼容旧接口：namespace/key/value
function M.upsert(namespace, key, value, opts)
    return M.upsert_record(ns_to_type(namespace), key, value, opts)
end

function M.save_to_disk()
    local ok, err = persistence.write_atomic(FILE_PATH, "w", function(f)
        local h_ok, h_err = f:write("NOTEBOOK_V2\n")
        if not h_ok then return false, h_err end

        for id, rec in pairs(M.records) do
            local line = table.concat({
                tostring(id),
                esc(rec.type or ""),
                esc(rec.entity or ""),
                esc(rec.entity_norm or ""),
                esc(rec.value or ""),
                esc(rec.evidence or ""),
                esc(rec.status or "active"),
                tostring(rec.confidence or 0.7),
                tostring(rec.created_at or 0),
                tostring(rec.updated_at or 0),
                tostring(rec.deleted_at or 0),
                tostring(rec.first_turn or 0),
                tostring(rec.last_turn or 0),
                esc(turns_to_csv(rec.turns)),
                tostring(rec.hits or 0),
                esc(rec.source or ""),
                tostring(rec.version or 1),
            }, SEP)
            local w_ok, w_err = f:write(line .. "\n")
            if not w_ok then return false, w_err end
        end
        return true
    end)
    if not ok then
        return false, err
    end
    return true
end

local function load_v2_line(fields)
    if #fields < 17 then return nil end
    local id = tonumber(fields[1])
    local type_name = normalize_type(unesc(fields[2]))
    local entity = normalize_entity(unesc(fields[3]))
    local entity_norm = normalize_entity_key(unesc(fields[4]) ~= "" and unesc(fields[4]) or entity)
    local value = unesc(fields[5])
    if not id or not type_name or entity == "" or value == "" then return nil end

    local rec = {
        id = id,
        type = type_name,
        entity = entity,
        entity_norm = entity_norm,
        value = value,
        evidence = unesc(fields[6]),
        status = (unesc(fields[7]) == "deleted") and "deleted" or "active",
        confidence = clamp01(tonumber(fields[8]), 0.7),
        created_at = tonumber(fields[9]) or 0,
        updated_at = tonumber(fields[10]) or 0,
        deleted_at = tonumber(fields[11]) or 0,
        first_turn = tonumber(fields[12]) or 0,
        last_turn = tonumber(fields[13]) or 0,
        turns = parse_turns_csv(unesc(fields[14])),
        hits = tonumber(fields[15]) or 0,
        source = unesc(fields[16]),
        version = tonumber(fields[17]) or 1,
    }
    return rec
end

function M.load()
    M.reset()
    local f = io.open(FILE_PATH, "r")
    if not f then return end

    local header = f:read("*l")
    if not header then
        f:close()
        return
    end
    header = trim(header)
    if header ~= "NOTEBOOK_V2" then
        f:close()
        print("[Notebook] 检测到旧版 notebook.txt，已跳过加载（仅支持 NOTEBOOK_V2）")
        return
    end

    for line in f:lines() do
        if line ~= "" then
            local fields = split_by_sep(line, SEP)
            local rec = load_v2_line(fields)

            if rec then
                ensure_type_tables(rec.type)
                M.records[rec.id] = rec
                M.by_type_entity[rec.type][rec.entity_norm] = rec.id
                index_doc(rec.id, rec)
                if rec.id >= M.next_id then
                    M.next_id = rec.id + 1
                end
            end
        end
    end
    f:close()
end

return M
