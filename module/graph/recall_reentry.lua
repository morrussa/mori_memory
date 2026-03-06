local util = require("module.graph.util")
local config = require("module.config")
local memory = require("module.memory.store")

local M = {}

local PRESETS = {
    thread_continuity = {
        preferred_type = "Project",
        allowed_types = { "Project", "Artifact", "Status", "Fact" },
        blocked_types = { "User", "Person", "Preference" },
    },
    prior_decisions = {
        preferred_type = "Decision",
        allowed_types = { "Decision", "Constraint", "Status", "Project", "Fact" },
        blocked_types = { "User", "Person", "Preference" },
    },
    artifact_history = {
        preferred_type = "Artifact",
        allowed_types = { "Artifact", "Project", "Decision", "Constraint", "Status" },
        blocked_types = { "User", "Person", "Preference" },
    },
    constraints = {
        preferred_type = "Constraint",
        allowed_types = { "Constraint", "Decision", "Status", "Project", "Fact" },
        blocked_types = { "User", "Person", "Preference" },
    },
    status = {
        preferred_type = "Status",
        allowed_types = { "Status", "Decision", "Project", "Artifact", "Fact" },
        blocked_types = { "User", "Person", "Preference" },
    },
}

local READ_TOOLS = {
    list_files = true,
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
    code_outline = true,
    project_structure = true,
    code_symbols = true,
}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function reentry_cfg()
    return ((graph_cfg().recall or {}).reentry or {})
end

local function clone_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function trim_anchor(text)
    return util.utf8_take(util.trim(text or ""), 120)
end

local function normalize_type_list(raw, fallback)
    local filter = memory.build_type_filter(raw)
    local out = {}
    local seen = {}

    if filter then
        for _, type_name in ipairs(memory.get_allowed_type_names()) do
            if filter[type_name] and not seen[type_name] then
                seen[type_name] = true
                out[#out + 1] = type_name
            end
        end
    end

    if #out <= 0 then
        for _, type_name in ipairs(fallback or {}) do
            local canonical = memory.match_type_name(type_name) or memory.normalize_type_name(type_name)
            if canonical and not seen[canonical] then
                seen[canonical] = true
                out[#out + 1] = canonical
            end
        end
    end

    return out
end

local function collect_common_anchors(state)
    local out = {}
    local seen = {}

    local function push(text)
        local anchor = trim_anchor(text)
        if anchor ~= "" and not seen[anchor] then
            seen[anchor] = true
            out[#out + 1] = anchor
        end
    end

    push((((state or {}).input or {}).message) or "")
    push((((((state or {}).session or {}).active_task) or {}).goal) or "")
    push(((((state or {}).working_memory) or {}).current_plan) or "")
    push(((((state or {}).working_memory) or {}).last_tool_batch_summary) or "")
    push((((((state or {}).episode or {}).recent) or {}).summary) or "")

    return out
end

local function extract_path_anchor(args)
    args = type(args) == "table" and args or {}
    local candidates = {
        args.path,
        args.prefix,
        args.target,
        args.name,
        args.symbol,
    }
    for _, raw in ipairs(candidates) do
        local path = trim_anchor(raw)
        if path ~= "" then
            local basename = tostring(path):match("([^/]+)$")
            if trim_anchor(basename) ~= "" then
                return trim_anchor(basename)
            end
            return path
        end
    end
    return ""
end

local function continuity_eligible(state)
    local kind = util.trim((((state or {}).task or {}).decision or {}).kind or "")
    local mode = util.trim((((state or {}).recall or {}).policy or {}).mode or "")
    return kind == "same_task_step"
        or kind == "same_task_refine"
        or kind == "meta_turn"
        or mode == "project"
        or mode == "coding"
        or mode == "meta"
end

function M.ensure(state)
    state.recall = state.recall or {}
    local reentry = state.recall.reentry
    if type(reentry) ~= "table" then
        reentry = {}
        state.recall.reentry = reentry
    end

    local cfg = reentry_cfg()
    reentry.pending = reentry.pending == true
    reentry.used = tonumber(reentry.used) or 0
    reentry.max_per_turn = math.max(0, math.floor(tonumber(reentry.max_per_turn) or tonumber(cfg.max_per_turn) or 1))
    reentry.kind = tostring(reentry.kind or "")
    reentry.phase = tostring(reentry.phase or "")
    reentry.reason = tostring(reentry.reason or "")
    reentry.source_error = tostring(reentry.source_error or "")
    reentry.requested_by = tostring(reentry.requested_by or "")
    reentry.preferred_type = tostring(reentry.preferred_type or "")
    reentry.allowed_types = type(reentry.allowed_types) == "table" and reentry.allowed_types or {}
    reentry.blocked_types = type(reentry.blocked_types) == "table" and reentry.blocked_types or {}
    reentry.anchors = type(reentry.anchors) == "table" and reentry.anchors or {}
    reentry.context = tostring(reentry.context or "")
    reentry.last_kind = tostring(reentry.last_kind or "")
    reentry.last_phase = tostring(reentry.last_phase or "")
    reentry.last_source_error = tostring(reentry.last_source_error or "")
    reentry.last_reason = tostring(reentry.last_reason or "")
    reentry.last_anchor_count = tonumber(reentry.last_anchor_count) or 0
    return reentry
end

function M.is_enabled()
    return reentry_cfg().enabled ~= false
end

function M.can_request(state)
    local reentry = M.ensure(state)
    return M.is_enabled()
        and continuity_eligible(state)
        and reentry.pending ~= true
        and reentry.used < reentry.max_per_turn
end

local function normalize_request(req)
    req = type(req) == "table" and req or {}
    local kind = util.trim(req.kind or "")
    local preset = PRESETS[kind]
    if not preset then
        return nil, "invalid_kind"
    end

    local anchors = {}
    local seen = {}
    for _, raw in ipairs(req.anchors or {}) do
        local anchor = trim_anchor(raw)
        if anchor ~= "" and not seen[anchor] then
            seen[anchor] = true
            anchors[#anchors + 1] = anchor
        end
    end

    local min_anchor_count = math.max(1, math.floor(tonumber(reentry_cfg().min_anchor_count) or 3))
    if #anchors < min_anchor_count then
        return nil, "insufficient_anchors"
    end

    return {
        kind = kind,
        phase = util.trim(req.phase or ""),
        reason = trim_anchor(req.reason or ""),
        source_error = trim_anchor(req.source_error or ""),
        requested_by = util.trim(req.requested_by or ""),
        anchors = anchors,
        preferred_type = memory.match_type_name(req.preferred_type) or preset.preferred_type,
        allowed_types = normalize_type_list(req.allowed_types, preset.allowed_types),
        blocked_types = normalize_type_list(req.blocked_types, preset.blocked_types),
    }
end

function M.request(state, req)
    if not M.can_request(state) then
        return false, "reentry_unavailable"
    end

    local reentry = M.ensure(state)
    local normalized, err = normalize_request(req)
    if not normalized then
        return false, err
    end

    if reentry.last_kind == normalized.kind
        and reentry.last_phase == normalized.phase
        and reentry.last_source_error == normalized.source_error then
        return false, "duplicate_request"
    end

    reentry.pending = true
    reentry.kind = normalized.kind
    reentry.phase = normalized.phase
    reentry.reason = normalized.reason
    reentry.source_error = normalized.source_error
    reentry.requested_by = normalized.requested_by
    reentry.preferred_type = tostring(normalized.preferred_type or "")
    reentry.allowed_types = normalized.allowed_types
    reentry.blocked_types = normalized.blocked_types
    reentry.anchors = normalized.anchors
    reentry.context = ""
    return true
end

function M.clear_pending(state)
    local reentry = M.ensure(state)
    reentry.pending = false
    reentry.kind = ""
    reentry.phase = ""
    reentry.reason = ""
    reentry.source_error = ""
    reentry.requested_by = ""
    reentry.preferred_type = ""
    reentry.allowed_types = {}
    reentry.blocked_types = {}
    reentry.anchors = {}
    reentry.context = ""
    return reentry
end

function M.consume(state, context_text)
    local reentry = M.ensure(state)
    local context = tostring(context_text or "")
    reentry.pending = false
    reentry.used = reentry.used + 1
    reentry.context = context
    reentry.last_kind = reentry.kind
    reentry.last_phase = reentry.phase
    reentry.last_source_error = reentry.source_error
    reentry.last_reason = reentry.reason
    reentry.last_anchor_count = #(reentry.anchors or {})
    return reentry
end

function M.build_query_text(state)
    local reentry = M.ensure(state)
    local lines = {
        "secondary recall",
        "kind=" .. tostring(reentry.kind or ""),
    }
    for _, anchor in ipairs(reentry.anchors or {}) do
        lines[#lines + 1] = anchor
    end
    return table.concat(lines, "\n")
end

function M.request_from_tool_exec(state)
    if not M.can_request(state) then
        return false, "reentry_unavailable"
    end

    local results = ((((state or {}).tool_exec) or {}).results) or {}
    if type(results) ~= "table" or #results <= 0 then
        return false, "no_tool_results"
    end

    for _, row in ipairs(results) do
        if row.ok ~= true then
            local tool_name = util.trim((row or {}).tool or "")
            local err_text = tostring((row or {}).error or "")
            local err_lower = err_text:lower()
            local args = type((row or {}).args) == "table" and row.args or {}
            local artifact_anchor = extract_path_anchor(args)
            local anchors = collect_common_anchors(state)
            if artifact_anchor ~= "" then
                table.insert(anchors, 1, artifact_anchor)
            end

            local is_missing_path = err_lower:find("missing `path`", 1, true) ~= nil
                or err_lower:find("missing path", 1, true) ~= nil
            local is_not_found = err_lower:find("not found", 1, true) ~= nil
                or err_lower:find("no such file", 1, true) ~= nil
                or err_lower:find("enoent", 1, true) ~= nil
            local is_unknown_ref = err_lower:find("unknown", 1, true) ~= nil
                or err_lower:find("ambiguous", 1, true) ~= nil
                or err_lower:find("missing target", 1, true) ~= nil

            if READ_TOOLS[tool_name] and artifact_anchor ~= "" and (is_missing_path or is_not_found) then
                return M.request(state, {
                    kind = "artifact_history",
                    phase = "tool_exec",
                    requested_by = "tool_exec",
                    reason = "read tool failed and artifact continuity is missing",
                    source_error = err_text,
                    anchors = anchors,
                })
            end

            if READ_TOOLS[tool_name] and is_unknown_ref then
                return M.request(state, {
                    kind = "thread_continuity",
                    phase = "tool_exec",
                    requested_by = "tool_exec",
                    reason = "read tool failed because the target reference is unresolved",
                    source_error = err_text,
                    anchors = anchors,
                })
            end
        end
    end

    return false, "no_memory_gap_failure"
end

function M.request_from_repair(state)
    if not M.can_request(state) then
        return false, "reentry_unavailable"
    end

    local last_error = util.trim((((state or {}).repair) or {}).last_error or "")
    if last_error == "" then
        return false, "no_repair_error"
    end

    local mode = util.trim((((state or {}).recall or {}).policy or {}).mode or "")
    local anchors = collect_common_anchors(state)

    if last_error == "missing_terminal_signal" then
        local kind = (mode == "meta") and "status" or "prior_decisions"
        return M.request(state, {
            kind = kind,
            phase = "repair",
            requested_by = "repair",
            reason = "planner failed to produce a complete next step and may be missing historical state",
            source_error = last_error,
            anchors = anchors,
        })
    end

    return false, "repair_error_not_memory_gap"
end

return M
