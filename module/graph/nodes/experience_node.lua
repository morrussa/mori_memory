local experience = require("module.experience")
local util = require("module.graph.util")

local M = {}

local HINT_LABELS = {
    "优先",
    "次优",
    "兜底",
}

local function build_hint_text(items)
    local lines = {}

    for idx, item in ipairs(items or {}) do
        local label = HINT_LABELS[idx]
        if not label then
            break
        end

        local text = util.trim((item or {}).description or "")
        if text ~= "" then
            lines[#lines + 1] = string.format("%s：%s", label, text)
        end
    end

    local joined = table.concat(lines, "\n")
    if joined == "" then
        return ""
    end

    local clipped = util.utf8_take(joined, 600)
    if clipped ~= joined then
        clipped = clipped .. "..."
    end
    return clipped
end

function M.run(state, _ctx)
    experience.init()

    local user_input = tostring((((state or {}).input or {}).message) or "")
    local read_only = (((state or {}).input or {}).read_only) == true
    local uploads = (((state or {}).uploads) or {})
    local features = experience.retriever.extract_query_features(user_input)
    local context_signature = {
        domain = features.domain or nil,
        language = features.language or nil,
        read_only = read_only,
        has_uploads = #uploads > 0,
    }

    local items, strategy = experience.retrieve({
        text = user_input,
        task_type = features.task_type,
        domain = features.domain,
        language = features.language,
        context_signature = context_signature,
    }, {
        type = "success",
        limit = 3,
        context_signature = context_signature,
        domain = features.domain,
        language = features.language,
        read_only = read_only,
    })

    local ids = {}
    for _, item in ipairs(items or {}) do
        if item and item.id then
            ids[#ids + 1] = item.id
        end
    end

    local hints = build_hint_text(items)

    state.experience = state.experience or {}
    state.experience.query = {
        task_type = features.task_type,
        domain = features.domain,
        language = features.language,
        context_signature = context_signature,
    }
    state.experience.retrieved = {
        items = items or {},
        ids = ids,
        strategy = tostring((strategy or {}).strategy or ""),
    }
    state.experience.hints = hints
    state.experience.feedback = state.experience.feedback or { effective_ids = {} }
    state.experience.writeback = state.experience.writeback or { written = false }

    state.context = state.context or {}
    state.context.experience_context = hints
    return state
end

return M
