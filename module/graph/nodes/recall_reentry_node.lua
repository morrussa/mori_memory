local tool = require("module.tool")
local recall = require("module.memory.recall")
local util = require("module.graph.util")
local reentry = require("module.graph.recall_reentry")

local M = {}

local function append_secondary_context(existing, secondary)
    local base = util.trim(existing or "")
    local extra = util.trim(secondary or "")
    if extra == "" then
        return base
    end
    if base == "" then
        return "[SecondaryRecall]\n" .. extra
    end
    return table.concat({
        base,
        "[SecondaryRecall]",
        extra,
    }, "\n\n")
end

function M.run(state, _ctx)
    state.recall = state.recall or {}
    state.context = state.context or {}
    state.repair = state.repair or {}

    local current = reentry.ensure(state)
    if current.pending ~= true then
        return state
    end

    local query_text = reentry.build_query_text(state)
    local query_vec = tool.get_embedding_query(query_text)
    local secondary_context = recall.check_and_retrieve(query_text, query_vec, {
        read_only = (((state or {}).input or {}).read_only) == true,
        force = true,
        suppress = false,
        allowed_types = current.allowed_types,
        blocked_types = current.blocked_types,
        preferred_type = util.trim(current.preferred_type or "") ~= "" and current.preferred_type or nil,
        policy_decided = true,
    })

    secondary_context = util.trim(secondary_context)
    reentry.consume(state, secondary_context)

    if secondary_context ~= "" then
        state.context.memory_context = append_secondary_context(state.context.memory_context, secondary_context)
        local planner_note = string.format(
            "Secondary recall executed: kind=%s phase=%s reason=%s anchors=%d",
            tostring(current.kind or ""),
            tostring(current.phase or ""),
            tostring(current.reason or ""),
            #(current.anchors or {})
        )
        local existing_planner_context = util.trim(state.context.planner_context or "")
        if existing_planner_context ~= "" then
            state.context.planner_context = planner_note .. "\n" .. existing_planner_context
        else
            state.context.planner_context = planner_note
        end
        state.recall.triggered = true
        state.recall.context = state.context.memory_context
    else
        local note = string.format(
            "Secondary recall returned empty context: kind=%s phase=%s anchors=%d",
            tostring(current.kind or ""),
            tostring(current.phase or ""),
            #(current.anchors or {})
        )
        local existing_planner_context = util.trim(state.context.planner_context or "")
        if existing_planner_context ~= "" then
            state.context.planner_context = note .. "\n" .. existing_planner_context
        else
            state.context.planner_context = note
        end
    end

    state.repair.pending = false
    state.repair.retry_requested = false
    return state
end

return M
