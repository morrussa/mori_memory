
-- module/experience/init.lua
-- Experience模块入口点

local M = {}
local util = require("module.graph.util")

-- 子模块
M.store_v2 = require("module.experience.store_v2")
M.retriever = require("module.experience.retriever")
M.adaptive = require("module.experience.adaptive")
M.run_builder = require("module.experience.run_builder")
M.policy = require("module.experience.policy")

M._initialized = false

-- ==================== 初始化 ====================

function M.init()
    if M._initialized == true then
        return
    end

    M.adaptive.load()
    M.store_v2.init()
    M._initialized = true

    print("[Experience] Module initialized")
end

-- 保存状态（退出时调用）
function M.finalize()
    if M._initialized ~= true then
        return true
    end
    M.adaptive.save_to_disk()
    M.store_v2.save()
    print("[Experience] Module finalized")
    return true
end

-- ==================== 核心API ====================

function M.observe_v2(observation)
    return M.store_v2.observe_v2(observation)
end

function M.retrieve_v2(query, options)
    return M.store_v2.retrieve_v2(query, options)
end

-- 记录效用反馈
function M.record_utility_feedback(retrieved_ids, effective_ids)
    M.retriever.record_utility_feedback(retrieved_ids, effective_ids)
end

function M.match_behavior_to_candidate(observation, candidates_or_ids)
    return M.store_v2.match_behavior_to_candidate(observation, candidates_or_ids)
end

function M.compose_planner_prior(candidates, recommendation, risks)
    local lines = { "[ExperienceCandidatesV2]" }

    if type(candidates) == "table" and #candidates > 0 then
        for idx, row in ipairs(candidates) do
            if idx > 5 then
                break
            end
            local patch_summary = M.policy.summarize_runtime_policy(((row or {}).macro_patch) or {})
            if util.trim(patch_summary or "") == "" then
                patch_summary = "policy=auto"
            end
            lines[#lines + 1] = string.format(
                "%d. id=%s score=%.3f support=%d patch=%s",
                idx,
                tostring((row or {}).id or ""),
                tonumber((row or {}).score) or 0,
                tonumber((row or {}).support_count) or 0,
                patch_summary:gsub("\n", "; ")
            )
            local flags = ((row or {}).risk_flags) or {}
            if type(flags) == "table" and #flags > 0 then
                lines[#lines + 1] = "   risk=" .. table.concat(flags, ",")
            end
        end
    else
        lines[#lines + 1] = "none"
    end

    local rec_id = util.trim((((recommendation or {}).id) or ""))
    if rec_id ~= "" then
        lines[#lines + 1] = string.format(
            "recommended=id=%s confidence=%.3f reason=%s",
            tostring((recommendation or {}).id or ""),
            tonumber((recommendation or {}).confidence) or 0,
            tostring((recommendation or {}).reason or "")
        )
    else
        lines[#lines + 1] = "recommended=none"
    end

    local warnings = {}
    for _, item in ipairs(risks or {}) do
        local text = util.trim(item or "")
        if text ~= "" then
            warnings[#warnings + 1] = text
        end
    end
    if #warnings > 0 then
        lines[#lines + 1] = "[FailureWarnings]"
        for _, row in ipairs(warnings) do
            lines[#lines + 1] = "- " .. row
        end
    end
    return table.concat(lines, "\n")
end

function M.save_all()
    local ok_v2, err_v2 = M.store_v2.save()
    if not ok_v2 then
        return false, err_v2
    end
    return true
end

-- ==================== 统计信息 ====================

function M.get_stats()
    local store_v2_stats = M.store_v2.get_stats()
    local adaptive_stats = M.adaptive.get_stats()
    return {
        store_v2 = store_v2_stats,
        adaptive = adaptive_stats
    }
end

return M
