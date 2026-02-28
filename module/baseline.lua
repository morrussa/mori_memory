local M = {}

local persistence = require("module.persistence")
local history = require("module.history")
local topic = require("module.topic")
local recall = require("module.recall")
local tool = require("module.tool")

local function csv_escape(v)
    local s = tostring(v or "")
    if s:find("[,\n\"]") then
        s = s:gsub("\"", "\"\"")
        return "\"" .. s .. "\""
    end
    return s
end

function M.freeze_snapshot(out_path)
    out_path = tostring(out_path or "memory/baseline_metrics.csv")
    recall.reset_metrics()

    local total_turns = tonumber(history.get_turn()) or 0
    for turn = 1, total_turns do
        local user_text = history.get_turn_text(turn, "user")
        if user_text and user_text ~= "" then
            local qv = tool.get_embedding_query(user_text)
            local current_info = topic.get_topic_for_turn and topic.get_topic_for_turn(turn) or nil
            local current_anchor = topic.get_topic_anchor and topic.get_topic_anchor(turn) or nil
            recall.check_and_retrieve(user_text, qv, {
                current_turn = turn,
                current_info = current_info,
                current_anchor = current_anchor,
                freeze_mode = true,
            })
        end
    end

    local snap = recall.get_metrics_snapshot()
    local ok, err = persistence.write_atomic(out_path, "w", function(f)
        local header = "query_turns,hits_same,empty_query_rate,target_recall_recent,p95_sim_ops\n"
        local line = table.concat({
            csv_escape(snap.query_turns),
            csv_escape(snap.hits_same),
            csv_escape(string.format("%.8f", tonumber(snap.empty_query_rate) or 0.0)),
            csv_escape(string.format("%.8f", tonumber(snap.target_recall_recent) or 0.0)),
            csv_escape(string.format("%.8f", tonumber(snap.p95_sim_ops) or 0.0)),
        }, ",") .. "\n"

        local h_ok, h_err = f:write(header)
        if not h_ok then return false, h_err end
        local w_ok, w_err = f:write(line)
        if not w_ok then return false, w_err end
        return true
    end)
    if not ok then
        return false, err
    end

    print(string.format(
        "[Baseline] 已冻结基线到 %s | query_turns=%d hits_same=%d empty_query_rate=%.4f target_recall_recent=%.4f p95_sim_ops=%.2f",
        out_path,
        tonumber(snap.query_turns) or 0,
        tonumber(snap.hits_same) or 0,
        tonumber(snap.empty_query_rate) or 0.0,
        tonumber(snap.target_recall_recent) or 0.0,
        tonumber(snap.p95_sim_ops) or 0.0
    ))
    return true, snap
end

return M
