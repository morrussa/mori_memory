local config = require("module.config")

local ingest_node = require("module.graph.nodes.ingest_node")
local context_node = require("module.graph.nodes.context_node")
local recall_node = require("module.graph.nodes.recall_node")
local agent_node = require("module.graph.nodes.agent_node")
local tools_node = require("module.graph.nodes.tools_node")
local finalize_node = require("module.graph.nodes.finalize_node")
local writeback_node = require("module.graph.nodes.writeback_node")
local persist_node = require("module.graph.nodes.persist_node")
local end_node = require("module.graph.nodes.end_node")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

function M.build()
    local nodes = {
        ingest_node = ingest_node,
        recall_node = recall_node,
        context_node = context_node,
        agent_node = agent_node,
        tools_node = tools_node,
        finalize_node = finalize_node,
        writeback_node = writeback_node,
        persist_node = persist_node,
        ["end"] = end_node,
    }

    local function next_node(current, state)
        local cfg = graph_cfg()
        local tool_loop_max = math.max(1, math.floor(tonumber(cfg.tool_loop_max) or 5))

        if current == "ingest_node" then
            return "recall_node"
        end
        if current == "recall_node" then
            return "context_node"
        end
        if current == "context_node" then
            return "agent_node"
        end
        if current == "agent_node" then
            local loop_count = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0
            local pending_calls = (((state or {}).agent_loop or {}).pending_tool_calls) or {}
            local stop_reason = tostring((((state or {}).agent_loop or {}).stop_reason) or "")

            if stop_reason ~= "" then
                return "finalize_node"
            end

            if #pending_calls > 0 then
                if loop_count >= tool_loop_max then
                    state.agent_loop = state.agent_loop or {}
                    state.agent_loop.pending_tool_calls = {}
                    state.agent_loop.stop_reason = "tool_loop_max_exceeded"
                    return "finalize_node"
                end
                return "tools_node"
            end

            return "finalize_node"
        end
        if current == "tools_node" then
            return "agent_node"
        end
        if current == "finalize_node" then
            return "writeback_node"
        end
        if current == "writeback_node" then
            return "persist_node"
        end
        if current == "persist_node" then
            return "end"
        end
        return "end"
    end

    return {
        start_node = "ingest_node",
        nodes = nodes,
        next_node = next_node,
    }
end

return M
