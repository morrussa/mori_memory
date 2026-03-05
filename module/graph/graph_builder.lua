local config = require("module.config")

local ingest_node = require("module.graph.nodes.ingest_node")
local context_node = require("module.graph.nodes.context_node")
local router_node = require("module.graph.nodes.router_node")
local recall_node = require("module.graph.nodes.recall_node")
local planner_node = require("module.graph.nodes.planner_node")
local tool_exec_node = require("module.graph.nodes.tool_exec_node")
local repair_node = require("module.graph.nodes.repair_node")
local responder_node = require("module.graph.nodes.responder_node")
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
        context_node = context_node,
        router_node = router_node,
        recall_node = recall_node,
        planner_node = planner_node,
        tool_exec_node = tool_exec_node,
        repair_node = repair_node,
        responder_node = responder_node,
        writeback_node = writeback_node,
        persist_node = persist_node,
        ["end"] = end_node,
    }

    local function next_node(current, state)
        local cfg = graph_cfg()
        local tool_loop_max = math.max(1, math.floor(tonumber(cfg.tool_loop_max) or 5))

        if current == "ingest_node" then
            return "context_node"
        end
        if current == "context_node" then
            return "router_node"
        end
        if current == "router_node" then
            local route = tostring((((state or {}).router_decision or {}).route) or "respond")
            if route == "tool_loop" then
                return "recall_node"
            end
            return "responder_node"
        end
        if current == "recall_node" then
            return "planner_node"
        end
        if current == "planner_node" then
            local loop_count = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0
            local calls = (((state or {}).planner or {}).tool_calls) or {}
            if #calls <= 0 then
                return "responder_node"
            end
            if loop_count >= tool_loop_max then
                return "responder_node"
            end
            return "tool_exec_node"
        end
        if current == "tool_exec_node" then
            local failed = tonumber((((state or {}).tool_exec or {}).failed) or 0) or 0
            local loop_count = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0
            local repair_attempts = tonumber((((state or {}).repair or {}).attempts) or 0) or 0
            local repair_max = tonumber((((state or {}).repair or {}).max_attempts) or 2) or 2
            if failed > 0 then
                if repair_attempts < repair_max then
                    return "repair_node"
                end
                return "responder_node"
            end
            if loop_count >= tool_loop_max then
                return "responder_node"
            end
            return "planner_node"
        end
        if current == "repair_node" then
            local loop_count = tonumber((((state or {}).tool_exec or {}).loop_count) or 0) or 0
            if loop_count >= tool_loop_max then
                return "responder_node"
            end
            local failed = tonumber((((state or {}).tool_exec or {}).failed) or 0) or 0
            if failed > 0 then
                local calls = (((state or {}).planner or {}).tool_calls) or {}
                if #calls > 0 then
                    return "tool_exec_node"
                end
                return "responder_node"
            end
            -- If failures were repaired to zero, continue planning.
            return "planner_node"
        end
        if current == "responder_node" then
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
