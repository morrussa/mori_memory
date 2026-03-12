local util = require("module.graph.util")
local config = require("module.config")
local context_manager = require("module.graph.context_manager")
local project_knowledge = require("module.graph.project_knowledge")

local M = {}

local function default_graph_cfg()
    return ((config.settings or {}).graph or {})
end

local function ctx_cfg()
    return (((default_graph_cfg() or {}).context_manager) or {})
end

local function count_tokens(messages)
    local n = py_pipeline:count_chat_tokens(messages)
    return math.max(0, math.floor(tonumber(n) or 0))
end

local function extract_history_pairs(conversation_history)
    local out = {}
    if type(conversation_history) ~= "table" then
        return out
    end

    local idx = 2
    local turn = 0
    while idx <= #conversation_history do
        local user_msg = conversation_history[idx]
        local assistant_msg = conversation_history[idx + 1]
        if type(user_msg) == "table" and tostring(user_msg.role or "") == "user" then
            turn = turn + 1
            local pair = {
                turn = turn,
                user = tostring(user_msg.content or ""),
                assistant = "",
            }
            if type(assistant_msg) == "table" and tostring(assistant_msg.role or "") == "assistant" then
                pair.assistant = tostring(assistant_msg.content or "")
                idx = idx + 2
            else
                idx = idx + 1
            end
            out[#out + 1] = pair
        else
            idx = idx + 1
        end
    end

    return out
end

local function summarize_working_memory(state)
    local active_task = ((((state or {}).session or {}).active_task) or {})
    local task_decision = ((((state or {}).task or {}).decision) or {})
    local memory = ((state or {}).working_memory) or {}
    local read_count = 0
    local written_count = 0
    for _, _ in pairs(memory.files_read_set or {}) do
        read_count = read_count + 1
    end
    for _, _ in pairs(memory.files_written_set or {}) do
        written_count = written_count + 1
    end

    local lines = {
        "[ActiveTask]",
        string.format("task_id=%s", tostring(active_task.task_id or "")),
        string.format("goal=%s", tostring(active_task.goal or "")),
        string.format("status=%s", tostring(active_task.status or "")),
        string.format("profile=%s", tostring(active_task.profile or "")),
    }
    if util.trim(task_decision.kind or "") ~= "" then
        lines[#lines + 1] = string.format("decision=%s", tostring(task_decision.kind or ""))
    end
    if util.trim(active_task.carryover_summary or "") ~= "" then
        lines[#lines + 1] = string.format("carryover=%s", tostring(active_task.carryover_summary))
    end
    local contract = ((((state or {}).task or {}).contract) or active_task.contract or {})
    if util.trim((contract or {}).goal or "") ~= "" then
        lines[#lines + 1] = "[TaskContract]"
        lines[#lines + 1] = string.format("goal=%s", tostring((contract or {}).goal or ""))
        for _, item in ipairs((contract or {}).deliverables or {}) do
            lines[#lines + 1] = "deliverable=" .. tostring(item)
        end
        for _, item in ipairs((contract or {}).acceptance_criteria or {}) do
            lines[#lines + 1] = "acceptance=" .. tostring(item)
        end
        for _, item in ipairs((contract or {}).non_goals or {}) do
            lines[#lines + 1] = "non_goal=" .. tostring(item)
        end
    end
    lines[#lines + 1] = "[WorkingMemory]"
    lines[#lines + 1] = string.format("current_plan=%s", tostring(memory.current_plan or ""))
    lines[#lines + 1] = string.format("plan_step_index=%s", tostring(memory.plan_step_index or 0))
    lines[#lines + 1] = string.format("files_read=%d", read_count)
    lines[#lines + 1] = string.format("files_written=%d", written_count)
    if util.trim(memory.last_tool_batch_summary or "") ~= "" then
        lines[#lines + 1] = "last_tool_batch:"
        lines[#lines + 1] = util.utf8_take(tostring(memory.last_tool_batch_summary), 800)
    end
    if util.trim(memory.last_repair_error or "") ~= "" then
        lines[#lines + 1] = string.format("last_repair_error=%s", tostring(memory.last_repair_error))
    end
    return table.concat(lines, "\n")
end

local function compose_system_prompt(base_system_prompt, state)
    local context = ((state or {}).context) or {}
    local lines = { tostring(base_system_prompt or "") }

    local pk_overview = project_knowledge.get_project_knowledge(state)
    if util.trim(pk_overview or "") ~= "" then
        lines[#lines + 1] = ""
        lines[#lines + 1] = pk_overview
    end

    lines[#lines + 1] = summarize_working_memory(state)

    if util.trim((context or {}).memory_context or "") ~= "" then
        lines[#lines + 1] = "[MemoryContext]"
        lines[#lines + 1] = tostring(context.memory_context)
    end
    if util.trim((context or {}).task_context or "") ~= "" then
        lines[#lines + 1] = tostring(context.task_context)
    end
    local recent_episode_summary = util.trim((((state or {}).episode or {}).recent or {}).summary or "")
    if recent_episode_summary ~= "" then
        lines[#lines + 1] = "[RecentEpisodes]"
        lines[#lines + 1] = tostring(recent_episode_summary)
    end
    if util.trim((context or {}).tool_context or "") ~= "" then
        lines[#lines + 1] = "[ToolContext]"
        lines[#lines + 1] = tostring(context.tool_context)
    end
    if util.trim((context or {}).planner_context or "") ~= "" then
        lines[#lines + 1] = "[PlannerContext]"
        lines[#lines + 1] = tostring(context.planner_context)
    end

    if (context or {})._budget_warning then
        lines[#lines + 1] = ""
        lines[#lines + 1] = "[System Note: " .. context._budget_warning .. "]"
    end

    return table.concat(lines, "\n\n")
end

local function build_messages(system_prompt, user_input, history_pairs)
    local msgs = {
        { role = "system", content = tostring(system_prompt or "") },
    }
    for _, pair in ipairs(history_pairs or {}) do
        if util.trim(pair.user or "") ~= "" then
            msgs[#msgs + 1] = { role = "user", content = tostring(pair.user or "") }
        end
        if util.trim(pair.assistant or "") ~= "" then
            msgs[#msgs + 1] = { role = "assistant", content = tostring(pair.assistant or "") }
        end
    end
    msgs[#msgs + 1] = { role = "user", content = tostring(user_input or "") }
    return msgs
end

local function compress_text(text, max_chars)
    local raw = util.trim(text or "")
    max_chars = math.max(0, math.floor(tonumber(max_chars) or 0))
    if raw == "" or max_chars <= 0 then
        return ""
    end
    local snippet = util.utf8_take(raw, max_chars)
    if #snippet < #raw then
        return snippet .. "..."
    end
    return snippet
end

local function pair_weight(weights, variant)
    local w = tonumber((weights or {})[variant])
    if w == nil then
        return 0
    end
    return math.max(0, w)
end

local function make_variant_pair(pair, variant, cfg)
    if variant == "none" then
        return nil
    end

    local user_text = util.trim(pair.user or "")
    local assistant_text = util.trim(pair.assistant or "")
    local total_chars = #user_text + #assistant_text
    if total_chars <= 0 then
        return {
            turn = pair.turn,
            user = "",
            assistant = "",
            _variant = variant,
        }
    end

    if variant == "full" then
        return {
            turn = pair.turn,
            user = user_text,
            assistant = assistant_text,
            _variant = variant,
        }
    end

    local ratio = tonumber(cfg.history_compress_ratio_slight) or 0.65
    if variant == "heavy" then
        ratio = tonumber(cfg.history_compress_ratio_heavy) or 0.30
    end
    ratio = math.max(0.05, math.min(0.95, ratio))
    local budget = math.max(40, math.floor(total_chars * ratio))
    local user_budget = math.max(20, math.floor(budget * 0.45))
    local assistant_budget = math.max(20, budget - user_budget)

    return {
        turn = pair.turn,
        user = compress_text(user_text, user_budget),
        assistant = compress_text(assistant_text, assistant_budget),
        _variant = variant,
    }
end

local function build_pair_variants(pair, cfg)
    return {
        full = make_variant_pair(pair, "full", cfg),
        slight = make_variant_pair(pair, "slight", cfg),
        heavy = make_variant_pair(pair, "heavy", cfg),
        none = nil,
    }
end

local function select_variant_that_fits(pair, variants, index_from_oldest, total_pairs, cfg, system_prompt, user_input, kept)
    local weights = cfg.history_variant_weights or {}
    local decay = tonumber(cfg.history_recency_decay) or 0.90
    decay = math.max(0.01, math.min(1.0, decay))

    local recency_index = math.max(0, total_pairs - index_from_oldest)
    local recency_factor = decay ^ recency_index

    local ranked = {
        { name = "full", score = pair_weight(weights, "full") * recency_factor },
        { name = "slight", score = pair_weight(weights, "slight") * recency_factor },
        { name = "heavy", score = pair_weight(weights, "heavy") * recency_factor },
        { name = "none", score = pair_weight(weights, "none") },
    }

    table.sort(ranked, function(a, b)
        if a.score == b.score then
            local order = { full = 1, slight = 2, heavy = 3, none = 4 }
            return (order[a.name] or 99) < (order[b.name] or 99)
        end
        return a.score > b.score
    end)

    for _, item in ipairs(ranked) do
        if item.name == "none" then
            return nil, "none", nil, nil
        end
        local selected_variant = variants[item.name]
        if selected_variant ~= nil then
            local candidate = { selected_variant }
            for k = 1, #kept do
                candidate[#candidate + 1] = kept[k]
            end
            local candidate_messages = build_messages(system_prompt, user_input, candidate)
            local candidate_tokens = count_tokens(candidate_messages)
            return selected_variant, item.name, candidate_messages, candidate_tokens
        end
    end

    return nil, "none", nil, nil
end

function M.build_chat_messages(state)
    local graph_cfg = default_graph_cfg()
    local cfg = ctx_cfg()
    local token_budget = math.max(256, math.floor(tonumber(graph_cfg.input_token_budget) or 12000))

    local conversation_history = (((state or {}).messages or {}).conversation_history) or {}
    local base_system_prompt = (((state or {}).messages or {}).system_prompt) or ""
    if util.trim(base_system_prompt) == "" and type(conversation_history[1]) == "table" then
        if tostring(conversation_history[1].role or "") == "system" then
            base_system_prompt = tostring(conversation_history[1].content or "")
        end
    end

    local system_prompt = compose_system_prompt(base_system_prompt, state)
    local user_input = tostring((((state or {}).input or {}).message) or "")

    local pairs = extract_history_pairs(conversation_history)
    local pair_variants = {}
    for i, pair in ipairs(pairs) do
        pair_variants[i] = build_pair_variants(pair, cfg)
    end

    local kept = {}
    local dropped = {}
    local variant_counts = { full = 0, slight = 0, heavy = 0, none = 0 }

    local messages = build_messages(system_prompt, user_input, kept)
    local total_tokens = count_tokens(messages)

    for i = #pairs, 1, -1 do
        local selected_variant, variant_name, candidate_messages, candidate_tokens =
            select_variant_that_fits(pairs[i], pair_variants[i], i, #pairs, cfg, system_prompt, user_input, kept)
        if selected_variant ~= nil and candidate_messages and candidate_tokens and candidate_tokens <= token_budget then
            local next_kept = { selected_variant }
            for k = 1, #kept do
                next_kept[#next_kept + 1] = kept[k]
            end
            kept = next_kept
            messages = candidate_messages
            total_tokens = candidate_tokens
            variant_counts[variant_name] = (variant_counts[variant_name] or 0) + 1
        else
            dropped[#dropped + 1] = pairs[i]
            variant_counts.none = (variant_counts.none or 0) + 1
        end
    end

    local optimized_messages, opt_stats = context_manager.optimize_runtime_messages(messages, 4000)

    if total_tokens > token_budget then
        for i, msg in ipairs(optimized_messages) do
            if msg.role == "tool" and #(msg.content or "") > 2000 then
                optimized_messages[i].content = util.utf8_take(msg.content, 2000)
                    .. "\n[Truncated for context budget]"
            end
        end
        total_tokens = count_tokens(optimized_messages)
    end

    if total_tokens > token_budget then
        print(string.format("[GraphContext] Warning: token budget exceeded total=%d budget=%d", total_tokens, token_budget))
    end

    return optimized_messages, {
        token_budget = token_budget,
        total_tokens = total_tokens,
        kept_pairs = #kept,
        dropped_pairs = #dropped,
        compressed_pairs = (variant_counts.slight or 0) + (variant_counts.heavy or 0),
        history_variant_counts = variant_counts,
        optimized_messages = opt_stats.truncated_count,
    }
end

return M
