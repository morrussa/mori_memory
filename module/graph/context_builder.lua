local util = require("module.graph.util")
local config = require("module.config")
local context_manager = require("module.graph.context_manager")
local project_knowledge = require("module.graph.project_knowledge")
local topic = require("module.memory.topic")

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

local function unique_topic_list(items)
    local out = {}
    local seen = {}
    for _, item in ipairs(items or {}) do
        local key = util.trim(item or "")
        if key ~= "" and not seen[key] then
            seen[key] = true
            out[#out + 1] = key
        end
    end
    return out
end

local function build_topic_score_map(recall_state)
    local scores = {}
    local topic_rows = (((recall_state or {}).topic_debug or {}).topic_rows) or {}
    for _, row in ipairs(topic_rows or {}) do
        local anchor = util.trim((row or {}).anchor or "")
        if anchor ~= "" then
            local score = tonumber((row or {}).total_score)
            if score == nil then
                score = tonumber((row or {}).score) or tonumber((row or {}).route_score) or 0.0
            end
            scores[anchor] = score
        end
    end
    for anchor, score in pairs((recall_state or {}).candidate_topics or {}) do
        local key = util.trim(anchor or "")
        if key ~= "" and scores[key] == nil then
            scores[key] = tonumber(score) or 0.0
        end
    end
    return scores
end

local function topic_overlap_score(fp_a, fp_b)
    local weights_a = (fp_a or {}).weights or {}
    local weights_b = (fp_b or {}).weights or {}
    local total = 0.0
    for cid, wa in pairs(weights_a) do
        local wb = weights_b[cid]
        if wb ~= nil then
            local a = tonumber(wa) or 0.0
            local b = tonumber(wb) or 0.0
            total = total + math.min(a, b)
        end
    end
    return total
end

local function rank_topics(candidates, scores)
    local out = {}
    local seen = {}
    for _, anchor in ipairs(candidates or {}) do
        local key = util.trim(anchor or "")
        if key ~= "" and not seen[key] then
            seen[key] = true
            out[#out + 1] = {
                anchor = key,
                score = tonumber(scores[key]) or -1e9,
            }
        end
    end
    table.sort(out, function(a, b)
        if (a.score or 0.0) ~= (b.score or 0.0) then
            return (a.score or 0.0) > (b.score or 0.0)
        end
        return tostring(a.anchor) < tostring(b.anchor)
    end)
    return out
end

local function select_topics_with_saturation(semantic_ranked, adjacent_ranked, current_anchor, scores)
    local tg_cfg = (config.settings or {}).topic_graph or {}
    local total_budget = math.max(1, math.floor(tonumber(tg_cfg.max_return_topics) or 4))
    local overlap_threshold = tonumber(tg_cfg.context_overlap_threshold) or 0.55

    local selected = {}
    local selected_set = {}
    local fp_cache = {}

    local function get_fp(anchor)
        local key = util.trim(anchor or "")
        if key == "" then
            return nil
        end
        if fp_cache[key] == nil then
            fp_cache[key] = (topic.get_topic_fingerprint and topic.get_topic_fingerprint(key)) or {}
        end
        return fp_cache[key]
    end

    local function is_saturated(anchor)
        if overlap_threshold <= 0 then
            return false
        end
        local fp = get_fp(anchor)
        if not fp or not fp.weights then
            return false
        end
        for _, picked in ipairs(selected) do
            local fp_b = get_fp(picked)
            if fp_b and fp_b.weights then
                if topic_overlap_score(fp, fp_b) >= overlap_threshold then
                    return true
                end
            end
        end
        return false
    end

    local function try_pick(anchor)
        if util.trim(anchor or "") == "" then
            return false
        end
        if anchor == current_anchor then
            return false
        end
        if selected_set[anchor] then
            return false
        end
        if is_saturated(anchor) then
            return false
        end
        selected_set[anchor] = true
        selected[#selected + 1] = anchor
        return true
    end

    local semantic_selected = {}
    for _, row in ipairs(semantic_ranked or {}) do
        if #selected >= total_budget then
            break
        end
        if try_pick(row.anchor) then
            semantic_selected[#semantic_selected + 1] = row.anchor
        end
    end

    local adjacent_selected = {}
    for _, row in ipairs(adjacent_ranked or {}) do
        if #selected >= total_budget then
            break
        end
        if try_pick(row.anchor) then
            adjacent_selected[#adjacent_selected + 1] = row.anchor
        end
    end

    return semantic_selected, adjacent_selected
end

local function format_topic_entry(anchor, score, max_summary_chars)
    if util.trim(anchor or "") == "" then
        return ""
    end
    local fp = (topic.get_topic_fingerprint and topic.get_topic_fingerprint(anchor)) or {}
    local ts = tonumber((fp or {}).start)
    local ts_text = ts and tostring(ts) or "?"
    local line = string.format("t=%s | anchor=%s", ts_text, tostring(anchor))
    if score ~= nil then
        line = line .. string.format(" | score=%.3f", tonumber(score) or 0.0)
    end
    local summary = util.trim((fp or {}).summary or "")
    if summary ~= "" then
        summary = util.utf8_take(summary, max_summary_chars)
        line = line .. "\nsummary=" .. summary
    end
    return line
end

local function build_topic_blocks(state)
    local recall_state = ((state or {}).recall) or {}
    local current_anchor = util.trim(recall_state.topic_anchor or "")
    if current_anchor == "" then
        current_anchor = util.trim((((state or {}).episode or {}).current or {}).topic_anchor or "")
    end

    local scores = build_topic_score_map(recall_state)
    local semantic_topics = unique_topic_list(recall_state.predicted_topics or {})
    local adjacent_topics = unique_topic_list(recall_state.bridge_topics or {})
    local semantic_ranked = rank_topics(semantic_topics, scores)
    local adjacent_ranked = rank_topics(adjacent_topics, scores)
    local semantic_selected, adjacent_selected = select_topics_with_saturation(
        semantic_ranked,
        adjacent_ranked,
        current_anchor,
        scores
    )

    local semantic_lines = {}
    local max_summary_chars = 220
    for _, anchor in ipairs(semantic_selected) do
        local entry = format_topic_entry(anchor, scores[anchor], max_summary_chars)
        if entry ~= "" then
            semantic_lines[#semantic_lines + 1] = entry
        end
    end

    local adjacent_lines = {}
    for _, anchor in ipairs(adjacent_selected) do
        local entry = format_topic_entry(anchor, scores[anchor], max_summary_chars)
        if entry ~= "" then
            adjacent_lines[#adjacent_lines + 1] = entry
        end
    end

    local past_blocks = {}
    if #semantic_lines > 0 then
        past_blocks[#past_blocks + 1] = "[SemanticTopics]\n" .. table.concat(semantic_lines, "\n\n")
    end
    if #adjacent_lines > 0 then
        past_blocks[#past_blocks + 1] = "[AdjacentTopics]\n" .. table.concat(adjacent_lines, "\n\n")
    end

    local past_block = ""
    if #past_blocks > 0 then
        past_block = "[PastTopics]\n" .. table.concat(past_blocks, "\n\n")
    end

    local current_block = ""
    if current_anchor ~= "" then
        state.context = state.context or {}
        if util.trim(state.context.current_topic_anchor or "") ~= current_anchor
            or util.trim(state.context.current_topic_block or "") == "" then
            local entry = format_topic_entry(current_anchor, scores[current_anchor], max_summary_chars)
            if entry ~= "" then
                state.context.current_topic_anchor = current_anchor
                state.context.current_topic_block = "[CurrentTopic]\n" .. entry
            else
                state.context.current_topic_anchor = current_anchor
                state.context.current_topic_block = ""
            end
        end
        current_block = util.trim(state.context.current_topic_block or "")
    end

    return past_block, current_block
end

-- Static system prompt prefix: must be as stable as possible for KV-cache reuse.
local function compose_static_system_prompt(base_system_prompt, state)
    local lines = { tostring(base_system_prompt or "") }

    local pk_overview = project_knowledge.get_project_knowledge(state)
    if util.trim(pk_overview or "") ~= "" then
        lines[#lines + 1] = ""
        lines[#lines + 1] = pk_overview
    end

    return table.concat(lines, "\n\n")
end

-- Dynamic runtime context: changes frequently; keep it OUT of the system message.
local function compose_dynamic_context(state)
    local context = ((state or {}).context) or {}
    local lines = { "[DynamicContext]" }

    local past_topics, current_topic = build_topic_blocks(state)
    if util.trim(past_topics or "") ~= "" then
        lines[#lines + 1] = past_topics
    end
    if util.trim(current_topic or "") ~= "" then
        lines[#lines + 1] = current_topic
    end

    if util.trim((context or {}).memory_context or "") ~= "" then
        local memory_text = tostring(context.memory_context or "")
        if not memory_text:find("【相关主题】", 1, true)
            or (util.trim(past_topics or "") == "" and util.trim(current_topic or "") == "") then
            lines[#lines + 1] = "[MemoryContext]"
            lines[#lines + 1] = memory_text
        end
    end
    if util.trim((context or {}).task_context or "") ~= "" then
        lines[#lines + 1] = "[TaskContext]"
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

    lines[#lines + 1] = summarize_working_memory(state)

    if (context or {})._budget_warning then
        lines[#lines + 1] = "[System Note: " .. context._budget_warning .. "]"
    end

    return table.concat(lines, "\n\n")
end

local function build_messages(system_prompt, dynamic_context, user_input, history_pairs)
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

    local ctx_text = util.trim(dynamic_context or "")
    if ctx_text ~= "" then
        msgs[#msgs + 1] = { role = "user", content = ctx_text }
        msgs[#msgs + 1] = { role = "assistant", content = "Context noted." }
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

local function select_variant_that_fits(pair, variants, index_from_oldest, total_pairs, cfg, system_prompt, dynamic_context, user_input, kept)
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
            local candidate_messages = build_messages(system_prompt, dynamic_context, user_input, candidate)
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

    local system_prompt = compose_static_system_prompt(base_system_prompt, state)
    local dynamic_context = compose_dynamic_context(state)
    local user_input = tostring((((state or {}).input or {}).message) or "")

    local pairs = extract_history_pairs(conversation_history)
    local pair_variants = {}
    for i, pair in ipairs(pairs) do
        pair_variants[i] = build_pair_variants(pair, cfg)
    end

    local kept = {}
    local dropped = {}
    local variant_counts = { full = 0, slight = 0, heavy = 0, none = 0 }

    local messages = build_messages(system_prompt, dynamic_context, user_input, kept)
    local total_tokens = count_tokens(messages)

    for i = #pairs, 1, -1 do
        local selected_variant, variant_name, candidate_messages, candidate_tokens =
            select_variant_that_fits(pairs[i], pair_variants[i], i, #pairs, cfg, system_prompt, dynamic_context, user_input, kept)
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
