local util = require("module.graph.util")
local config = require("module.config")
local tool_registry = require("module.graph.tool_registry_v2")
local context_manager = require("module.graph.context_manager")

local M = {}

-- 最大tool消息累积限制
local MAX_TOOL_MESSAGES = 50
-- 最大runtime_messages总长度
local MAX_RUNTIME_MESSAGES_CHARS = 100000

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

-- 检查对象是否可调用（支持 Lua function 和 Lupa userdata）
local function is_callable(obj)
    if type(obj) == "function" then
        return true
    end
    if obj == nil then
        return false
    end

    local obj_type = type(obj)
    local ok, callable = pcall(function()
        local mt = getmetatable(obj)
        if obj_type == "table" then
            local direct = rawget(obj, "__call")
            if direct ~= nil then
                return direct
            end
        else
            local direct = obj.__call
            if direct ~= nil then
                return direct
            end
        end
        return mt and mt.__call or nil
    end)
    if ok and callable ~= nil then
        return true
    end

    if obj_type == "userdata" then
        local ok2, _ = pcall(function()
            return obj({})
        end)
        return ok2
    end
    return false
end

-- 发送流式事件
local function emit_stream(state, event_name, payload)
    local sink = state.stream_sink
    if not is_callable(sink) then
        return
    end
    local ok, err = pcall(sink, {
        event = event_name,
        data = payload or {},
    })
    if not ok then
        print(string.format("[AgentNode][WARN] stream emit failed: %s", tostring(err)))
    end
end

-- 分块发送文本作为 token 事件
local function emit_tokens(state, text, chunk_chars)
    local s = tostring(text or "")
    if s == "" then
        return
    end
    local n = math.max(1, math.floor(tonumber(chunk_chars) or 24))
    local buf = {}
    local count = 0
    for ch in s:gmatch("[%z\1-\127\194-\244][\128-\191]*") do
        buf[#buf + 1] = ch
        count = count + 1
        if count >= n then
            emit_stream(state, "token", { token = table.concat(buf) })
            buf = {}
            count = 0
        end
    end
    if #buf > 0 then
        emit_stream(state, "token", { token = table.concat(buf) })
    end
end

local function has_method(runtime, method_name)
    if runtime == nil then
        return false
    end
    local ok, attr = pcall(function()
        return runtime[method_name]
    end)
    return ok and attr ~= nil
end

local function get_field(obj, key)
    if obj == nil then
        return nil
    end
    if type(obj) == "table" then
        local v = obj[key]
        if v ~= nil then
            return v
        end
        return obj[tostring(key)]
    end

    local ok, v = pcall(function()
        return obj[key]
    end)
    if ok and v ~= nil then
        return v
    end
    ok, v = pcall(function()
        return obj[tostring(key)]
    end)
    if ok and v ~= nil then
        return v
    end
    return nil
end

local function get_seq_item(seq, index1)
    local ok, v = pcall(function()
        return seq[index1]
    end)
    if ok and v ~= nil then
        return v
    end
    ok, v = pcall(function()
        return seq[index1 - 1]
    end)
    if ok then
        return v
    end
    return nil
end

local function clone_seq_rows(src, limit)
    local out = {}
    if src == nil then
        return out
    end
    if type(src) == "table" then
        for i, item in ipairs(src) do
            out[#out + 1] = item
            if limit and #out >= limit then
                break
            end
        end
        return out
    end

    local n = 0
    local ok_len, len_or_err = pcall(function()
        return #src
    end)
    if ok_len then
        n = math.max(0, math.floor(tonumber(len_or_err) or 0))
    end

    if n > 0 then
        for i = 1, n do
            local item = get_seq_item(src, i)
            if item ~= nil then
                out[#out + 1] = item
                if limit and #out >= limit then
                    break
                end
            end
        end
        return out
    end

    local probe_limit = math.max(1, math.floor(tonumber(limit) or 128))
    for i = 1, probe_limit do
        local item = get_seq_item(src, i)
        if item == nil then
            break
        end
        out[#out + 1] = item
    end
    return out
end

local function normalize_tool_call(item, idx)
    local fn = get_field(item, "function")
    local name = util.trim(
        get_field(item, "name")
        or get_field(item, "tool")
        or (type(fn) == "table" and fn.name)
        or get_field(fn, "name")
    )
    if name == "" then
        return nil
    end

    local args = get_field(item, "args")
    if args == nil then
        args = get_field(item, "arguments")
    end
    if args == nil and fn ~= nil then
        args = get_field(fn, "arguments")
    end
    if type(args) ~= "table" then
        args = {}
    end

    local call_id = util.trim(
        get_field(item, "id")
        or get_field(item, "call_id")
    )
    if call_id == "" then
        call_id = string.format("tool_call_%d", tonumber(idx) or 0)
    end

    return {
        tool = name,
        args = args,
        call_id = call_id,
    }
end

local function normalize_tool_calls(raw_calls)
    local rows = clone_seq_rows(raw_calls, 256)
    local out = {}
    for i, item in ipairs(rows) do
        local call = normalize_tool_call(item, i)
        if call then
            out[#out + 1] = call
        end
    end
    return out
end

local READ_EVIDENCE_TOOLS = {
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
}

local function lower_text(v)
    return tostring(v or ""):lower()
end

local function has_uploads(state)
    local rows = ((state or {}).uploads) or {}
    local result = type(rows) == "table" and #rows > 0
    print(string.format("[HasUploads][DEBUG] state.uploads type=%s len=%d result=%s",
        type(rows), #rows, tostring(result)))
    return result
end

local function collect_history_rows(state)
    local rows = ((((state or {}).messages or {}).conversation_history) or {})
    if type(rows) ~= "table" then
        return {}
    end
    return rows
end

local function has_history_tool_path(state)
    local rows = collect_history_rows(state)
    for i = #rows, 1, -1 do
        local row = rows[i]
        if type(row) == "table" then
            local content = tostring(row.content or "")
            if content:find("tool_path=download/", 1, true) or content:find("download/", 1, true) then
                return true
            end
        end
    end
    return false
end

local function has_file_intent(user_input)
    local q = lower_text(user_input)
    if q == "" then
        return false
    end
    local keywords = {
        "读取", "读一下", "读一读", "查看", "前几行", "几行", "第几行",
        "文件", "代码", "模型设置", "配置", "search", "grep", "read", "line", "lines",
        "list_files", "read_file", "read_lines", "search_file", "search_files",
        "download/", ".py", ".lua", ".txt",
    }
    for _, key in ipairs(keywords) do
        if q:find(key, 1, true) then
            return true
        end
    end
    return false
end

local function has_followup_read_intent(user_input)
    -- 简化：移除大部分关键词检测
    -- 现在 agent 应该通过 continue_task 工具来表达继续意图
    -- 这里只保留最基本的检测，用于处理用户明确要求继续的情况
    local q = lower_text(user_input)
    if q == "" then
        return false
    end
    return q == "再试一次" or q == "继续"
end

local function should_require_file_read(state)
    local has_up = has_uploads(state)
    local has_hist = has_history_tool_path(state)
    print(string.format("[ShouldRequire][DEBUG] has_uploads=%s has_history_tool_path=%s",
        tostring(has_up), tostring(has_hist)))
    
    if has_up then
        return true
    end
    if not has_hist then
        return false
    end
    local user_input = tostring((((state or {}).input or {}).message) or "")
    if has_file_intent(user_input) then
        return true
    end
    return has_followup_read_intent(user_input)
end

local function normalize_tool_path(raw)
    return util.normalize_tool_path(raw)
end

local function extract_tool_paths_from_text(text, out, seen)
    local s = tostring(text or "")
    if s == "" then
        return
    end

    for path in s:gmatch("tool_path=([^,%s%)]+)") do
        local p = normalize_tool_path(path)
        if p ~= "" and not seen[p] then
            seen[p] = true
            out[#out + 1] = p
        end
    end

    for path in s:gmatch("download/[0-9A-Za-z%._%-%/]+") do
        local p = normalize_tool_path(path)
        if p ~= "" and not seen[p] then
            seen[p] = true
            out[#out + 1] = p
        end
    end
end

local function collect_known_tool_paths(state)
    local out = {}
    local seen = {}

    local uploads = ((state or {}).uploads) or {}
    print(string.format("[CollectPaths][DEBUG] state.uploads count=%d", #uploads))
    for i, item in ipairs(uploads) do
        local p = normalize_tool_path((item or {}).tool_path or (item or {}).path)
        print(string.format("[CollectPaths][DEBUG] upload[%d]: tool_path=%s path=%s normalized=%s",
            i, tostring((item or {}).tool_path or "?"), tostring((item or {}).path or "?"), tostring(p)))
        if p ~= "" and not seen[p] then
            seen[p] = true
            out[#out + 1] = p
        end
    end

    local rows = collect_history_rows(state)
    for i = #rows, 1, -1 do
        local row = rows[i]
        if type(row) == "table" then
            extract_tool_paths_from_text(row.content or "", out, seen)
        end
    end

    extract_tool_paths_from_text((((state or {}).input or {}).message) or "", out, seen)
    print(string.format("[CollectPaths][DEBUG] total known_paths=%d", #out))
    return out
end

local function has_read_evidence(state)
    if (tonumber((((state or {}).tool_exec or {}).read_evidence_total) or 0) or 0) > 0 then
        return true
    end
    local tool_ctx = tostring((((state or {}).context or {}).tool_context) or "")
    if tool_ctx:find("[Tool:read_file]", 1, true)
        or tool_ctx:find("[Tool:read_lines]", 1, true)
        or tool_ctx:find("[Tool:search_file]", 1, true)
        or tool_ctx:find("[Tool:search_files]", 1, true) then
        return true
    end
    return false
end

local function has_read_call(calls)
    for _, row in ipairs(calls or {}) do
        if READ_EVIDENCE_TOOLS[tostring((row or {}).tool or "")] then
            return true
        end
    end
    return false
end

local function build_forced_read_calls(state, max_calls)
    local calls = {}
    local limit = math.max(1, math.floor(tonumber(max_calls) or 1))
    local known_paths = collect_known_tool_paths(state)
    local q = lower_text((((state or {}).input or {}).message) or "")
    local wants_lines = (
        q:find("前几行", 1, true) ~= nil
        or q:find("几行", 1, true) ~= nil
        or q:find("line", 1, true) ~= nil
        or q:find("lines", 1, true) ~= nil
        or q:find("配置", 1, true) ~= nil
        or q:find("config", 1, true) ~= nil
        or q:find("模型设置", 1, true) ~= nil
    )

    -- 从 tool_exec.read_files 中获取已读取的文件路径
    local already_read = state.tool_exec.read_files or {}
    local already_read_count = 0
    for _ in pairs(already_read) do
        already_read_count = already_read_count + 1
    end

    -- DEBUG: 打印known_paths详细信息
    print(string.format("[BuildForced][DEBUG] known_paths=%d already_read=%d limit=%d",
        #known_paths, already_read_count, limit))
    for i, p in ipairs(known_paths) do
        print(string.format("[BuildForced][DEBUG] known_paths[%d]=%s", i, tostring(p)))
    end

    -- 为每个未读取的文件构建读取调用
    local call_idx = 0
    for _, path in ipairs(known_paths) do
        if call_idx >= limit then
            break
        end
        -- 跳过已经读取过的文件
        if not already_read[path] then
            if wants_lines then
                calls[#calls + 1] = {
                    tool = "read_lines",
                    args = {
                        path = path,
                        start_line = 1,
                        max_lines = 220,
                    },
                    call_id = string.format("guard_read_lines_%d", call_idx + 1),
                }
            else
                calls[#calls + 1] = {
                    tool = "read_file",
                    args = {
                        path = path,
                        max_chars = 2400,
                    },
                    call_id = string.format("guard_read_file_%d", call_idx + 1),
                }
            end
            call_idx = call_idx + 1
            print(string.format("[BuildForced][DEBUG] Added read call for: %s", path))
        end
    end

    -- 如果没有找到任何路径，尝试列出download目录
    if #calls == 0 and #known_paths == 0 then
        calls[#calls + 1] = {
            tool = "list_files",
            args = { prefix = "download" },
            call_id = "guard_list_files_1",
        }
    end

    return calls
end

local function enforce_file_read_guard(state, tool_calls)
    local calls = tool_calls or {}
    
    -- DEBUG: 打印uploads详细信息
    local uploads_debug = {}
    for i, item in ipairs(state.uploads or {}) do
        uploads_debug[#uploads_debug + 1] = string.format("[%d] name=%s tool_path=%s path=%s",
            i,
            tostring(item.name or "?"),
            tostring(item.tool_path or "?"),
            tostring(item.path or "?")
        )
    end
    print(string.format("[EnforceGuard][DEBUG] state.uploads count=%d items=%s",
        #(state.uploads or {}),
        table.concat(uploads_debug, "; ")))
    
    -- 如果 agent 调用了 finish_turn，不要强制添加其他工具
    for _, call in ipairs(calls) do
        if call.tool == "finish_turn" or call.tool == "continue_task" then
            print("[EnforceGuard][DEBUG] Agent called finish_turn/continue_task, returning as-is")
            return calls
        end
    end
    
    if not should_require_file_read(state) then
        print("[EnforceGuard][DEBUG] should_require_file_read returned false, returning as-is")
        return calls
    end
    
    -- 检查是否有上传文件和已读取文件数量
    local uploads_count = #(state.uploads or {})
    local read_files = state.tool_exec.read_files or {}
    local read_count = 0
    for _ in pairs(read_files) do
        read_count = read_count + 1
    end
    
    -- DEBUG
    print(string.format("[EnforceGuard][DEBUG] uploads=%d read_count=%d calls=%d",
        uploads_count, read_count, #calls))
    
    -- 如果有上传文件，且读取文件数少于上传数量，说明还有未读取的文件
    local has_unread_uploads = uploads_count > 0 and read_count < uploads_count
    
    -- 如果没有未读取的上传文件，且已有读取证据，则不需要强制
    if not has_unread_uploads and has_read_evidence(state) then
        print("[EnforceGuard][DEBUG] No unread uploads, skipping")
        return calls
    end
    
    print(string.format("[EnforceGuard][DEBUG] Has unread uploads or no evidence, forcing reads: %s", tostring(has_unread_uploads)))

    local max_calls = math.max(1, math.floor(tonumber((((graph_cfg() or {}).planner or {}).max_calls_per_loop) or 6)))
    if #calls <= 0 then
        local forced = build_forced_read_calls(state, max_calls)
        if #forced > 0 then
            return forced
        end
        return calls
    end

    if has_read_call(calls) then
        return calls
    end
    if #calls >= max_calls then
        return calls
    end

    local forced = build_forced_read_calls(state, 1)
    if #forced > 0 then
        calls[#calls + 1] = forced[1]
    end
    return calls
end



local function parse_model_output(raw)
    if type(raw) == "string" then
        local parsed = util.parse_lua_table_literal(raw)
        if parsed and type(parsed) == "table" then
            raw = parsed
        else
            return {
                content = util.trim(raw),
                tool_calls = {},
                raw = util.trim(raw),
            }
        end
    end

    local content_v = get_field(raw, "content")
    local text_v = get_field(raw, "text")
    local tool_calls_v = get_field(raw, "tool_calls")
    local raw_v = get_field(raw, "raw")

    local content = util.trim(content_v or text_v)
    local calls = normalize_tool_calls(tool_calls_v)
    local raw_text = util.trim(raw_v)

    -- Python dict/list objects may arrive as userdata proxy via Lupa.
    -- In that case, try field extraction first; only fallback to tostring
    -- when no structured fields are available.
    if content == "" and #calls <= 0 and raw_text == "" and raw ~= nil then
        local fallback = util.trim(tostring(raw))
        return {
            content = fallback,
            tool_calls = {},
            raw = fallback,
        }
    end

    return {
        content = content,
        tool_calls = calls,
        raw = raw_text,
    }
end

local function call_agent_with_tools(state)
    local runtime = _G.py_pipeline
    if runtime == nil then
        return nil, "python_runtime_unavailable"
    end
    if not has_method(runtime, "generate_chat_with_tools_sync") then
        return nil, "python_method_unavailable"
    end

    local cfg = (graph_cfg().agent or {})
    local params = {
        max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 1024)),
        temperature = tonumber(cfg.temperature) or 0.6,
        seed = tonumber(cfg.seed) or 42,
    }

    local tools = tool_registry.get_tool_schemas()
    local tool_choice = cfg.tool_choice
    if util.trim(tool_choice) == "" then
        tool_choice = "auto"
    end
    local parallel_tool_calls = util.to_bool(cfg.parallel_tool_calls, true)

    local ok, result_or_err = pcall(function()
        return runtime:generate_chat_with_tools_sync(
            (((state or {}).messages or {}).runtime_messages) or {},
            params,
            tools,
            tool_choice,
            parallel_tool_calls
        )
    end)
    if not ok then
        return nil, tostring(result_or_err or "agent_llm_call_failed")
    end

    local out = parse_model_output(result_or_err)
    return out, nil
end

local function fallback_without_tools(state)
    local cfg = (graph_cfg().agent or {})
    local ok, text_or_err = pcall(function()
        return py_pipeline:generate_chat_sync(
            (((state or {}).messages or {}).runtime_messages) or {},
            {
                max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 1024)),
                temperature = tonumber(cfg.temperature) or 0.6,
                seed = tonumber(cfg.seed) or 42,
            }
        )
    end)
    if not ok then
        return nil, tostring(text_or_err or "agent_llm_fallback_failed")
    end
    return {
        content = util.trim(tostring(text_or_err or "")),
        tool_calls = {},
        raw = "",
    }, nil
end

function M.run(state, _ctx)
    state.messages = state.messages or {}
    state.messages.runtime_messages = state.messages.runtime_messages or {}
    state.agent_loop = state.agent_loop or {
        remaining_steps = 25,
        pending_tool_calls = {},
        stop_reason = "",
        iteration = 0,
    }
    state.planner = state.planner or { tool_calls = {}, errors = {}, raw = "", force_reason = "" }
    state.router_decision = state.router_decision or { route = "respond", raw = "", reason = "" }

    -- DEBUG: 打印当前状态
    print(string.format("[AgentNode][DEBUG] iteration=%d remaining_steps=%d uploads_count=%d stop_reason='%s'",
        tonumber(state.agent_loop.iteration) or 0,
        tonumber(state.agent_loop.remaining_steps) or 0,
        #(state.uploads or {}),
        tostring(state.agent_loop.stop_reason or "")
    ))

    -- 检查上下文预算
    local stats = context_manager.get_context_stats(state)
    local budget = math.max(256, math.floor(tonumber((graph_cfg().input_token_budget) or 12000)))
    local budget_status, budget_warning = context_manager.check_budget(stats.estimated_tokens, budget)

    -- 如果预算超限，尝试优化runtime_messages
    if budget_status == "exceeded" then
        local optimized, opt_stats = context_manager.optimize_runtime_messages(
            state.messages.runtime_messages,
            2000 -- 更激进的截断
        )
        state.messages.runtime_messages = optimized
        -- 重新计算
        stats = context_manager.get_context_stats(state)
        budget_status, budget_warning = context_manager.check_budget(stats.estimated_tokens, budget)
    end

    -- 限制tool消息数量，防止上下文无限增长
    local runtime_msgs = state.messages.runtime_messages
    local tool_msg_count = 0
    for i = #runtime_msgs, 1, -1 do
        if runtime_msgs[i] and runtime_msgs[i].role == "tool" then
            tool_msg_count = tool_msg_count + 1
            if tool_msg_count > MAX_TOOL_MESSAGES then
                -- 标记需要清理旧消息
                state.context._tool_messages_overflow = true
            end
        end
    end

    -- 如果tool消息过多，清理最旧的
    if tool_msg_count > MAX_TOOL_MESSAGES then
        local cleaned_msgs = {}
        local kept_tools = 0
        for _, msg in ipairs(runtime_msgs) do
            if msg.role ~= "tool" then
                cleaned_msgs[#cleaned_msgs + 1] = msg
            elseif kept_tools < MAX_TOOL_MESSAGES then
                cleaned_msgs[#cleaned_msgs + 1] = msg
                kept_tools = kept_tools + 1
            end
        end
        state.messages.runtime_messages = cleaned_msgs
    end

    local out, err = call_agent_with_tools(state)
    if not out then
        out, err = fallback_without_tools(state)
    end

    if not out then
        state.agent_loop.pending_tool_calls = {}
        state.agent_loop.stop_reason = "model_call_failed"
        state.planner.tool_calls = {}
        state.planner.raw = ""
        state.planner.errors = state.planner.errors or {}
        state.planner.errors[#state.planner.errors + 1] = tostring(err or "agent_call_failed")
        return state
    end

    local tool_calls = enforce_file_read_guard(state, out.tool_calls or {})
    local assistant_content = tostring(out.content or "")

    -- DEBUG: 打印LLM响应信息
    print(string.format("[AgentNode][DEBUG] LLM response: content_len=%d tool_calls_count=%d",
        #assistant_content,
        #tool_calls
    ))
    for i, call in ipairs(tool_calls) do
        print(string.format("[AgentNode][DEBUG] tool_call[%d]: %s", i, tostring(call.tool or "unknown")))
    end

    -- 如果预算紧张，在assistant消息中添加提示
    if budget_status == "warning" then
        assistant_content = assistant_content .. "\n\n[System: Context budget is high. Consider summarizing previous results.]"
    end

    state.messages.runtime_messages[#state.messages.runtime_messages + 1] = {
        role = "assistant",
        content = assistant_content,
        tool_calls = tool_calls,
    }

    -- 流式发送 assistant 回复（如果有内容且无工具调用）
    if #tool_calls == 0 and assistant_content ~= "" then
        local stream_cfg = graph_cfg().streaming or {}
        local chunk_chars = tonumber(stream_cfg.token_chunk_chars) or 24
        -- 标记已发送，避免在 graph_runtime 中重复发送
        state._streaming_sent = true
        emit_tokens(state, assistant_content, chunk_chars)
    end

    state.agent_loop.iteration = (tonumber(state.agent_loop.iteration) or 0) + 1
    state.agent_loop.remaining_steps = math.max(0, (tonumber(state.agent_loop.remaining_steps) or 0) - 1)
    state.agent_loop.pending_tool_calls = tool_calls

    state.planner.tool_calls = tool_calls
    state.planner.raw = tostring(out.raw or "")

    if #tool_calls > 0 then
        state.router_decision.route = "tool_loop"
    else
        state.router_decision.route = "respond"
    end

    if #tool_calls > 0 and (tonumber(state.agent_loop.remaining_steps) or 0) <= 0 then
        -- 步数已用尽，但允许执行已生成的工具调用
        -- 设置停止原因，以便工具执行后循环停止
        state.agent_loop.stop_reason = "remaining_steps_exhausted"
        -- 不清空 pending_tool_calls，让工具可以执行
    end

    -- 记录上下文统计
    state.metrics = state.metrics or {}
    state.metrics.context_stats = context_manager.get_context_stats(state)

    return state
end

return M
