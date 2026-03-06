local context_builder = require("module.graph.context_builder")
local util = require("module.graph.util")
local config = require("module.config")

local M = {}

local function graph_cfg()
    return ((config.settings or {}).graph or {})
end

-- 发送流式事件
local function emit_stream(state, event_name, payload)
    local sink = state.stream_sink
    if type(sink) ~= "function" then
        return
    end
    local ok, err = pcall(sink, {
        event = event_name,
        data = payload or {},
    })
    if not ok then
        print(string.format("[ResponderNode][WARN] stream emit failed: %s", tostring(err)))
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

local FILE_TOOLS = {
    list_files = true,
    read_file = true,
    read_lines = true,
    search_file = true,
    search_files = true,
}

local function has_uploads(state)
    local rows = ((state or {}).uploads) or {}
    return type(rows) == "table" and #rows > 0
end

local function has_file_tool_evidence(state)
    local rows = (((state or {}).tool_exec or {}).results) or {}
    for _, row in ipairs(rows) do
        if row.ok == true and FILE_TOOLS[tostring(row.tool or "")] then
            return true
        end
    end

    local tool_context = tostring((((state or {}).context or {}).tool_context) or "")
    if tool_context ~= "" then
        for tool_name, _ in pairs(FILE_TOOLS) do
            if tool_context:find("[Tool:" .. tool_name .. "]", 1, true) then
                return true
            end
        end
    end
    return false
end

local function build_responder_user_payload(user_input, tool_results, uploads_present)
    local lines = {
        "[UserRequest]",
        tostring(user_input or ""),
        "",
        "[ToolExecutionSummary]",
    }

    local rows = tool_results or {}
    if type(rows) == "table" and #rows > 0 then
        for _, row in ipairs(rows) do
            local tool_name = util.trim((row or {}).tool)
            if tool_name == "" then
                tool_name = "unknown"
            end
            if row.ok == true then
                local result = util.utf8_take(util.trim((row or {}).result), 320)
                lines[#lines + 1] = string.format("- %s: ok | %s", tool_name, result)
            else
                local err = util.utf8_take(util.trim((row or {}).error), 200)
                lines[#lines + 1] = string.format("- %s: failed | %s", tool_name, err)
            end
        end
    else
        lines[#lines + 1] = "- none"
    end

    lines[#lines + 1] = ""
    lines[#lines + 1] = "[ResponsePolicy]"
    lines[#lines + 1] = "1) 只能依据 ToolExecutionSummary 中真实结果描述工具行为。"
    lines[#lines + 1] = "2) 当 ToolExecutionSummary 为 none 时，禁止声称已经执行任何工具。"
    lines[#lines + 1] = "3) 工具失败时需明确失败并给出可执行的下一步。"
    if uploads_present then
        lines[#lines + 1] = "4) 用户已上传附件；若没有文件工具结果，不要编造附件内容。"
    end

    return table.concat(lines, "\n")
end

local function contains_tool_claim(text)
    local s = tostring(text or ""):lower()
    return s:find("list_files", 1, true) ~= nil
        or s:find("read_file", 1, true) ~= nil
        or s:find("read_lines", 1, true) ~= nil
        or s:find("search_file", 1, true) ~= nil
        or s:find("search_files", 1, true) ~= nil
end

function M.run(state, _ctx)
    local cfg = graph_cfg().responder or {}
    local tool_results = (((state or {}).tool_exec or {}).results) or {}
    local original_user = tostring((((state or {}).input or {}).message) or "")
    local uploads_present = has_uploads(state)
    local strict_tool_honesty = util.to_bool(cfg.strict_tool_honesty, true)
    local file_tool_evidence = has_file_tool_evidence(state)

    if strict_tool_honesty and uploads_present and not file_tool_evidence then
        local fallback_msg = "我还没有成功读取你上传的附件内容。请重试这次请求，我会先执行文件工具再给出基于附件的结论。"
        state.final_response = {
            message = fallback_msg,
            context_meta = {
                policy = "missing_upload_tool_evidence",
            },
        }
        -- 流式发送 fallback 消息
        local stream_cfg = graph_cfg().streaming or {}
        local chunk_chars = tonumber(stream_cfg.token_chunk_chars) or 24
        emit_tokens(state, fallback_msg, chunk_chars)
        state._streaming_sent = true
        return state
    end

    local merged_user = build_responder_user_payload(original_user, tool_results, uploads_present)

    local original = (((state or {}).input or {}).message) or ""
    state.input.message = merged_user
    local messages, meta = context_builder.build_chat_messages(state)
    state.input.message = original

    local final_text = py_pipeline:generate_chat_sync(messages, {
        max_tokens = math.max(64, math.floor(tonumber(cfg.max_tokens) or 1024)),
        temperature = tonumber(cfg.temperature) or 0.75,
        seed = tonumber(cfg.seed) or math.random(1, 2147483647),
    })

    final_text = util.trim(final_text)
    if final_text == "" then
        final_text = "好的，已处理。"
    end

    if strict_tool_honesty and (type(tool_results) ~= "table" or #tool_results == 0) and contains_tool_claim(final_text) then
        final_text = "当前回合没有任何工具执行结果可用。我可以先执行工具，再基于真实结果回答。"
    end

    state.final_response = {
        message = final_text,
        context_meta = meta,
    }

    -- 流式发送最终回复
    if final_text ~= "" then
        local stream_cfg = graph_cfg().streaming or {}
        local chunk_chars = tonumber(stream_cfg.token_chunk_chars) or 24
        emit_tokens(state, final_text, chunk_chars)
        state._streaming_sent = true
    end

    return state
end

return M
