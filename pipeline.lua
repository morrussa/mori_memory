local tool = require("module.tool")
local history = require("module.history")
local topic   = require("module.topic")
local memory = require("module.memory")
local heat = require("module.heat")
local math = require("math")
local cluster = require("module.cluster")
local saver =require("module.saver")
local recall = require("module.recall")
local tool_calling = require("module.tool_calling")
local notebook = require("module.notebook")
local adaptive = require("module.adaptive")
local app_config = require("module.config")
local run_mode = tostring(MORI_RUN_MODE or "cli")

-- 1. 配置模型路径
local base = "/home/morusa/AI/mori/model/"
local config = {
    large_model = base .. "Qwen3.5-9B-Q6_K.gguf",
    embedding_model = base .. "Qwen3-Embedding-0.6B-Q8_0.gguf"
}

print("[Lua] large_model path: " .. tostring(config.large_model))
print("[Lua] embedding_model path: " .. tostring(config.embedding_model))
py_pipeline:load_models(config.large_model, config.embedding_model)

memory.load()
cluster.load()
cluster.update_hot_status()
heat.load()
saver.mark_dirty()
history.load()
adaptive.load()
recall.init_all_sentiment_vectors()
notebook.load()

print("[Lua] Pipeline started.")

topic.init()

-- ====================== 对话历史 + 滑动窗口管理 ======================
print("[Lua] keyring tool mode: two_step")

local base_prompt = [[
你叫 Mori，是一名天才AI极客少女，常用颜文字 (´･ω･')ﾉ
你喜欢有趣和有创意的对话，对于用户的提问会尽力给出有帮助的回答。
当遇到你不确定或觉得信息不足的问题时，你会要求用户提供更多信息，而不是直接拒绝。
你尊重每一个认真提问的人。

你有 keyring 长期记忆系统，系统会在后台二阶段自动执行工具调用。
系统会在每轮自动注入 LongTermPlan BOM（来自已存 long_term_plan 记录）。

规则：
1. 正常回复用户，不要输出任何 {act="..."} 工具调用行。
2. 若你判断需要长期保存事实、更新计划、或检索旧记录，直接在正文自然表达意图即可，后台会自动处理。
3. 回答时优先保持与 LongTermPlan BOM 一致；若信息冲突，先向用户确认。
]]

local conversation_history = {
    { role = "system", content = base_prompt }
}

local MAX_HISTORY_TURNS = 12

local function build_conversation_context(user_input)
    local msgs = { conversation_history[1] }
    local start_idx = math.max(2, #conversation_history - MAX_HISTORY_TURNS * 2 + 1)
    for i = start_idx, #conversation_history do
        table.insert(msgs, conversation_history[i])
    end
    table.insert(msgs, { role = "user", content = user_input })
    return msgs
end

local function add_to_history(user_msg, assistant_msg)
    table.insert(conversation_history, { role = "user", content = user_msg })
    table.insert(conversation_history, { role = "assistant", content = assistant_msg })
end
-- =================================================================



if run_mode ~= "webui" then
    -- 3. 测试相似度接口
    print("\n[Lua] Testing Similarity...")
    local text_a = "Python is terrible."
    local text_b = "I hate Python."
    local emb_a = tool.get_embedding(text_a)
    local emb_b = tool.get_embedding(text_b)
    local score = tool.cosine_similarity(emb_a, emb_b)
    print(string.format("[Lua] Similarity between '%s' and '%s' is: %.4f", text_a, text_b, score))

    -- 4. 聊天生成接口演示（保留原样）
    local function demo_callback(result_text)
        local cot_rm = tool.remove_cot(result_text)
        print("\n[Lua Callback] Generation Finished! Result: " .. cot_rm)
    end

    local messages = {
        { role = "system", content = "你是一个诗人，擅长写短诗。" },
        { role = "user",   content = "写一首关于讨厌Python的短诗。" }
    }

    local gen_params = {
        max_tokens = 1024,
        temperature = 0.5,
        seed = math.random(114, 514)
    }

    print("\n[Lua] Calling Large Model (GPU) with chat template...")
    py_pipeline:generate_chat(messages, gen_params, demo_callback)
end


local function shutdown_pipeline()
    print("[Lua] 正在原子保存所有内存数据...")
    saver.on_exit()
end

local function process_user_input(line, stream_sink, session_id, read_only)
    local user_input = tostring(line or ""):match("^%s*(.-)%s*$")
    if user_input == "" then
        return ""
    end

    local ro = (read_only == true)

    local current_turn = history.get_turn() + 1
    local user_vec_q = tool.get_embedding_query(user_input)-- 检索向量（query）
    local user_vec_p = tool.get_embedding_passage(user_input)-- 写入/主题向量（passage）
    -- local user_vec = tool.get_embedding(user_input)
    if not ro then
        topic.add_turn(current_turn, user_input, user_vec_p)
    end

    -- ========== 插入记忆召回逻辑 ==========
    local memory_context = recall.check_and_retrieve(user_input, user_vec_q)
    local plan_bom = tool_calling.get_long_term_plan_bom()
    local tool_context = ""
    if not ro then
        tool_context = tool_calling.consume_pending_system_context_for_turn(current_turn)
    end
    local messages_for_llm = build_conversation_context(user_input)
    local insert_pos = 2
    if plan_bom and plan_bom ~= "" then
        table.insert(messages_for_llm, insert_pos, { role = "system", content = plan_bom })
        insert_pos = insert_pos + 1
        print("[Lua] LongTermPlan BOM 已注入本轮上下文")
    end
    if tool_context and tool_context ~= "" then
        table.insert(messages_for_llm, insert_pos, { role = "system", content = tool_context })
        insert_pos = insert_pos + 1
        print("[Lua] query_record 结果已注入本轮上下文")
    end
    if memory_context and memory_context ~= "" then
        -- 将召回的记忆作为额外的系统消息插入（紧跟原始系统提示之后）
        table.insert(messages_for_llm, insert_pos, { role = "system", content = memory_context })
        print("[Lua] 记忆召回已加入上下文")
    end
    -- ====================================

    local params = { 
        max_tokens = 1024, 
        temperature = 0.75,
        seed = math.random(1, 2147483647),
    }
    if stream_sink then
        params.stream = true
    end

    local final_result = ""
    local function chat_callback(result)
        final_result = select(1, tool_calling.handle_chat_result({
            user_input = user_input,
            current_turn = current_turn,
            read_only = ro,
            add_to_history = function(user_msg, assistant_msg)
                add_to_history(user_msg, assistant_msg)
            end,
        }, result)) or ""
    end

    py_pipeline:generate_chat(messages_for_llm, params, chat_callback, stream_sink)
    return final_result
end

_G.mori_handle_user_input = process_user_input
_G.mori_shutdown = shutdown_pipeline

if run_mode == "webui" then
    print("[Lua] MORI_RUN_MODE=webui，交互式控制台已禁用（等待 HTTP 请求）")
else
    -- ========== 交互式主循环 ==========
    print("\n[Lua] Entering interactive loop. Type 'exit' to quit.")

    while true do
        io.write("> ")
        local line = io.read()
        if not line then
            print("\n[Lua] EOF received, exiting.")
            shutdown_pipeline()
            break
        end

        line = line:match("^%s*(.-)%s*$")
        if line == "" then goto continue end

        if line == "exit" then
            shutdown_pipeline()
            print("[Lua] Exiting.")
            break
        end

        process_user_input(line, nil, nil, false)

        ::continue::
    end
end

print("[Lua] Pipeline execution finished.")
