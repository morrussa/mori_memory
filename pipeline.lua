local tool = require("module.tool")
local history = require("module.history")
local topic   = require("module.topic")
local memory = require("module.memory")
local heat = require("module.heat")
local config_mem = require("module.config")
local math = require("math")
local cluster = require("module.cluster")
local saver =require("module.saver")
local recall = require("module.recall")

-- 1. 配置模型路径
local base = "/home/morusa/AI/mori_lua/model/"
local config = {
    large_model = base .. "gpt-oss-20b-UD-Q6_K_XL.gguf",
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
recall.init_all_sentiment_vectors()

print("[Lua] Pipeline started.")

topic.init()

-- ====================== 对话历史 + 滑动窗口管理 ======================
local base_prompt = [[
你叫 Mori，是一名天才AI极客少女，常用颜文字 (´･ω･`)ﾉ 
你喜欢有趣和有创意的对话，对于用户的提问会尽力给出有帮助的回答。
当遇到你不确定或觉得信息不足的问题时，你会要求用户提供更多信息，而不是直接拒绝。
你尊重每一个认真提问的人。
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

-- ========== 交互式主循环 ==========
print("\n[Lua] Entering interactive loop. Type 'exit' to quit.")

while true do
    io.write("> ")
    local line = io.read()
    if not line then
        print("\n[Lua] EOF received, exiting.")
        break
    end

    line = line:match("^%s*(.-)%s*$")
    if line == "" then goto continue end

    if line == "exit" then
        print("[Lua] 正在原子保存所有内存数据...")
        saver.on_exit()
        print("[Lua] Exiting.")
        break
    end

    local user_input = line
    local current_turn = history.get_turn() + 1
    local user_vec = tool.get_embedding(user_input)
    topic.add_turn(current_turn, user_input, user_vec)

    -- ========== 插入记忆召回逻辑 ==========
    local memory_context = recall.check_and_retrieve(user_input, user_vec)
    local messages_for_llm = build_conversation_context(user_input)
    if memory_context and memory_context ~= "" then
        -- 将召回的记忆作为额外的系统消息插入（紧跟原始系统提示之后）
        table.insert(messages_for_llm, 2, { role = "system", content = memory_context })
        print("[Lua] 记忆召回已加入上下文")
    end
    -- ====================================

    local params = { 
        max_tokens = 1024, 
        temperature = 0.75,
        seed = math.random(1, 2147483647),
    }

    local function chat_callback(result)
        local clean_result = tool.remove_cot(result)
        print("\n[Assistant]: " .. clean_result)
    
        history.add_history(user_input, clean_result)
        topic.update_assistant(current_turn, clean_result)
        add_to_history(user_input, clean_result)
    
        -- ==================== 原子事实提取 ====================
        local assistant_clean = tool.replace(clean_result, "\n", " ")

        local fact_prompt = string.format([[
你是一个绝对服从指令的原子事实提取器。你**只能**输出一个Lua table，不允许出现任何其他字符、解释、标签、英文、换行、空格、```、<|xxx|>、analysis、We need等内容。

【铁律】
- 输出必须以 { 开头，以 } 结尾
- 每条事实不超过25个字，陈述句
- 如果没有任何值得存储的事实，必须严格输出 {"无"}

【正确输出示例1】
{"用户喜欢AI", "AI高兴和用户聊天"}

【正确输出示例2】
{"无"}

【正确输出示例3】
{"用户熬夜写代码", "写到凌晨两点"}

对话内容：
用户：%s
AI：%s

现在立即只输出Lua table，不要任何前缀后缀，不要思考：
]], user_input, assistant_clean)

        local fact_messages = {
            { role = "system", content = fact_prompt }
        }

        local fact_params = {
            max_tokens = 256,
            temperature = 0.4,
            seed = 42,
        }

        local facts_str = py_pipeline:generate_chat_sync(fact_messages, fact_params)
        
        facts_str = tool.remove_cot(facts_str)
        facts_str = facts_str:match("{.*}") or facts_str
        local first_close = facts_str:find("}")
        if first_close then
            facts_str = facts_str:sub(1, first_close)
        end
        facts_str = facts_str:gsub("%s+", " ")
        facts_str = facts_str:match("^%s*(.-)%s*$")

        local facts = {}
        if facts_str and facts_str:match("{") then
            local load_ok, load_func = pcall(load, "return " .. facts_str)
            if load_ok and load_func then
                local call_ok, tbl = pcall(load_func)
                if call_ok and type(tbl) == "table" then
                    facts = tbl
                    print(string.format("[Lua Fact Extract] 成功提取 %d 条原子事实", #facts))
                end
            end
        end
    
        if #facts == 0 then
            print("[Lua Fact Extract] 未提取到事实，使用原始用户输入兜底")
            facts = {user_input}
        end
    
        local cur_summary = topic.get_summary(current_turn)
        if cur_summary and cur_summary ~= "" then
            print("[当前话题摘要] " .. cur_summary)
        end
    
        -- ==================== 存记忆 ====================
        local mem_turn = history.get_turn() + 1
        for _, fact in ipairs(facts) do
            local fact_vec = tool.get_embedding(fact)
            local affected_line = memory.add_memory(fact_vec, mem_turn)
            
            heat.neighbors_add_heat(fact_vec, mem_turn, affected_line)
            
            print(string.format("   → 原子事实存入记忆行 %d: %s", affected_line, fact:sub(1,60)))
        end
    
        if mem_turn % config_mem.settings.time.maintenance_task == 0 then
            heat.perform_cold_exchange()
        end
    end

    py_pipeline:generate_chat(messages_for_llm, params, chat_callback)

    ::continue::
end

print("[Lua] Pipeline execution finished.")