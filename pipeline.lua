local tool = require("module.tool")
local history = require("module.memory.history")
local topic   = require("module.memory.topic")
local memory = require("module.memory.store")
local heat = require("module.memory.heat")
local math = require("math")
local cluster = require("module.memory.cluster")
local saver =require("module.memory.saver")
local recall = require("module.memory.recall")
local notebook = require("module.agent.notebook")
local adaptive = require("module.memory.adaptive")
local app_config = require("module.config")
local agent_runtime = require("module.agent.runtime")
local run_mode = tostring(MORI_RUN_MODE or "cli")

local runtime_cfg = (app_config.settings or {}).runtime or {}
local runtime_defaults = (app_config.defaults or {}).runtime
if type(runtime_defaults) ~= "table" then
    runtime_defaults = runtime_cfg
end
local model_cfg = runtime_cfg.models or {}
local model_defaults = runtime_defaults.models or {}
local demo_cfg = runtime_cfg.demo_chat or {}
local demo_defaults = runtime_defaults.demo_chat or {}
local webui_stream_max_steps = tonumber(MORI_WEBUI_STREAM_MAX_STEPS)
if run_mode == "webui" and webui_stream_max_steps == nil then
    webui_stream_max_steps = 1
end
if webui_stream_max_steps ~= nil then
    webui_stream_max_steps = math.floor(webui_stream_max_steps)
end

local function join_model_path(base_dir, path_or_name)
    local path = tostring(path_or_name or "")
    if path == "" then
        return path
    end
    if path:sub(1, 1) == "/" then
        return path
    end

    local base = tostring(base_dir or "")
    if base == "" then
        return path
    end
    if base:sub(-1) ~= "/" then
        base = base .. "/"
    end
    return base .. path
end

-- 1. 配置模型路径（统一从 config.lua 读取）
local model_base = tostring(model_cfg.base_dir or model_defaults.base_dir or "")
local large_model_path = join_model_path(model_base, model_cfg.large_model or model_defaults.large_model)
local embedding_model_path = join_model_path(model_base, model_cfg.embedding_model or model_defaults.embedding_model)

print("[Lua] large_model path: " .. tostring(large_model_path))
print("[Lua] embedding_model path: " .. tostring(embedding_model_path))
py_pipeline:load_models(large_model_path, embedding_model_path)

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

-- ====================== 对话历史 + Token 窗口管理 ======================

local base_prompt = [[
你叫 Mori，是一名天才AI极客少女，常用颜文字 (´･ω･')ﾉ
你喜欢有趣和有创意的对话，对于用户的提问会尽力给出有帮助的回答。
当遇到你不确定或觉得信息不足的问题时，你会要求用户提供更多信息，而不是直接拒绝。
你尊重每一个认真提问的人。

你有 keyring 长期记忆系统，系统会在后台二阶段自动执行工具调用。
系统会在每轮自动注入 LongTermPlan BOM（来自已存 long_term_plan 记录）。

规则：
1. 正常回复用户，不要输出任何工具调用格式（包括 `{act="..."}`、`<tool_call>...</tool_call>`、`✿FUNCTION✿/✿ARGS✿`）。
2. 若你判断需要长期保存事实、更新计划、或检索旧记录，直接在正文自然表达意图即可，后台会自动处理。
3. 回答时优先保持与 LongTermPlan BOM 一致；若信息冲突，先向用户确认。
4. 若用户上传文件且上下文给出了附件目录路径（MORI_AGENT_FILES_DIR，默认 ./workspace），不要假设你已读完整正文；先在正文说明你将分段检索/读取，再由后台 planner 按需调用 list_agent_files/read_agent_file/read_agent_file_lines/search_agent_file/search_agent_files。
5. 若系统在当前轮要求你输出计划信号，请严格按要求仅输出一个 Lua table 信号（{act="plan"} 或 {act="no_plan"}），并放在回复最后一行。
]]

local conversation_history = {
    { role = "system", content = base_prompt }
}

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

    -- 4. 聊天生成接口演示
    local function demo_callback(result_text)
        local cot_rm = tool.remove_cot(result_text)
        print("\n[Lua Callback] Generation Finished! Result: " .. cot_rm)
    end

    local messages = {
        { role = "system", content = "你是一个诗人，擅长写短诗。" },
        { role = "user",   content = "写一首关于讨厌Python的短诗。" }
    }

    local demo_seed_min = math.floor(tonumber(demo_cfg.seed_min) or tonumber(demo_defaults.seed_min) or 114)
    local demo_seed_max = math.floor(tonumber(demo_cfg.seed_max) or tonumber(demo_defaults.seed_max) or 514)
    if demo_seed_max < demo_seed_min then
        demo_seed_min, demo_seed_max = demo_seed_max, demo_seed_min
    end
    local gen_params = {
        max_tokens = math.floor(tonumber(demo_cfg.max_tokens) or tonumber(demo_defaults.max_tokens) or 1024),
        temperature = tonumber(demo_cfg.temperature) or tonumber(demo_defaults.temperature) or 0.5,
        seed = math.random(demo_seed_min, demo_seed_max)
    }

    print("\n[Lua] Calling Large Model (GPU) with chat template...")
    py_pipeline:generate_chat(messages, gen_params, demo_callback)
end


local function shutdown_pipeline()
    print("[Lua] 正在原子保存所有内存数据...")
    saver.on_exit()
end

local function process_user_input(line, stream_sink, _thread_id, read_only)
    local user_input = tostring(line or ""):match("^%s*(.-)%s*$")
    if user_input == "" then
        return ""
    end

    local turn_args = {
        user_input = user_input,
        stream_sink = stream_sink,
        read_only = (read_only == true),
        conversation_history = conversation_history,
        add_to_history = add_to_history,
    }
    if run_mode == "webui" and stream_sink and webui_stream_max_steps and webui_stream_max_steps > 0 then
        turn_args.max_steps_override = webui_stream_max_steps
    end

    return agent_runtime.run_turn(turn_args)
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
