local tool = require("module.tool")
local history = require("module.memory.history")
local topic   = require("module.memory.topic")
local memory = require("module.memory.store")
local ghsom = require("module.memory.ghsom")
local math = require("math")
local saver =require("module.memory.saver")
local recall = require("module.memory.recall")
local adaptive = require("module.memory.adaptive")
local topic_predictor = require("module.memory.topic_predictor")
local app_config = require("module.config")
local graph_runtime = require("module.graph.graph_runtime")
local run_mode = tostring(MORI_RUN_MODE or "cli")
local graph_cfg = ((app_config.settings or {}).graph or {})
local cli_boot_demo = false

local function read_env_int(name, fallback, min_value)
    local raw = os.getenv(name)
    if not raw or raw == "" then
        return tonumber(fallback)
    end
    local n = tonumber(raw)
    if not n then
        return tonumber(fallback)
    end
    if min_value and n < min_value then
        n = min_value
    end
    return math.floor(n)
end

local function read_env_bool(name, fallback)
    local raw = os.getenv(name)
    if not raw or raw == "" then
        return fallback == true
    end
    local v = tostring(raw):lower()
    if v == "1" or v == "true" or v == "yes" or v == "on" then
        return true
    end
    if v == "0" or v == "false" or v == "no" or v == "off" then
        return false
    end
    return fallback == true
end

do
    graph_cfg.tool_loop_max = read_env_int("MORI_GRAPH_TOOL_LOOP_MAX", graph_cfg.tool_loop_max or 5, 1)
    graph_cfg.max_nodes_per_run = read_env_int("MORI_GRAPH_MAX_NODES_PER_RUN", graph_cfg.max_nodes_per_run or 128, 20)
    graph_cfg.agent = graph_cfg.agent or {}
    graph_cfg.agent.remaining_steps = read_env_int(
        "MORI_GRAPH_AGENT_REMAINING_STEPS",
        (graph_cfg.agent or {}).remaining_steps or 25,
        1
    )
    graph_cfg.agent.max_tokens = read_env_int(
        "MORI_GRAPH_AGENT_MAX_TOKENS",
        (graph_cfg.agent or {}).max_tokens or 1024,
        64
    )
    graph_cfg.streaming = graph_cfg.streaming or {}
    graph_cfg.streaming.token_chunk_chars = read_env_int(
        "MORI_GRAPH_STREAM_TOKEN_CHUNK_CHARS",
        (graph_cfg.streaming or {}).token_chunk_chars or 24,
        1
    )
    graph_cfg.repair = graph_cfg.repair or {}
    graph_cfg.repair.max_attempts = read_env_int(
        "MORI_GRAPH_REPAIR_MAX_ATTEMPTS",
        (graph_cfg.repair or {}).max_attempts or 2,
        0
    )
    graph_cfg.cli = graph_cfg.cli or {}
    graph_cfg.cli.debug_trace = read_env_bool("MORI_GRAPH_DEBUG_TRACE", (graph_cfg.cli or {}).debug_trace == true)
    cli_boot_demo = read_env_bool("MORI_CLI_BOOT_DEMO", false)
end

local runtime_cfg = (app_config.settings or {}).runtime or {}
local runtime_defaults = (app_config.defaults or {}).runtime
if type(runtime_defaults) ~= "table" then
    runtime_defaults = runtime_cfg
end
local model_cfg = runtime_cfg.models or {}
local model_defaults = runtime_defaults.models or {}
local demo_cfg = runtime_cfg.demo_chat or {}
local demo_defaults = runtime_defaults.demo_chat or {}

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
ghsom.load()
saver.mark_dirty()
history.load()
adaptive.load()
topic_predictor.load()
recall.init_all_sentiment_vectors()

print("[Lua] Pipeline started.")

topic.init()



if run_mode ~= "webui" and cli_boot_demo then
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

local function process_user_input(line, stream_sink, _unused, read_only, uploads)
    local user_input = tostring(line or ""):match("^%s*(.-)%s*$")
    if user_input == "" then
        return ""
    end

    local turn_args = {
        user_input = user_input,
        stream_sink = stream_sink,
        read_only = (read_only == true),
        uploads = uploads or {},
    }

    local text = graph_runtime.run_turn(turn_args)
    local debug_trace = (((graph_cfg or {}).cli or {}).debug_trace) == true
    if debug_trace then
        local trace_summary = _G.mori_last_trace_summary
        if type(trace_summary) == "table" then
            print(string.format("[GraphTrace] run_id=%s node_count=%s tool_loops=%s repair_attempts=%s",
                tostring(trace_summary.run_id or ""),
                tostring(trace_summary.node_count or 0),
                tostring(trace_summary.tool_loops or 0),
                tostring(trace_summary.repair_attempts or 0)
            ))
        end
    end
    return text
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

        local assistant_text = process_user_input(line, nil, nil, false)
        assistant_text = tostring(assistant_text or "")
        if assistant_text ~= "" then
            print(assistant_text)
        end

        ::continue::
    end
end

print("[Lua] Pipeline execution finished.")
