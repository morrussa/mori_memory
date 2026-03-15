local tool = require("module.tool")
local app_config = require("module.config")
local memory = require("mori_memory")

local run_mode = tostring(MORI_RUN_MODE or "cli"):lower()

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

-- 配置模型路径（统一从 config.lua 读取）
local model_base = tostring(model_cfg.base_dir or model_defaults.base_dir or "")
local large_model_path = join_model_path(model_base, model_cfg.large_model or model_defaults.large_model)
local embedding_model_path = join_model_path(model_base, model_cfg.embedding_model or model_defaults.embedding_model)
local draft_enabled = (model_cfg.draft_enabled ~= nil) and (model_cfg.draft_enabled == true) or (model_defaults.draft_enabled == true)
local draft_model_path = ""
if draft_enabled then
    draft_model_path = join_model_path(model_base, model_cfg.draft_model or model_defaults.draft_model)
end
local spec_cfg = {
    enabled = draft_enabled == true and draft_model_path ~= "",
    draft_gpu_layers = tonumber(model_cfg.draft_gpu_layers or model_defaults.draft_gpu_layers or 0) or 0,
    draft_max = tonumber(model_cfg.draft_max or model_defaults.draft_max or 0) or 0,
    draft_min = tonumber(model_cfg.draft_min or model_defaults.draft_min or 0) or 0,
    draft_p_min = tonumber(model_cfg.draft_p_min or model_defaults.draft_p_min or 0) or 0,
    draft_ctx_size = tonumber(model_cfg.draft_ctx_size or model_defaults.draft_ctx_size or 0) or 0,
}

print("[Lua] large_model path: " .. tostring(large_model_path))
print("[Lua] embedding_model path: " .. tostring(embedding_model_path))
print("[Lua] draft_model path: " .. tostring(draft_model_path))
py_pipeline:load_models(large_model_path, embedding_model_path, draft_model_path, spec_cfg)

print("[Lua] Pipeline started (memory-only).")

local function trim(s)
    return tostring(s or ""):match("^%s*(.-)%s*$")
end

local function get_gen_params()
    local demo_seed_min = math.floor(tonumber(demo_cfg.seed_min) or tonumber(demo_defaults.seed_min) or 114)
    local demo_seed_max = math.floor(tonumber(demo_cfg.seed_max) or tonumber(demo_defaults.seed_max) or 514)
    if demo_seed_max < demo_seed_min then
        demo_seed_min, demo_seed_max = demo_seed_max, demo_seed_min
    end
    return {
        max_tokens = math.floor(tonumber(demo_cfg.max_tokens) or tonumber(demo_defaults.max_tokens) or 1024),
        temperature = tonumber(demo_cfg.temperature) or tonumber(demo_defaults.temperature) or 0.6,
        seed = math.random(demo_seed_min, demo_seed_max),
    }
end

local function shutdown_pipeline()
    print("[Lua] 正在原子保存 memory 状态...")
    memory.shutdown()
end

local function process_user_input(line, _stream_sink, _unused, read_only, _uploads)
    local user_input = trim(line)
    if user_input == "" then
        return ""
    end

    local system_prompt = os.getenv("MORI_SYSTEM_PROMPT") or "你是一个简洁高效的对话助手。"
    local ctx_blocks = memory.compile_context({ user_input = user_input })
    local messages = {
        { role = "system", content = system_prompt },
    }
    for _, msg in ipairs(ctx_blocks or {}) do
        messages[#messages + 1] = msg
    end
    messages[#messages + 1] = { role = "user", content = user_input }

    local ok, assistant_text = pcall(function()
        return py_pipeline:generate_chat_sync(messages, get_gen_params()) or ""
    end)
    if not ok then
        return "[ERROR] generate_chat_sync failed: " .. tostring(assistant_text)
    end

    assistant_text = trim(tool.remove_cot(assistant_text))

    if read_only ~= true then
        memory.ingest_turn({
            user_input = user_input,
            assistant_text = assistant_text,
        })
    end

    return assistant_text
end

_G.mori_handle_user_input = process_user_input
_G.mori_shutdown = shutdown_pipeline

if run_mode == "lib" or run_mode == "server" or run_mode == "webui" then
    print(string.format("[Lua] MORI_RUN_MODE=%s，交互式控制台已禁用", run_mode))
else
    print("\n[Lua] Entering interactive loop. Type 'exit' to quit.")

    while true do
        io.write("> ")
        local line = io.read()
        if not line then
            print("\n[Lua] EOF received, exiting.")
            shutdown_pipeline()
            break
        end

        line = trim(line)
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
