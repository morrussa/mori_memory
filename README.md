# mori

*mori就是mori，memeto mori!*

这是heat_mem的重置版，因为我不喜欢python，就这么简单。

要求lupa支持luaJIT，因为向量计算余弦近似度我是用c库+FFI强算的

# 可能有用的命令

先退出虚拟环境（如果当前还在里面）
deactivate

rm -rf .venv

python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt


CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120" \
CUDACXX=/usr/local/cuda-13.1/bin/nvcc \
CC=/usr/bin/gcc-14 \
CXX=/usr/bin/g++-14 \
FORCE_CMAKE=1 \
cmake -S . -B build $CMAKE_ARGS

--这个是我自己的cuda和gcc环境

git clone https://luajit.org/git/luajit-2.0.git luajit
cd luajit
make
cd ..
rm -rf build dist lupa.egg-info lupa/*.so lupa/*.c
pip install lupa --no-binary :all: --verbose --no-cache-dir --force-reinstall

qwen3.5支持pr：

## 本地 No-Build WebUI（当前默认 webui 模式）

`MORI_RUN_MODE=webui` 现在不再代理 `llama.cpp` 自带页面，而是直接托管仓库内
`module/frontend`（no-build 前端），聊天走 Mori 原生接口。

启动示例：

```bash
MORI_RUN_MODE=webui \
MORI_WEBUI_BRIDGE_HOST=127.0.0.1 \
MORI_WEBUI_BRIDGE_PORT=8080 \
.venv/bin/python main.py
```

访问：

```text
http://127.0.0.1:8080
```

常用环境变量：

- `MORI_RUN_MODE=cli|webui`：运行模式（默认 `cli`）
- `MORI_WEBUI_BRIDGE_HOST` / `MORI_WEBUI_BRIDGE_PORT`：本地 WebUI 监听地址（默认 `127.0.0.1:8080`）
- `MORI_FRONTEND_ROOT`：本地前端目录（默认 `module/frontend`）
- `MORI_LARGE_SERVER_HOST` / `MORI_LARGE_SERVER_PORT`：底层主模型 `llama-server` 地址（默认 `127.0.0.1` + 自动端口）
- `MORI_LARGE_SERVER_WEBUI=0|1`：主模型服务是否启用自带 WebUI（`MORI_RUN_MODE=webui` 时强制 `0`）
- `MORI_LARGE_SERVER_JINJA=0|1`：主模型 `--jinja`（默认 `1`）
- `MORI_LARGE_SERVER_API_KEY`：上游 `llama-server` API key（未设置时自动生成）
- `MORI_LARGE_SERVER_GPU_LAYERS`：主模型 `--gpu-layers`（默认 `all`）
- `MORI_EMBED_SERVER_GPU_LAYERS`：embedding 模型 `--gpu-layers`（默认 `0`）
- `MORI_LARGE_SERVER_GPU_FALLBACK_CPU=0|1`：主模型 `gpu_layers=all` OOM 时是否自动回退 `0`（默认 `1`）
- `MORI_LLAMA_SERVER_LOG_TO_FILE=0|1`：是否将 `llama-server` 输出写入 `logs/llama_server_*.log`（默认 `1`）
- `MORI_WEBUI_STREAM_MAX_STEPS`：仅对流式请求生效的 `agent_runtime.max_steps` 覆盖（默认 `3`，设 `0` 关闭覆盖）

本地 WebUI API：

- `GET /health` -> `{"status":"ok"}`
- `GET /mori/session/status` -> 单会话状态（`mode=single`, `session=mori`），并包含 `model_name`、`upload_dir` 与 `upload_limits`
- `POST /mori/chat`：非流式  
  请求：`{"message":"...", "thread_id":"mori", "files":[...]}`  
  `files[]` 支持 `{"name":"a.txt","content_base64":"..."}`（可仅上传文件不填 message）
- `POST /mori/chat/stream`：SSE 流式  
  请求同上；事件：`token / uploads / status / thinking / tool_call / tool_result / done`，结束 `{"type":"done"}` + `[DONE]`
- `POST /v1/chat/completions`：已废弃，返回 `410` 并提示迁移到 `/mori/chat` 或 `/mori/chat/stream`

说明：

- `MORI_RUN_MODE=webui` 下，后端是单会话写入模型（固定会话 `mori`），不再使用 primary/observer 分流。
- 不再依赖 `llama.cpp` 自带 WebUI 页面注入与 IndexedDB 历史回灌。
- WebUI 上传文件会自动写入 `./workspace/download/`（可用文件工具从 `download/` 前缀读取）。
- 上传限额可通过环境变量调整：  
  `MORI_WEBUI_UPLOAD_MAX_FILES`、`MORI_WEBUI_UPLOAD_MAX_FILE_BYTES`、`MORI_WEBUI_UPLOAD_MAX_TOTAL_BYTES`、`MORI_WEBUI_MAX_BODY_BYTES`
- 以下变量在新 webui 模式下已废弃并忽略：  
  `MORI_WEBUI_PRIMARY_SESSION`、`MORI_WEBUI_NON_PRIMARY_POLICY`、`MORI_WEBUI_SESSION_HEADER`、`MORI_WEBUI_THREAD_HEADER`、`MORI_WEBUI_PRIMARY_IDLE_TTL_SEC`、`MORI_WEBUI_SESSION_DEBUG`。

## AgentRuntime / ToolRegistry（一次性替换版）

当前主链路已切到（按职责分层）：
- `module/agent/runtime.lua`：多步闭环状态机（循环 `BUILD_CONTEXT -> GENERATE -> PLAN_TOOLS -> EXECUTE_TOOLS`，最后 `PERSIST`）
  - 每轮绑定平级子步骤：`general-purpose | explore | plan`（支持扩展注册）
  - 支持 qwen-agent 风格 “Action -> Observation” 轨迹回注
  - 主模型仅负责聊天/总结；工具调用统一由二阶段 `tool_planner` 产出并执行
- `module/agent/tool_planner.lua`：二阶段工具规划（只产出工具调用，显式消费上一轮 observation/trace）
- `module/agent/tool_registry.lua`：工具执行与 pending context（仅 notebook 工具）
- `module/agent/context_window.lua`：按 token 预算裁剪上下文（模板后精确计数）
- `module/agent/substep.lua`：子步骤注册表与路由器（默认内置 `general-purpose / explore / plan`）
- `module/agent/tool_parser.lua`：统一工具调用解析层（兼容 Lua table / Qwen 符号 / ReAct 风格；不再解析 XML/JSON 调用协议）
- `module/memory/*.lua`：记忆与检索子系统（history/topic/recall/store/heat/cluster/adaptive/saver）
- 工具参数协议：Lua table 是第一公民（`arguments={...}`）；JSON 仅保留 Python 边界兼容解析，不作为模型/规划器协议。

路径约束：
- 仅保留新路径：`module/agent/*.lua` 与 `module/memory/*.lua`。

### Token 滑窗（精确计数）

- 已移除 `pipeline.lua` 固定轮数窗口（`MAX_HISTORY_TURNS`）主路径
- 现在使用 `py_pipeline:count_chat_tokens(messages)` 做模板后精确计数：
  - 先调用 `/apply-template`
  - 再调用 `/tokenize`
- Fail-close：token 计数失败会直接抛错，不回退近似估算
- 默认优先丢弃：`memory_context -> tool_context`，`plan_bom` 默认固定保留（必要时自动压缩）
- 当历史对话被滑窗丢弃时，会自动注入 `【历史自动压缩】` 块，保留早期任务/计划线索

### 新配置（`module/config.lua`）

```lua
runtime = {
    models = {
        base_dir = "/home/morusa/AI/mori/model/",
        large_model = "Qwen3.5-9B-Q6_K.gguf",
        embedding_model = "Qwen3-Embedding-0.6B-Q8_0.gguf",
    },
    demo_chat = {
        max_tokens = 1024,
        temperature = 0.5,
        seed_min = 114,
        seed_max = 514,
    },
},

agent = {
    max_steps = 4,
    input_token_budget = 12000,
    completion_reserve_tokens = 1024,
    substep_default = "general-purpose",
    substep_auto_route = true,
    substep_route = {
        auto_route = true,
        plan_keywords = { "架构", "规划", "方案", "设计", "plan", "roadmap", "architecture" },
        explore_keywords = { "探索", "搜索", "查找", "定位", "关键词", "代码库", "文件", "grep", "rg", "ripgrep" },
    },
    substeps = {
        ["general-purpose"] = {
            label = "general-purpose",
            description = "通用任务处理、代码搜索、多步骤任务",
            system_prompt = "优先给出可执行结果；必要时分步推进，并保持每一步可验证。",
            planner = {},
        },
        explore = {
            label = "Explore",
            description = "快速探索代码库、查找文件、搜索关键词",
            system_prompt = "先快速定位范围并收集证据，再基于证据给出结论。",
            planner = {
                planner_gate_mode = "always",
                planner_default_when_missing = true,
            },
        },
        plan = {
            label = "Plan",
            description = "架构规划、实现方案设计",
            system_prompt = "先给出架构与实施路径，再细化模块边界、风险和验收方法。",
            planner = {},
        },
    },
    token_count_mode = "templated_exact",
    context_drop_order = "memory_tool_plan",
    plan_bom_pinned = true,
    plan_bom_compact_min_chars = 120,
    history_auto_compress = true,
    history_auto_compress_min_dropped_pairs = 1,
    history_auto_compress_max_pairs = 24,
    history_auto_compress_user_chars = 64,
    history_auto_compress_assistant_chars = 96,
    history_auto_compress_max_chars = 1400,
    history_auto_compress_min_chars = 220,
    continue_on_tool_context = true,
    max_context_refine_steps = 2,
    continue_on_tool_failure = true,
    max_failure_refine_steps = 2,
    llm_temperature = 0.75,
    llm_seed_min = 1,
    llm_seed_max = 2147483647,
    planner_gate_mode = "assistant_signal",
    planner_default_when_missing = false,
    function_choice = "auto",
    supported_tool_acts = {
        upsert_record = true,
        query_record = true,
        delete_record = true,
        list_agent_files = true,
        read_agent_file = true,
        read_agent_file_lines = true,
        search_agent_file = true,
        search_agent_files = true,
    },
    parallel_function_calls = true,
    include_tool_observation_trace = true,
    tool_trace_max_steps = 4,
    tool_trace_max_chars = 1200,
}
```

补充语义：
- `upsert_max_per_turn` / `query_max_per_turn` 为同一 `turn` 内跨 `step` 累计预算，不会在多步循环中按 step 重置。
- 工具调用统一走二阶段 planner；主回复只负责给出 Lua table 计划信号（`{act="plan"}` / `{act="no_plan"}`）。
- 计划信号提示采用运行时动态注入：仅首轮注入，进入后续子步骤轮后不再重复注入。
- 子步骤为平级架构：`general-purpose`（通用任务）、`explore`（快速探索代码/文件/关键词）、`plan`（架构与实现方案）。
- 子步骤只影响 planner/tool 策略，不会把“子步骤展开文本”注入主 LLM 上下文；主 LLM 只消费常规上下文与工具返回结果。
- `settings.agent.substeps` 可注册更多子步骤；`settings.agent.substep_default` 控制默认类型；`settings.agent.substep_route` 控制自动路由关键词；每个子步骤可通过 `planner` 字段覆盖规划策略。
- `settings.agent.planner_gate_mode = "assistant_signal"`：按主回复末行 Lua table 计划信号决定是否进入 planner（可选 `always`）。
- `settings.agent.planner_default_when_missing = false`：缺失 `{act="plan"}` / `{act="no_plan"}` 时默认不进入 planner（建议保持 `false`）。
- `settings.agent.function_choice`：`auto|none|query_record|upsert_record|delete_record`，用于约束本轮可执行工具。
- `settings.agent.parallel_function_calls=false`：单轮只保留第一条工具调用（qwen-agent 对齐）。
- `settings.keyring.tool_calling.parallel_execute_enabled = true`：连续 `query_record` 会按批次并行调度（默认批大小 `parallel_query_batch_size=4`）。
- `settings.keyring.tool_calling.retry_transient_max = 1`：仅“可恢复错误”自动重试；预算/参数类错误默认不重试（`retry_budget_max=0`、`retry_validation_max=0`）。
- 记忆通道文件策略（专业建议）：
  - `settings.keyring.memory_input.max_chars = 2048`：记忆检索/事实提取输入硬上限
  - `settings.keyring.memory_input.recall_file_payload_mode = "ignore"`：召回阶段忽略附件正文
  - `settings.keyring.memory_input.fact_file_payload_mode = "ignore"`：原子事实提取阶段忽略附件正文
  - 可选 `filename_only`：仅保留文件名清单，不传正文
- WebUI 附件策略（默认）：
  - 用户附件正文会被抽离并保存到 `./workspace/<thread>/...`
  - 用户输入上下文仅保留“附件清单 + 路径”，不再整段注入正文
  - agent 可通过 `list_agent_files` / `read_agent_file` / `read_agent_file_lines` / `search_agent_file` / `search_agent_files` 按需读取与定位

### 新日志字段

- `context_tokens`
- `kept_pairs`
- `dropped_blocks`
- `tool_calls_count`
- `tool_calls_count source=planner_pass|planner_skipped|planner_error|none`
- `tool_exec ... parallel_batches=... retries=...`
- `continue reason`
- `工具上下文无增量，提前收敛`
- `agent_state_end`



——————————

# 1. 先更新到最新版（Blackwell 支持在近期 commit 才完整）
git pull origin master   # 或 git checkout master && git pull

# 2. 清舊 build 目錄，避免殘留 CMake cache 干擾
rm -rf build

# 3. 設定環境變數（確保 nvcc 正確）
export CUDACXX=/usr/local/cuda-13.0/bin/nvcc
export CC=/usr/bin/gcc-14
export CXX=/usr/bin/g++-14
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# 3.5 傻逼黄仁勋bug不修
sudo cp /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h.bak

sudo sed -i 's/rsqrt(double x);/rsqrt(double x) noexcept(true);/' /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h
sudo sed -i 's/rsqrtf(float x);/rsqrtf(float x) noexcept(true);/' /usr/local/cuda-13.0/targets/x86_64-linux/include/crt/math_functions.h

# 4. CMake 配置（關鍵旗標）
cmake -S . -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DGGML_NATIVE=OFF \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA_FORCE_CUBLAS=ON \
  -DLLAMA_CURL=OFF

# 如果 120 報 unsupported，試換成 120f（Blackwell 優化版，常見於 2025-12 後 commit）
# -DCMAKE_CUDA_ARCHITECTURES=120f

# 5. 建置（用多核加速，nproc 自動取核心數）
cmake --build build --config Release -j$(nproc)

# 6. 可選：安裝到系統（或直接用 build/bin/ 下的 binary）
# cmake --install build   # 如果想裝到 /usr/local/bin 等


---

source .venv/bin/activate
MORI_RUN_MODE=webui \
MORI_WEBUI_BRIDGE_HOST=127.0.0.1 \
MORI_WEBUI_BRIDGE_PORT=8080 \
.venv/bin/python main.py
