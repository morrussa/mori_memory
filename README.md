# mori

*mori就是mori，memeto mori!*

这是heat_mem的重置版，因为我不喜欢python，就这么简单。

要求lupa支持luaJIT，因为向量计算余弦近似度我是用FFI强算的

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

## llama.cpp WebUI 接入（已接到当前链路）

现在有两种 WebUI 方式：

1) 直连模式（不经过 Mori 链路）  
默认 `python main.py`（CLI 交互）时，主模型 `llama-server` 仍可开启自带 WebUI。

2) 链路模式（经过 Mori 实际链路，推荐）  
启动 `MORI_RUN_MODE=webui` 后，会起一个 HTTP 代理：
- 页面与静态资源仍来自 `llama-server` 自带 WebUI
- 但 `/v1/chat/completions` 会改走 `pipeline.lua` 的真实链路（记忆/召回/tool_calling）

链路模式启动示例：

```bash
MORI_RUN_MODE=webui \
MORI_WEBUI_BRIDGE_HOST=127.0.0.1 \
MORI_WEBUI_BRIDGE_PORT=8080 \
python main.py
```

启动后访问：

```text
http://127.0.0.1:8080
```

常用环境变量：

- `MORI_RUN_MODE=cli|webui`：运行模式（默认 `cli`）
- `MORI_WEBUI_BRIDGE_HOST` / `MORI_WEBUI_BRIDGE_PORT`：链路模式下对外 WebUI 地址（默认 `127.0.0.1:8080`）
- `MORI_WEBUI_PRIMARY_SESSION`：固定主会话 key（设置后仅该会话写入 Mori 链路）
- `MORI_WEBUI_NON_PRIMARY_POLICY=readonly_chain|readonly_upstream|reject`：非主会话策略（默认 `readonly_chain`，`readonly_upstream` 兼容映射为 `readonly_chain`）
- `MORI_WEBUI_SESSION_HEADER`：自定义会话 header（默认 `X-Mori-Session`）
- `MORI_WEBUI_THREAD_HEADER`：自定义 thread header（默认 `X-Mori-Thread`）
- `MORI_WEBUI_PRIMARY_IDLE_TTL_SEC`：自动占有模式下主会话空闲接管 TTL（默认 `1800` 秒）
- `MORI_WEBUI_SESSION_DEBUG=0|1`：打印会话 key 来源与主/非主判定（默认 `0`）
- `MORI_LARGE_SERVER_HOST` / `MORI_LARGE_SERVER_PORT`：底层主模型 `llama-server` 地址（默认 `127.0.0.1` + 自动端口）
- `MORI_LARGE_SERVER_WEBUI=1`：主模型服务启用 WebUI（链路模式会强制开启）
- `MORI_LARGE_SERVER_JINJA=1`：主模型启用 `--jinja`（默认开启）
- `MORI_LARGE_SERVER_API_KEY`：上游 `llama-server` API key（链路模式默认自动生成随机 key）
- `MORI_RUN_MODE=webui` 时默认不打印上游端口 URL，避免误进直连口

会话路由说明（WebUI 模式）：

- 主会话：走 Mori 链路（会写入 topic/history/memory/tool_calling）
- 非主会话：
  - 默认 `readonly_chain`：仍走 Mori 链路，但启用只读（只读记忆，不写 history/topic/memory/tool_calling）
  - 可选 `reject`：直接返回 409
- 会话 key 提取顺序：
  - `payload.metadata.{conversation_id,chat_id,session_id}`
  - `payload.{conversation_id,chat_id,session_id}`
  - `header[MORI_WEBUI_SESSION_HEADER]`
  - 若设置了固定主会话 key：回退到主会话 key（默认 `mori`）
  - 否则回退到指纹 `sha1(first_system + first_user + client_ip + user_agent)` 并告警
- 线程 key（用于前端透传显示，不参与后端状态分桶）提取顺序：
  - `payload.metadata.thread_id`
  - `payload.thread_id`
  - `header[MORI_WEBUI_THREAD_HEADER]`
  - 缺失时回退到当前会话 key（`fallback.session_key`）
- WebUI 启动引导（桥接层自动注入到 `/`）：
  - 检查前端 IndexedDB `Conversations` 是否存在名为 `Mori` 的会话
  - 不存在则创建主会话（ID = 主会话 key，默认 `mori`）
  - 若 `memory/history.txt` 存在且为 `HIST_V2`，会从该文件重建 `Mori` 会话消息树
- WebUI 请求自动透传当前会话：
  - 请求头：`X-Mori-Session`、`X-Mori-Thread`
  - 请求体 metadata：`conversation_id`、`thread_id`
  - 值来源：当前 URL hash `#/chat/<id>`，缺失时回退主会话 key
- 后端状态模式：单进程单会话（不按前端会话分桶）
- 长期记忆策略：
  - 非主会话只读，不写全局长期记忆
  - 主会话提取到的事实按现有流程全部写入全局长期记忆

诊断接口：

- `GET /mori/session/status`
  - 返回：`{ primary_session, policy, mode, ttl_sec, last_seen_ts, session_header, thread_header, primary_alias }`
- 聊天响应头：
  - `X-Mori-Session-Key`
  - `X-Mori-Session-Role=primary|observer`
  - `X-Mori-Session-Source`
  - `X-Mori-Thread-Key`
  - `X-Mori-Thread-Source`

### 额外依赖（venv）

- 对 `llama-server` 自带 WebUI 本身：通常不需要额外 Python 依赖（不是 venv 里的包）
- 关键是你已经正确编译了 `llama-server` 可执行文件，并确保当前 `LLAMA_SERVER_BIN` 指向它
- 建议确保在 `llama.cpp` 目录执行过：`cmake --build build -j --target llama-server`
- 已修复 `Jinja Exception: No user query found in messages`（内部二阶段调用已改为带 `user` 消息）
- 在 `MORI_RUN_MODE=webui` 下，底层上游端口会被 API key 保护（避免误打开直连端口绕过 Mori 链路）

## AgentRuntime / ToolRegistry（一次性替换版）

当前主链路已切到（按职责分层）：
- `module/agent/runtime.lua`：多步闭环状态机（循环 `BUILD_CONTEXT -> GENERATE -> PLAN_TOOLS -> EXECUTE_TOOLS`，最后 `PERSIST`）
- `module/agent/tool_planner.lua`：二阶段工具规划（只产出工具调用）
- `module/agent/tool_registry.lua`：工具执行与 pending context（仅 notebook 工具）
- `module/agent/context_window.lua`：按 token 预算裁剪上下文（模板后精确计数）
- `module/memory/*.lua`：记忆与检索子系统（history/topic/recall/store/heat/cluster/adaptive/saver）

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
agent = {
    max_steps = 4,
    input_token_budget = 12000,
    completion_reserve_tokens = 1024,
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
}
```

补充语义：
- `upsert_max_per_turn` / `query_max_per_turn` 为同一 `turn` 内跨 `step` 累计预算，不会在多步循环中按 step 重置。
- 记忆通道文件策略（专业建议）：
  - `settings.keyring.memory_input.max_chars = 2048`：记忆检索/事实提取输入硬上限
  - `settings.keyring.memory_input.recall_file_payload_mode = "ignore"`：召回阶段忽略附件正文
  - `settings.keyring.memory_input.fact_file_payload_mode = "ignore"`：原子事实提取阶段忽略附件正文
  - 可选 `filename_only`：仅保留文件名清单，不传正文

### 新日志字段

- `context_tokens`
- `kept_pairs`
- `dropped_blocks`
- `tool_calls_count`
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
python main.py
