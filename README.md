# Mori Graph V1（LangGraph 风格断代版）

## 1. 目标

本版本已将旧 Agent 链路重构为 `StateGraph` 风格执行内核，核心目标：

- 可追踪：节点级 trace（JSONL）
- 可恢复：节点级 checkpoint（二进制快照）
- 可测试：统一图语义回归入口

仅保留 `/mori/*` 路径兼容，协议已断代。

## 2. 执行图拓扑

主图固定节点顺序：

1. `ingest_node`
2. `context_node`
3. `router_node`
4. `recall_node`
5. `planner_node`
6. `tool_exec_node`
7. `repair_node`
8. `responder_node`
9. `writeback_node`
10. `persist_node`
11. `end`

循环约束：

- `tool_loop` 最大 `5` 次（`graph.tool_loop_max`）
- `repair` 最大 `2` 次（`graph.repair.max_attempts`）

## 3. 目录结构（核心）

- `module/graph/state_schema.lua`：`GraphStateV1`
- `module/graph/graph_runtime.lua`：图执行引擎
- `module/graph/graph_builder.lua`：节点与转移
- `module/graph/nodes/*`：节点实现
- `module/graph/checkpoint_store.lua`：FFI 二进制快照
- `module/graph/trace_writer.lua`：JSONL trace
- `module/graph/tool_registry_v2.lua`：工具执行层（file + external provider）
- `module/graph/providers/*`：provider 抽象与 qwen 实现

旧 `module/agent/*` 核心链路已删除。

## 4. API（断代后）

### `POST /mori/chat`

- 仅支持 `multipart/form-data`
- 字段：`message` + `files[]`
- 返回：
  - `message`
  - `run_id`
  - `trace`
  - `uploads`（`name/path/bytes`）

### `POST /mori/chat/stream`

- 仅支持 `multipart/form-data`
- SSE 语义事件（`event:`）：
  - `run_start`
  - `node_start`
  - `node_end`
  - `tool_call`
  - `tool_result`
  - `token`
  - `status`
  - `error`
  - `uploads`
  - `done`
- 结束信号仅 `done`，不再发送 `[DONE]`

### `GET /mori/session/status`

返回 Graph V1 会话状态（无 thread/session 多实例语义）。

### `POST /v1/chat/completions`

在 `MORI_RUN_MODE=webui` 下返回 `410`（弃用）。

## 5. Tool 命名（V1）

文件工具统一命名为：

- `list_files`
- `read_file`
- `read_lines`
- `search_file`
- `search_files`

external provider 默认关闭，必须显式白名单启用。

## 6. Checkpoint 与 Trace

- Checkpoint：`memory/v3/graph/checkpoints/*`
- Trace：`memory/v3/graph/traces/*`
- 保留策略：不清理
- 启动策略：检测旧 `memory/state.zst` 时忽略旧执行态，不迁移

## 7. 环境变量（Graph）

- `MORI_GRAPH_TOOL_LOOP_MAX`
- `MORI_GRAPH_REPAIR_MAX_ATTEMPTS`
- `MORI_GRAPH_MAX_NODES_PER_RUN`
- `MORI_GRAPH_STREAM_TOKEN_CHUNK_CHARS`
- `MORI_GRAPH_DEBUG_TRACE`

## 8. 运行

### CLI

```bash
MORI_RUN_MODE=cli python3 main.py
```

### WebUI

```bash
MORI_RUN_MODE=webui python3 main.py
```

默认本地地址：`http://127.0.0.1:8080`

## 9. 测试

统一入口（LuaJIT）：

```bash
luajit scripts/run_graph_tests.lua
```

当前回归集合：

- 总数：60
- 分布：Memory 20 / File 20 / No-tool 10 / External 10
- 指标：命中率阈值 `>= 90%`（脚本内强校验）

## 10. 兼容性说明

- 保留路径兼容：`/mori/*`
- 不保留旧请求体/旧流协议/旧 thread 语义
- 全局入口保持：
  - `mori_handle_user_input`
  - `mori_shutdown`
