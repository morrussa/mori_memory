# Mori Memory Core

这个仓库已被收敛成一个**可嵌入的 memory 核心库**：接收每一轮对话的元信息（尤其是向量），并根据当前 topic 状态返回**编译后的上下文序列**。

- 不包含 agent loop / LangGraph / WebUI
- 不负责 embedding 计算（由宿主系统提供 `user_vec` / `query_vec`）
- 保留加速：HNSW（可选）+ SIMD 点积/余弦（可选）

## API（LuaJIT）

```lua
local memory = require("mori_memory")

-- 写入一轮（当 user_input 非空时必须提供 user_vec）
local ingest = memory.ingest_turn({
  user_input = "你好",
  assistant_text = "你好！",
  user_vec = user_vec, -- float[] table
})

-- 编译下一轮上下文（推荐提供 query_vec；否则会用 user_vec 或当前 topic 质心）
local blocks = memory.compile_context({
  turn = (ingest.turn or 0) + 1,
  user_input = "继续说",
  query_vec = query_vec, -- float[] table
  max_selected_turns = 6,
})

-- blocks: { {role="system", content="..."}, ... }
```

进程退出前可调用 `memory.shutdown()` 触发落盘。

## Native 加速

### HNSW（topic centroid 索引，可选）

构建 `module/mori_hnsw.so`：

```bash
./scripts/build_hnsw_module.sh
```

如果该 `.so` 不存在，系统不会报错，只会退回纯 Lua 扫描（更慢）。

### SIMD 点积/余弦（可选）

`module/simdc_math.so` 会在加载 `module/tool.lua` 时自动尝试加载；失败则使用 LuaJIT fallback。

## 数据落盘

默认写入到：

- `memory/history.txt`
- `memory/topic.bin`
- `memory/v4/topic_graph/`

