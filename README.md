# Mori Memory Core

这个仓库已被收敛成一个**可嵌入的 memory 核心库**：接收每一轮对话的元信息（尤其是向量），并根据当前 topic 状态返回**编译后的上下文序列**。

- 不包含 agent loop / LangGraph / WebUI
- embedding 由宿主系统提供（传入 `user_vec` / `query_vec`，或注入 embedder 回调）
- 保留加速：HNSW（可选）+ SIMD 点积/余弦（可选）

## API（LuaJIT）

```lua
local memory = require("mori_memory")

-- 方式 A：写入时直接传入 embedding
local ingest = memory.ingest_turn({
  user_input = "你好",
  assistant_text = "你好！",
  user_vec = user_vec, -- float[] table
})

-- 方式 B：注入 embedder（memory 内部会在缺 vec 时调用）
-- memory.set_embedder(function(text, mode) ... return float[] end)

-- 编译下一轮上下文（推荐提供 query_vec；否则会用 embedder/user_vec/topic 质心）
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

构建 `module/simdc_math.so`：

```bash
./scripts/build_simdc_math.sh
```

也可以一次性构建两个 native 模块：

```bash
./scripts/build_native_modules.sh
```

加载 `module/tool.lua` 时会自动尝试加载 `module/simdc_math.so`；失败则使用 LuaJIT fallback。

## 数据落盘

默认写入到：

- `memory/history.txt`
- `memory/topic.bin`
- `memory/v4/topic_graph/`
