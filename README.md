# Mori

Mori is memeto..ol.. ri.

Mori 的核心机制并非选取agent想要查询的 prompt，而是以 `memory` 为中心，在每一轮推理前围绕任务目标、工作状态与历史证据重新组织上下文。

从 `module/` 的实现出发，可以将系统概括为四个部分：

- `memory`：负责对话状态分层、长期记忆写入与按需召回
- `graph`：负责单轮执行流程、工具循环、上下文装配与收尾
- `episode / checkpoint / session`：负责任务连续性恢复与运行态持久化
- `frontend / native`：提供交互界面与检索性能支撑

## 系统主流程

单轮请求的执行链路如下：

```text
ingest_node
  -> task_node
  -> recall_node
  -> context_node
  -> planner_node
       -> tool_exec_node <-> planner_node
       -> repair_node -> planner_node / responder_node / finalize_node
  -> responder_node
  -> finalize_node
  -> writeback_node
  -> persist_node
```

在这条链路中，`memory` 贯穿前后两个阶段：

- 前半段负责判断是否需要召回记忆，以及应当召回哪些证据
- 后半段负责将本轮对话沉淀为可复用状态，并更新记忆反馈关系

## 记忆系统结构

记忆系统并非单一存储，而是由四层状态共同构成。

### 1. `history`

文件：`module/memory/history.lua`

- 以 turn 为单位保存完整的 user / assistant 对话
- 存储文件为 `memory/history.txt`
- 其职责是保存原始对话账本，而非承担长期语义记忆
- recall 阶段最终会回到该层，将命中的 memory 展开为可读对话片段

### 2. `topic`

文件：`module/memory/topic.lua`

- 每轮基于用户输入的 embedding 更新话题状态
- 通过上一轮相似度与头尾漂移共同判断是否发生话题切换
- 每个话题对应稳定 anchor，形式为 `S:<start_turn>`
- 该层负责描述“本轮属于哪条语义轨道”，而不是直接保存事实内容

### 3. `topic_graph`

文件：`module/memory/topic_graph.lua`

`topic_graph` 是长期记忆的主体，`module/memory/store.lua` 仅提供兼容性封装。

该模块保存的不是整段对话文本，而是带结构的 memory 节点与 topic 图：

- memory 节点：`vec + turns + text + facets + topic_anchor + cluster_id`
- topic 节点：每个 anchor 下的 centroid、memory 列表与局部状态 `local_state`
- topic 边：`transition / recall / adopt`
- `memory_next`：描述跨轮 memory 的接续关系
- HNSW 索引：用于 topic centroid 的快速 seed 检索
- Deep ARTMAP 局部状态：用于 topic 内候选收集与反馈强化

该层有两个重要特征：

- 长期记忆写入对象是“原子事实”，而不是整段 assistant 回复
- 当新事实与既有 memory 的相似度高于 `merge_limit` 时，会执行合并并补充 turn，而不是无条件新增节点

### 4. `episode`

文件：`module/episode/*`

- 每轮执行结束后会生成一个 episode
- episode 保存任务目标、工具链路、文件读写、最终回复与 memory writeback 摘要
- 该层服务于任务连续性恢复，而非长期事实记忆本体

## 记忆写入链路

一轮结束后，记忆相关写入按照如下顺序发生。

### 1. 更新 topic

`persist_node` 先计算 `current_turn = history.get_turn() + 1`，随后执行：

- `topic.add_turn(current_turn, user_input, user_embedding)`
- 基于局部断裂与全局漂移判断是否切分话题
- 为本轮建立或延续稳定 topic anchor

因此，topic 的更新先于 history 的写入。

### 2. 写入 history

随后执行：

- `history.add_history(user_input, final_text)`
- `topic.update_assistant(current_turn, final_text)`

因此，完整对话仍然保存在 `history` 中，后续 recall 所展开的“第 N 轮用户/助手说了什么”也来自这一层。

### 3. 抽取原子事实并写入长期记忆

`writeback_node` 调用 `module/graph/memory_core.lua` 完成事实抽取：

- 输入为本轮 `user_input + final_text`
- LLM 被约束为只输出 Lua 字符串数组
- 提取目标是可长期复用的事实，优先覆盖偏好、约束、身份、长期计划与持续需求
- 若抽取失败，会进入 repair；repair 仍失败时，使用用户输入进行兜底

之后 `save_ingest_items` 会：

- 对每条 fact 计算 embedding
- 调用 `topic_graph.add_memory(...)`
- 将该 fact 挂接到本 turn 对应的 topic anchor
- 自动生成 facets
- 在满足相似度条件时执行 merge，否则新增 memory 节点

### 4. 识别本轮实际采用的记忆

该步骤并非人工标注，而是基于回复结果反推：

- `recall.infer_adopted_memories(final_text, recall_state)`
- 对最终回复与 recall fragments 进行 embedding 相似度比较
- 推断本轮回答中实际采用过的 memory

### 5. 进行反馈学习

`topic_graph.observe_feedback(...)` 会同步更新多类关系：

- 当前 topic 下 memory 的 prior
- topic 之间的 `recall / adopt / transition` 边
- `memory_next` 序列连接
- topic 内 Deep ARTMAP 的 bundle / memory 强化

因此，记忆系统并非仅在写入阶段追加数据，还会根据“召回了什么”与“真正采用了什么”持续修正其内部路由关系。

## 记忆召回链路

### 1. Recall gating

入口位于 `module/memory/recall.lua`。

系统不会在每轮请求上无条件召回长期记忆，而是先计算 recall score。评分信号包括：

- 是否出现“之前 / 上次 / 以前 / recall / remember”等显式回忆意图
- 是否包含技术关键词
- 输入长度是否达到阈值
- 是否表现出焦虑或求助语气 embedding
- 是否出现“不要回忆 / 不查记忆”等抑制信号
- 对过短且不具备“继续 / 刚才”等延续意图的输入进行扣分

只有当分数超过阈值后，系统才进入 `topic_graph.retrieve(...)`。

### 2. 先检索 topic，而非直接检索全库 memory

`topic_graph.retrieve(...)` 的第一阶段是选取 seed topic：

- 使用 HNSW 在 topic centroid 上进行近邻搜索
- 对当前 topic 施加额外加权
- 沿 `topic_edges` 执行 bridge 扩展
- 在受限的 load budget 内加载候选 topic

因此，召回的第一层单位是 topic，而不是裸 memory 向量。

### 3. 在 topic 内选择 evidence memory

对于每个候选 topic，系统会进一步执行局部检索：

- 先通过 Deep ARTMAP 的 `collect_candidates(...)` 收集局部候选
- 若候选 topic 即为当前 topic，还会叠加 `memory_next` 提供的单轮动量
- 仅在局部候选不足时，才退回该 topic 下的全部 memory

候选 memory 会综合以下因素重新排序：

- query 与 memory 的语义相似度
- topic 内局部匹配分
- facets 覆盖增益
- cluster / bundle 多样性
- 已选 memory 之间的相似度饱和惩罚

因此，Mori 的 recall 机制本质上是“topic 路由 + topic 内证据选择”，而不是简单的向量 top-k。

### 4. 召回结果会重新展开为历史证据

命中 memory 后，系统会依据其关联的 `turns` 回到 `history`：

- 选取最相关的 turn
- 生成 `fragments`
- 产出 `memory_context`

注入 prompt 的内容形式如下：

```text
【相关记忆】
第12轮 用户：...
助手：...
```

因此，大模型最终接收到的是可读的历史证据，而不是抽象 memory id 或单纯 embedding 命中项。

## 记忆如何进入上下文

上下文组装由 `module/graph/context_builder.lua` 完成。

system prompt 的主要组成块包括：

- Base System Prompt
- Project Knowledge
- ActiveTask / TaskContract
- WorkingMemory
- MemoryContext
- TaskContext
- RecentEpisodes
- ToolContext
- PlannerContext

其中与记忆系统直接相关的部分主要有两类：

- `MemoryContext`：来自 recall 阶段展开的历史证据
- `WorkingMemory`、`RecentEpisodes`、`ToolContext`：与 recall 共同参与最终上下文装配

此外，`history` 虽在磁盘中完整保留，但进入 prompt 时并非完整回放，而是经过预算感知的裁剪：

- 最近对话对优先保留
- 支持 `full / slight / heavy / none` 多档压缩
- 超出预算后优先丢弃更早的 pair
- 工具结果还会经过 `context_manager` 的截断、去重与预算控制

## 持久化状态

系统主要涉及以下存储对象：

```text
memory/history.txt
memory/topic.bin
memory/v4/topic_graph/state.lua
memory/v4/topic_graph/vectors.bin
memory/v4/topic_graph/hnsw/
memory/episodes/items/*.lua
memory/episodes/index.lua
memory/v3/graph/checkpoints/*.bin
```

需要区分“内存状态更新”与“实际刷盘”两个时机：

- `topic.lua` 在每轮更新时直接保存 `memory/topic.bin`
- `episode.store.save()` 在 `persist_node` 内完成当轮写盘
- `history`、`topic_graph` 与 `graph checkpoint` 通过 `module/memory/saver.lua` 统一 flush
- pipeline 正常退出时会调用 `saver.on_exit()`，并执行 `py_pipeline:pack_state()`

因此，并非所有记忆相关状态都会在 `persist_node` 中立即完成全量落盘。

## 其他模块概览

### `graph`

- `module/graph/graph_runtime.lua` 负责整轮执行与节点跳转
- `task_node` 负责将输入分类为 `same_task_step / same_task_refine / hard_shift / meta_turn`
- `planner_node + tool_exec_node + repair_node` 负责工具循环
- 每个节点结束后都会记录 checkpoint 与 trace，以支持恢复与调试

### `episode`

- 不承担长期事实存储
- 更接近任务运行快照
- 为后续任务连续性判断、carryover 与 recent episode summary 提供恢复材料

### `frontend`

- `module/frontend/` 提供轻量本地前端
- 负责交互、消息展示与模型 worker 调用
- 不属于记忆系统主体逻辑

### `native/hnsw`

- 为 topic centroid 检索提供近邻搜索加速
- 是 `topic_graph` 的性能支撑模块，而非业务主体

## 总结

Mori 的记忆系统可以概括为以下四个步骤：

1. 将对话拆分为 `history + topic + topic_graph + episode`
2. 在 recall 阶段先定位 topic，再在 topic 内选择证据 memory
3. 将命中的 memory 重新展开为历史片段并注入 prompt
4. 在回合结束后回写原子事实、采用反馈与任务快照

因此，Mori 的核心并不是将一段长期记忆固定塞入 prompt，而是围绕任务语义对上下文进行按需重组，并持续修正其内部记忆路由。
