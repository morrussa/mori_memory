# Thread-First Memory Design (LuaJIT First)

## 1. 固定目标

这份设计以以下约束为前提，这些约束不改：

- 不把“召回准确率”作为第一目标。
- 把“在合适情景下拉回相似、可用、能帮助继续理解的过去记忆”作为第一目标。
- 不把系统做成长期保存原始对话的日记系统。
- 在线主链路以 LuaJIT 为主，尽量不引入 Python。

这意味着系统的首要风险不是 miss，而是 contamination：

- 相似语义但不同互动流被混到一起。
- 不同用户/不同子会话的原子记忆被提前合并。
- LLM 看到“像同一个话题，但不是同一个互动关系”的上下文后发生理解混乱。


## 2. 设计结论

推荐路线：

`thread-first + delayed commit + layered associative memory + short-lived WAL`

关键判断：

- `thread / interaction stream` 是一级对象。
- `topic` 是二级投影，不是一级路由键。
- 允许消息在一段时间内处于“未决”状态。
- 只把已经稳定的片段投影成长期关联记忆。
- 落盘保存“可恢复状态”和“决策轨迹”，不是永久原始日记。

## 3. 当前实现状态

这份文档现在分两类内容：

- `已完成`：仓库里已经落地并跑过真实直播回放的能力。
- `下一步`：还没做完，或者只做了第一版的能力。

当前已经完成的部分：

- `thread-first` 主链路已经落地，`disentangle.lua` 不再只是 topic-like 分流器。
- 已有 `thread_id / thread_key / segment_key / keep_pending / orphan pending`。
- 已有三层记忆主干：`thread-local / actor-local / scope aggregate`。
- 已有 scope-aware merge 和 scope-aware retrieve。
- 已有运行时 `checkpoint + WAL replay`。
- 已有 `CommittedChunk` 提交路径。
- 已有轻量 reply-link scorer 特征：`mention / addressee hint / reply cue / head-tail similarity / stability`。
- 已有直播特化兜底：`reaction fallback + ambient/shared live thread`。
- 已有真实 Bilibili 直播回放验证，不再只依赖模拟器。

当前还没完成的部分：

- 群聊场景的专门策略还没做，尤其是 `group/thread` 关系和参与者稀疏切换。
- 单用户、群聊、直播三种模式的回归矩阵还没统一。
- ambient 目前还是短期上下文层，没有做压缩晋升。
- topic projection 还有优化空间，但已经降级为二级视图，不再是一级事实源。

## 4. 总体架构图

### 4.1 Ingest Path

```text
host meta + user_vec + text
        |
        v
+------------------+
| core.lua         |
| ingest_turn()    |
+------------------+
        |
        v
+------------------+      same scope only
| thread router    |----> recent thread frontier
| (disentangle v2) |----> reply/addressee hints
|                  |----> reaction / ambient fallback
+------------------+----> temporal window
        |
        |  outcome:
        |  attach / open / keep_pending / orphan / ambient
        v
+------------------+
| thread runtime   |
| pending buffers  |
| segment state    |
+------------------+
        |
        | stable enough?
        +------------------------ no -------------------+
        |                                               |
       yes                                              v
        |                                     pending local context
        v
+------------------+
| chunk commit     |
| fact extraction  |
+------------------+
        |
        v
+------------------+      +------------------+
| associative mem  |----->| topic projection |
| topic_graph v2   |      | topic.lua        |
+------------------+      +------------------+
        |
        v
  materialized store
```

### 4.2 Recall Path

```text
query_vec + scope + actor
        |
        v
+------------------+
| core.lua         |
| compile_context  |
+------------------+
        |
        v
+------------------+
| thread router    |
| select live flow |
+------------------+
        |
        v
+----------------------------+
| retrieval planner          |
| 1) thread-local committed  |
| 2) pending local context   |
| 3) actor-local memories    |
| 4) scope aggregate/topic   |
+----------------------------+
        |
        v
 compiled blocks for LLM
```


## 5. 为什么是 Thread First

直播场景里最常见的失败模式不是“没人聊这个”，而是：

- 两拨人同时聊相似主题。
- 大量短句缺少明确主语。
- 话题相似，但回复关系、对象关系、节奏关系不同。

所以一级对象必须是“谁在和谁继续同一条互动流”，而不是“这些句子语义像不像同一话题”。

对 Mori 来说，这意味着：

- `topic_anchor` 不能再承担一级分流职责。
- `sequence_key` 不能只表示“局部片段”，还要能稳定表示 thread。
- `topic` 应该是 thread 的投影结果，不是 thread 的替身。


## 6. 模块收敛图

### 6.1 当前模块与状态

| 当前模块 | 状态 | 当前职责 |
| --- | --- | --- |
| `mori_memory/core.lua` | 已完成一版 | 编排 ingest / compile / shutdown / recovery |
| `module/memory/disentangle.lua` | 已完成一版 | `thread router`，含 pending / orphan / ambient |
| `module/memory/topic_graph.lua` | 已完成一版 | 已提交关联记忆的 merge / recall |
| `module/memory/topic.lua` | 已降级 | topic 摘要、指纹、投影 |
| `module/persistence.lua` | 已沿用 | 原子替换 checkpoint 文件 |
| `module/memory/recovery_log.lua` | 已新增 | 运行时 WAL |
| `module/memory/thread_checkpoint.lua` | 已新增 | 活跃 thread/pending 状态快照 |

### 6.2 当前目录

```text
module/
  persistence.lua
  memory/
    disentangle.lua          # 保留文件名，内部职责升级为 thread router
    recovery_log.lua         # 新增
    thread_checkpoint.lua    # 新增
    topic_graph.lua
    topic.lua
```

说明：

- 这里刻意保留 `disentangle.lua` 入口，避免对 `core.lua` 和外部 API 造成大改动。
- Python 保留给离线模拟、评测、数据标注工具，不进入在线热路径。


## 7. 核心数据模型

这一节保留为概念模型，用来约束职责边界。

- 它描述的是“系统应维护什么状态”，不是要求当前代码字段逐字一致。
- 已完成的能力以第 3 节和第 12、13 节为准。

### 7.1 ScopeRuntime

```lua
{
  scope_key = "bili:room:123",
  next_thread_id = 17,
  next_segment_id = 42,
  threads = { [thread_id] = ThreadRuntime },
  frontier = { tail_turn_ids... },
  last_checkpoint_seq = 1203,
}
```

### 7.2 ThreadRuntime

```lua
{
  thread_id = 9,
  thread_key = "bili:room:123$th:9",
  state = "tentative", -- tentative / confirmed / idle / closed
  segment_id = 21,
  segment_key = "bili:room:123$th:9/seg:21",
  head_turn = 401,
  tail_turn = 417,
  last_turn = 417,
  participants = { ["uid:1"] = true, ["uid:2"] = true },
  addressee_hints = { ["uid:2"] = 3 },
  pending = { PendingTurn, ... },
  committed_turns = { 401, 402, 405, ... },
  centroid_head = {...},
  centroid_tail = {...},
  stability = 0.82,
  confidence = 0.77,
}
```

### 7.3 PendingTurn

```lua
{
  turn = 418,
  actor_key = "uid:3",
  text = "...",              -- 仅短期保留，用于恢复和 pending context
  vec = {...},
  top_candidates = {
    { parent_turn = 417, thread_id = 9, score = 0.77 },
    { parent_turn = 410, thread_id = 6, score = 0.71 },
  },
  decision = "keep_pending", -- attach / open / keep_pending / orphan
  created_at_turn = 418,
}
```

### 7.4 CommittedChunk

```lua
{
  chunk_id = 88,
  scope_key = "bili:room:123",
  thread_key = "bili:room:123$th:9",
  segment_key = "bili:room:123$th:9/seg:21",
  turns = { 409, 410, 411 },
  actor_set = { ["uid:1"] = true, ["uid:2"] = true },
  facts = { ... },
  summary = "...",
}
```

### 7.5 AssociativeMemory

```lua
{
  memory_id = 301,
  scope_key = "bili:room:123",
  actor_key = "uid:1",       -- 可为空；个人事实必须带 actor
  thread_key = "bili:room:123$th:9",
  origin_topics = { ... },   -- 仅作为投影/回溯信息
  facets = { ... },
  vec = {...},
  type = "fact",
}
```

关键规则：

- `thread_key` 与 `topic_anchor` 必须解耦。
- 个人事实默认只在 `actor-local` 层合并。
- 只有跨 turn 稳定、或跨 actor 被重复支持的事实，才允许上浮到 `scope aggregate`。


## 8. 在线状态机

### 8.1 消息路由状态机

```text
NEW MESSAGE
    |
    v
candidate generation
    |
    +--> high confidence attach ------> append to thread
    |
    +--> medium confidence -----------> keep_pending
    |
    +--> no good parent --------------> open new thread
    |
    +--> short reaction --------------> ambient pending
    |
    +--> impossible / overflow -------> orphan pending
```

### 8.2 Thread 状态机

```text
tentative
  |
  +--> confirmed   (连续支撑、参与者稳定、边置信度足够)
  |
  +--> closed      (超时且没有继续)

confirmed
  |
  +--> idle        (短时间无更新)
  |
  +--> split       (出现明显新子流，segment_id 递增)
  |
  +--> closed

idle
  |
  +--> confirmed   (同一 thread 被重新激活)
  |
  +--> closed      (超过 TTL)
```

### 8.3 提交原则

- `attach` 后不等于立即进入长期记忆。
- 至少满足以下之一才 commit 成 `CommittedChunk`：
  - 同一 thread 在后续 `K` 轮内继续被支持。
  - 同一 segment 已经形成最小闭包。
  - 该片段被 recall/adopt 反向验证过。
- `keep_pending` 的内容可以参与局部上下文，但不能进入长期聚合层。


## 9. 路由与检索打分

### 9.1 已完成的轻量特征

在线打分建议从轻量特征开始，不先上大模型主裁决：

- `same_scope`
- `delta_turn`
- `same_actor`
- `explicit_mention / addressee hint`
- `reply cue`
- `head similarity`
- `tail similarity`
- `actor continuity`
- `thread stability`
- `margin(best - second_best)`
- `reaction-like fallback`

### 9.2 当前决策

- 高置信度：直接 attach。
- 中置信度：先 pending，等待后续 disambiguation。
- 低置信度短反应：优先进入 `ambient pending` 或附着到最近稳定流。
- 低置信度且流已满：进入 `orphan pending`，带 TTL。

这一步的重点不是“立刻分对”，而是“不要立刻分错”。


## 10. 记忆分层

### 10.1 三层结构

```text
Layer 1: thread-local episodic
Layer 2: actor-local associative
Layer 3: scope/topic aggregate
```

### 10.2 当前召回顺序

```text
1. 当前 live thread 的 committed chunks
2. 当前 live thread 的 pending local context
3. 当前 actor 的历史事实
4. 当前 scope 下的公共关联记忆
5. topic summary / fingerprint 投影
```

### 10.3 不同层的用途

- `thread-local`：保证 LLM 先理解“现在谁在接谁的话”。
- `actor-local`：保存个人偏好、立场、长期习惯。
- `scope aggregate`：保存房间公共背景、反复出现的共识。
- `topic`：只做摘要、聚类、路由提示，不直接代表 thread。


## 11. 落盘设计

### 11.1 原则

- 不把原始 turn 永久保留成日记。
- 但必须让系统能从故障中恢复复杂中间态。
- 因此要区分：
  - `durability log`
  - `long-term memory`

前者只是恢复手段，后者才是长期语义记忆。

### 11.2 当前文件布局

```text
memory/v4/runtime/
  manifest.lua
  thread_checkpoint.lua
  thread_wal.log

memory/v4/topic_graph/
  state.lua
  vectors.bin
  hnsw/...
```

### 11.3 Checkpoint 内容

`thread_checkpoint.lua` 只存运行时必需状态：

- 每个 scope 的活跃 thread
- frontier
- pending 队列
- thread/segment id 游标
- 各 thread 的 centroid / participants / confidence
- 最近一次成功 replay 的序号

不应该把以下内容无限膨胀地存进去：

- 全量原始文本历史
- 已经被 compact 的旧 pending 文本
- 可由 materialized memory 重新导出的冗余投影

### 11.4 WAL 内容

`thread_wal.log` 追加记录：

- `turn_seen`
- `route_selected`
- `pending_added`
- `thread_opened`
- `thread_attached`
- `segment_rotated`
- `chunk_committed`
- `thread_closed`

每条日志最少记录：

- `seq`
- `turn`
- `scope_key`
- `thread_key`
- `segment_key`
- `decision`
- `score / margin`
- `actor_key`
- `short_text` 或引用
- `config_version`
- `embed_version`

### 11.5 LuaJIT 序列化格式

推荐优先级：

1. `Lua table literal + length-prefixed frame`
2. `Lua table literal + atomic checkpoint`

原因：

- 现有代码已经有 `util.encode_lua_value()` 和 `util.parse_lua_table_literal()`。
- 不需要在线引入 Python。
- 与当前 `topic_graph.lua` 的持久化风格一致。

WAL 不推荐直接做成“纯文本无边界拼接”；建议至少带：

- frame length
- frame payload
- 可选 CRC32

这样崩溃恢复时可以安全截断损坏尾帧。

### 11.6 恢复流程

```text
startup
  -> load materialized topic_graph
  -> load thread_checkpoint
  -> replay thread_wal after checkpoint seq
  -> rebuild live frontier / pending context
  -> continue serving
```

### 11.7 压缩与清理

一旦 `CommittedChunk` 已经被写入 `topic_graph`，并且对应 checkpoint 已成功刷新：

- 删除旧 pending 文本
- 删除已 compact 的旧 WAL 段
- 只保留 chunk 级摘要、事实和向量

这一步是“反日记化”的关键。


## 12. 当前已完成与下一步

### 12.1 已完成

- `disentangle.lua`
  - 已有 `thread_id / thread_key / segment_key`
  - 已有 `keep_pending / orphan pending / ambient pending`
  - 已有轻量 scorer 和直播 fallback
- `core.lua`
  - 已按 `pending -> thread-local -> associative -> topic` 顺序编排上下文
  - 已接上 runtime recovery
- `topic_graph.lua`
  - 已只接 committed chunk/fact
  - 已支持 scope-aware merge / retrieve
- `topic.lua`
  - 已降级为 projection/view

### 12.2 下一步

- 单用户回归确认
  - 需要验证多用户兼容没有破坏原本单会话路径。
- 群聊策略
  - 需要把“直播并行流”和“群聊多子会话”分开建模。
- ambient 压缩晋升
  - 当前 ambient 主要是短期缓冲，还没形成稳定公共记忆。
- topic projection 微调
  - 当前可用，但还不是下一阶段重点。


## 13. 实施阶段状态

### Phase 1

已完成：

- 引入 `thread_id / thread_key`
- 保留现有接口，不动 `core.lua` 对外 API
- 增加 `keep_pending` 和 `orphan pending`

### Phase 2

已完成：

- 新增 `recovery_log.lua` 和 `thread_checkpoint.lua`
- 持久化活跃 thread 和 pending
- 继续复用当前 `topic_graph` 作为长期 materialized store

### Phase 3

已完成：

- commit 单位从“单轮”提升为 `CommittedChunk`
- 将 `actor-local` 和 `scope aggregate` 分开

### Phase 4

部分完成：

- 已有更强的一版轻量 reply-link scorer
- 已有直播 `reaction fallback + ambient thread`
- topic projection 的进一步优化未完成


## 14. 评测指标

不要再只看命中率，推荐看这些：

- `useful_associative_recall_rate`
- `cross_thread_pollution_rate`
- `cross_actor_personal_leak_rate`
- `pending_resolution_rate`
- `orphan_recovery_rate`
- `replay_consistency_rate`
- `compile_context_p95_ms`
- `ingest_turn_p95_ms`


## 15. 下一阶段约束

下一步推进群组时，默认继续保持这三个前提不变：

- 不能为了群聊把单用户路径重新打散。
- 不能为了直播兜底把群聊和单会话都退化成大量 `ambient/local_only`。
- 仍然坚持 `thread-first`，不退回 `topic-first`。


## 16. 最终选型摘要

如果只保留一句话，这套设计就是：

`用 LuaJIT 在线维护 thread 级活状态，让消息先属于 thread，再让稳定片段投影成 topic 和长期关联记忆；用短期 WAL 保证恢复，但不把原始对话永久变成日记。`
