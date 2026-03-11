# Mori 记忆系统

Mori 的记忆不是把“所有历史 + 所有记忆”一次性塞进 prompt，而是每一轮都根据当前任务、当前输入、当前工作状态，重新拼接出一份当下最需要的上下文。

这套系统的核心不是“保存一大段固定上下文”，而是“维护一组可重组的状态源”，然后在推理前动态装配。

## 核心理念：完全动态拼接上下文

所谓“完全动态拼接”，指的是：

1. 每一轮输入到来后，系统先判断这轮到底是在继续原任务、微调原任务，还是已经切换到新任务。
2. 只有在这轮确实需要记忆时，才触发召回，而不是无条件把长期记忆常驻在 prompt 里。
3. prompt 中的不同信息块来自不同状态源，例如任务状态、工作记忆、最近 episode、工具结果、历史对话、长期记忆，它们彼此独立、按需组合。
4. 上下文超预算时，系统优先压缩、裁剪、丢弃低价值部分，而不是让 prompt 无限增长。
5. 本轮结束后，系统把这轮对话重新写回成“可供未来重组的状态”，下一轮再重新装配，而不是简单续接上一轮 prompt。

换句话说，Mori 保存的不是一个固定 prompt，而是一个可以在每轮推理前重新编译的上下文状态机。

## 一轮上下文是怎么拼出来的

主流程如下：

```text
用户输入
  -> task_node      判断任务连续性
  -> recall_node    按需召回长期记忆
  -> context_node   动态组装 system/history/user
  -> planner/tool   进入工具循环，持续更新 tool context / working memory
  -> responder      生成最终答复
  -> writeback      抽取原子事实、写回记忆、记录反馈
  -> persist        保存 topic / history / episode / session
```

### 1. 先判断“这轮是什么任务”

系统不会默认把每条新消息都当作独立请求处理。`task_node` 会先做任务连续性判断，把当前输入归类为：

- `same_task_step`：继续执行同一个任务
- `same_task_refine`：仍是同一个任务，但约束变了
- `hard_shift`：切到新任务
- `meta_turn`：只是问进度或状态

这个判断直接决定后续上下文要继承哪些内容，例如：

- 要不要保留旧的 task contract
- 要不要沿用 working memory
- 要不要恢复上个 episode 的连续性
- 要不要把这轮视作全新任务重新起步

所以 Mori 的上下文不是“按历史顺序机械累加”，而是先过一层任务语义路由。

### 2. 再决定“要不要回忆”

`recall_node` 不会无条件检索记忆，而是先做一次 recall gating。

它会综合判断：

- 用户输入里是否出现“之前 / 上次 / 以前 / remember / recall”之类的回忆意图
- 是否包含技术关键词
- 输入长度是否足够
- 是否出现“不要回忆 / 不查记忆”这样的抑制信号
- 输入是否是“继续 / 刚才”这类上下文延续型请求

只有分数过阈值，才会进入 `topic_graph.retrieve(...)`。

这一步很关键，因为它让长期记忆从“常驻背景噪音”变成“按需激活的上下文部件”。

### 3. 记忆召回不是直接拿全文，而是先过 topic graph

长期记忆的核心不是简单向量 top-k，而是以 topic anchor 为中心的图式检索：

- 先根据当前 query 和当前 topic 选出 seed topics
- 再沿 topic 迁移边扩展候选 topic
- 在候选 topic 下挑选记忆
- 再把记忆展开回历史 turn
- 最终生成 `selected_memories`、`selected_turns`、`fragments` 和 `memory_context`

最终注入 prompt 的不是裸 memory 向量，而是类似“第 N 轮用户说了什么、助手答了什么”的可读片段。

也就是说，Mori 召回的是“与当前任务相关的对话证据”，而不是单纯的 embedding 命中结果。

### 4. 真正进入 prompt 的上下文块

`context_builder` 会在每轮重新生成一份 system prompt。它拼接的不是固定模板，而是多个动态区块：

```text
Base System Prompt
+ Project Knowledge
+ [ActiveTask]
+ [TaskContract]
+ [WorkingMemory]
+ [MemoryContext]
+ [TaskContext]
+ [RecentEpisodes]
+ [ToolContext]
+ [PlannerContext]
+ Budget Warning
```

这些区块分别回答不同问题：

- `ActiveTask / TaskContract`：当前到底在做什么
- `WorkingMemory`：已经读了哪些文件、写了哪些文件、上一步计划是什么、最近一批工具做了什么
- `MemoryContext`：从长期记忆里动态召回出来的相关历史片段
- `RecentEpisodes`：过去几个 episode 的摘要，用于任务恢复和跨轮延续
- `ToolContext`：本轮工具执行后沉淀出的高价值结果摘要
- `PlannerContext`：上一轮工具失败、协议错误、修复提示等控制信息

这里最重要的一点是：

**这些块不是常驻不变的，它们都是每轮根据 state 现算、现选、现拼。**

### 5. 对话历史也不是全量保留

除了 system prompt，`context_builder` 还会处理历史对话：

- 只从最近的 user/assistant 对开始向前保留
- 能放进预算就保留
- 放不进时先尝试压缩
- 压缩后还超预算就直接丢弃

因此 Mori 的历史上下文是“预算感知的最近优先保留”，而不是无限增长的聊天记录回放。

### 6. 工具结果会进入上下文，但不是原样堆积

在工具循环里，`tool_exec_node` 会持续更新 working memory，并把工具结果交给 `context_manager` 处理：

- 大结果先截断
- 重复内容做去重提示
- 多条结果合并成受预算限制的 `tool_context`
- 超大的 tool message 再次裁剪

这意味着 workspace 证据也属于动态上下文的一部分，但它同样是“按预算筛选后的摘要”，而不是原始输出的无边界堆积。

## 为什么说它不是传统的“记忆注入”

很多记忆系统的做法是：

- 先把长期记忆检索出来
- 然后简单 prepend/append 到 prompt
- 下一轮继续在上一轮 prompt 基础上叠加

Mori 不是这样。

Mori 的运行方式更像：

1. 持久化保存一组结构化状态
2. 每轮先判断当前任务语义
3. 再按需激活记忆、episode、workspace、working memory
4. 在 token 预算内重新组装消息
5. 本轮结束后再把新的事实和反馈写回状态

所以它的关键不是“存了多少”，而是“每轮能不能只拿出当前真正需要的那部分”。

## 本轮结束后，系统会写回什么

为了让下一轮还能继续动态组装，这一轮结束后会把结果写回多个层级：

- `history`：保存本轮 user / assistant 对话
- `topic`：更新当前 turn 的 topic anchor 和摘要
- `memory_core`：从本轮对话中抽取原子事实，写入长期记忆
- `topic_graph.observe_feedback(...)`：根据本轮真正采用了哪些记忆，强化 topic 和 memory 之间的连接
- `episode`：把这一轮整理成可恢复的任务连续性摘要
- `session / working_memory`：保存当前计划、读写文件记录、最近工具摘要等工作状态

因此下一轮开始时，系统并不是“拿到上一轮完整 prompt”，而是拿到这些被拆开的状态，再重新选择、重新拼装。

## 这种设计带来的好处

- 不会让 prompt 随对话轮数线性膨胀
- 能把“任务连续性”和“语义回忆”分开处理
- 能把工具使用痕迹变成工作记忆，而不是散落在历史消息里
- 能在恢复中断任务时重建上下文，而不是依赖完整原始对话
- 能根据最终回答反向学习哪些记忆真的被采用过

## 关键模块

- `module/graph/nodes/task_node.lua`：任务连续性判断
- `module/graph/nodes/recall_node.lua`：回忆触发与召回入口
- `module/memory/topic_graph.lua`：基于 topic 的记忆检索与反馈学习
- `module/graph/context_builder.lua`：最终消息拼装
- `module/graph/context_manager.lua`：上下文预算、截断、去重、压缩
- `module/graph/nodes/tool_exec_node.lua`：工具执行后的工作记忆与上下文回写
- `module/graph/nodes/persist_node.lua`：history / topic / memory / episode 的持久化

## 一句话总结

Mori 的“记忆”不是一段被长期塞在 prompt 里的文本，而是一套在每一轮推理前，围绕当前任务实时重组上下文的动态系统。
