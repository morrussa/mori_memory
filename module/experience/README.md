
# Experience Module

## 概述

Experience模块现在承担的是 `policy memory` 角色：保存可复用的策略偏好、成功路径和失败教训，用于给 planner 注入“惯性”。

主图中的 run 级执行记录已经拆向独立 `episode` 层；`builder.lua`/`bridge.lua` 保留为旧 topic 流程的兼容接口，不再是主链路。

## 设计理念

### 核心原则

1. **策略独立于事实**：长期事实留在memory core，Experience只存策略层信号
2. **主链路基于Run写入**：graph run 结束后由 `run_builder.lua` 生成策略记忆
3. **上下文感知**：策略检索考虑当前上下文，避免错误迁移
4. **多维度索引**：支持按类型、任务、上下文、工具等维度检索

### 架构

```
┌─────────────────────────────────────────┐
│         Memory Core (现有)             │
│  - 长期记忆存储                       │
│  - 主题系统 (topic)                   │
│  - 聚类系统 (cluster)                 │
│  - 热度系统 (heat)                   │
└─────────────────────────────────────────┘
              ↓ topic事件
┌─────────────────────────────────────────┐
│      Experience Builder                │
│  - 监听topic生命周期                  │
│  - 从topic提取经验                    │
│  - 构建经验结构                      │
└─────────────────────────────────────────┘
              ↓ 构建经验
┌─────────────────────────────────────────┐
│      Experience Store                   │
│  - 独立的经验存储                    │
│  - 多维度索引                        │
│  - 持久化存储                        │
└─────────────────────────────────────────┘
              ↓ 检索经验
┌─────────────────────────────────────────┐
│      Experience Retriever               │
│  - 上下文感知检索                    │
│  - 失败案例检查                      │
│  - 成功案例推荐                      │
└─────────────────────────────────────────┘
              ↓ 使用经验
┌─────────────────────────────────────────┐
│         Agent System                   │
│  - 使用经验指导决策                   │
└─────────────────────────────────────────┘
```

## 模块组成

### 1. store.lua - 经验存储系统

负责经验的持久化存储和多维度索引。

**主要功能：**
- 经验的添加和存储
- 多维度索引（类型、任务、上下文、工具、时间）
- 相似度计算（上下文相似度、向量相似度）
- 持久化和加载

**数据结构：**
```lua
experience = {
    id = "exp_xxx",                    -- 经验ID
    type = "success|failure",          -- 经验类型
    created_at = timestamp,            -- 创建时间

    -- Topic关联
    topic_id = 123,                    -- 关联的topic ID
    topic_anchor = "T:123",           -- topic锚点
    topic_turn_range = {start, end},   -- topic轮次范围

    -- 上下文特征
    context_signature = {...},         -- 上下文签名
    task_type = "coding",             -- 任务类型
    tools_used = {tool_name: count},  -- 工具使用统计

    -- 内容
    description = "...",               -- 经验描述
    patterns = [...],                 -- 提取的模式
    lessons = [...],                  -- 提取的教训

    -- 结果
    outcome = {...},                  -- 执行结果
    success_rate = 0.85,             -- 成功率

    -- 向量（可选）
    embedding = vec                   -- 向量表示
}
```

### 2. builder.lua - 经验构建器

基于topic生命周期动态构建经验。

**主要功能：**
- 监听topic生命周期事件
- 从topic中提取上下文特征
- 分析topic执行结果
- 构建经验对象
- 提取模式和教训

**Topic事件处理：**
- `on_topic_start`: 记录初始上下文
- `on_topic_update`: 增量更新上下文（工具使用、错误等）
- `on_topic_end`: 构建完整经验并存储

### 3. retriever.lua - 经验检索器

智能检索和排序agent经验。

**主要功能：**
- 多种检索策略（按类型、任务、上下文、工具）
- 混合检索（综合多种因素）
- 失败案例风险评估
- 成功案例推荐

**检索策略：**
- `retrieve_by_type_and_task`: 按类型和任务类型检索
- `retrieve_by_context`: 上下文感知检索
- `retrieve_by_tool`: 工具使用经验检索
- `retrieve_hybrid`: 混合检索（综合多种因素）

## 使用示例

### 初始化

```lua
local experience = require("module.experience")

-- 初始化模块
experience.init()
```

### 添加经验

```lua
-- 手动添加经验
local exp = {
    type = "success",
    task_type = "coding",
    context_signature = {
        language = "python",
        domain = "coding",
        task_category = "debugging"
    },
    tools_used = {
        ["code_analyzer"] = 3,
        ["debugger"] = 2
    },
    description = "成功调试Python代码",
    outcome = {
        success = true,
        metrics = {
            duration = 5,
            tool_calls = 5
        }
    }
}

local ok, exp_id = experience.add_experience(exp)
```

### 检索经验

```lua
-- 检索相关经验
local results = experience.retrieve_experience({
    task_type = "coding",
    context_signature = {
        language = "python",
        domain = "coding"
    },
    limit = 5
})

-- 检查失败风险
local risks = experience.check_failure_risk({
    tools = {
        ["code_analyzer"] = true
    }
}, {
    language = "python",
    domain = "coding"
})

-- 获取成功案例
local successes = experience.get_success_cases(
    "调试Python代码",
    {
        language = "python",
        domain = "coding"
    }
)
```

### 查看统计

```lua
local stats = experience.get_stats()
print(string.format("总经验数: %d", stats.total_experiences))
```

## 与Agent集成

### 在Agent Loop中使用

```lua
local experience = require("module.experience")

-- 在执行任务前检查失败风险
local risks = experience.check_failure_risk(
    proposed_solution,
    current_context
)

if #risks > 0 then
    -- 调整方案
    adjust_solution_based_on_risks(risks)
end

-- 执行任务
execute_task(task)

-- 任务完成后，topic会自动构建经验
-- (需要集成topic事件监听)
```

### 在工具选择中使用

```lua
-- 检索工具使用经验
local tool_experiences = experience.retriever.retrieve_by_tool(
    "code_analyzer",
    {limit = 5}
)

-- 根据经验选择最佳工具
select_best_tool_based_on_experience(tool_experiences)
```

## 配置

在config中添加experience配置：

```lua
{
    settings = {
        experience = {
            retriever = {
                weights = {
                    context_similarity = 0.35,
                    vector_similarity = 0.30,
                    success_rate = 0.20,
                    recency = 0.15
                },
                recency_half_life = 30
            }
        }
    }
}
```

## 存储结构

```
memory/experiences/
├── experience_index.txt      -- 索引文件
└── experiences/              -- 经验文件目录
    ├── exp_xxx.lua
    ├── exp_yyy.lua
    └── ...
```

## 与Topic系统集成

Experience模块通过松散耦合的方式与Topic系统集成：

### 集成方式

1. **桥接层（bridge.lua）**
   - 作为topic和experience之间的轻量级连接层
   - 处理topic生命周期事件
   - 提供错误隔离和可选启用

2. **Topic钩子**
   - topic.lua在关键位置调用bridge的事件处理函数
   - 使用延迟加载机制，避免硬依赖
   - experience模块不可用时，topic系统仍能正常工作

3. **事件流**
   ```
   Topic开始 -> bridge.on_topic_start -> builder记录初始上下文
   Topic更新 -> bridge.on_topic_update -> builder增量更新
   Topic结束 -> bridge.on_topic_end -> builder构建并存储经验
   ```

### 集成特点

✅ **松散耦合**：单向依赖，避免循环
✅ **可选集成**：experience模块不可用时，topic系统正常工作
✅ **错误隔离**：experience的错误不会影响topic核心功能
✅ **延迟加载**：只在需要时才加载experience_bridge

### 使用方式

在系统初始化时：
```lua
local experience = require("module.experience")
experience.init()  -- 初始化experience模块
```

Topic系统会自动检测并使用experience_bridge，无需额外配置。

## 后续迭代方向

1. **向量化支持**
   - 为经验添加向量表示
   - 支持向量相似度检索

2. **经验评估**
   - 添加经验质量评估机制
   - 自动淘汰低质量经验

3. **经验压缩**
   - 对相似经验进行合并
   - 提取通用原则

4. **元学习**
   - 从经验中学习学习策略
   - 快速适应新任务

5. **与Agent深度集成**
   - 在agent决策流程中集成经验检索
   - 自动记录和评估agent行为

## 学术支持

本设计基于以下研究：

1. **情景记忆（Episodic Memory）**
   - Tulving, E. (2002). Episodic memory: From mind to brain.

2. **层次化学习（Hierarchical Learning）**
   - Sutton, R. S., et al. (1999). Between MDPs and semi-MDPs.

3. **经验回放（Experience Replay）**
   - Mnih, V., et al. (2013). Playing Atari with Deep RL.

4. **元学习（Meta-Learning）**
   - Finn, C., et al. (2017). MAML for Fast Adaptation.
