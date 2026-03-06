
# Experience模块集成文档

> 注：本文档中的 topic bridge 流程仅用于兼容旧方案。当前主图已经改为 `graph run -> run_builder -> policy memory`，run 级执行历史应写入独立 episode 层。

## 概述

本文档说明Experience模块与Topic系统的集成方式，以及如何在Agent中使用Experience模块。

## 集成架构

```
┌─────────────────────────────────────────┐
│         Topic System                  │
│  - 话题生命周期管理                  │
│  - 事件钩子（轻量级）              │
└─────────────────────────────────────────┘
              ↓ 事件通知
┌─────────────────────────────────────────┐
│      Experience Bridge                │
│  - 事件转换和路由                   │
│  - 错误隔离                         │
│  - 可选启用                         │
└─────────────────────────────────────────┘
              ↓ 事件处理
┌─────────────────────────────────────────┐
│      Experience Builder               │
│  - 从topic构建经验                   │
│  - 提取上下文和模式                 │
└─────────────────────────────────────────┘
              ↓ 存储
┌─────────────────────────────────────────┐
│      Experience Store                 │
│  - 经验持久化                       │
│  - 多维度索引                       │
└─────────────────────────────────────────┘
              ↓ 检索
┌─────────────────────────────────────────┐
│      Experience Retriever             │
│  - 智能检索                         │
│  - 风险评估                         │
└─────────────────────────────────────────┘
```

## 集成细节

### 1. Topic系统修改

#### 1.1 添加Experience桥接引用

在topic.lua开头添加：
```lua
-- Experience桥接（可选，轻量级集成）
local experience_bridge = nil
local function get_experience_bridge()
    if not experience_bridge then
        local ok, mod = pcall(require, "module.experience.bridge")
        if ok then
            experience_bridge = mod
            print("[Topic] Experience bridge loaded")
        else
            experience_bridge = false  -- 标记为不可用
        end
    end
    return experience_bridge
end
```

#### 1.2 Topic开始钩子

在开启新话题时调用：
```lua
-- Experience桥接：通知experience系统
local bridge = get_experience_bridge()
if bridge then
    bridge.on_topic_start({
        topic_idx = turn,
        start = turn,
        context = {
            user_text = user_text,
            vector = vector
        }
    })
end
```

#### 1.3 Topic更新钩子

在update_assistant函数中调用：
```lua
-- Experience桥接：通知experience系统话题更新
local bridge = get_experience_bridge()
if bridge then
    bridge.on_topic_update({
        topic_idx = M.active_topic.start or turn,
        intermediate_state = {
            turn = turn,
            assistant_text = text
        }
    })
end
```

#### 1.4 Topic结束钩子

在close_current_topic函数中调用：
```lua
-- Experience桥接：通知experience系统
local bridge = get_experience_bridge()
if bridge then
    local topic_data = {
        topic_idx = M.active_topic.start,
        end_turn = end_turn,
        anchor = "C:" .. tostring(#M.topics + 1),
        summary = summary,
        outcome = {
            success = true,
            metrics = {
                duration = end_turn - M.active_topic.start + 1
            }
        },
        errors = {}
    }
    bridge.on_topic_end(topic_data)
end
```

### 2. Experience Bridge实现

bridge.lua提供三个核心事件处理函数：

```lua
-- Topic开始事件
function M.on_topic_start(topic_data)
    -- 记录初始上下文
    experience.builder.on_topic_start(topic_data)
end

-- Topic更新事件
function M.on_topic_update(topic_data)
    -- 增量更新上下文
    experience.builder.on_topic_update(topic_data)
end

-- Topic结束事件
function M.on_topic_end(topic_data)
    -- 构建并存储经验
    experience.builder.on_topic_end(topic_data)
end
```

## 使用方式

### 初始化

在系统启动时初始化Experience模块：

```lua
local experience = require("module.experience")
experience.init()
```

Topic系统会自动检测并使用experience_bridge，无需额外配置。

### 在Agent中使用

#### 1. 检索相关经验

```lua
-- 检索当前任务相关的经验
local experiences = experience.retrieve_experience({
    task_type = "coding",
    context_signature = {
        language = "python",
        domain = "coding"
    },
    limit = 5
})

-- 使用经验指导决策
for _, exp in ipairs(experiences) do
    print(string.format("经验: %s, 成功率: %.2f", 
        exp.description, exp.success_rate))
end
```

#### 2. 检查失败风险

```lua
-- 在执行任务前检查失败风险
local risks = experience.check_failure_risk(
    proposed_solution,
    current_context
)

if #risks > 0 then
    -- 根据风险调整方案
    adjust_solution(risks)
end
```

#### 3. 获取成功案例

```lua
-- 获取成功案例以指导当前任务
local successes = experience.get_success_cases(
    task_description,
    current_context
)

-- 学习成功经验
for _, success in ipairs(successes) do
    apply_success_factors(success)
end
```

## 设计原则

### 1. 松散耦合

- 单向依赖：topic -> experience
- 避免循环依赖
- 模块可以独立演进

### 2. 可选集成

- experience模块不可用时，topic系统正常工作
- 使用pcall进行错误隔离
- 提供启用/禁用控制

### 3. 错误隔离

- experience的错误不会影响topic核心功能
- 每个事件处理都有独立的错误处理
- 失败时只打印日志，不中断流程

### 4. 延迟加载

- 只在需要时才加载experience_bridge
- 减少启动时的依赖
- 提高系统启动速度

## 配置选项

### Experience模块配置

在config中添加：
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

### Bridge配置

在bridge.lua中：
```lua
-- 是否启用experience构建
local ENABLED = true

-- 启用/禁用
experience.bridge.set_enabled(true/false)
```

## 调试和监控

### 日志输出

Topic系统会输出以下日志：
- `[Topic] Experience bridge loaded` - bridge加载成功
- `[ExperienceBuilder] Topic X started` - topic开始
- `[ExperienceBuilder] Created experience exp_xxx from topic X` - 经验创建成功

### 统计信息

查看Experience模块统计：
```lua
local stats = experience.get_stats()
print(string.format("总经验数: %d", stats.total_experiences))
```

## 故障排查

### Experience模块未加载

检查：
1. experience模块是否存在于module/experience/
2. 是否调用了experience.init()
3. 查看日志中是否有错误信息

### 经验未创建

检查：
1. topic是否正常结束
2. bridge是否正确加载
3. 查看ExperienceBuilder日志

### 检索结果为空

检查：
1. 是否有足够的经验数据
2. 检索条件是否合理
3. 索引是否正确构建

## 后续优化

1. **性能优化**
   - 添加缓存机制
   - 优化索引结构
   - 批量操作支持

2. **功能增强**
   - 添加向量化支持
   - 实现经验评估
   - 支持经验压缩

3. **监控和诊断**
   - 添加性能指标
   - 实现经验质量评估
   - 提供诊断工具

## 总结

Experience模块通过松散耦合的方式与Topic系统集成，提供了强大的经验学习和检索能力，同时保持了系统的灵活性和可维护性。集成设计遵循了以下原则：

- ✅ 松散耦合
- ✅ 可选集成
- ✅ 错误隔离
- ✅ 延迟加载

这种设计使得Experience模块可以独立演进，而不会影响Topic系统的核心功能。
