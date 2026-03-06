
-- module/experience/builder.lua
-- 经验构建器：基于topic动态构建agent经验

local M = {}

local topic = require("module.memory.topic")
local memory = require("module.memory.store")
local store = require("module.experience.store")
local adaptive = require("module.experience.adaptive")
local tool = require("module.tool")
local config = require("module.config")

-- 经验类型（与store保持一致）
local EXP_TYPES = {
    SUCCESS = "success",
    FAILURE = "failure",
    PATTERN = "pattern",
    LESSON = "lesson"
}

-- 输出策略类型
local OUTPUT_STRATEGIES = {
    STRUCTURED = "structured",       -- 结构化输出（列表、表格）
    NARRATIVE = "narrative",         -- 叙述性输出
    CODE_FIRST = "code_first",       -- 代码优先
    STEP_BY_STEP = "step_by_step",   -- 步骤式输出
    CONCISE = "concise",             -- 简洁输出
    DETAILED = "detailed"            -- 详细输出
}

-- 活跃的topic上下文
M.active_contexts = {}

local function clamp(v, lo, hi)
    if v < lo then return lo end
    if v > hi then return hi end
    return v
end

local function count_tool_results(tool_calls)
    local success_count = 0
    local failure_count = 0

    for _, call in ipairs(tool_calls or {}) do
        if call.result and call.result.success == true then
            success_count = success_count + 1
        elseif call.result ~= nil then
            failure_count = failure_count + 1
        end
    end

    return success_count, failure_count
end

-- ==================== 初始化 ====================

function M.init()
    -- 初始化store
    store.init()

    -- 注册topic事件监听器
    M.register_topic_listeners()
end

-- ==================== Topic事件监听 ====================

function M.register_topic_listeners()
    -- 在topic系统中注册回调
    -- 注意：这需要topic.lua提供相应的钩子
    -- 这里先定义接口，实际集成时需要修改topic.lua

    -- 示例：
    -- topic.register_listener("on_topic_start", M.on_topic_start)
    -- topic.register_listener("on_topic_end", M.on_topic_end)
    -- topic.register_listener("on_topic_update", M.on_topic_update)
end

-- ==================== Topic生命周期处理 ====================

-- Topic开始时：记录初始上下文
function M.on_topic_start(topic_data)
    local topic_id = topic_data.topic_idx or topic_data.id
    if not topic_id then return end

    local context = {
        topic_id = topic_id,
        start_turn = topic_data.start or topic_data.turn,
        initial_context = M.extract_context_signature(topic_data),
        task_type = M.detect_task_type(topic_data),
        tools_used = {},
        tool_calls = {},
        intermediate_states = {},
        errors = {}
    }

    M.active_contexts[topic_id] = context

    print(string.format("[ExperienceBuilder] Topic %d started, task_type=%s", 
        topic_id, context.task_type))
end

-- Topic更新时：增量更新上下文
function M.on_topic_update(topic_data)
    local topic_id = topic_data.topic_idx or topic_data.id
    if not topic_id then return end

    local context = M.active_contexts[topic_id]
    if not context then return end

    -- 更新工具使用记录
    if topic_data.tool_used then
        context.tools_used[topic_data.tool_used] = 
            (context.tools_used[topic_data.tool_used] or 0) + 1

        table.insert(context.tool_calls, {
            tool = topic_data.tool_used,
            args = topic_data.tool_args,
            result = topic_data.tool_result,
            timestamp = os.time()
        })
    end

    -- 更新中间状态
    if topic_data.intermediate_state then
        table.insert(context.intermediate_states, topic_data.intermediate_state)
    end

    -- 记录错误
    if topic_data.error then
        table.insert(context.errors, topic_data.error)
    end
end

-- Topic结束时：构建完整经验
function M.on_topic_end(topic_data)
    local topic_id = topic_data.topic_idx or topic_data.id
    if not topic_id then return end

    local context = M.active_contexts[topic_id]
    if not context then return end

    -- 构建经验对象
    local experience = M.build_experience(topic_data, context)

    -- 存储到Experience Store
    local ok, exp_id = store.add(experience)

    if ok then
        print(string.format("[ExperienceBuilder] Created experience %s from topic %d", 
            exp_id, topic_id))
    else
        print(string.format("[ExperienceBuilder] Failed to create experience from topic %d: %s", 
            topic_id, exp_id))
    end

    -- 清理临时上下文
    M.active_contexts[topic_id] = nil
end

-- ==================== 经验构建 ====================

function M.build_experience(topic_data, context)
    -- 分析topic结果
    local outcome = M.analyze_topic_outcome(topic_data, context)

    -- 提取关键模式
    local patterns = M.extract_patterns(topic_data, context)

    -- 提取教训
    local lessons = M.extract_lessons(topic_data, context)

    -- [新增] 检测输出策略
    local output_strategy = M.detect_output_strategy(topic_data, context)

    local success_rate = M.compute_success_rate(topic_data, context, outcome)
    local utility_prior = M.compute_utility_prior(topic_data, context, outcome, success_rate)
    local error_info = (#context.errors > 0) and {
        type = (context.errors[1] and context.errors[1].type) or "unknown",
        count = #context.errors,
    } or nil
    local success_key = outcome.success and M.build_success_key(context, output_strategy) or nil

    -- 构建经验对象
    local experience = {
        id = nil,  -- 由store分配

        -- 基本信息
        type = outcome.success and EXP_TYPES.SUCCESS or EXP_TYPES.FAILURE,
        created_at = os.time(),

        -- Topic关联
        topic_id = context.topic_id,
        topic_anchor = topic_data.anchor or ("T:" .. context.topic_id),
        topic_turn_range = {
            start = context.start_turn,
            ["end"] = topic_data.end_turn or topic_data.turn  -- end是保留字，需用字符串形式
        },

        -- 上下文特征
        context_signature = context.initial_context,
        task_type = context.task_type,
        domain = context.initial_context and context.initial_context.domain or nil,
        language = context.initial_context and context.initial_context.language or nil,
        tools_used = context.tools_used,

        -- [新增] 输出策略
        output_strategy = output_strategy,

        -- 内容
        description = M.generate_description(topic_data, context, outcome),
        patterns = patterns,
        lessons = lessons,

        -- 结果
        outcome = outcome,
        success_rate = success_rate,
        utility_prior = utility_prior,
        error_info = error_info,
        success_key = success_key,

        -- 向量（用于检索）
        embedding = nil  -- 可选：后续可以添加向量化
    }

    experience.embedding = tool.get_embedding_passage(store.build_retrieval_text(experience))

    -- [新增] 记录输出策略结果到自适应系统
    adaptive.record_output_strategy(
        output_strategy,
        context.initial_context,
        context.task_type,
        outcome.success
    )

    return experience
end

-- ==================== 上下文分析 ====================

function M.extract_context_signature(topic_data)
    local sig = {}

    -- 提取语言特征
    sig.language = M.detect_language(topic_data)

    -- 提取领域特征
    sig.domain = M.detect_domain(topic_data)

    -- 提取任务特征
    sig.task_category = M.detect_task_category(topic_data)

    -- 提取环境特征
    sig.environment = M.extract_environment_features(topic_data)

    return sig
end

-- [新增] 检测输出策略
function M.detect_output_strategy(topic_data, context)
    -- 从中间状态分析输出模式
    local states = context.intermediate_states or {}
    local ai_responses = {}
    
    for _, state in ipairs(states) do
        if state.assistant_text then
            ai_responses[#ai_responses + 1] = state.assistant_text
        end
    end

    -- 如果没有足够的回复，从 topic_data 中获取
    if #ai_responses == 0 and topic_data.summary then
        ai_responses[1] = topic_data.summary
    end

    if #ai_responses == 0 then
        return OUTPUT_STRATEGIES.NARRATIVE  -- 默认
    end

    local combined_text = table.concat(ai_responses, " ")

    -- 检测结构化输出（列表、表格）
    local list_count = select(2, combined_text:gsub("\n[%s]*[%d]+%.", ""))
    local bullet_count = select(2, combined_text:gsub("\n[%s]*[%-%*]", ""))
    local table_count = select(2, combined_text:gsub("|", ""))
    
    if list_count >= 3 or bullet_count >= 3 or table_count >= 5 then
        return OUTPUT_STRATEGIES.STRUCTURED
    end

    -- 检测代码优先
    local code_blocks = select(2, combined_text:gsub("```", ""))
    if code_blocks >= 2 then
        return OUTPUT_STRATEGIES.CODE_FIRST
    end

    -- 检测步骤式输出
    local step_patterns = {
        "第一步", "第二步", "第三步", "步骤", "Step", "step",
        "首先", "然后", "最后", "接下来"
    }
    local step_count = 0
    for _, pattern in ipairs(step_patterns) do
        step_count = step_count + select(2, combined_text:gsub(pattern, ""))
    end
    if step_count >= 3 then
        return OUTPUT_STRATEGIES.STEP_BY_STEP
    end

    -- 检测简洁输出
    local text_len = #combined_text
    if text_len < 200 then
        return OUTPUT_STRATEGIES.CONCISE
    end

    -- 检测详细输出
    if text_len > 1000 or #ai_responses > 3 then
        return OUTPUT_STRATEGIES.DETAILED
    end

    -- 默认叙述性输出
    return OUTPUT_STRATEGIES.NARRATIVE
end

function M.detect_language(topic_data)
    -- 简单实现：检测文本中的语言标记
    local text = topic_data.content or topic_data.description or ""

    if text:match("python") or text:match("Python") then
        return "python"
    elseif text:match("javascript") or text:match("JavaScript") or text:match("js") then
        return "javascript"
    elseif text:match("lua") or text:match("Lua") then
        return "lua"
    else
        return "unknown"
    end
end

function M.detect_domain(topic_data)
    local text = topic_data.content or topic_data.description or ""

    if text:match("代码") or text:match("编程") or text:match("函数") then
        return "coding"
    elseif text:match("分析") or text:match("统计") then
        return "analysis"
    elseif text:match("对话") or text:match("聊天") then
        return "conversation"
    else
        return "general"
    end
end

function M.detect_task_category(topic_data)
    local text = topic_data.content or topic_data.description or ""

    if text:match("调试") or text:match("错误") or text:match("bug") then
        return "debugging"
    elseif text:match("重构") or text:match("优化") then
        return "refactoring"
    elseif text:match("生成") or text:match("创建") then
        return "generation"
    elseif text:match("解释") or text:match("说明") then
        return "explanation"
    else
        return "general"
    end
end

function M.extract_environment_features(topic_data)
    local env = {}

    -- 从topic_data中提取环境信息
    if topic_data.environment then
        env = topic_data.environment
    else
        -- 默认环境
        env.os = "unknown"
        env.framework = "unknown"
    end

    return env
end

function M.detect_task_type(topic_data)
    -- 检测任务类型
    local text = topic_data.content or topic_data.description or ""

    if text:match("代码") or text:match("编程") then
        return "coding"
    elseif text:match("分析") then
        return "analysis"
    elseif text:match("对话") then
        return "conversation"
    else
        return "general"
    end
end

-- ==================== 结果分析 ====================

function M.analyze_topic_outcome(topic_data, context)
    local tool_success_count, tool_failure_count = count_tool_results(context.tool_calls)
    local tool_calls = #context.tool_calls
    local tool_success_rate = tool_calls > 0 and (tool_success_count / tool_calls) or nil
    local explicit_success = topic_data.outcome and type(topic_data.outcome.success) == "boolean"
        and topic_data.outcome.success
        or nil

    local outcome = {
        success = false,
        reason = "",
        metrics = {}
    }

    if explicit_success ~= nil then
        outcome.success = explicit_success
        outcome.reason = explicit_success and "explicit_success" or "explicit_failure"
    elseif #context.errors > 0 then
        outcome.success = false
        outcome.reason = "encountered_errors"
    elseif tool_calls > 0 and tool_success_count == 0 then
        outcome.success = false
        outcome.reason = "tool_execution_failed"
    else
        outcome.success = true
        outcome.reason = "completed_successfully"
    end

    if #context.errors > 0 then
        outcome.errors = context.errors
    end

    -- 提取指标
    outcome.metrics = {
        duration = (topic_data.end_turn or topic_data.turn) - context.start_turn,
        tool_calls = tool_calls,
        tool_success_count = tool_success_count,
        tool_failure_count = tool_failure_count,
        tool_success_rate = tool_success_rate,
        iterations = #context.intermediate_states,
        errors = #context.errors
    }

    return outcome
end

-- ==================== 模式提取 ====================

function M.extract_patterns(topic_data, context)
    local patterns = {}

    -- 工具使用模式
    for tool, count in pairs(context.tools_used) do
        patterns[#patterns + 1] = {
            type = "tool_usage",
            key = tool,
            tool = tool,
            frequency = count
        }
    end

    local task_pattern = M.detect_task_pattern(topic_data)
    -- 任务模式
    patterns[#patterns + 1] = {
        type = "task",
        key = task_pattern,
        pattern = task_pattern,
        success = context.errors == nil or #context.errors == 0
    }

    -- 错误模式
    for _, error in ipairs(context.errors) do
        local error_pattern = M.extract_error_pattern(error)
        patterns[#patterns + 1] = {
            type = "error",
            key = error_pattern.type or "unknown",
            pattern = error_pattern,
            context = error.context
        }
    end

    return patterns
end

function M.detect_task_pattern(topic_data)
    local text = topic_data.content or topic_data.description or ""

    -- 简单模式检测
    if text:match("调用.*函数") then
        return "function_call"
    elseif text:match("分析.*数据") then
        return "data_analysis"
    else
        return "general"
    end
end

function M.extract_error_pattern(error)
    -- 提取错误模式
    local pattern = {
        type = error.type or "unknown",
        message = error.message or "",
        context = error.context or {}
    }

    return pattern
end

-- ==================== 教训提取 ====================

function M.extract_lessons(topic_data, context)
    local lessons = {}

    -- 成功案例：提取成功因素
    if #context.errors == 0 then
        lessons[#lessons + 1] = {
            type = "success_factor",
            content = M.extract_success_factors(topic_data, context)
        }
    end

    -- 失败案例：提取失败原因和解决方案
    if #context.errors > 0 then
        lessons[#lessons + 1] = {
            type = "failure_cause",
            content = M.extract_failure_cause(topic_data, context)
        }

        lessons[#lessons + 1] = {
            type = "solution",
            content = M.extract_solution(topic_data, context)
        }
    end

    -- 通用教训
    lessons[#lessons + 1] = {
        type = "general",
        content = M.extract_general_lesson(topic_data, context)
    }

    return lessons
end

function M.extract_success_factors(topic_data, context)
    local factors = {}

    -- 提取工具使用成功因素
    for tool, count in pairs(context.tools_used) do
        if count > 0 then
            factors[#factors + 1] = string.format("工具 %s 使用了 %d 次", tool, count)
        end
    end

    return factors
end

function M.extract_failure_cause(topic_data, context)
    local causes = {}

    -- 提取错误原因
    for _, error in ipairs(context.errors) do
        causes[#causes + 1] = error.message or error.type or "unknown_error"
    end

    return causes
end

function M.extract_solution(topic_data, context)
    local solutions = {}

    -- 从topic_data中提取解决方案
    if topic_data.solution then
        solutions[#solutions + 1] = topic_data.solution
    end

    return solutions
end

function M.extract_general_lesson(topic_data, context)
    -- 提取通用教训
    local lesson = ""
    local tools = {}
    for tool, count in pairs(context.tools_used) do
        tools[#tools + 1] = tool
    end

    if #context.errors > 0 then
        lesson = string.format("在 %s 任务中遇到了错误，需要避免类似情况", context.task_type)
    else
        lesson = string.format("在 %s 任务中成功使用了工具 %s", 
            context.task_type,
            table.concat(tools, ", "))
    end

    return lesson
end

-- ==================== 描述生成 ====================

function M.generate_description(topic_data, context, outcome)
    local parts = {}

    -- 添加任务类型
    parts[#parts + 1] = string.format("任务类型: %s", context.task_type)

    -- 添加工具使用
    local tools = {}
    for tool, count in pairs(context.tools_used) do
        tools[#tools + 1] = string.format("%s(%d)", tool, count)
    end
    if #tools > 0 then
        parts[#parts + 1] = string.format("使用的工具: %s", table.concat(tools, ", "))
    end

    -- 添加结果
    parts[#parts + 1] = string.format("结果: %s", outcome.success and "成功" or "失败")

    -- 添加错误信息
    if #context.errors > 0 then
        parts[#parts + 1] = string.format("错误: %d 个", #context.errors)
    end

    return table.concat(parts, "; ")
end

-- ==================== 成功率计算 ====================

function M.compute_success_rate(topic_data, context, outcome)
    if type(outcome) ~= "table" then
        outcome = M.analyze_topic_outcome(topic_data, context)
    end

    local success_count = count_tool_results(context.tool_calls)
    local tool_calls = #context.tool_calls
    local tool_success_rate = tool_calls > 0 and (success_count / tool_calls) or nil
    local outcome_signal = outcome.success and 1.0 or 0.0

    local base
    if tool_success_rate ~= nil then
        base = 0.75 * tool_success_rate + 0.25 * outcome_signal
    else
        base = outcome.success and 0.55 or 0.20
    end

    local error_penalty = math.min(0.35, #context.errors * 0.12)
    return clamp(base - error_penalty, 0.0, 1.0)
end

function M.compute_utility_prior(topic_data, context, outcome, success_rate)
    local prior = tonumber(success_rate) or 0.5

    if outcome.success then
        prior = 0.55 + 0.45 * prior
    else
        local diagnostic_bonus = math.min(0.18, #context.errors * 0.05)
        local solution_bonus = topic_data.solution and 0.12 or 0.0
        prior = 0.30 + 0.30 * prior + diagnostic_bonus + solution_bonus
    end

    return clamp(prior, 0.0, 1.0)
end

function M.build_success_key(context, output_strategy)
    local key_parts = {
        tostring(context.task_type or "general"),
        tostring(output_strategy or "default"),
    }

    local tools = {}
    for tool_name in pairs(context.tools_used or {}) do
        tools[#tools + 1] = tostring(tool_name)
    end

    if #tools > 0 then
        table.sort(tools)
        key_parts[#key_parts + 1] = table.concat(tools, "+")
    end

    return table.concat(key_parts, "|")
end

-- ==================== 手动触发 ====================

-- 手动触发经验构建（用于测试）
function M.build_from_topic(topic_data)
    -- 模拟topic开始
    M.on_topic_start(topic_data)

    -- 模拟topic更新（如果有）
    if topic_data.updates then
        for _, update in ipairs(topic_data.updates) do
            M.on_topic_update(update)
        end
    end

    -- 模拟topic结束
    M.on_topic_end(topic_data)
end

return M
