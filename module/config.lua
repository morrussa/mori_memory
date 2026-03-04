local M = {}

M.settings = {
    merge_limit = 0.95,--两个记忆如果相似度>0.95，则直接合并为一个，并且对应多条记忆。
    dim = 1024,
    heat = {
        total_heat = 10000000,--总热力池大小
        new_memory_heat = 43000,--新记忆的热力
        max_neighbors = 5,--新记忆存入时，最多平分给几个邻居
        neighbors_heat = 26000,--邻居分配的热力，是平分的
        softmax = true,--当softmax = true时，自动禁用热力池设定，每一句话和它的邻居更新都触发全局归一化
        tolerance = 500,--softmax的热力归一化以后的容差
        cold_cluster = {
            neighbor_multiplier = 2.2,   -- 冷簇时邻居热力倍数（1.8~2.5 都行）
            wake_multiplier     = 1.8,   -- 冷簇唤醒热力倍数
            extra_neighbor_heat = 18000, -- 额外给邻居的热力（冷簇专用）
        },
    },
    cluster = {
        cluster_sim = 0.72,--新记忆首先计算与其他簇的质心的相似度，如果>cluster_sim，那么就进入簇内。如果没有，那么就将这个向量本身作为质心。
        hot_cluster_ratio = 0.65,--热簇占比超过hot_cluster_ratio则为热簇，反之为冷簇。
        cluster_heat_cap = 180000,--簇的热力cap（软cap）
        cluster_heat_floor = 6500,
        hierarchical_cluster_enabled = true, -- 开启三层路由（memory -> cluster -> supercluster）；关闭后退回普通簇扫描。
        supercluster_min_clusters = 64,      -- 只有簇数量达到这个阈值，才启用 supercluster（避免小样本下分层开销反而更大）。
        supercluster_target_size = 64,       -- 每个 supercluster 期望容纳的簇数（影响分层粒度与召回粗筛强度）。
        supercluster_sim = 0.52,             -- 新簇挂接到已有 supercluster 的最低相似度；低于该值会新建 supercluster。
        supercluster_max_size_mult = 1.8,    -- supercluster 最大容量倍率，实际 cap = target_size * 该倍率。
        supercluster_topn_add = 3,           -- 写入 add_memory 路径：仅在 topN supercluster 里找候选簇。
        supercluster_topn_query = 4,         -- 查询 recall 路径：仅在 topN supercluster 里找候选簇。
        supercluster_topn_scale = 0.20,      -- 动态扩展系数：簇越多，topN 会按 log2(cluster_count/min_clusters) 增长。
        supercluster_rebuild_every = 600,    -- 增量挂接多少新簇后强制重建 supercluster 索引（抑制索引漂移）。
    },
    time = {
        loss_turn = 50,--距离失去搜索加权的轮数
        time_boost = 0.2,--搜索加权
        maintenance_task = 75,--维护任务的定时器。
    },
    ai_query = {
        max_memory = 5, --基础数字，根据AI搜寻关键词进行精细度查询。如果AI发送了两个关键词query，那么则返回max_memory*2个top
        max_turns = 10,
        expand = 2,--获取所有的命中的top并将每一个memory关联的history turn展开
        flat = false,--如果flat为true，那么每一个和memory关联的history turn则都以1打分，如果flat为false，那么每一个和memory关联的history turn则以它们在自己query内的语义相关度为分数。
        recall_base = 5.3,--总分recall_base时触发回忆
        history_search_bonus = 4.3,--包含明确历史/过去关键词加history_search_bonus分
        technical_term_bonus = 1.3,--包含专业词汇时加technical_term_bonus分
        length_limit = 20,--用户输入的长度达到length_limit时加length_bonus分（你字长跟你混）
        length_bonus = 1.4,
        anxiety_multi = 2.4,--用户的语气很急*embedding检测相似度
        help_cry_multi = 5.4,--用户的语气像求救*embedding检测相似度
        past_talk_multi = -2.1,--用户谈及过去*embedding检测相似度
        explicit_recall_bonus = 2.2, -- 包含明确“记得/回顾/历史”意图时的额外加分
        context_link_bonus = 1.1, -- “继续/接着/刚才”等上下文延续词加分
        fresh_task_penalty = 1.6, -- 典型新任务指令在无回忆意图时扣分，降低误触发
        suppress_recall_penalty = 8.0, -- 检测到“不要回忆/不查记忆”等指令时强制抑制
        short_query_penalty_len = 8, -- 过短输入阈值（字符）
        short_query_penalty = 0.65, -- 过短且无上下文意图时扣分
        technical_bonus_scale_when_no_recall_intent = 0.30, -- 无回忆信号时技术词加分缩放
        length_bonus_scale_when_no_recall_intent = 0.35, -- 无回忆信号时长度加分缩放
        explicit_recall_threshold_scale = 0.72, -- 明确回忆意图时阈值缩放
        context_recall_threshold_scale = 0.88, -- 上下文延续意图时阈值缩放
        recall_cooldown_turns = 1, -- 触发后冷却轮数（无明确回忆意图时提高阈值）
        recall_cooldown_threshold_mult = 1.18, -- 冷却期间阈值倍率
        past_talk_multi_explicit_scale = 0.65, -- 明确回忆时，past_talk 负权重转正的缩放系数
        recall_trigger_debug = false, -- true 时打印触发器分项打分
        topic_same_boost = 1.15,
        topic_similar_boost = 1.02,
        topic_cross_penalty = 0.9,
        topic_sim_threshold = 0.7,
        topic_cross_quota_ratio = 0.25, -- topic 约束召回时，跨 topic 兜底配额比例（0~0.5）
        max_keywords = 4,
        keyword_weight = 0.55,    -- 关键词权重
        keyword_queries = 2,      -- 每轮 query 展开的检索向量数量（含主向量）
        keyword_noise_mix = 0.20, -- 派生 query 向量的噪声混合比例
        keyword_perf_mode = "lossless", -- lossless=零损优化；near_lossless=近零损（次查询优先重排主查询候选）
        near_lossless_min_primary_candidates = 64, -- near_lossless 触发阈值：主查询候选数达到该值才启用
        near_lossless_secondary_cap = 256, -- near_lossless 次查询重排候选上限；0 表示不封顶
        -- [实现方法] keyword 候选聚合防偏：
        -- 关键词只在“主查询召回出的候选池”内重排；若关键词向量与主向量夹角过大则直接跳过，避免 LLM 误生成把结果拖偏。
        keyword_align_reject = 0.1, -- 与主向量相似度低于该值时直接丢弃该关键词查询（硬拒绝）。
        keyword_align_floor = 0.3,  -- 通过硬拒绝后，低于该值仍按 0 权重处理；高于该值才进入平滑降权。
        keyword_align_gamma = 1.4,  -- 对齐度权重曲线指数，越大越保守（中低对齐关键词被压得更狠）。
        memory_drop_sim = 0.60,   -- memory 展开前硬阈值；sim < 阈值的 memory 整条丢弃
        min_sim_gate = 0.58,      -- 硬过滤，直接丢掉泛化噪声
        power_suppress = 1.80,    -- 非线性压制
        -- [实现方法] Smart Preload + EnhancedWeak：
        smart_preload_enabled = true,
        preload_budget_per_query = 5, -- 每次 query 预加载簇数预算
        preload_heat_amount = 25000, -- 旧逻辑兼容字段（簇缓存模式不使用）
        preload_topic_confidence = 0.50,
        preload_use_vector_prediction = true,
        preload_max_io_per_turn = 8, -- 每回合最多预加载簇数
        preload_low_hot_ratio_threshold = 0.15,
        soft_gate_enabled = true,
        soft_gate_margin = 0.10,
        expected_recall_enabled = true,
        cluster_hit_rate_alpha = 0.10,
        route_score_bonus_scale = 0.15,
        empty_gate_decay = 0.98,
        empty_gate_decay_aggressive = 0.95,
        hit_gate_boost = 1.002,
        min_gate_floor = 0.25,
        max_gate_ceiling = 0.85,
        -- [实现方法] learning curve：
        -- recall 每轮根据 progress=lerp(warmup->full) 动态插值 min_gate/power/max_memory/max_turns/keyword_weight/super_topn。
        learning_curve_enabled = true,      -- 开启查询参数“随轮次收敛”；关闭后直接使用静态参数。
        learning_warmup_turns = 100,        -- aggressive 冷启动：更早进入学习插值区间。
        learning_full_turns = 1600,         -- aggressive 冷启动：更快收敛到稳定参数。
        learning_query_noise_extra = 0.18,  -- 早期额外噪声注入上限：噪声=base+(1-progress)*extra。
        learning_min_sim_gate_start = 0.42, -- 早期最小相似度门限（后续收敛到 min_sim_gate）。
        learning_power_suppress_start = 1.15,-- 早期非线性压制指数（后续收敛到 power_suppress）。
        learning_topic_cross_quota_start = 0.48,-- 早期跨topic配额（后续收敛到 topic_cross_quota_ratio）。
        learning_max_memory_start = 3,      -- 早期每query扫描 memory 数量起点（后续收敛到 max_memory）。
        learning_max_turns_start = 14,      -- 早期输出 turn 数量起点（后续收敛到 max_turns）。
        learning_keyword_weight_start = 0.78,-- 早期关键词子查询权重起点（后续收敛到 keyword_weight）。
        learning_super_topn_query_start = 2,-- 早期 supercluster 查询 topN 起点（后续收敛到 supercluster_topn_query）。

        -- [实现方法] cold start boost：
        -- 仅在早期 turn 生效：降低 need_recall 阈值、放宽召回门限并提高检索预算；到 cold_start_turns 后自动退场。
        cold_start_enabled = true,
        cold_start_turns = 1600,
        cold_start_recall_base_scale = 0.82, -- 早期 recall_base 缩放下限（越小越容易触发回忆）。
        cold_start_min_gate_drop = 0.04, -- 早期额外下调 min_sim_gate 幅度。
        cold_start_power_drop = 0.18, -- 早期额外下调 power_suppress 幅度。
        cold_start_max_memory_boost = 2, -- 早期额外增加 max_memory。
        cold_start_max_turns_boost = 2, -- 早期额外增加 max_turns。
        cold_start_probe_clusters_boost = 2, -- 早期额外增加 supercluster_topn_query / 探测簇预算。

        -- [实现方法] refinement：
        -- recall 把候选样本与正/负样本送入 adaptive.update_after_recall，在线更新 learned_min_gate / online_merge_limit / route_score。
        refinement_enabled = true, -- 启用在线自适应（gate、merge_limit、route bias 都会随着召回反馈调整）。
        refinement_start_turn = 80, -- aggressive 冷启动：更早开始在线自适应。
        refinement_sample_mem_topk = 48, -- 每轮用于 refinement 的候选 memory 样本上限（按相似度截断）。
        refinement_route_lr = 0.10, -- 簇路由分数学习率（越大越快改变 route_score）。
        refinement_gate_lr = 0.08, -- learned_min_gate 学习率（控制门限收敛速度）。
        refinement_merge_lr = 0.05, -- online_merge_limit 学习率（控制合并阈值收敛速度）。
        refinement_route_bias_scale = 0.08, -- route_score 参与簇排序时的偏置强度（0=不使用路由偏置）。
        refinement_probe_clusters_start = 8, -- 早期每query探测簇数（高探索）。
        refinement_probe_clusters_end = 2, -- 后期每query探测簇数（高精度）。
        refinement_probe_per_cluster_limit = 12, -- 单簇最多扫描候选 memory 数（限制大簇成本）。

        -- [实现方法] persistent explore：
        -- recall 在主召回外随机/周期探测额外簇，避免长期陷入同一路由局部最优。
        persistent_explore_enabled = true, -- 开启持久探索策略（对抗“只搜热簇”导致的召回盲区）。
        persistent_explore_epsilon = 0.01, -- 每轮随机触发探索概率（epsilon-greedy）。
        persistent_explore_period_turns = 0, -- 周期触发间隔；0 表示关闭周期触发，仅保留 epsilon 触发。
        persistent_explore_extra_clusters = 1, -- 触发探索时，额外补探测的簇数量。
        persistent_explore_candidate_cap = 32, -- 探索簇单簇扫描上限（防止探索带来过高成本）。
        uncertain_recency_enabled = true, -- 灰度默认开启
        uncertain_top1_sim_threshold = 0.62,
        uncertain_top12_gap_threshold = 0.03,
        uncertain_min_candidates = 6,
        uncertain_recency_bonus = 0.035,
        uncertain_recency_half_life = 120,
        uncertain_recency_pool_cap = 32,

        -- [实现方法] topic 分桶与预加载：
        -- recall 先按 same/near/cross 进行桶内选取；当 topic 稳定时可随机预热少量同topic memory 进入 cache 通道。
        use_topic_buckets = false, -- true=按 same/near/cross 分桶再配额；false=全局分数直接截断。
        stable_warmup_turns = 6, -- topic 连续稳定轮数阈值（达到后才允许 random lift）。
        stable_min_pair_sim = 0.72, -- 连续轮 query 向量平均相似度阈值（稳定性判断）。
        topic_random_lift_interval = 3, -- 每隔多少轮尝试一次随机 lift（>1 时做取模触发）。
        topic_random_lift_count = 2, -- 每次 lift 放入 cache 的 memory 数量。
        topic_random_lift_prob = 0.85, -- 满足稳定条件后，本轮实际执行 lift 的概率。
        topic_random_lift_only_cold = true, -- true=只从冷记忆里挑 lift 候选，避免热区重复。
        topic_cache_weight = 1.02, -- cache 命中的分数增益系数（用于轻微偏置同topic记忆）。

        -- [实现方法] 查询驱动冷救援（delayed rescue queue）：
        -- 召回 miss/弱命中时入队冷记忆，按延迟到期后在 maintenance tick 执行唤醒 + 邻居加热。
        cold_rescue_delay_min = 24, -- 入队后最小延迟轮数（防止即时抖动）。
        cold_rescue_delay_max = 120, -- 入队后最大延迟轮数（与 min 组成随机延迟区间）。
        cold_rescue_topn = 3, -- 单次入队最多挑选多少条冷记忆候选。
        cold_rescue_batch = 24, -- 每次 maintenance tick 最多执行多少条到期救援任务。
        cold_rescue_on_empty_only = false, -- true=仅空召回时入队；false=命中不足(hits<=0)也会入队。
        cold_rescue_max_queue = 50000, -- 冷救援队列硬上限，超过后停止入队以保护内存与I/O。
    },
    keyring = {
        long_term_plan = {
            max_value_chars = 200,     -- 单条计划 value 最大长度
            max_evidence_chars = 160,  -- 单条计划 evidence 最大长度
            bom_max_items = 3,         -- 每轮注入 BOM 的最多条数
            bom_max_chars = 800,       -- BOM 注入文本最大长度
        },
        tool_calling = {
            upsert_min_confidence = 0.82,   -- upsert 最低置信度
            upsert_max_per_turn = 1,        -- 每轮最多 upsert 次数
            query_max_per_turn = 2,         -- 每轮最多 query 次数
            delete_enabled = false,         -- 4B 默认禁用 delete
            query_max_types = 3,            -- query types 最多允许几个
            query_fetch_limit = 18,         -- query 先取较大召回池
            query_inject_top = 3,           -- 注入前重排后只保留 top N
            query_inject_max_chars = 800,   -- query 注入文本最大字符数
            tool_pass_temperature = 0.15,   -- 二阶段工具调用温度
            tool_pass_max_tokens = 128,     -- 二阶段工具调用长度
            tool_pass_seed = 42,            -- 二阶段工具调用随机种子
        },
        fact_extractor = {
            -- 默认对齐 simu/dialog_hf_realflow_pipeline.py 的 high_recall_v1 + verify pass。
            prompt_style = "balanced_v3", -- 可选：high_recall_v1 / strict_v2 / balanced_v3 / balanced_en_v1 / baseline
            verify_pass = true,              -- 开启二次质检，控制噪声
            max_facts = 8,                   -- 单轮最多入库原子事实数
            max_parse_items = 12,            -- 单次解析最多保留候选数量
            max_item_chars = 64,             -- 单条事实长度上限
            extract_max_tokens = 320,        -- 首轮抽取长度
            extract_temperature = 0.15,
            extract_seed = 42,
            repair_max_tokens = 192,         -- 修复轮长度
            repair_temperature = 0.0,
            repair_seed = 43,
            verify_max_tokens = 192,         -- 质检轮长度
            verify_temperature = 0.0,
            verify_seed = 46,
        },
    },
    agent = {
        max_steps = 4,
        input_token_budget = 12000,
        completion_reserve_tokens = 1024,
        token_count_mode = "templated_exact", -- 固定：模板后精确计数
        context_drop_order = "memory_tool_plan", -- 超预算时先丢 memory_context，再 tool_context，再 plan_bom
        plan_bom_pinned = true, -- true=滑窗时尽量保留 plan_bom，不可接受的情况下先压缩而不是直接丢弃
        plan_bom_compact_min_chars = 120, -- plan 保底压缩长度
        history_auto_compress = true, -- 超预算丢弃历史时，自动生成历史压缩块注入上下文
        history_auto_compress_min_dropped_pairs = 1, -- 至少丢弃多少历史 pair 才触发自动压缩
        history_auto_compress_max_pairs = 24, -- 自动压缩最多覆盖多少条被丢弃历史 pair
        history_auto_compress_user_chars = 64, -- 压缩块内单条 user 文本截断长度
        history_auto_compress_assistant_chars = 96, -- 压缩块内单条 assistant 文本截断长度
        history_auto_compress_max_chars = 1400, -- 压缩块初始最大长度
        history_auto_compress_min_chars = 220, -- 压缩块最小长度下限
        continue_on_tool_context = true, -- 工具产生新上下文时继续下一步重生成
        max_context_refine_steps = 2, -- 单轮最多因工具上下文增量触发多少次重生成
        continue_on_tool_failure = true, -- 工具失败时触发自修正重试
        max_failure_refine_steps = 2, -- 单轮最多因工具失败触发多少次重修
    },
    topic = {--在退出时，如果topic没有闭合，在topic.bin的第一行写下 current_turn\x1F<topic_head_vec>\x1F<topic_now_vec> ，topic.bin的第一行永远留给这个用途。
        make_cluster1 = 4,--当一个topic建立时，它向前make_cluster1步以建立头质心，
        make_cluster2 = 3,--当头质心建立完成后，它会向后make_cluster2步以建立尾质心，并且每一次建立完成后都与头质心对比
        topic_limit = 0.62,--尾质心与头质心对比低于这个数字以后，它就会判定topic结束，然后为这个topic建立总结、topic内对话整体的质心文件、
        break_limit = 0.48, -- 话语对断裂阈值：当前轮与上一轮相似度低于此值，视为发生“断裂”
        confirm_limit = 0.55, -- 话题无关阈值：断裂发生时，当前轮与头质心相似度低于此值，确认为新话题
        min_topic_length =2,--防止话题被切的太碎
        summary_max_tokens = 192, -- topic 摘要生成的最大输出长度（默认加大，减少截断）。
        rebuild = true,--当异常退出时，如果rebuild为true，那么就找到文件一开始写的current_turn上一个topic的末尾，然后自己根据history.txt的输出自动重建整个topic。这个过程会阻塞主pipeline，因为如果不阻塞就不安全。
    },
    storage_v3 = {
        root = "memory/v3",
        cluster_cache_cap = 24,
        enable_lru = true,
    },
}

return M
