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
        topic_same_boost = 1.15,
        topic_similar_boost = 1.02,
        topic_cross_penalty = 0.9,
        topic_sim_threshold = 0.7,
        topic_cross_quota_ratio = 0.25, -- topic 约束召回时，跨 topic 兜底配额比例（0~0.5）
        max_keywords = 4,
        keyword_weight = 0.55,    -- 关键词权重
        min_sim_gate = 0.58,      -- 硬过滤，直接丢掉泛化噪声
        power_suppress = 1.80,    -- 非线性压制
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
    },
    topic = {--在退出时，如果topic没有闭合，在topic.bin的第一行写下 current_turn\x1F<topic_head_vec>\x1F<topic_now_vec> ，topic.bin的第一行永远留给这个用途。
        make_cluster1 = 4,--当一个topic建立时，它向前make_cluster1步以建立头质心，
        make_cluster2 = 3,--当头质心建立完成后，它会向后make_cluster2步以建立尾质心，并且每一次建立完成后都与头质心对比
        topic_limit = 0.62,--尾质心与头质心对比低于这个数字以后，它就会判定topic结束，然后为这个topic建立总结、topic内对话整体的质心文件、
        break_limit = 0.48, -- 话语对断裂阈值：当前轮与上一轮相似度低于此值，视为发生“断裂”
        confirm_limit = 0.55, -- 话题无关阈值：断裂发生时，当前轮与头质心相似度低于此值，确认为新话题
        min_topic_length =2,--防止话题被切的太碎
        rebuild = true,--当异常退出时，如果rebuild为true，那么就找到文件一开始写的current_turn上一个topic的末尾，然后自己根据history.txt的输出自动重建整个topic。这个过程会阻塞主pipeline，因为如果不阻塞就不安全。
    },
}

return M
