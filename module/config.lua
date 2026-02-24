local M = {}

M.settings = {
    merge_limit = 0.95,--两个记忆如果相似度>0.95，则直接合并为一个，并且对应多条记忆。
    heat = {
        total_heat = 10000000,--总热力池大小
        heat_pool_ratio = 1000000,--热力池的大小，用于给新记忆存入
        new_memory_heat = 50000,--新记忆的热力
        max_neighbors = 5,--新记忆存入时，最多平分给几个邻居
        neighbors_heat = 20000,--邻居分配的热力，是平分的
        in_cluster_heat_boost = 0.2,--簇内最大的热力搜索加权
        heat_limit = 0.05--单个记忆的热力如果超过总热力*cluster_limit，则不再能够获得热力。
    },
    cluster = {
        cluster_sim = 0.75,--新记忆首先计算与其他簇的质心的相似度，如果>cluster_sim，那么就进入簇内。如果没有，那么就将这个向量本身作为质心。
        heat_limit = 0.25,--单个簇的热力如果超过总热力*cluster_limit，则不再能够获得热力。
        hot_cluster_ratio = 0.65,--热簇占比超过50%则为热簇，反之为冷簇。
    },
    time = {
        loss_turn = 50,--距离失去搜索加权的轮数
        time_boost = 0.2,--搜索加权
        maintenance_task = 100,--维护任务的定时器。
    },
    ai_query = {
        max_memory = 5 --一次原子事实查询
    },
}

return M