local topic_graph = require("module.memory.topic_graph")

local M = {}

function M.load()
    return topic_graph.load()
end

function M.save_to_disk()
    return topic_graph.save_to_disk()
end

function M.begin_turn(turn)
    return topic_graph.begin_turn(turn)
end

function M.get_total_lines()
    return topic_graph.get_total_lines()
end

function M.get_turns(line)
    return topic_graph.get_turns(line)
end

function M.get_topic_anchor(line)
    return topic_graph.get_topic_anchor(line)
end

function M.get_cluster_id(line)
    return topic_graph.get_cluster_id(line)
end

function M.return_mem_vec(line)
    return topic_graph.return_mem_vec(line)
end

function M.iter_topic_lines(topic_anchor)
    return topic_graph.iter_topic_lines(topic_anchor)
end

function M.iterate_all()
    return topic_graph.iterate_all()
end

function M.find_similar_all_fast(query_vec, max_results)
    return topic_graph.find_similar_all_fast(query_vec, max_results)
end

function M.add_memory(vec, turn, opts)
    return topic_graph.add_memory(vec, turn, opts)
end

function M.store_vector(line, cluster_id, vec)
    return topic_graph.store_vector(line, cluster_id, vec)
end

function M.update_topic_anchor(line, new_anchor)
    return topic_graph.update_topic_anchor(line, new_anchor)
end

function M.get_cached_cluster_ids()
    return {}
end

function M.get_preload_io_count()
    return 0
end

function M.reserve_preload_io(requested, max_per_turn)
    requested = math.max(0, math.floor(tonumber(requested) or 0))
    max_per_turn = math.max(0, math.floor(tonumber(max_per_turn) or requested))
    return math.min(requested, max_per_turn)
end

function M.is_cluster_cached(_cluster_id)
    return false
end

function M.preload_clusters(_cluster_ids, _opts)
    return {}
end

function M.preload_lines(_lines, _opts)
    return {}
end

function M.preload_prediction(prediction, _opts)
    local topics = {}
    for _, anchor in ipairs(((prediction or {}).predicted_topics) or {}) do
        topics[#topics + 1] = anchor
    end
    return topics
end

return M
