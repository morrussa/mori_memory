local M = {}

M.store = require("module.episode.store")
M.run_builder = require("module.episode.run_builder")

M._initialized = false

function M.init()
    if M._initialized == true then
        return
    end

    M.store.init()
    M._initialized = true
    print("[Episode] Module initialized")
end

function M.finalize()
    if M._initialized ~= true then
        return true
    end
    return M.store.save()
end

function M.add_episode(episode)
    return M.store.add(episode)
end

function M.get_recent_by_task(task_id, limit)
    return M.store.get_recent_by_task(task_id, limit)
end

return M
