local util = require("module.graph.util")
local config = require("module.config")
local persistence = require("module.persistence")

local M = {}

M.episodes = {}
M.index = nil
M._loaded = false
M._dirty = false
M._dirty_ids = {}

local function storage_root()
    local cfg = ((config.settings or {}).episode or {}).storage or {}
    local root = tostring(cfg.root or "memory/episodes")
    if root == "" then
        root = "memory/episodes"
    end
    return root
end

local function episodes_dir()
    return storage_root() .. "/items"
end

local function index_path()
    return storage_root() .. "/index.lua"
end

local function episode_path(episode_id)
    return string.format("%s/%s.lua", episodes_dir(), tostring(episode_id or "unknown"))
end

local function default_index()
    return {
        ids = {},
        by_task = {},
        by_run = {},
        updated_at_ms = 0,
    }
end

local function shallow_copy_array(src)
    local out = {}
    for i = 1, #(src or {}) do
        out[i] = src[i]
    end
    return out
end

local function shallow_copy_map(src)
    local out = {}
    for k, v in pairs(src or {}) do
        out[k] = v
    end
    return out
end

local function sort_strings(values)
    table.sort(values, function(a, b)
        return tostring(a) < tostring(b)
    end)
    return values
end

local function copy_tool_results(rows)
    local out = {}
    for _, row in ipairs(rows or {}) do
        if type(row) == "table" then
            out[#out + 1] = {
                call_id = tostring(row.call_id or ""),
                tool = tostring(row.tool or ""),
                ok = row.ok == true,
                is_control = row.is_control == true,
                result_preview = util.utf8_take(tostring(row.result_preview or row.result or ""), 320),
                error_preview = util.utf8_take(tostring(row.error_preview or row.error or ""), 240),
            }
        end
    end
    return out
end

local function copy_uploads(rows)
    local out = {}
    for _, row in ipairs(rows or {}) do
        if type(row) == "table" then
            out[#out + 1] = {
                name = tostring(row.name or ""),
                path = tostring(row.path or ""),
                tool_path = tostring(row.tool_path or ""),
                bytes = tonumber(row.bytes) or 0,
            }
        end
    end
    return out
end

local function generate_id()
    return string.format("ep_%d_%04x", util.now_ms(), math.random(0, 0xffff))
end

local function append_unique(seq, value)
    local needle = tostring(value or "")
    if needle == "" then
        return
    end
    for _, existing in ipairs(seq) do
        if tostring(existing) == needle then
            return
        end
    end
    seq[#seq + 1] = needle
end

local function normalize_episode(episode)
    local normalized = {
        id = tostring((episode or {}).id or ""),
        run_id = tostring((episode or {}).run_id or ""),
        task_id = tostring((episode or {}).task_id or ""),
        goal = tostring((episode or {}).goal or ""),
        profile = tostring((episode or {}).profile or ""),
        status = tostring((episode or {}).status or ""),
        stop_reason = tostring((episode or {}).stop_reason or ""),
        success = ((episode or {}).success) == true,
        read_only = ((episode or {}).read_only) == true,
        created_at = tonumber((episode or {}).created_at) or os.time(),
        created_at_ms = tonumber((episode or {}).created_at_ms) or util.now_ms(),
        turn_index = tonumber((episode or {}).turn_index) or 0,
        topic_anchor = tostring((episode or {}).topic_anchor or ""),
        user_input = tostring((episode or {}).user_input or ""),
        final_text = tostring((episode or {}).final_text or ""),
        summary = tostring((episode or {}).summary or ""),
        tool_sequence = shallow_copy_array((episode or {}).tool_sequence or {}),
        tools_used = shallow_copy_map((episode or {}).tools_used or {}),
        tool_results = copy_tool_results((episode or {}).tool_results or {}),
        files_read = shallow_copy_array((episode or {}).files_read or {}),
        files_written = shallow_copy_array((episode or {}).files_written or {}),
        uploads = copy_uploads((episode or {}).uploads or {}),
        retrieved_policy_ids = shallow_copy_array((episode or {}).retrieved_policy_ids or {}),
        effective_policy_ids = shallow_copy_array((episode or {}).effective_policy_ids or {}),
        policy_writeback = shallow_copy_map((episode or {}).policy_writeback or {}),
        memory_writeback = {
            facts = shallow_copy_array((((episode or {}).memory_writeback) or {}).facts or {}),
            saved = tonumber((((episode or {}).memory_writeback) or {}).saved) or 0,
        },
        metrics = shallow_copy_map((episode or {}).metrics or {}),
    }

    if normalized.id == "" then
        normalized.id = generate_id()
    end
    if normalized.summary == "" then
        normalized.summary = util.utf8_take(
            normalized.final_text ~= "" and normalized.final_text or normalized.goal,
            240
        )
    end

    normalized.files_read = sort_strings(normalized.files_read)
    normalized.files_written = sort_strings(normalized.files_written)
    normalized.retrieved_policy_ids = sort_strings(normalized.retrieved_policy_ids)
    normalized.effective_policy_ids = sort_strings(normalized.effective_policy_ids)

    return normalized
end

local function rebuild_index()
    local index = default_index()
    local ids = {}

    for id, episode in pairs(M.episodes) do
        ids[#ids + 1] = id
        if tostring((episode or {}).task_id or "") ~= "" then
            local task_id = tostring(episode.task_id)
            index.by_task[task_id] = index.by_task[task_id] or {}
            append_unique(index.by_task[task_id], id)
        end
        if tostring((episode or {}).run_id or "") ~= "" then
            index.by_run[tostring(episode.run_id)] = id
        end
    end

    sort_strings(ids)
    index.ids = ids

    for _, rows in pairs(index.by_task) do
        table.sort(rows, function(a, b)
            local ep_a = M.episodes[a] or {}
            local ep_b = M.episodes[b] or {}
            local t_a = tonumber(ep_a.created_at_ms or ep_a.created_at) or 0
            local t_b = tonumber(ep_b.created_at_ms or ep_b.created_at) or 0
            if t_a == t_b then
                return tostring(a) < tostring(b)
            end
            return t_a > t_b
        end)
    end

    index.updated_at_ms = util.now_ms()
    M.index = index
end

function M.init()
    util.ensure_dir(storage_root())
    util.ensure_dir(episodes_dir())
    M.load()
end

function M.add(episode)
    if type(episode) ~= "table" then
        return false, "invalid_episode"
    end

    local normalized = normalize_episode(episode)
    M.episodes[normalized.id] = normalized
    M._dirty_ids[normalized.id] = true
    M._dirty = true
    rebuild_index()
    return true, normalized.id
end

function M.get(id)
    return M.episodes[tostring(id or "")]
end

function M.get_recent_by_task(task_id, limit)
    local task = tostring(task_id or "")
    if task == "" then
        return {}
    end

    local rows = ((M.index or {}).by_task or {})[task] or {}
    local out = {}
    local cap = math.max(1, math.floor(tonumber(limit) or 5))
    for i = 1, math.min(cap, #rows) do
        local episode = M.episodes[rows[i]]
        if episode then
            out[#out + 1] = episode
        end
    end
    return out
end

function M.count()
    local count = 0
    for _ in pairs(M.episodes) do
        count = count + 1
    end
    return count
end

function M.load()
    M.episodes = {}
    M.index = default_index()
    M._dirty = false
    M._dirty_ids = {}

    util.ensure_dir(storage_root())
    util.ensure_dir(episodes_dir())

    local f = io.open(index_path(), "rb")
    if not f then
        M._loaded = true
        print("[EpisodeStore] index.lua 不存在，使用空 episode 存储")
        return
    end

    local raw = f:read("*a") or ""
    f:close()

    local parsed, err = util.parse_lua_table_literal(raw)
    if type(parsed) ~= "table" then
        M._loaded = true
        print(string.format("[EpisodeStore] index.lua 解析失败，使用空 episode 存储: %s", tostring(err)))
        return
    end

    local ids = shallow_copy_array(parsed.ids or {})
    for _, id in ipairs(ids) do
        local path = episode_path(id)
        local ep_file = io.open(path, "rb")
        if ep_file then
            local ep_raw = ep_file:read("*a") or ""
            ep_file:close()
            local episode = util.parse_lua_table_literal(ep_raw)
            if type(episode) == "table" then
                local normalized = normalize_episode(episode)
                M.episodes[normalized.id] = normalized
            end
        end
    end

    rebuild_index()
    M._loaded = true
    print(string.format("[EpisodeStore] Loaded %d episodes", M.count()))
end

function M.save()
    if not M._dirty then
        return true
    end

    util.ensure_dir(storage_root())
    util.ensure_dir(episodes_dir())

    for episode_id in pairs(M._dirty_ids) do
        local episode = M.episodes[episode_id]
        if episode then
            local ok, err = persistence.write_atomic(episode_path(episode_id), "wb", function(f)
                return f:write(util.encode_lua_value(episode, 0))
            end)
            if not ok then
                return false, err
            end
        end
    end

    rebuild_index()
    local ok, err = persistence.write_atomic(index_path(), "wb", function(f)
        return f:write(util.encode_lua_value(M.index or default_index(), 0))
    end)
    if not ok then
        return false, err
    end

    M._dirty = false
    M._dirty_ids = {}
    return true
end

function M.get_stats()
    local task_count = 0
    for _ in pairs(((M.index or {}).by_task) or {}) do
        task_count = task_count + 1
    end
    return {
        total_episodes = M.count(),
        indexed_tasks = task_count,
        loaded = M._loaded == true,
    }
end

return M
