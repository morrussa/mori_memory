local util = require("module.graph.util")
local qwen_provider_mod = require("module.graph.providers.qwen_provider")

local M = {}

local PROVIDER_BUILDERS = {
    qwen = function(opts)
        return qwen_provider_mod.new(opts or {})
    end,
}

local function normalize_name(name)
    local n = util.trim(name)
    if n == "" then
        return "qwen"
    end
    return n
end

function M.create(provider_name, opts)
    local name = normalize_name(provider_name)
    local builder = PROVIDER_BUILDERS[name]
    if type(builder) ~= "function" then
        return nil, "provider_not_registered"
    end
    local ok, provider_or_err = pcall(builder, opts or {})
    if not ok or type(provider_or_err) ~= "table" then
        return nil, tostring(provider_or_err or "provider_init_failed")
    end
    return provider_or_err
end

return M
