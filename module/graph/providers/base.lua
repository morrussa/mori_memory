local M = {}

function M.new()
    local obj = {}

    function obj:list_tools()
        return {}
    end

    function obj:call(_tool_name, _tool_args)
        return false, "provider_call_not_implemented"
    end

    return obj
end

return M
