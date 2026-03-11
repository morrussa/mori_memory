local context_builder = require("module.graph.context_builder")

_G.py_pipeline = {
    count_chat_tokens = function(_, messages)
        local chars = 0
        for _, row in ipairs(messages or {}) do
            chars = chars + #(tostring((row or {}).content or ""))
        end
        return math.floor(chars / 4) + 1
    end,
}

local state = {
    messages = {
        system_prompt = "You are assistant",
        conversation_history = {
            { role = "system", content = "You are assistant" },
            { role = "user", content = string.rep("old question ", 30) },
            { role = "assistant", content = string.rep("old answer ", 40) },
            { role = "user", content = string.rep("recent question ", 25) },
            { role = "assistant", content = string.rep("recent answer ", 28) },
        },
    },
    context = {},
    input = { message = "new question" },
}

local messages, meta = context_builder.build_chat_messages(state)
assert(type(messages) == "table" and #messages >= 2, "messages should be generated")
assert(type(meta) == "table", "meta should exist")
assert(type(meta.history_variant_counts) == "table", "variant stats should exist")

local vc = meta.history_variant_counts
assert((vc.full or 0) + (vc.slight or 0) + (vc.heavy or 0) + (vc.none or 0) >= 2, "should classify each history pair")

print("context_builder variant selection test passed")
