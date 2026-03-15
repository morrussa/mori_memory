from __future__ import annotations

from pathlib import Path
from typing import Any


class MoriMemoryBridge:
    def __init__(self, *, lua_root: Path, py_pipeline: object | None = None) -> None:
        try:
            from lupa.luajit21 import LuaRuntime  # type: ignore[import-not-found]
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "Missing dependency: lupa. Create/activate venv, then `pip install -r requirements.txt`."
            ) from e

        lua_root = lua_root.resolve()
        if not lua_root.is_dir():
            raise FileNotFoundError(f"mori_memory lua_root not found: {lua_root}")

        self._lua_root = lua_root
        self.lua = LuaRuntime(unpack_returned_tuples=True)

        append_path = self.lua.eval("function(p) package.path = package.path .. ';' .. p end")
        append_path(str(lua_root / "?.lua"))
        append_path(str(lua_root / "?/init.lua"))

        if py_pipeline is not None:
            self.lua.globals().py_pipeline = py_pipeline

        require = self.lua.eval("require")
        self.memory = require("mori_memory")

    def ingest_turn(self, meta: dict[str, Any]) -> Any:
        return self.memory.ingest_turn(meta)

    def compile_context(self, meta: dict[str, Any]) -> Any:
        return self.memory.compile_context(meta)

    def shutdown(self) -> None:
        self.memory.shutdown()
