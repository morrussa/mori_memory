import lupa.luajit21 as lupa
from lupa.luajit21 import LuaRuntime
import numpy as np
import zstandard as zstd
import argparse
import errno
import os
import io
import tarfile
import shutil
import atexit
import json
import gzip
import hashlib
import mimetypes
import re
import socket
import subprocess
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from email import policy as email_policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

try:
    import json5
except Exception:
    json5 = None


class LlamaCppServerClient:
    def __init__(
        self,
        server_bin: str,
        model_path: str,
        ctx_size: int,
        embedding: bool = False,
        draft_model_path: str | None = None,
        draft_max: int | None = None,
        draft_min: int | None = None,
        draft_p_min: float | None = None,
        ctx_size_draft: int | None = None,
        draft_gpu_layers="0",
        host: str = "127.0.0.1",
        port: int = None,
        enable_webui: bool = False,
        enable_jinja: bool = True,
        api_key: str = "",
        log_ready_url: bool = True,
        log_to_file: bool = True,
        startup_timeout: int = 600,
        gpu_layers="all",
    ):
        self.server_bin = server_bin
        self.model_path = model_path
        self.ctx_size = int(ctx_size)
        self.embedding = bool(embedding)
        self.draft_model_path = str(draft_model_path).strip() if draft_model_path else ""
        self.draft_max = int(draft_max) if draft_max is not None else None
        self.draft_min = int(draft_min) if draft_min is not None else None
        self.draft_p_min = float(draft_p_min) if draft_p_min is not None else None
        self.ctx_size_draft = int(ctx_size_draft) if ctx_size_draft is not None else None
        self.server_host = str(host or "127.0.0.1")
        self.request_host = "127.0.0.1" if self.server_host in ("0.0.0.0", "::") else self.server_host
        self.enable_webui = bool(enable_webui) and not self.embedding
        self.enable_jinja = bool(enable_jinja) and not self.embedding
        self.api_key = str(api_key or "").strip()
        self.log_ready_url = bool(log_ready_url)
        self.log_to_file = bool(log_to_file)
        self.startup_timeout = int(startup_timeout)
        raw_gpu_layers = str(gpu_layers if gpu_layers is not None else "all").strip().lower()
        if raw_gpu_layers in {"all", "-1"}:
            self.gpu_layers = "all"
        else:
            try:
                parsed_layers = int(raw_gpu_layers)
            except ValueError:
                parsed_layers = -1
            self.gpu_layers = max(0, parsed_layers) if parsed_layers >= 0 else "all"

        raw_draft_gpu_layers = str(draft_gpu_layers if draft_gpu_layers is not None else "0").strip().lower()
        if raw_draft_gpu_layers in {"all", "-1"}:
            self.draft_gpu_layers = "all"
        else:
            try:
                parsed_layers = int(raw_draft_gpu_layers)
            except ValueError:
                parsed_layers = 0
            self.draft_gpu_layers = max(0, parsed_layers)

        self.model_name = os.path.basename(model_path) or "local-model"
        self.port = int(port) if port else self._find_free_port(self.request_host)
        self.base_url = f"http://{self.request_host}:{self.port}"
        self.process = None
        self.log_path = None
        self._log_file = None
        self._start_server()

    @staticmethod
    def _find_free_port(host: str) -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((host, 0))
            return int(sock.getsockname()[1])

    def get_webui_url(self) -> str:
        if not self.enable_webui:
            return ""
        host = self.server_host
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{self.port}"

    def _start_server(self):
        role = "embedding" if self.embedding else "chat"
        if self.log_to_file:
            os.makedirs("logs", exist_ok=True)
            self.log_path = os.path.join("logs", f"llama_server_{role}_{self.port}.log")
            self._log_file = open(self.log_path, "a", encoding="utf-8")
        else:
            self.log_path = None
            self._log_file = None

        cmd = [
            self.server_bin,
            "--model",
            self.model_path,
            "--host",
            self.server_host,
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.ctx_size),
        ]
        cmd.extend(["--gpu-layers", str(self.gpu_layers)])

        if self.draft_model_path and not self.embedding:
            cmd.extend(["--model-draft", self.draft_model_path])
            if self.ctx_size_draft is not None and self.ctx_size_draft > 0:
                cmd.extend(["--ctx-size-draft", str(int(self.ctx_size_draft))])
            if self.draft_max is not None and self.draft_max > 0:
                cmd.extend(["--draft-max", str(int(self.draft_max))])
            if self.draft_min is not None and self.draft_min > 0:
                cmd.extend(["--draft-min", str(int(self.draft_min))])
            if self.draft_p_min is not None and self.draft_p_min > 0:
                cmd.extend(["--draft-p-min", str(float(self.draft_p_min))])
            cmd.extend(["--gpu-layers-draft", str(self.draft_gpu_layers)])

        if not self.embedding:
            cmd.extend(["--reasoning-format", "none"])
            if self.enable_jinja:
                cmd.append("--jinja")

        if not self.enable_webui:
            cmd.append("--no-webui")

        if self.embedding:
            cmd.append("--embeddings")

        if self.api_key:
            cmd.extend(["--api-key", self.api_key])

        env = os.environ.copy()
        lib_dir = os.path.dirname(self.server_bin)
        old_ld_path = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{lib_dir}:{old_ld_path}" if old_ld_path else lib_dir

        stdout_target = self._log_file if self._log_file is not None else subprocess.DEVNULL
        self.process = subprocess.Popen(
            cmd,
            stdout=stdout_target,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._wait_until_ready()
        if self.log_ready_url:
            print(f"[Python] llama-server ready ({role}) at {self.base_url}")
        else:
            print(f"[Python] llama-server ready ({role})")

    def _wait_until_ready(self):
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                log_tail = self._tail_log()
                log_hint = (
                    f"log tail:\n{log_tail}"
                    if self.log_to_file
                    else "file logging disabled (set MORI_LLAMA_SERVER_LOG_TO_FILE=1 to enable)"
                )
                raise RuntimeError(
                    f"llama-server exited early (code={self.process.returncode}) for model: {self.model_path}\n"
                    f"{log_hint}"
                )

            try:
                status, _ = self._raw_http("GET", "/health", timeout=2)
                if status == 200:
                    return
            except Exception:
                pass

            time.sleep(0.5)

        if self.log_to_file:
            raise TimeoutError(
                f"Timed out waiting llama-server ({self.model_path}) on {self.base_url}. "
                f"Check logs: {self.log_path}"
            )
        raise TimeoutError(
            f"Timed out waiting llama-server ({self.model_path}) on {self.base_url}. "
            "file logging disabled (set MORI_LLAMA_SERVER_LOG_TO_FILE=1 to enable)"
        )

    def _tail_log(self, lines: int = 40) -> str:
        if not self.log_path or not os.path.exists(self.log_path):
            return ""
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.readlines()
            return "".join(data[-lines:]).strip()
        except Exception:
            return ""

    def _raw_http(self, method: str, endpoint: str, payload=None, timeout: int = 600):
        url = f"{self.base_url}{endpoint}"
        data = None
        if payload is not None:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
                return int(resp.status), body
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return int(e.code), body

    def _request_json(self, method: str, endpoint: str, payload=None, timeout: int = 600):
        status, body = self._raw_http(method, endpoint, payload=payload, timeout=timeout)
        if status >= 400:
            raise RuntimeError(
                f"llama-server request failed ({status}) {endpoint}: {body[:4000]}"
            )
        if not body:
            return {}
        try:
            return json.loads(body)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"Invalid JSON from llama-server {endpoint}: {e}; body={body[:1000]}"
            ) from e

    def create_chat_completion(
        self,
        messages,
        max_tokens=128,
        temperature=0.7,
        stop=None,
        seed=None,
        tools=None,
        tool_choice=None,
        parallel_tool_calls=None,
    ):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)
        if tools:
            payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            payload["parallel_tool_calls"] = bool(parallel_tool_calls)
        return self._request_json(
            "POST",
            "/v1/chat/completions",
            payload=payload,
            timeout=3600,
        )

    @staticmethod
    def _extract_delta_text(delta) -> str:
        if not isinstance(delta, dict):
            return ""
        content = delta.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts)
        if content is None:
            return ""
        return str(content)

    def create_chat_completion_stream(self, messages, max_tokens=128, temperature=0.7, stop=None, seed=None):
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "stream": True,
        }
        if stop:
            payload["stop"] = list(stop)
        if seed is not None:
            payload["seed"] = int(seed)

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                if int(resp.status) >= 400:
                    body = resp.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"llama-server stream failed ({resp.status}): {body[:4000]}")

                while True:
                    raw_line = resp.readline()
                    if not raw_line:
                        break
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if not data_str:
                        continue
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    choices = obj.get("choices") if isinstance(obj, dict) else None
                    if not choices or not isinstance(choices[0], dict):
                        continue
                    delta = choices[0].get("delta")
                    piece = self._extract_delta_text(delta)
                    if piece:
                        yield piece
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"llama-server stream failed ({e.code}): {body[:4000]}") from e

    def create_embedding(self, texts):
        payload = {
            "model": self.model_name,
            "input": texts,
            "encoding_format": "float",
        }
        return self._request_json(
            "POST",
            "/v1/embeddings",
            payload=payload,
            timeout=600,
        )

    def apply_chat_template(self, messages):
        payload = {
            "messages": messages,
        }
        output = self._request_json(
            "POST",
            "/apply-template",
            payload=payload,
            timeout=120,
        )
        prompt = output.get("prompt")
        if prompt is None:
            raise RuntimeError(f"Invalid /apply-template response: {output}")
        return str(prompt)

    def tokenize_text(self, text: str, add_special: bool = True, parse_special: bool = True):
        payload = {
            "content": str(text or ""),
            "add_special": bool(add_special),
            "parse_special": bool(parse_special),
        }
        output = self._request_json(
            "POST",
            "/tokenize",
            payload=payload,
            timeout=120,
        )
        tokens = output.get("tokens")
        if not isinstance(tokens, list):
            raise RuntimeError(f"Invalid /tokenize response: {output}")
        return len(tokens)

    def stop(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
            self.process = None

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None


class AIPipeline:
    """Lua-facing bridge for model IO, tool adapters, and state archive lifecycle."""
    _tool_args_lua_runtime = None
    _tool_args_lua_parser = None

    def __init__(self):
        self.lua_runtime = None
        self.llm_large = None   # GPU 大模型（生成用）
        self.llm_embed = None   # GGUF Embedding 模型
        self.suppress_large_webui_log = False
        self.quiet_server_urls = False
        self._qwen_tool_instances = {}
        self._qwen_tool_specs = {}
        self.llama_cpp_root = os.environ.get("LLAMA_CPP_ROOT", "/home/morusa/AI/llama-cpp")
        self.large_server_host = os.environ.get("MORI_LARGE_SERVER_HOST", "127.0.0.1")
        self.large_server_port = self._get_env_port("MORI_LARGE_SERVER_PORT", default=None)
        self.large_server_webui = self._get_env_bool("MORI_LARGE_SERVER_WEBUI", True)
        self.large_server_jinja = self._get_env_bool("MORI_LARGE_SERVER_JINJA", True)
        self.large_server_api_key = str(os.environ.get("MORI_LARGE_SERVER_API_KEY", "") or "").strip()
        self.large_server_gpu_layers = str(os.environ.get("MORI_LARGE_SERVER_GPU_LAYERS", "all") or "all").strip()
        self.embed_server_gpu_layers = str(os.environ.get("MORI_EMBED_SERVER_GPU_LAYERS", "0") or "0").strip()
        self.large_server_gpu_fallback_cpu = self._get_env_bool("MORI_LARGE_SERVER_GPU_FALLBACK_CPU", True)
        self.large_server_spec_enabled = self._get_env_bool("MORI_LARGE_SERVER_SPEC_ENABLED", False)
        self.large_server_draft_model = str(os.environ.get("MORI_LARGE_SERVER_DRAFT_MODEL", "") or "").strip()
        self.large_server_draft_gpu_layers = str(
            os.environ.get("MORI_LARGE_SERVER_DRAFT_GPU_LAYERS", "") or ""
        ).strip()
        self.large_server_draft_max = self._get_env_int("MORI_LARGE_SERVER_DRAFT_MAX", 0)
        self.large_server_draft_min = self._get_env_int("MORI_LARGE_SERVER_DRAFT_MIN", 0)
        self.large_server_draft_p_min = self._get_env_float("MORI_LARGE_SERVER_DRAFT_P_MIN", 0.0)
        self.large_server_draft_ctx_size = self._get_env_int("MORI_LARGE_SERVER_DRAFT_CTX_SIZE", 0)
        self.llama_server_log_to_file = self._get_env_bool("MORI_LLAMA_SERVER_LOG_TO_FILE", True)
        self.agent_files_root = os.path.abspath(
            str(os.environ.get("MORI_AGENT_FILES_DIR", "workspace") or "workspace")
        )
        self.agent_virtual_root = "/mori/workspace"
        os.makedirs(self.agent_files_root, exist_ok=True)
        self.llama_server_bin = os.environ.get(
            "LLAMA_SERVER_BIN",
            os.path.join(self.llama_cpp_root, "build", "bin", "llama-server"),
        )
        atexit.register(self.shutdown)

    @staticmethod
    def _get_env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _get_env_port(name: str, default=None):
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        if value <= 0:
            return default
        return value

    @staticmethod
    def _get_env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return int(default)
        try:
            return int(raw)
        except ValueError:
            return int(default)

    @staticmethod
    def _get_env_float(name: str, default: float) -> float:
        raw = os.environ.get(name)
        if raw is None:
            return float(default)
        try:
            return float(raw)
        except ValueError:
            return float(default)

    @staticmethod
    def _normalize_embedding_vec(embedding):
        emb_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tolist()

    @staticmethod
    def _extract_chat_text(output: dict) -> str:
        text, _tool_calls = AIPipeline._extract_chat_text_and_tool_calls(output)
        return text

    def _luaify(self, value):
        lua_runtime = getattr(self, "lua_runtime", None)
        if lua_runtime is None:
            return value
        try:
            return _py_to_lua_value(lua_runtime, value)
        except Exception:
            return value

    @staticmethod
    def _extract_chat_text_and_tool_calls(output: dict):
        choices = output.get("choices") if isinstance(output, dict) else None
        if not choices:
            raise RuntimeError(f"Invalid chat completion response: {output}")

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            text = "".join(texts)
        else:
            text = str(content)

        def _normalize_tool_call(item):
            if not isinstance(item, dict):
                return None
            fn = item.get("function")
            if not isinstance(fn, dict):
                return None
            name = str(fn.get("name", "") or "").strip()
            if not name:
                return None
            return {
                "id": str(item.get("id", "") or ""),
                "type": str(item.get("type", "function") or "function"),
                "function": {
                    "name": name,
                    "arguments": str(fn.get("arguments", "") or ""),
                },
            }

        tool_calls_raw = message.get("tool_calls")
        tool_calls = []
        if isinstance(tool_calls_raw, list):
            for item in tool_calls_raw:
                norm = _normalize_tool_call(item)
                if norm:
                    tool_calls.append(norm)

        # Legacy/兼容分支：部分服务只返回 function_call，不返回 tool_calls。
        if not tool_calls:
            fc = message.get("function_call")
            if isinstance(fc, dict):
                name = str(fc.get("name", "") or "").strip()
                if name:
                    tool_calls.append(
                        {
                            "id": str(message.get("tool_call_id", "") or "1"),
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": str(fc.get("arguments", "") or ""),
                            },
                        }
                    )

        return text, tool_calls

    @staticmethod
    def _coerce_lua_value(value, _depth: int = 0):
        if _depth > 24:
            return None
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            out = {}
            for k, v in value.items():
                out[str(k)] = AIPipeline._coerce_lua_value(v, _depth + 1)
            return out
        if isinstance(value, (list, tuple)):
            return [AIPipeline._coerce_lua_value(v, _depth + 1) for v in value]

        try:
            keys = value.keys()
        except Exception:
            keys = None
        if keys is not None:
            out = {}
            for k in keys:
                out[str(k)] = AIPipeline._coerce_lua_value(value[k], _depth + 1)
            return out

        try:
            n = len(value)
        except Exception:
            n = None
        if isinstance(n, int) and n >= 0:
            seq = []
            ok = True
            for i in range(1, n + 1):
                try:
                    seq.append(AIPipeline._coerce_lua_value(value[i], _depth + 1))
                except Exception:
                    ok = False
                    break
            if ok:
                return seq

            seq = []
            ok = True
            for i in range(0, n):
                try:
                    seq.append(AIPipeline._coerce_lua_value(value[i], _depth + 1))
                except Exception:
                    ok = False
                    break
            if ok:
                return seq

        return value

    @staticmethod
    def _coerce_messages(messages):
        messages_list = []

        try:
            length = len(messages)
            if length > 0:
                for i in range(1, length + 1):
                    msg = messages[i]
                    if msg and "role" in msg and "content" in msg:
                        content = AIPipeline._coerce_lua_value(msg["content"])
                        if content is None:
                            content = ""
                        row = {
                            "role": str(msg["role"]),
                            "content": content,
                        }
                        for extra_key in ("name", "tool_call_id", "tool_calls", "reasoning_content", "function_call", "function_id"):
                            if extra_key in msg and msg[extra_key] is not None:
                                row[extra_key] = AIPipeline._coerce_lua_value(msg[extra_key])
                        messages_list.append(
                            row
                        )
        except TypeError:
            if isinstance(messages, dict) and "role" in messages and "content" in messages:
                content = AIPipeline._coerce_lua_value(messages["content"])
                if content is None:
                    content = ""
                row = {
                    "role": str(messages["role"]),
                    "content": content,
                }
                for extra_key in ("name", "tool_call_id", "tool_calls", "reasoning_content", "function_call", "function_id"):
                    if extra_key in messages and messages[extra_key] is not None:
                        row[extra_key] = AIPipeline._coerce_lua_value(messages[extra_key])
                messages_list.append(
                    row
                )

        if not messages_list and isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                if "role" not in msg or "content" not in msg:
                    continue
                content = AIPipeline._coerce_lua_value(msg.get("content"))
                if content is None:
                    content = ""
                row = {
                    "role": str(msg.get("role")),
                    "content": content,
                }
                for extra_key in ("name", "tool_call_id", "tool_calls", "reasoning_content", "function_call", "function_id"):
                    if msg.get(extra_key) is not None:
                        row[extra_key] = AIPipeline._coerce_lua_value(msg.get(extra_key))
                messages_list.append(
                    row
                )

        return AIPipeline._normalize_fncall_messages_for_oai(messages_list)

    @staticmethod
    def _normalize_fncall_messages_for_oai(messages):
        normalized = []

        def _normalize_tool_calls(tool_calls):
            out = []
            if not isinstance(tool_calls, list):
                return out
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    continue
                fn = tc.get("function")
                if not isinstance(fn, dict):
                    continue
                name = str(fn.get("name", "") or "").strip()
                if not name:
                    continue
                out.append(
                    {
                        "id": str(tc.get("id", "") or ""),
                        "type": str(tc.get("type", "function") or "function"),
                        "function": {
                            "name": name,
                            "arguments": str(fn.get("arguments", "") or ""),
                        },
                    }
                )
            return out

        for msg in messages or []:
            if not isinstance(msg, dict):
                continue
            row = dict(msg)
            role = str(row.get("role", "") or "").strip().lower()

            if role == "function":
                row["role"] = "tool"
                if row.get("tool_call_id") is None:
                    row["tool_call_id"] = str(row.get("function_id", "") or row.get("id", "") or "1")
                if row.get("content") is None:
                    row["content"] = ""
                normalized.append(row)
                continue

            fc = row.get("function_call")
            if role == "assistant" and isinstance(fc, dict):
                fn_name = str(fc.get("name", "") or "").strip()
                if fn_name:
                    tc = {
                        "id": str(
                            row.get("tool_call_id", "")
                            or row.get("function_id", "")
                            or row.get("id", "")
                            or "1"
                        ),
                        "type": "function",
                        "function": {
                            "name": fn_name,
                            "arguments": str(fc.get("arguments", "") or ""),
                        },
                    }
                    if normalized and str(normalized[-1].get("role", "")).lower() == "assistant":
                        prev_tcs = _normalize_tool_calls(normalized[-1].get("tool_calls"))
                        prev_tcs.append(tc)
                        normalized[-1]["tool_calls"] = prev_tcs
                        if (
                            row.get("content")
                            and (not normalized[-1].get("content"))
                        ):
                            normalized[-1]["content"] = row.get("content")
                    else:
                        normalized.append(
                            {
                                "role": "assistant",
                                "content": row.get("content", ""),
                                "tool_calls": [tc],
                                **(
                                    {"reasoning_content": row.get("reasoning_content")}
                                    if row.get("reasoning_content") is not None
                                    else {}
                                ),
                            }
                        )
                    continue

            if role == "assistant" and row.get("tool_calls") is not None:
                row["tool_calls"] = _normalize_tool_calls(row.get("tool_calls"))
            normalized.append(row)
        return normalized

    @staticmethod
    def _extract_first_json_object(text: str) -> str:
        s = str(text or "")
        start = s.find("{")
        if start < 0:
            return ""

        depth = 0
        quote = None
        escaped = False
        for idx in range(start, len(s)):
            ch = s[idx]
            if quote is not None:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == quote:
                    quote = None
                continue

            if ch in {'"', "'"}:
                quote = ch
                continue
            if ch == "{":
                depth += 1
                continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return s[start:idx + 1]
        return ""

    @staticmethod
    def _extract_first_lua_table(text: str) -> str:
        return AIPipeline._extract_first_json_object(text)

    @staticmethod
    def _get_tool_args_lua_parser():
        parser = AIPipeline._tool_args_lua_parser
        if parser is False:
            return None
        if parser is not None:
            return parser
        try:
            runtime = LuaRuntime(unpack_returned_tuples=True)
            parser = runtime.eval(
                "function(src)\n"
                "  local text = tostring(src or '')\n"
                "  text = text:gsub('^%s*(.-)%s*$', '%1')\n"
                "  if text == '' then return nil end\n"
                "  local chunk = load('return ' .. text, 'tool_args', 't', {})\n"
                "  if not chunk then return nil end\n"
                "  local ok, value = pcall(chunk)\n"
                "  if not ok or type(value) ~= 'table' then return nil end\n"
                "  return value\n"
                "end"
            )
            AIPipeline._tool_args_lua_runtime = runtime
            AIPipeline._tool_args_lua_parser = parser
            return parser
        except Exception:
            AIPipeline._tool_args_lua_parser = False
            return None

    @staticmethod
    def _parse_lua_table_arguments(text: str):
        if not isinstance(text, str):
            return None
        parser = AIPipeline._get_tool_args_lua_parser()
        if parser is None:
            return None

        raw = text.strip()
        if not raw:
            return {}

        candidates = [raw]
        first_tbl = AIPipeline._extract_first_lua_table(raw)
        if first_tbl and first_tbl != raw:
            candidates.append(first_tbl)

        for candidate in candidates:
            try:
                parsed = parser(candidate)
            except Exception:
                parsed = None
            parsed = AIPipeline._coerce_lua_value(parsed)
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _parse_tool_arguments_relaxed(args_raw):
        if isinstance(args_raw, dict):
            return args_raw
        if not isinstance(args_raw, str):
            return {}

        text = args_raw.strip()
        if not text:
            return {}

        parsed_lua = AIPipeline._parse_lua_table_arguments(text)
        if isinstance(parsed_lua, dict):
            return parsed_lua

        parsers = [json.loads]
        if json5 is not None:
            parsers.append(json5.loads)

        for parser in parsers:
            try:
                parsed = parser(text)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        first_obj = AIPipeline._extract_first_json_object(text)
        if first_obj:
            for parser in parsers:
                try:
                    parsed = parser(first_obj)
                    if isinstance(parsed, dict):
                        return parsed
                except Exception:
                    continue

        return {}

    @staticmethod
    def _coerce_qwen_tool_entries(tool_entries):
        raw = AIPipeline._coerce_lua_value(tool_entries)
        if raw is None:
            return []
        if isinstance(raw, (str, dict)):
            return [raw]
        if isinstance(raw, list):
            return raw
        return []

    @staticmethod
    def _normalize_qwen_tool_entry(entry):
        if isinstance(entry, str):
            name = entry.strip()
            if not name:
                return None, None
            return name, None
        if isinstance(entry, dict):
            if isinstance(entry.get("function"), dict):
                fn = entry.get("function") or {}
                name = str(fn.get("name", "") or "").strip()
                if not name:
                    return None, None
                cfg = entry.get("cfg")
                if cfg is None:
                    cfg = entry.get("config")
                if cfg is None:
                    cfg = entry
                if isinstance(cfg, dict):
                    cfg = dict(cfg)
                    cfg.pop("function", None)
                return name, cfg if isinstance(cfg, dict) else None
            name = str(entry.get("name", "") or "").strip()
            if not name:
                return None, None
            cfg = dict(entry)
            cfg.pop("name", None)
            return name, cfg if cfg else None
        return None, None

    def list_qwen_tool_names(self):
        try:
            from qwen_agent.tools import TOOL_REGISTRY
        except Exception as e:
            raise RuntimeError(f"qwen-agent tools unavailable: {e}") from e
        return sorted([str(x) for x in TOOL_REGISTRY.keys()])

    def get_qwen_tool_schemas(self, tool_entries=None):
        try:
            from qwen_agent.tools import TOOL_REGISTRY
        except Exception as e:
            raise RuntimeError(f"qwen-agent tools unavailable: {e}") from e

        entries = self._coerce_qwen_tool_entries(tool_entries)
        if not entries:
            entries = sorted(TOOL_REGISTRY.keys())

        schemas = []
        new_instances = {}
        new_specs = {}

        for entry in entries:
            tool_name, tool_cfg = self._normalize_qwen_tool_entry(entry)
            if not tool_name:
                continue
            cls = TOOL_REGISTRY.get(tool_name)
            if cls is None:
                print(f"[Python][WARN] qwen tool not registered: {tool_name}")
                continue
            try:
                inst = cls(tool_cfg)
            except Exception as e:
                print(f"[Python][WARN] qwen tool init failed: {tool_name}: {e}")
                continue

            fn = getattr(inst, "function", None)
            if not isinstance(fn, dict):
                fn = {
                    "name": tool_name,
                    "description": str(getattr(inst, "description", "") or ""),
                    "parameters": getattr(inst, "parameters", {"type": "object", "properties": {}}),
                }
            fn_name = str(fn.get("name", "") or tool_name).strip()
            if not fn_name:
                continue

            schema = {
                "type": "function",
                "function": {
                    "name": fn_name,
                    "description": str(fn.get("description", "") or ""),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                },
            }
            schemas.append(schema)
            new_instances[fn_name] = inst
            new_specs[fn_name] = schema

        self._qwen_tool_instances = new_instances
        self._qwen_tool_specs = new_specs
        return schemas

    @staticmethod
    def _normalize_qwen_tool_result(result):
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            return json.dumps(result, ensure_ascii=False, indent=2)
        if isinstance(result, list):
            packed = []
            for item in result:
                if hasattr(item, "model_dump"):
                    packed.append(item.model_dump())
                    continue
                if isinstance(item, dict):
                    packed.append(item)
                    continue
                text = getattr(item, "text", None)
                file_v = getattr(item, "file", None)
                image_v = getattr(item, "image", None)
                audio_v = getattr(item, "audio", None)
                if text is not None or file_v is not None or image_v is not None or audio_v is not None:
                    packed.append(
                        {
                            "text": text,
                            "file": file_v,
                            "image": image_v,
                            "audio": audio_v,
                        }
                    )
                else:
                    packed.append(str(item))
            return json.dumps(packed, ensure_ascii=False, indent=2)
        return str(result)

    def call_qwen_tool(self, tool_name, tool_args="{}"):
        name = str(tool_name or "").strip()
        if not name:
            raise ValueError("tool_name is empty")

        tool = self._qwen_tool_instances.get(name)
        if tool is None:
            self.get_qwen_tool_schemas([name])
            tool = self._qwen_tool_instances.get(name)
        if tool is None:
            raise RuntimeError(f"qwen tool `{name}` is not available")

        args = self._coerce_lua_value(tool_args)
        tool_args_raw = "{}"
        if isinstance(args, dict):
            tool_args_raw = json.dumps(args, ensure_ascii=False)
        else:
            raw = str(args or "").strip()
            if raw:
                parsed = self._parse_tool_arguments_relaxed(raw)
                if isinstance(parsed, dict) and parsed:
                    tool_args_raw = json.dumps(parsed, ensure_ascii=False)
                else:
                    tool_args_raw = raw

        try:
            result = tool.call(tool_args_raw)
        except Exception as e:
            raise RuntimeError(f"qwen tool `{name}` call failed: {e}") from e
        return self._normalize_qwen_tool_result(result)

    def _agent_root_display(self) -> str:
        return self.agent_virtual_root

    def _agent_display_path(self, rel_path: str = "") -> str:
        base = self._agent_root_display().rstrip("/")
        rel = str(rel_path or "").strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        if not rel:
            return base
        return f"{base}/{rel}"

    def _normalize_agent_rel_path(self, raw_path: str) -> str:
        raw = str(raw_path or "").strip().replace("\\", "/")
        if not raw:
            return ""

        while raw.startswith("./"):
            raw = raw[2:]

        virtual_root = self.agent_virtual_root.rstrip("/")
        if raw == virtual_root:
            return ""
        if raw.startswith(virtual_root + "/"):
            raw = raw[len(virtual_root) + 1 :]
        elif raw.startswith("workspace/") or raw.startswith("agent_files/"):
            return ""
        elif raw.startswith("/"):
            return ""

        normalized = os.path.normpath(raw).replace("\\", "/")
        if normalized in {".", ""}:
            return ""
        if normalized == ".." or normalized.startswith("../") or normalized.startswith("/"):
            return ""
        return normalized

    def _resolve_agent_fs_path(self, raw_path: str):
        raw = str(raw_path or "").strip()
        if not raw:
            return "", ""

        rel = self._normalize_agent_rel_path(raw)
        root = os.path.abspath(self.agent_files_root)
        if rel == "":
            if raw.replace("\\", "/").rstrip("/") in {"", self.agent_virtual_root.rstrip("/")}:
                return "", root
            return "", ""

        abs_path = os.path.abspath(os.path.join(root, rel))
        if not (abs_path.startswith(root + os.sep) or abs_path == root):
            return "", ""
        return rel, abs_path

    @staticmethod
    def _coerce_int_arg(value, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _coerce_bool_arg(value, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            s = value.strip().lower()
            if s in {"1", "true", "yes", "on"}:
                return True
            if s in {"0", "false", "no", "off"}:
                return False
        return bool(default)

    @staticmethod
    def _alias_tool_output(text: str, old_name: str, new_name: str) -> str:
        value = str(text or "")
        if not old_name or not new_name:
            return value
        return value.replace(str(old_name), str(new_name))

    def _coerce_workspace_tool_args(self, args_raw, direct_key: str = ""):
        args = self._parse_tool_arguments_relaxed(args_raw)
        if isinstance(args, dict) and args:
            return args
        if isinstance(args_raw, dict):
            return dict(args_raw)
        if isinstance(args_raw, str):
            raw = args_raw.strip()
            if raw and direct_key and not raw.startswith("{"):
                return {direct_key: raw}
        return {}

    # Graph V1 file tools (new names)
    def list_files(self, args_raw="{}", default_limit=12, hard_limit=64):
        out = self.list_agent_files(args_raw, default_limit, hard_limit)
        return self._alias_tool_output(out, "list_agent_files", "list_files")

    def read_file(self, args_raw="{}", default_max_chars=3000, hard_max_chars=12000):
        out = self.read_agent_file(args_raw, default_max_chars, hard_max_chars)
        return self._alias_tool_output(out, "read_agent_file", "read_file")

    def read_lines(self, args_raw="{}", default_max_lines=220, hard_max_lines=1200):
        out = self.read_agent_file_lines(args_raw, default_max_lines, hard_max_lines)
        return self._alias_tool_output(out, "read_agent_file_lines", "read_lines")

    def search_file(self, args_raw="{}", default_max_hits=20, hard_max_hits=200):
        out = self.search_agent_file(args_raw, default_max_hits, hard_max_hits)
        return self._alias_tool_output(out, "search_agent_file", "search_file")

    def search_files(
        self,
        args_raw="{}",
        default_max_hits=30,
        hard_max_hits=400,
        default_max_files=24,
        hard_max_files=200,
        default_per_file_hits=5,
        hard_per_file_hits=20,
    ):
        out = self.search_agent_files(
            args_raw,
            default_max_hits,
            hard_max_hits,
            default_max_files,
            hard_max_files,
            default_per_file_hits,
            hard_per_file_hits,
        )
        return self._alias_tool_output(out, "search_agent_files", "search_files")

    def write_file(self, args_raw="{}"):
        out = self.write_agent_file(args_raw)
        return self._alias_tool_output(out, "write_agent_file", "write_file")

    def apply_patch(self, args_raw="{}"):
        out = self.apply_agent_patch(args_raw)
        return self._alias_tool_output(out, "apply_agent_patch", "apply_patch")

    def exec_command(self, args_raw="{}"):
        out = self.exec_agent_command(args_raw)
        return self._alias_tool_output(out, "exec_agent_command", "exec_command")

    def list_agent_files(self, args_raw="{}", default_limit=12, hard_limit=64):
        args = self._coerce_workspace_tool_args(args_raw, direct_key="prefix")

        limit = int(args.get("limit") or default_limit or 12)
        hard = max(1, int(hard_limit or 64))
        if limit < 1:
            limit = int(default_limit or 12)
        limit = min(limit, hard)

        prefix = str(args.get("prefix") or args.get("path_prefix") or "").strip()
        prefix_norm = self._normalize_agent_rel_path(prefix)
        prefix_norm = os.path.normpath(prefix_norm).replace("\\", "/") if prefix_norm else ""
        if prefix_norm in {".", ".."}:
            prefix_norm = ""
        if prefix_norm.startswith("../") or prefix_norm.startswith("/"):
            return "list_agent_files error: invalid prefix"

        root = os.path.abspath(self.agent_files_root)
        root_display = self._agent_root_display()
        if not os.path.isdir(root):
            return f"list_agent_files: {root_display} is empty"

        rows = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                try:
                    stat = os.stat(abs_path)
                except OSError:
                    continue
                rel = os.path.relpath(abs_path, root).replace("\\", "/")
                if prefix_norm:
                    if rel == prefix_norm:
                        pass
                    elif not rel.startswith(prefix_norm.rstrip("/") + "/"):
                        continue
                rows.append((stat.st_mtime, rel, int(stat.st_size)))

        if not rows:
            if prefix_norm:
                return f"list_agent_files: no files under {self._agent_display_path(prefix_norm)}"
            return f"list_agent_files: no files in {root_display}"

        rows.sort(key=lambda x: x[0], reverse=True)
        shown = rows[:limit]
        lines = [f"[agent_files] root={root_display} showing {len(shown)}/{len(rows)} files"]
        for idx, (_mtime, rel, size) in enumerate(shown, 1):
            lines.append(f"{idx}) {self._agent_display_path(rel)} | bytes={size}")
        if len(rows) > len(shown):
            lines.append(f"... ({len(rows) - len(shown)} more)")
        return "\n".join(lines)

    def read_agent_file(self, args_raw="{}", default_max_chars=3000, hard_max_chars=12000):
        args = self._coerce_workspace_tool_args(args_raw, direct_key="path")

        raw_path = str(args.get("path") or args.get("file") or args.get("target") or "").strip()
        if not raw_path:
            return "read_agent_file error: missing `path`"

        rel, abs_path = self._resolve_agent_fs_path(raw_path)
        root_display = self._agent_root_display()
        if not abs_path or abs_path == os.path.abspath(self.agent_files_root):
            return f"read_agent_file error: path is outside {root_display}"
        if not os.path.isfile(abs_path):
            return f"read_agent_file error: file not found: {self._agent_display_path(rel)}"

        max_chars = int(args.get("max_chars") or default_max_chars or 3000)
        hard = max(128, int(hard_max_chars or 12000))
        if max_chars < 1:
            max_chars = int(default_max_chars or 3000)
        max_chars = min(max_chars, hard)

        start_char = int(args.get("start_char") or args.get("offset_char") or 1)
        if start_char < 1:
            start_char = 1

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
        except Exception as e:
            return f"read_agent_file error: failed to read file: {e}"

        total = len(text)
        start_idx = min(total, start_char - 1)
        end_idx = min(total, start_idx + max_chars)
        chunk = text[start_idx:end_idx]
        truncated = end_idx < total

        header = (
            f"[read_agent_file] path={self._agent_display_path(rel)} "
            f"start_char={start_char} max_chars={max_chars} "
            f"returned_chars={len(chunk)} total_chars={total} truncated={'yes' if truncated else 'no'}"
        )
        if chunk:
            return header + "\n" + chunk
        return header + "\n[empty slice]"

    def read_agent_file_lines(self, args_raw="{}", default_max_lines=220, hard_max_lines=1200):
        args = self._coerce_workspace_tool_args(args_raw, direct_key="path")

        raw_path = str(args.get("path") or args.get("file") or args.get("target") or "").strip()
        if not raw_path:
            return "read_agent_file_lines error: missing `path`"

        rel, abs_path = self._resolve_agent_fs_path(raw_path)
        root_display = self._agent_root_display()
        if not abs_path or abs_path == os.path.abspath(self.agent_files_root):
            return f"read_agent_file_lines error: path is outside {root_display}"
        if not os.path.isfile(abs_path):
            return f"read_agent_file_lines error: file not found: {self._agent_display_path(rel)}"

        max_lines = self._coerce_int_arg(args.get("max_lines"), int(default_max_lines or 220))
        hard = max(16, int(hard_max_lines or 1200))
        if max_lines < 1:
            max_lines = int(default_max_lines or 220)
        max_lines = min(max_lines, hard)

        start_line = self._coerce_int_arg(
            args.get("start_line") or args.get("line_start") or args.get("from_line"),
            1,
        )
        if start_line < 1:
            start_line = 1
        raw_end = args.get("end_line") or args.get("line_end") or args.get("to_line")
        has_end = raw_end is not None and str(raw_end).strip() != ""
        end_line = self._coerce_int_arg(raw_end, 0)

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except Exception as e:
            return f"read_agent_file_lines error: failed to read file: {e}"

        total_lines = len(lines)
        if has_end and end_line < start_line:
            end_line = start_line
        requested_end = end_line if has_end else (start_line + max_lines - 1)
        effective_end = min(total_lines, requested_end, start_line + max_lines - 1)
        start_idx = min(max(0, start_line - 1), total_lines)

        out_rows = []
        if total_lines > 0 and start_idx < total_lines and effective_end >= start_line:
            width = max(3, len(str(total_lines)))
            for line_no in range(start_line, effective_end + 1):
                out_rows.append(f"{line_no:>{width}} | {lines[line_no - 1]}")

        returned_lines = len(out_rows)
        truncated = (requested_end > effective_end) or ((not has_end) and effective_end < total_lines)
        header = (
            f"[read_agent_file_lines] path={self._agent_display_path(rel)} "
            f"start_line={start_line} end_line={effective_end if effective_end > 0 else 0} "
            f"max_lines={max_lines} returned_lines={returned_lines} total_lines={total_lines} "
            f"truncated={'yes' if truncated else 'no'}"
        )
        if out_rows:
            return header + "\n" + "\n".join(out_rows)
        return header + "\n[empty slice]"

    def search_agent_file(self, args_raw="{}", default_max_hits=20, hard_max_hits=200):
        args = self._coerce_workspace_tool_args(args_raw, direct_key="path")

        raw_path = str(args.get("path") or args.get("file") or args.get("target") or "").strip()
        if not raw_path:
            return "search_agent_file error: missing `path`"
        pattern = str(args.get("pattern") or args.get("query") or args.get("string") or "").strip()
        if not pattern:
            return "search_agent_file error: missing `pattern`"

        rel, abs_path = self._resolve_agent_fs_path(raw_path)
        root_display = self._agent_root_display()
        if not abs_path or abs_path == os.path.abspath(self.agent_files_root):
            return f"search_agent_file error: path is outside {root_display}"
        if not os.path.isfile(abs_path):
            return f"search_agent_file error: file not found: {self._agent_display_path(rel)}"

        max_hits = self._coerce_int_arg(args.get("max_hits"), int(default_max_hits or 20))
        hard = max(8, int(hard_max_hits or 200))
        if max_hits < 1:
            max_hits = int(default_max_hits or 20)
        max_hits = min(max_hits, hard)

        context_lines = self._coerce_int_arg(args.get("context_lines"), 0)
        if context_lines < 0:
            context_lines = 0
        if context_lines > 30:  # 增加上限以支持 Codex 风格的上下文窗口
            context_lines = 30

        case_sensitive = self._coerce_bool_arg(args.get("case_sensitive"), False)
        regex_mode = self._coerce_bool_arg(args.get("regex"), False)
        start_line = self._coerce_int_arg(
            args.get("start_line") or args.get("line_start") or args.get("from_line"),
            1,
        )
        if start_line < 1:
            start_line = 1
        raw_end = args.get("end_line") or args.get("line_end") or args.get("to_line")
        has_end = raw_end is not None and str(raw_end).strip() != ""
        end_line = self._coerce_int_arg(raw_end, 0)

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except Exception as e:
            return f"search_agent_file error: failed to read file: {e}"

        total_lines = len(lines)
        if has_end and end_line < start_line:
            end_line = start_line
        scan_end = min(total_lines, end_line if has_end else total_lines)
        scan_start = min(max(1, start_line), total_lines + 1)

        matcher = None
        if regex_mode:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                matcher = re.compile(pattern, flags=flags)
            except re.error as e:
                return f"search_agent_file error: invalid regex: {e}"
        elif not case_sensitive:
            pattern_low = pattern.lower()
        else:
            pattern_low = pattern

        total_hits = 0
        shown = 0
        blocks = []
        width = max(3, len(str(max(1, total_lines))))

        for line_no in range(scan_start, scan_end + 1):
            line = lines[line_no - 1]
            if regex_mode:
                matched = matcher.search(line) is not None
            elif case_sensitive:
                matched = pattern_low in line
            else:
                matched = pattern_low in line.lower()
            if not matched:
                continue

            total_hits += 1
            if shown >= max_hits:
                continue

            shown += 1
            from_no = max(scan_start, line_no - context_lines)
            to_no = min(scan_end, line_no + context_lines)
            block = [f"#{shown} line={line_no}"]
            for n in range(from_no, to_no + 1):
                mark = ">" if n == line_no else " "
                block.append(f"{mark}{n:>{width}} | {lines[n - 1]}")
            blocks.append("\n".join(block))

        pattern_view = pattern if len(pattern) <= 120 else (pattern[:120] + "...")
        header = (
            f"[search_agent_file] path={self._agent_display_path(rel)} pattern={json.dumps(pattern_view, ensure_ascii=False)} "
            f"regex={'yes' if regex_mode else 'no'} case_sensitive={'yes' if case_sensitive else 'no'} "
            f"start_line={scan_start if total_lines > 0 else 0} end_line={scan_end if total_lines > 0 else 0} "
            f"hits={total_hits} shown={shown} context_lines={context_lines} truncated={'yes' if total_hits > shown else 'no'}"
        )
        if not blocks:
            return header + "\n[no matches]"
        return header + "\n" + "\n\n".join(blocks)

    def search_agent_files(
        self,
        args_raw="{}",
        default_max_hits=30,
        hard_max_hits=400,
        default_max_files=24,
        hard_max_files=200,
        default_per_file_hits=5,
        hard_per_file_hits=20,
    ):
        args = self._coerce_workspace_tool_args(args_raw, direct_key="prefix")

        pattern = str(args.get("pattern") or args.get("query") or args.get("string") or "").strip()
        if not pattern:
            return "search_agent_files error: missing `pattern`"

        prefix = str(args.get("prefix") or args.get("path_prefix") or "").strip()
        prefix_norm = self._normalize_agent_rel_path(prefix)
        prefix_norm = os.path.normpath(prefix_norm).replace("\\", "/") if prefix_norm else ""
        if prefix_norm in {".", ".."}:
            prefix_norm = ""
        if prefix_norm.startswith("../") or prefix_norm.startswith("/"):
            return "search_agent_files error: invalid prefix"

        max_hits = self._coerce_int_arg(args.get("max_hits"), int(default_max_hits or 30))
        hard_hits = max(8, int(hard_max_hits or 400))
        if max_hits < 1:
            max_hits = int(default_max_hits or 30)
        max_hits = min(max_hits, hard_hits)

        max_files = self._coerce_int_arg(args.get("max_files"), int(default_max_files or 24))
        hard_files = max(1, int(hard_max_files or 200))
        if max_files < 1:
            max_files = int(default_max_files or 24)
        max_files = min(max_files, hard_files)

        per_file_hits = self._coerce_int_arg(args.get("per_file_hits"), int(default_per_file_hits or 5))
        hard_per_file = max(1, int(hard_per_file_hits or 20))
        if per_file_hits < 1:
            per_file_hits = int(default_per_file_hits or 5)
        per_file_hits = min(per_file_hits, hard_per_file)

        context_lines = self._coerce_int_arg(args.get("context_lines"), 0)
        if context_lines < 0:
            context_lines = 0
        if context_lines > 30:  # 增加上限以支持 Codex 风格的上下文窗口
            context_lines = 30

        case_sensitive = self._coerce_bool_arg(args.get("case_sensitive"), False)
        regex_mode = self._coerce_bool_arg(args.get("regex"), False)
        start_line = self._coerce_int_arg(
            args.get("start_line") or args.get("line_start") or args.get("from_line"),
            1,
        )
        if start_line < 1:
            start_line = 1
        raw_end = args.get("end_line") or args.get("line_end") or args.get("to_line")
        has_end = raw_end is not None and str(raw_end).strip() != ""
        end_line = self._coerce_int_arg(raw_end, 0)

        root = os.path.abspath(self.agent_files_root)
        root_display = self._agent_root_display()
        if not os.path.isdir(root):
            return f"search_agent_files: {root_display} is empty"

        rows = []
        for dirpath, _, filenames in os.walk(root):
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                try:
                    stat = os.stat(abs_path)
                except OSError:
                    continue
                rel = os.path.relpath(abs_path, root).replace("\\", "/")
                if prefix_norm:
                    if rel == prefix_norm:
                        pass
                    elif not rel.startswith(prefix_norm.rstrip("/") + "/"):
                        continue
                rows.append((stat.st_mtime, rel, abs_path))

        if not rows:
            if prefix_norm:
                return f"search_agent_files: no files under {self._agent_display_path(prefix_norm)}"
            return f"search_agent_files: no files in {root_display}"

        rows.sort(key=lambda x: x[0], reverse=True)
        scan_rows = rows[:max_files]

        matcher = None
        if regex_mode:
            flags = 0 if case_sensitive else re.IGNORECASE
            try:
                matcher = re.compile(pattern, flags=flags)
            except re.error as e:
                return f"search_agent_files error: invalid regex: {e}"
        elif not case_sensitive:
            pattern_low = pattern.lower()
        else:
            pattern_low = pattern

        total_hits = 0
        shown = 0
        scanned_files = 0
        blocks = []

        for _, rel, abs_path in scan_rows:
            scanned_files += 1
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.read().splitlines()
            except Exception:
                continue

            total_lines = len(lines)
            if has_end and end_line < start_line:
                end_line = start_line
            scan_end = min(total_lines, end_line if has_end else total_lines)
            scan_start = min(max(1, start_line), total_lines + 1)
            width = max(3, len(str(max(1, total_lines))))
            shown_in_file = 0

            for line_no in range(scan_start, scan_end + 1):
                line = lines[line_no - 1]
                if regex_mode:
                    matched = matcher.search(line) is not None
                elif case_sensitive:
                    matched = pattern_low in line
                else:
                    matched = pattern_low in line.lower()
                if not matched:
                    continue

                total_hits += 1
                if shown >= max_hits or shown_in_file >= per_file_hits:
                    continue

                shown += 1
                shown_in_file += 1

                from_no = max(scan_start, line_no - context_lines)
                to_no = min(scan_end, line_no + context_lines)
                block = [f"#{shown} file={self._agent_display_path(rel)} line={line_no}"]
                for n in range(from_no, to_no + 1):
                    mark = ">" if n == line_no else " "
                    block.append(f"{mark}{n:>{width}} | {lines[n - 1]}")
                blocks.append("\n".join(block))

        pattern_view = pattern if len(pattern) <= 120 else (pattern[:120] + "...")
        header = (
            f"[search_agent_files] files_total={len(rows)} files_scanned={scanned_files} "
            f"pattern={json.dumps(pattern_view, ensure_ascii=False)} "
            f"regex={'yes' if regex_mode else 'no'} case_sensitive={'yes' if case_sensitive else 'no'} "
            f"max_files={max_files} max_hits={max_hits} per_file_hits={per_file_hits} "
            f"context_lines={context_lines} hits={total_hits} shown={shown} "
            f"truncated={'yes' if total_hits > shown or len(rows) > scanned_files else 'no'}"
        )
        if not blocks:
            return header + "\n[no matches]"
        return header + "\n" + "\n\n".join(blocks)

    def write_agent_file(self, args_raw="{}"):
        args = self._coerce_workspace_tool_args(args_raw, direct_key="path")
        raw_path = str(args.get("path") or args.get("file") or "").strip()
        if not raw_path:
            return "write_agent_file error: missing `path`"

        rel, abs_path = self._resolve_agent_fs_path(raw_path)
        root_display = self._agent_root_display()
        if not abs_path or abs_path == os.path.abspath(self.agent_files_root):
            return f"write_agent_file error: path is outside {root_display}"

        content = args.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = str(content)

        mode = str(args.get("mode") or "overwrite").strip().lower() or "overwrite"
        if mode not in {"overwrite", "append"}:
            return "write_agent_file error: invalid `mode`"

        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        try:
            if mode == "append":
                with open(abs_path, "a", encoding="utf-8", errors="replace") as f:
                    f.write(content)
            else:
                with open(abs_path, "w", encoding="utf-8", errors="replace") as f:
                    f.write(content)
        except Exception as e:
            return f"write_agent_file error: failed to write file: {e}"

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                total_chars = len(f.read())
        except Exception:
            total_chars = len(content)

        return (
            f"[write_agent_file] path={self._agent_display_path(rel)} "
            f"mode={mode} written_chars={len(content)} total_chars={total_chars}"
        )

    def apply_agent_patch(self, args_raw="{}"):
        args = self._coerce_workspace_tool_args(args_raw, direct_key="patch")
        patch_text = args.get("patch")
        if patch_text is None:
            patch_text = args.get("diff")
        patch_text = str(patch_text or "")
        if not patch_text.strip():
            return "apply_agent_patch error: missing `patch`"

        rewritten = []
        touched = []
        seen = set()
        for raw_line in patch_text.splitlines():
            line = raw_line
            if line.startswith("--- ") or line.startswith("+++ "):
                header = line[:4]
                target = line[4:].split("\t", 1)[0].strip()
                if target != "/dev/null":
                    rel, abs_path = self._resolve_agent_fs_path(target)
                    if not abs_path or abs_path == os.path.abspath(self.agent_files_root):
                        return "apply_agent_patch error: patch touches path outside /mori/workspace"
                    if rel not in seen:
                        seen.add(rel)
                        touched.append(self._agent_display_path(rel))
                    suffix = line[4 + len(target):]
                    line = header + rel + suffix
            rewritten.append(line)

        if not touched:
            return "apply_agent_patch error: no valid workspace targets found in patch"

        try:
            proc = subprocess.run(
                ["patch", "-p0", "--batch", "--forward", "--reject-file", "-"],
                input=("\n".join(rewritten) + "\n").encode("utf-8"),
                cwd=self.agent_files_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return "apply_agent_patch error: patch command timed out"
        except Exception as e:
            return f"apply_agent_patch error: failed to run patch: {e}"

        stdout_text = proc.stdout.decode("utf-8", errors="replace").strip()
        stderr_text = proc.stderr.decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            detail = stderr_text or stdout_text or "patch failed"
            return f"apply_agent_patch error: {detail}"

        summary = (
            f"[apply_agent_patch] touched={len(touched)} "
            f"targets={', '.join(touched)} exit_code={proc.returncode}"
        )
        if stdout_text:
            summary += "\n" + stdout_text[:2000]
        return summary

    def exec_agent_command(self, args_raw="{}"):
        args = self._coerce_workspace_tool_args(args_raw)
        argv = args.get("argv")
        if not isinstance(argv, list) or not argv:
            return "exec_agent_command error: missing `argv`"

        cmd = [str(item) for item in argv if str(item) != ""]
        if not cmd:
            return "exec_agent_command error: missing `argv`"

        raw_workdir = str(args.get("workdir") or self.agent_virtual_root).strip() or self.agent_virtual_root
        rel, abs_workdir = self._resolve_agent_fs_path(raw_workdir)
        root_abs = os.path.abspath(self.agent_files_root)
        if raw_workdir.rstrip("/") == self.agent_virtual_root.rstrip("/"):
            abs_workdir = root_abs
            rel = ""
        if not abs_workdir or not os.path.isdir(abs_workdir):
            return "exec_agent_command error: invalid `workdir`"

        timeout_ms = self._coerce_int_arg(args.get("timeout_ms"), 5000)
        if timeout_ms < 100:
            timeout_ms = 100
        if timeout_ms > 60000:
            timeout_ms = 60000

        max_output_chars = self._coerce_int_arg(args.get("max_output_chars"), 4000)
        if max_output_chars < 200:
            max_output_chars = 200
        if max_output_chars > 12000:
            max_output_chars = 12000

        try:
            proc = subprocess.run(
                cmd,
                cwd=abs_workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                timeout=timeout_ms / 1000.0,
                check=False,
            )
            timed_out = False
        except subprocess.TimeoutExpired as e:
            proc = e
            timed_out = True
        except Exception as e:
            return f"exec_agent_command error: failed to run command: {e}"

        stdout_text = ""
        stderr_text = ""
        exit_code = -1
        if timed_out:
            stdout_text = bytes(proc.stdout or b"").decode("utf-8", errors="replace")
            stderr_text = bytes(proc.stderr or b"").decode("utf-8", errors="replace")
        else:
            stdout_text = bytes(proc.stdout or b"").decode("utf-8", errors="replace")
            stderr_text = bytes(proc.stderr or b"").decode("utf-8", errors="replace")
            exit_code = int(proc.returncode)

        stdout_view = stdout_text[:max_output_chars]
        stderr_view = stderr_text[:max_output_chars]
        truncated = len(stdout_view) < len(stdout_text) or len(stderr_view) < len(stderr_text)

        header = (
            f"[exec_agent_command] cwd={self._agent_display_path(rel)} "
            f"exit_code={exit_code} timeout={'yes' if timed_out else 'no'} "
            f"truncated={'yes' if truncated else 'no'}"
        )
        if stdout_view:
            header += "\n[stdout]\n" + stdout_view
        if stderr_view:
            header += "\n[stderr]\n" + stderr_view
        return header

    def get_embedding(self, text: str, mode: str = "query"):
        """将文本转换为向量"""
        if not self.llm_embed:
            raise RuntimeError("Embedding model not loaded")
        
        text = str(text).strip()
        # Qwen3-Embedding 训练时需要的前缀（query 用于检索，passage 用于存储）
        if not any(text.startswith(p) for p in ["query: ", "passage: "]):
            prefix = "query: " if mode == "query" else "passage: "
            text = prefix + text
        
        response = self.llm_embed.create_embedding([text])
        embedding = response['data'][0]['embedding']
        return self._normalize_embedding_vec(embedding)

    def get_embeddings(self, texts, mode: str = "query"):
        """批量文本转向量，返回 list[list[float]]"""
        if not self.llm_embed:
            raise RuntimeError("Embedding model not loaded")

        seq = []
        if texts is None:
            return seq

        try:
            n = len(texts)
        except Exception:
            n = None

        if n is not None:
            try:
                for i in range(1, n + 1):
                    seq.append(str(texts[i] or "").strip())
            except Exception:
                seq = []

            if not seq:
                try:
                    for i in range(0, n):
                        seq.append(str(texts[i] or "").strip())
                except Exception:
                    seq = []
        else:
            seq = [str(texts).strip()]

        prefixed = []
        for text in seq:
            if text == "":
                continue
            if not any(text.startswith(p) for p in ["query: ", "passage: "]):
                prefix = "query: " if mode == "query" else "passage: "
                text = prefix + text
            prefixed.append(text)

        if not prefixed:
            return []

        response = self.llm_embed.create_embedding(prefixed)
        data = response.get("data") or []
        out = []
        for item in data:
            emb = item.get("embedding") if isinstance(item, dict) else None
            if emb is None:
                continue
            out.append(self._normalize_embedding_vec(emb))
        return out

    def apply_chat_template(self, messages):
        if not self.llm_large:
            raise RuntimeError("Large model not loaded")
        messages_list = self._coerce_messages(messages)
        if not messages_list:
            raise ValueError("Invalid messages format")
        return self.llm_large.apply_chat_template(messages_list)

    def tokenize_text(self, text: str, add_special: bool = True, parse_special: bool = True):
        if not self.llm_large:
            raise RuntimeError("Large model not loaded")
        return self.llm_large.tokenize_text(
            text=text,
            add_special=add_special,
            parse_special=parse_special,
        )

    def count_chat_tokens(self, messages):
        prompt = self.apply_chat_template(messages)
        return self.tokenize_text(prompt, add_special=True, parse_special=True)

    def generate_chat_sync(self, messages, params):
        """同步版本：直接返回生成文本（供原子事实提取使用）"""
        if not self.llm_large:
            raise RuntimeError("Large model not loaded")

        messages_list = self._coerce_messages(messages)
        if not messages_list:
            raise ValueError("Invalid messages format")
        
        params_dict = dict(params) if hasattr(params, 'keys') else params
        max_tokens = int(params_dict.get("max_tokens", 128))
        temperature = float(params_dict.get("temperature", 0.7))
        stop = [str(s) for s in params_dict.get("stop", []) if s is not None]
        
        seed = params_dict.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
                if seed < 0:
                    seed = None
            except (TypeError, ValueError):
                seed = None
        
        output = self.llm_large.create_chat_completion(
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed
        )
        return self._extract_chat_text(output)

    def _normalize_tool_choice(self, tool_choice):
        raw = self._coerce_lua_value(tool_choice)
        if raw is None:
            return "auto"
        if isinstance(raw, str):
            choice = raw.strip()
            if choice in {"", "auto"}:
                return "auto"
            if choice in {"none", "required"}:
                return choice
            return {
                "type": "function",
                "function": {"name": choice},
            }
        if isinstance(raw, dict):
            return raw
        return "auto"

    def generate_chat_with_tools_sync(
        self,
        messages,
        params,
        tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    ):
        if not self.llm_large:
            raise RuntimeError("Large model not loaded")

        messages_list = self._coerce_messages(messages)
        if not messages_list:
            raise ValueError("Invalid messages format")

        params_dict = dict(params) if hasattr(params, "keys") else params
        max_tokens = int(params_dict.get("max_tokens", 1024))
        temperature = float(params_dict.get("temperature", 0.6))
        stop = [str(s) for s in params_dict.get("stop", []) if s is not None]

        seed = params_dict.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
                if seed < 0:
                    seed = None
            except (TypeError, ValueError):
                seed = None

        raw_tools = self._coerce_lua_value(tools)
        if isinstance(raw_tools, dict):
            if "type" in raw_tools or "function" in raw_tools:
                tools_list = [raw_tools]
            else:
                indexed = []
                for key, value in raw_tools.items():
                    try:
                        idx = int(key)
                    except (TypeError, ValueError):
                        idx = None
                    if idx is None:
                        continue
                    indexed.append((idx, value))
                indexed.sort(key=lambda x: x[0])
                tools_list = [v for _, v in indexed if isinstance(v, dict)]
        elif isinstance(raw_tools, list):
            tools_list = [x for x in raw_tools if isinstance(x, dict)]
        else:
            tools_list = []

        output = self.llm_large.create_chat_completion(
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed,
            tools=tools_list if tools_list else None,
            tool_choice=self._normalize_tool_choice(tool_choice) if tools_list else None,
            parallel_tool_calls=self._coerce_bool_arg(parallel_tool_calls, True),
        )

        text, tool_calls = self._extract_chat_text_and_tool_calls(output)
        normalized_calls = []
        for idx, call in enumerate(tool_calls or [], 1):
            if not isinstance(call, dict):
                continue
            fn = call.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name", "") or "").strip()
            if not name:
                continue
            arguments = fn.get("arguments", "{}")
            args = self._parse_tool_arguments_relaxed(arguments)
            if not isinstance(args, dict):
                args = {}
            normalized_calls.append(
                {
                    "id": str(call.get("id", "") or f"tool_call_{idx}"),
                    "type": "tool_call",
                    "name": name,
                    "args": args,
                }
            )

        return self._luaify({
            "content": str(text or ""),
            "tool_calls": normalized_calls,
        })

    def load_models(self, large_model_path: str, embedding_model_path: str, draft_model_path=None, spec_cfg=None):
        print("[Python] Loading models...")

        self.shutdown()
        if not os.path.isfile(self.llama_server_bin):
            raise FileNotFoundError(f"llama-server not found: {self.llama_server_bin}")

        # 1. 大模型（GPU）
        if large_model_path:
            if not os.path.exists(large_model_path):
                raise FileNotFoundError(f"Large model not found: {large_model_path}")

            spec_cfg_value = self._coerce_lua_value(spec_cfg)
            if not isinstance(spec_cfg_value, dict):
                spec_cfg_value = {}

            draft_model_path_arg = str(draft_model_path or "").strip()
            spec_enabled = None
            if "enabled" in spec_cfg_value:
                spec_enabled = bool(spec_cfg_value.get("enabled"))
            if spec_enabled is None:
                spec_enabled = bool(draft_model_path_arg) or bool(self.large_server_spec_enabled)

            draft_model_candidate = ""
            if spec_enabled:
                draft_model_candidate = draft_model_path_arg or self.large_server_draft_model

            if draft_model_candidate and not os.path.exists(draft_model_candidate):
                print(
                    f"[Python][WARN] Draft model not found for speculative decoding: {draft_model_candidate}; "
                    "continuing without draft model."
                )
                draft_model_candidate = ""

            print(
                f"[Python] Loading Large LLM: {large_model_path} "
                f"(gpu_layers={self.large_server_gpu_layers})"
            )
            large_common_kwargs = {
                "server_bin": self.llama_server_bin,
                "model_path": large_model_path,
                "ctx_size": 30720,
                "embedding": False,
                "host": self.large_server_host,
                "port": self.large_server_port,
                "enable_webui": self.large_server_webui,
                "enable_jinja": self.large_server_jinja,
                "api_key": self.large_server_api_key,
                "log_ready_url": not self.quiet_server_urls,
                "log_to_file": self.llama_server_log_to_file,
            }
            configured_layers = self.large_server_gpu_layers
            draft_cfg_gpu_layers = spec_cfg_value.get("draft_gpu_layers")
            draft_cfg_max = spec_cfg_value.get("draft_max")
            draft_cfg_min = spec_cfg_value.get("draft_min")
            draft_cfg_p_min = spec_cfg_value.get("draft_p_min")
            draft_cfg_ctx = spec_cfg_value.get("draft_ctx_size")

            draft_gpu_layers = draft_cfg_gpu_layers
            if draft_gpu_layers is None:
                draft_gpu_layers = self.large_server_draft_gpu_layers or "0"

            draft_ctx_size = 0
            try:
                draft_ctx_size = int(draft_cfg_ctx) if draft_cfg_ctx is not None else int(self.large_server_draft_ctx_size or 0)
            except (TypeError, ValueError):
                draft_ctx_size = int(self.large_server_draft_ctx_size or 0)
            if draft_ctx_size <= 0:
                draft_ctx_size = int(large_common_kwargs["ctx_size"])

            draft_max = None
            try:
                parsed_max = int(draft_cfg_max) if draft_cfg_max is not None else int(self.large_server_draft_max or 0)
            except (TypeError, ValueError):
                parsed_max = int(self.large_server_draft_max or 0)
            if parsed_max > 0:
                draft_max = parsed_max

            draft_min = None
            try:
                parsed_min = int(draft_cfg_min) if draft_cfg_min is not None else int(self.large_server_draft_min or 0)
            except (TypeError, ValueError):
                parsed_min = int(self.large_server_draft_min or 0)
            if parsed_min > 0:
                draft_min = parsed_min

            draft_p_min = None
            try:
                parsed_p_min = float(draft_cfg_p_min) if draft_cfg_p_min is not None else float(self.large_server_draft_p_min or 0.0)
            except (TypeError, ValueError):
                parsed_p_min = float(self.large_server_draft_p_min or 0.0)
            if parsed_p_min > 0:
                draft_p_min = parsed_p_min
            try:
                self.llm_large = LlamaCppServerClient(
                    gpu_layers=configured_layers,
                    draft_model_path=draft_model_candidate or None,
                    ctx_size_draft=draft_ctx_size if draft_model_candidate else None,
                    draft_max=draft_max if draft_model_candidate else None,
                    draft_min=draft_min if draft_model_candidate else None,
                    draft_p_min=draft_p_min if draft_model_candidate else None,
                    draft_gpu_layers=draft_gpu_layers,
                    **large_common_kwargs,
                )
            except RuntimeError as e:
                err_text = str(e).lower()
                can_fallback = (
                    self.large_server_gpu_fallback_cpu
                    and str(configured_layers or "").strip().lower() in {"all", "-1"}
                    and "cuda" in err_text
                    and ("out of memory" in err_text or "failed to allocate" in err_text)
                )
                if not can_fallback:
                    raise
                print(
                    "[Python][WARN] Large LLM GPU OOM detected with gpu_layers=all; "
                    "retrying with gpu_layers=0 (CPU fallback)."
                )
                self.llm_large = LlamaCppServerClient(
                    gpu_layers=0,
                    draft_model_path=draft_model_candidate or None,
                    ctx_size_draft=draft_ctx_size if draft_model_candidate else None,
                    draft_max=draft_max if draft_model_candidate else None,
                    draft_min=draft_min if draft_model_candidate else None,
                    draft_p_min=draft_p_min if draft_model_candidate else None,
                    draft_gpu_layers=draft_gpu_layers,
                    **large_common_kwargs,
                )
            if self.large_server_webui and not self.suppress_large_webui_log:
                webui_url = self.llm_large.get_webui_url()
                if webui_url:
                    print(f"[Python] Large model WebUI: {webui_url}")

        # 2. Embedding 模型（GGUF）
        if embedding_model_path:
            if not os.path.exists(embedding_model_path):
                raise FileNotFoundError(f"Embedding model not found: {embedding_model_path}")
            print(f"[Python] Loading GGUF Embedding model: {embedding_model_path}")
            self.llm_embed = LlamaCppServerClient(
                server_bin=self.llama_server_bin,
                model_path=embedding_model_path,
                ctx_size=2048,
                embedding=True,
                gpu_layers=self.embed_server_gpu_layers,
                log_ready_url=not self.quiet_server_urls,
                log_to_file=self.llama_server_log_to_file,
            )

        print("[Python] All models loaded (llama.cpp + GGUF).")

    def shutdown(self):
        if self.llm_large is not None:
            try:
                self.llm_large.stop()
            except Exception as e:
                print(f"[Python][WARN] Stop large llama-server failed: {e}")
            self.llm_large = None

        if self.llm_embed is not None:
            try:
                self.llm_embed.stop()
            except Exception as e:
                print(f"[Python][WARN] Stop embedding llama-server failed: {e}")
            self.llm_embed = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def unpack_state(self):
        zst_path = "memory/state.zst"
        v3_manifest = "memory/v3/manifest.txt"
        os.makedirs("memory/v3/graph/checkpoints", exist_ok=True)
        os.makedirs("memory/v3/graph/traces", exist_ok=True)

        if os.path.exists(v3_manifest):
            print("[Python] 检测到 V3 raw state，直接使用")
            return

        if not os.path.exists(zst_path):
            print("[Python] 未找到 state.zst，使用现有 raw 或全新启动")
            return

        print("[Python] 检测到归档状态，正在解压 state.zst...")
        self._do_decompress(zst_path)
        if not os.path.exists(v3_manifest):
            print("[Python][WARN] state.zst 解压后仍无 V3 manifest，memory 将以空 V3 状态启动")

    def _ensure_v3_state(self, zst_path):
        v3_manifest = "memory/v3/manifest.txt"
        if os.path.exists(v3_manifest):
            return True

        print("[Python] 检测到旧时代 state.zst（无 V3 manifest）→ 执行断代清理")
        self._cleanup_legacy_raw_files(remove_v3=True)
        if os.path.exists(zst_path):
            try:
                os.remove(zst_path)
                print("[Python] 已作废旧 state.zst")
            except Exception as e:
                print(f"[Python][WARN] 删除旧 state.zst 失败: {e}")
        return False

    def _do_decompress(self, zst_path):
        print("[Python] 正在解压 state.zst → raw files...")
        with open(zst_path, "rb") as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        tar_bytes = dctx.decompress(compressed)
        tar_io = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_io, mode="r") as tar:
            self._safe_extract_tar(tar, "memory/")
        print(f"[Python] 解压完成（{len(compressed)/1024/1024:.1f} MB → raw）")

    def _safe_extract_tar(self, tar, target_dir):
        base_dir = os.path.realpath(target_dir)
        os.makedirs(base_dir, exist_ok=True)

        for member in tar.getmembers():
            if member.issym() or member.islnk():
                raise RuntimeError(f"Unsafe archive member (link): {member.name}")

            dest_path = os.path.realpath(os.path.join(base_dir, member.name))
            if dest_path != base_dir and not dest_path.startswith(base_dir + os.sep):
                raise RuntimeError(f"Unsafe archive member path: {member.name}")

        try:
            tar.extractall(base_dir, filter="data")
        except TypeError:
            tar.extractall(base_dir)

    def pack_state(self):
        print("[Python] 正在原子打包 state.zst...")
        os.makedirs("memory", exist_ok=True)
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
            for name in [
                "memory.bin",
                "clusters.bin",
                "history.txt",
                "topic.bin",
                "pending_cold.txt",
                "adaptive_state.txt",
            ]:
                path = f"memory/{name}"
                if os.path.exists(path):
                    tar.add(path, arcname=name)

            v3_root = "memory/v3"
            if os.path.isdir(v3_root):
                for root, _, files in os.walk(v3_root):
                    for fname in files:
                        full_path = os.path.join(root, fname)
                        arcname = os.path.relpath(full_path, "memory")
                        tar.add(full_path, arcname=arcname)
        tar_bytes.seek(0)
        cctx = zstd.ZstdCompressor(level=5)
        compressed = cctx.compress(tar_bytes.read())
        temp = "memory/state.zst.tmp"
        with open(temp, "wb") as f:
            f.write(compressed)
        os.replace(temp, "memory/state.zst")
        print(f"[Python] 状态已打包完成（{len(compressed)/1024/1024:.1f} MB）")

    def _cleanup_legacy_raw_files(self, remove_v3=False):
        for name in [
            "memory.bin",
            "clusters.bin",
            "history.txt",
            "topic.bin",
            "pending_cold.txt",
            "adaptive_state.txt",
        ]:
            path = f"memory/{name}"
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"[Python][WARN] 删除 {name} 失败: {e}")

        if remove_v3:
            v3_root = "memory/v3"
            if os.path.isdir(v3_root):
                try:
                    shutil.rmtree(v3_root)
                except Exception as e:
                    print(f"[Python][WARN] 删除 v3 目录失败: {e}")

    def cleanup_raw_files(self):
        print("[Python] 执行最终归档清理：删除所有 raw 文件...")
        for name in [
            "memory.bin",
            "clusters.bin",
            "history.txt",
            "topic.bin",
            "pending_cold.txt",
            "adaptive_state.txt",
        ]:
            path = f"memory/{name}"
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"   已删除 {name}")
                except Exception as e:
                    print(f"   删除 {name} 失败: {e}")

        print("[Python] V3 storage 与 Graph 状态目录已保留")
        print("[Python] 归档完成！保留 state.zst、memory/v3 与 graph 状态")

    def get_similarity(self, text1: str, text2: str) -> float:
        emb1 = self.get_embedding(text1)
        emb2 = self.get_embedding(text2)
        vec1 = np.array(emb1)
        vec2 = np.array(emb2)
        return float(np.dot(vec1, vec2))

    def generate_chat(self, messages, params, lua_callback, lua_stream_callback=None):
        if not self.llm_large:
            raise RuntimeError("Large model not loaded")

        messages_list = self._coerce_messages(messages)
    
        if not messages_list:
            raise ValueError("Invalid messages format")
    
        params_dict = dict(params) if hasattr(params, 'keys') else params
        max_tokens = int(params_dict.get("max_tokens", 128))
        temperature = float(params_dict.get("temperature", 0.7))
        stop = [str(s) for s in params_dict.get("stop", []) if s is not None]
        
        seed = params_dict.get("seed")
        if seed is not None:
            try:
                seed = int(seed)
                if seed < 0:
                    seed = None
            except (TypeError, ValueError):
                seed = None
    
        print(f"[Python] Generating chat with large model... (stop: {stop})")
    
        stream = bool(params_dict.get("stream")) and lua_stream_callback is not None

        if stream:
            chunks = []
            for piece in self.llm_large.create_chat_completion_stream(
                messages=messages_list,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                stop=stop,
            ):
                text_piece = str(piece or "")
                if not text_piece:
                    continue
                chunks.append(text_piece)
                lua_stream_callback(text_piece)
            text_result = "".join(chunks)
        else:
            output = self.llm_large.create_chat_completion(
                messages=messages_list,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                stop=stop
            )
            text_result = self._extract_chat_text(output)

        if lua_callback:
            lua_callback(text_result)
        return text_result


class MoriLocalWebUIBridge:
    """Serve local no-build frontend assets and Mori native chat endpoints."""

    def __init__(
        self,
        host: str,
        port: int,
        frontend_root: str,
        chat_handler,
        model_name: str,
        session_key: str = "mori",
        agent_files_root: str = "workspace",
    ):
        self.host = str(host or "127.0.0.1")
        self.port = int(port)
        self.frontend_root = os.path.abspath(str(frontend_root or "module/frontend"))
        self.chat_handler = chat_handler
        self.model_name = str(model_name or "mori-chain")
        self.session_key = str(session_key or "mori").strip() or "mori"
        self.agent_files_root = os.path.abspath(str(agent_files_root or "workspace"))
        self.workspace_virtual_root = "/mori/workspace"
        self.upload_download_root = os.path.abspath(os.path.join(self.agent_files_root, "download"))
        self.session_state_json_path = os.path.abspath(os.path.join(os.getcwd(), "memory/v3/graph/session_state.json"))
        self.upload_max_files = max(1, _read_env_int("MORI_WEBUI_UPLOAD_MAX_FILES", 8))
        self.upload_max_file_bytes = max(1024, _read_env_int("MORI_WEBUI_UPLOAD_MAX_FILE_BYTES", 10 * 1024 * 1024))
        self.upload_max_total_bytes = max(
            self.upload_max_file_bytes,
            _read_env_int("MORI_WEBUI_UPLOAD_MAX_TOTAL_BYTES", 30 * 1024 * 1024),
        )
        self.max_body_bytes = max(
            self.upload_max_total_bytes,
            _read_env_int("MORI_WEBUI_MAX_BODY_BYTES", 64 * 1024 * 1024),
        )
        self._httpd = None
        self.last_session_status = self._load_session_status_snapshot()
        active_task = dict((self.last_session_status or {}).get("active_task") or {})
        self.last_response_meta = {
            "task_status": str(active_task.get("status") or ""),
            "resumed": False,
        }

        if not os.path.isdir(self.frontend_root):
            raise FileNotFoundError(f"Frontend root not found: {self.frontend_root}")
        index_path = os.path.join(self.frontend_root, "index.html")
        if not os.path.isfile(index_path):
            raise FileNotFoundError(f"Frontend index not found: {index_path}")
        os.makedirs(self.upload_download_root, exist_ok=True)

    @staticmethod
    def _display_path(path: str) -> str:
        abs_path = os.path.abspath(str(path or ""))
        rel = os.path.relpath(abs_path, os.getcwd()).replace("\\", "/")
        if rel == ".":
            return "."
        if not rel.startswith("../"):
            return f"./{rel}"
        return abs_path.replace("\\", "/")

    def _agent_display_path(self, rel_path: str = "") -> str:
        """Return a display path prefixed with the workspace virtual root."""
        base = self.workspace_virtual_root.rstrip("/")
        rel = str(rel_path or "").strip().replace("\\", "/")
        while rel.startswith("./"):
            rel = rel[2:]
        rel = rel.lstrip("/")
        if not rel:
            return base
        return f"{base}/{rel}"

    def _default_session_status_snapshot(self):
        return {
            "api_version": "graph_v2",
            "session_mode": "single",
            "last_run_id": "",
            "active_task": {
                "task_id": "",
                "goal": "",
                "status": "",
                "carryover_summary": "",
                "last_user_message": "",
                "profile": "",
            },
            "recovery": {
                "resumable_run_id": "",
                "last_checkpoint_seq": 0,
                "next_node": "",
                "resumed_from_checkpoint": False,
            },
            "stats": {
                "files_read_count": 0,
                "files_written_count": 0,
            },
        }

    def _load_session_status_snapshot(self):
        snapshot = self._default_session_status_snapshot()
        path = self.session_state_json_path
        if not os.path.isfile(path):
            return snapshot
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            return snapshot
        if not isinstance(raw, dict):
            return snapshot
        merged = dict(snapshot)
        merged.update(raw)
        for key in ("active_task", "recovery", "stats", "working_memory", "last_trace_summary"):
            value = raw.get(key)
            if isinstance(value, dict):
                base = dict(snapshot.get(key) or {})
                base.update(value)
                merged[key] = base
        return merged

    @staticmethod
    def _sanitize_upload_token(value: str, fallback: str = "x", limit: int = 48) -> str:
        raw = str(value or "").strip()
        safe = re.sub(r"[^0-9A-Za-z._-]+", "_", raw).strip("._-")
        if not safe:
            safe = str(fallback or "x")
        if len(safe) > int(limit):
            safe = safe[: int(limit)]
        return safe

    @classmethod
    def _sanitize_upload_filename(cls, filename: str) -> str:
        raw = str(filename or "").strip().replace("\\", "/")
        base = os.path.basename(raw) or "upload.bin"
        safe = cls._sanitize_upload_token(base, fallback="upload.bin", limit=120)
        if "." not in safe:
            safe = safe + ".bin"
        return safe

    def _store_uploaded_files(self, raw_files):
        files = list(raw_files or [])
        if len(files) > self.upload_max_files:
            return None, f"Too many files: {len(files)} > max {self.upload_max_files}"

        saved = []
        total_bytes = 0
        thread_tag = self._sanitize_upload_token(self.session_key, fallback=self.session_key, limit=24)

        for idx, item in enumerate(files, 1):
            if not isinstance(item, dict):
                return None, f"files[{idx}] must be an object"

            original_name = str(item.get("name") or f"upload_{idx}.bin")
            safe_name = self._sanitize_upload_filename(original_name)
            raw_bytes = item.get("content")
            if not isinstance(raw_bytes, (bytes, bytearray)):
                return None, f"files[{idx}] has invalid binary payload"
            file_bytes = bytes(raw_bytes)

            size = len(file_bytes)
            if size > self.upload_max_file_bytes:
                return None, f"files[{idx}] is too large: {size} bytes > max {self.upload_max_file_bytes}"
            total_bytes += size
            if total_bytes > self.upload_max_total_bytes:
                return None, (
                    f"Total uploaded bytes exceed limit: {total_bytes} > max {self.upload_max_total_bytes}"
                )

            stamp = time.strftime("%Y%m%d_%H%M%S")
            final_name = f"{thread_tag}_{stamp}_{uuid.uuid4().hex[:8]}_{safe_name}"
            abs_path = os.path.join(self.upload_download_root, final_name)
            with open(abs_path, "wb") as f:
                f.write(file_bytes)

            rel_to_agent_root = os.path.relpath(abs_path, self.agent_files_root).replace("\\", "/")
            saved.append(
                {
                    "name": safe_name,
                    "original_name": original_name,
                    "path": self._agent_display_path(rel_to_agent_root),
                    "tool_path": self._agent_display_path(rel_to_agent_root),
                    "bytes": size,
                }
            )
        return saved, None

    def _build_upload_manifest_text(self, saved_files):
        if not saved_files:
            return ""
        lines = [f"[上传文件已保存到 {self._agent_display_path('download')}]"]
        for item in saved_files:
            lines.append(
                f"- {item.get('original_name')}: {item.get('path')} "
                f"(tool_path={item.get('tool_path')}, bytes={int(item.get('bytes') or 0)})"
            )
        lines.append("如需读取附件内容，请调用 list_files(prefix='/mori/workspace/download') 并按需 read_file。")
        return "\n".join(lines)

    @staticmethod
    def _is_client_disconnect_error(exc: Exception) -> bool:
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(exc, OSError):
            return exc.errno in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}
        return False

    def _send_json(self, handler: BaseHTTPRequestHandler, status: int, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            handler.send_response(int(status))
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            if handler.command != "HEAD":
                handler.wfile.write(body)
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return False
            raise
        return True

    def _send_oai_error(self, handler: BaseHTTPRequestHandler, status: int, message: str):
        err_type = "server_error" if int(status) >= 500 else "invalid_request_error"
        payload = {
            "error": {
                "code": int(status),
                "message": str(message),
                "type": err_type,
            }
        }
        return self._send_json(handler, status, payload)

    def _send_bytes(
        self,
        handler: BaseHTTPRequestHandler,
        status: int,
        body: bytes,
        content_type: str,
    ):
        payload = body or b""
        try:
            handler.send_response(int(status))
            handler.send_header("Content-Type", str(content_type))
            handler.send_header("Content-Length", str(len(payload)))
            handler.end_headers()
            if payload and handler.command != "HEAD":
                handler.wfile.write(payload)
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return False
            raise
        return True

    def _read_body(self, handler: BaseHTTPRequestHandler):
        content_length = handler.headers.get("Content-Length")
        if not content_length:
            return b"", None
        try:
            n = int(content_length)
        except ValueError:
            return None, "Invalid Content-Length header."
        if n <= 0:
            return b"", None
        if n > self.max_body_bytes:
            return None, f"Request body too large: {n} bytes > max {self.max_body_bytes} bytes."
        body = handler.rfile.read(n)
        if len(body) > self.max_body_bytes:
            return None, f"Request body too large: {len(body)} bytes > max {self.max_body_bytes} bytes."
        return body, None

    def _read_multipart_payload(self, handler: BaseHTTPRequestHandler):
        body, body_err = self._read_body(handler)
        if body_err:
            return None, body_err
        if not body:
            return None, "Missing multipart form data."

        content_type = str(handler.headers.get("Content-Type", "") or "")
        if not content_type.lower().startswith("multipart/form-data"):
            return None, "Only multipart/form-data is supported."

        try:
            mime_bytes = (
                f"Content-Type: {content_type}\r\n"
                "MIME-Version: 1.0\r\n"
                "\r\n"
            ).encode("utf-8") + body
            parsed = BytesParser(policy=email_policy.default).parsebytes(mime_bytes)
        except Exception as e:
            return None, f"Invalid multipart payload: {e}"
        if not parsed.is_multipart():
            return None, "Invalid multipart form data."

        message = ""
        raw_files = []
        for part in parsed.iter_parts():
            name = part.get_param("name", header="Content-Disposition")
            if not name:
                continue
            field_name = str(name).strip()
            filename = part.get_filename()
            payload_bytes = part.get_payload(decode=True)
            if payload_bytes is None:
                payload_bytes = b""

            if field_name == "message" and not filename:
                charset = part.get_content_charset() or "utf-8"
                try:
                    message_piece = payload_bytes.decode(charset, errors="replace")
                except LookupError:
                    message_piece = payload_bytes.decode("utf-8", errors="replace")
                message = (message + message_piece).strip()
                continue

            if field_name in {"files[]", "files"} and filename:
                raw_files.append(
                    {
                        "name": str(filename or "upload.bin"),
                        "content": bytes(payload_bytes),
                    }
                )
        return {"message": message, "files": raw_files}, None

    def _extract_chat_input(self, payload):
        payload = payload or {}
        message = str(payload.get("message", "")).strip()
        saved_files, file_err = self._store_uploaded_files(payload.get("files"))
        if file_err:
            return None, None, file_err

        if message == "" and not saved_files:
            return None, None, "Either 'message' or 'files[]' must be provided."

        manifest = self._build_upload_manifest_text(saved_files)
        if manifest:
            if message:
                message = message + "\n\n" + manifest
            else:
                message = "我上传了附件，请先读取后再回答。\n\n" + manifest
        return message, saved_files, None

    def _send_sse_event(self, handler: BaseHTTPRequestHandler, event_name: str, payload) -> bool:
        event = str(event_name or "message").strip() or "message"
        try:
            try:
                data_json = json.dumps(payload or {}, ensure_ascii=False)
            except Exception:
                data_json = json.dumps({"message": str(payload)}, ensure_ascii=False)
            line = (
                f"event: {event}\n"
                f"data: {data_json}\n\n"
            ).encode("utf-8")
            handler.wfile.write(line)
            handler.wfile.flush()
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return False
            raise
        return True

    def _handle_chat_sync(self, handler: BaseHTTPRequestHandler):
        payload, err = self._read_multipart_payload(handler)
        if err:
            status = 413 if str(err).lower().startswith("request body too large") else 400
            self._send_oai_error(handler, status, err)
            return

        user_text, saved_files, input_err = self._extract_chat_input(payload)
        if input_err:
            self._send_oai_error(handler, 400, input_err)
            return

        try:
            result = self.chat_handler(user_text, None, None, False, saved_files or [])
            if isinstance(result, tuple):
                assistant_text = str(result[0] or "")
                run_id = str(result[1] or "")
                trace = result[2] if len(result) > 2 else {}
                meta = result[3] if len(result) > 3 else {}
            else:
                assistant_text = str(result or "")
                run_id = ""
                trace = {}
                meta = {}
        except Exception as e:
            self._send_oai_error(handler, 500, f"Chain execution failed: {e}")
            return

        self._send_json(
            handler,
            200,
            {
                "message": assistant_text,
                "run_id": run_id,
                "trace": trace or {},
                "task_status": str((meta or {}).get("task_status") or ""),
                "resumed": bool((meta or {}).get("resumed")),
                "uploads": [
                    {
                        "name": item.get("name"),
                        "path": item.get("path"),
                        "bytes": int(item.get("bytes") or 0),
                    }
                    for item in (saved_files or [])
                ],
            },
        )

    def _handle_chat_stream(self, handler: BaseHTTPRequestHandler):
        payload, err = self._read_multipart_payload(handler)
        if err:
            status = 413 if str(err).lower().startswith("request body too large") else 400
            self._send_oai_error(handler, status, err)
            return

        user_text, saved_files, input_err = self._extract_chat_input(payload)
        if input_err:
            self._send_oai_error(handler, 400, input_err)
            return

        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
            handler.send_header("Cache-Control", "no-cache, no-transform")
            handler.send_header("X-Accel-Buffering", "no")
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.wfile.write(b": stream-open\n\n")
            handler.wfile.flush()
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return
            raise

        streamed = []
        run_done = False

        if saved_files:
            self._send_sse_event(
                handler,
                "uploads",
                {
                    "files": [
                        {
                            "name": item.get("name"),
                            "path": item.get("path"),
                            "bytes": int(item.get("bytes") or 0),
                        }
                        for item in saved_files
                    ],
                },
            )

        def emit_piece(piece):
            nonlocal run_done
            try:
                coerced = AIPipeline._coerce_lua_value(piece)
            except Exception:
                coerced = piece

            if isinstance(coerced, (bytes, bytearray)):
                coerced = bytes(coerced).decode("utf-8", errors="replace")

            if isinstance(coerced, dict):
                event_type = str(coerced.get("event", "") or "").strip()
                if event_type:
                    payload_obj = coerced.get("data")
                    if not isinstance(payload_obj, dict):
                        payload_obj = {}
                    if not self._send_sse_event(handler, event_type, payload_obj):
                        raise BrokenPipeError("client disconnected during stream")
                    if event_type == "token":
                        token_piece = payload_obj.get("token")
                        if isinstance(token_piece, str) and token_piece:
                            streamed.append(token_piece)
                    if event_type == "done":
                        run_done = True
                    return

                text_like = coerced.get("token")
                if not isinstance(text_like, str) or not text_like:
                    text_like = coerced.get("text")
                if not isinstance(text_like, str) or not text_like:
                    text_like = coerced.get("content")
                if isinstance(text_like, str) and text_like:
                    if not self._send_sse_event(handler, "token", {"token": text_like}):
                        raise BrokenPipeError("client disconnected during stream")
                    streamed.append(text_like)
                    return

                try:
                    fallback_msg = json.dumps(coerced, ensure_ascii=False)
                except Exception:
                    fallback_msg = str(coerced)
                if not self._send_sse_event(handler, "status", {"message": fallback_msg}):
                    raise BrokenPipeError("client disconnected during stream")
                return

            text = str(coerced or "")
            if not text:
                return
            if not self._send_sse_event(handler, "token", {"token": text}):
                raise BrokenPipeError("client disconnected during stream")
            streamed.append(text)

        try:
            result = self.chat_handler(user_text, emit_piece, None, False, saved_files or [])
            if isinstance(result, tuple):
                assistant_text = str(result[0] or "")
                run_id = str(result[1] or "")
                trace = result[2] if len(result) > 2 else {}
                meta = result[3] if len(result) > 3 else {}
            else:
                assistant_text = str(result or "")
                run_id = ""
                trace = {}
                meta = {}
        except Exception as e:
            self._send_sse_event(handler, "error", {"message": f"Chain execution failed: {e}"})
            self._send_sse_event(handler, "done", {"message": "", "run_id": "", "trace": {}})
            handler.close_connection = True
            return

        joined = "".join(streamed)
        if assistant_text and not joined:
            emit_piece(assistant_text)
        elif assistant_text and assistant_text.startswith(joined):
            remain = assistant_text[len(joined):]
            if remain:
                emit_piece(remain)
        elif assistant_text and assistant_text != joined:
            print("[LocalWebUI][WARN] stream token text mismatch with final response.")

        if not run_done:
            self._send_sse_event(
                handler,
                "done",
                {
                    "message": assistant_text,
                    "run_id": run_id,
                    "trace": trace or {},
                    "task_status": str((meta or {}).get("task_status") or ""),
                    "resumed": bool((meta or {}).get("resumed")),
                },
            )
        handler.close_connection = True

    def _resolve_static_path(self, request_path: str):
        raw = str(request_path or "/")
        if raw in {"/", ""}:
            rel = "index.html"
        else:
            rel = urllib.parse.unquote(raw.lstrip("/"))
            if rel == "":
                rel = "index.html"

        rel = rel.replace("\\", "/")
        normalized = os.path.normpath(rel)
        if normalized in {"", "."}:
            normalized = "index.html"
        if normalized == ".." or normalized.startswith("../") or os.path.isabs(normalized):
            return None

        abs_path = os.path.abspath(os.path.join(self.frontend_root, normalized))
        root_prefix = self.frontend_root + os.sep
        if abs_path != self.frontend_root and not abs_path.startswith(root_prefix):
            return None
        return abs_path

    def _serve_static(self, handler: BaseHTTPRequestHandler, request_path: str):
        abs_path = self._resolve_static_path(request_path)
        if not abs_path:
            self._send_oai_error(handler, 403, "Forbidden static path.")
            return
        if not os.path.isfile(abs_path):
            self._send_oai_error(handler, 404, f"Static file not found: {request_path}")
            return

        try:
            with open(abs_path, "rb") as f:
                body = f.read()
        except Exception as e:
            self._send_oai_error(handler, 500, f"Failed to read static asset: {e}")
            return

        ctype, _encoding = mimetypes.guess_type(abs_path)
        if not ctype:
            ctype = "application/octet-stream"
        if ctype.startswith("text/") or ctype in {"application/javascript", "application/json"}:
            ctype = f"{ctype}; charset=utf-8"
        self._send_bytes(handler, 200, body, ctype)

    def _handle_session_status(self, handler: BaseHTTPRequestHandler):
        snapshot = self._load_session_status_snapshot()
        self.last_session_status = snapshot
        active_task = dict(snapshot.get("active_task") or {})
        recovery = dict(snapshot.get("recovery") or {})
        stats = dict(snapshot.get("stats") or {})
        self._send_json(
            handler,
            200,
            {
                "api_version": str(snapshot.get("api_version") or "graph_v2"),
                "session_mode": "single",
                "model_name": self.model_name,
                "workspace_root": self.workspace_virtual_root,
                "upload_dir": self.workspace_virtual_root + "/download",
                "active_task": active_task,
                "resumable_run": bool(recovery.get("resumable_run_id")),
                "last_run_id": str(snapshot.get("last_run_id") or ""),
                "files_read_count": int(stats.get("files_read_count") or 0),
                "files_written_count": int(stats.get("files_written_count") or 0),
                "upload_limits": {
                    "max_files": int(self.upload_max_files),
                    "max_file_bytes": int(self.upload_max_file_bytes),
                    "max_total_bytes": int(self.upload_max_total_bytes),
                    "max_body_bytes": int(self.max_body_bytes),
                },
            },
        )

    def _handle_deprecated_openai_chat(self, handler: BaseHTTPRequestHandler):
        payload = {
            "error": {
                "code": 410,
                "type": "deprecated_endpoint",
                "message": (
                    "Deprecated endpoint '/v1/chat/completions' in MORI_RUN_MODE=webui. "
                    "Use '/mori/chat' or '/mori/chat/stream'."
                ),
            }
        }
        self._send_json(handler, 410, payload)

    def _make_handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                print(f"[LocalWebUI] {self.address_string()} - {fmt % args}")

            def _dispatch(self):
                path = urllib.parse.urlsplit(self.path).path

                if self.command == "OPTIONS":
                    self.send_response(204)
                    self.send_header("Allow", "GET,POST,OPTIONS,HEAD")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                if self.command == "GET" and path == "/health":
                    bridge._send_json(self, 200, {"status": "ok"})
                    return

                if self.command == "GET" and path == "/mori/session/status":
                    bridge._handle_session_status(self)
                    return

                if self.command == "POST" and path == "/mori/chat":
                    bridge._handle_chat_sync(self)
                    return

                if self.command == "POST" and path == "/mori/chat/stream":
                    bridge._handle_chat_stream(self)
                    return

                if self.command == "POST" and path == "/v1/chat/completions":
                    bridge._handle_deprecated_openai_chat(self)
                    return

                if self.command in {"GET", "HEAD"}:
                    bridge._serve_static(self, path)
                    return

                bridge._send_oai_error(self, 404, f"Unknown path: {path}")

            def do_GET(self):
                self._dispatch()

            def do_HEAD(self):
                self._dispatch()

            def do_POST(self):
                self._dispatch()

            def do_OPTIONS(self):
                self._dispatch()

        return Handler

    def serve_forever(self):
        handler = self._make_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.port), handler)
        self._httpd.daemon_threads = True
        self._httpd.serve_forever()

    def shutdown(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None


def _read_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        val = int(raw)
    except ValueError:
        return int(default)
    return val if val > 0 else int(default)


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _py_to_lua_value(lua_runtime: LuaRuntime, value):
    if isinstance(value, dict):
        t = lua_runtime.table()
        for k, v in value.items():
            t[str(k)] = _py_to_lua_value(lua_runtime, v)
        return t
    if isinstance(value, (list, tuple)):
        t = lua_runtime.table()
        for i, v in enumerate(value, start=1):
            t[i] = _py_to_lua_value(lua_runtime, v)
        return t
    return value


def _resolve_run_mode() -> str:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--mode", choices=["cli", "webui"], help="Run mode override.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--cli", action="store_true", help="Force CLI mode.")
    mode_group.add_argument("--webui", action="store_true", help="Force WebUI mode.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"[Python][WARN] Ignored unknown args: {' '.join(unknown)}")

    cli_mode = None
    if args.mode:
        cli_mode = str(args.mode).strip().lower()
        print(f"[Python] Run mode forced by --mode: {cli_mode}")
    elif args.cli:
        cli_mode = "cli"
        print("[Python] Run mode forced by --cli: cli")
    elif args.webui:
        cli_mode = "webui"
        print("[Python] Run mode forced by --webui: webui")

    if cli_mode is None:
        env_mode = os.environ.get("MORI_RUN_MODE")
        if env_mode is None:
            cli_mode = "cli"
            print("[Python] Run mode default: cli")
        else:
            cli_mode = str(env_mode).strip().lower()
            print(f"[Python] Run mode from MORI_RUN_MODE: {cli_mode}")

    if cli_mode not in {"cli", "webui"}:
        print(f"[Python][WARN] Unknown MORI_RUN_MODE={cli_mode}, fallback to cli")
        cli_mode = "cli"
    return cli_mode


def main():
    # 1) Resolve run mode and initialize shared Python<->Lua bridge objects.
    run_mode = _resolve_run_mode()

    lua = LuaRuntime(unpack_returned_tuples=True)
    pipeline = AIPipeline()
    pipeline.lua_runtime = lua
    lua.globals()['py_pipeline'] = pipeline
    lua.globals()['MORI_RUN_MODE'] = run_mode

    bridge = None
    bridge_host = os.environ.get("MORI_WEBUI_BRIDGE_HOST", "127.0.0.1")
    bridge_port = _read_env_int("MORI_WEBUI_BRIDGE_PORT", 8080)
    frontend_root = str(os.environ.get("MORI_FRONTEND_ROOT", "module/frontend") or "module/frontend")

    # 2) In webui mode, force local no-build UI and avoid port collisions.
    if run_mode == "webui":
        pipeline.suppress_large_webui_log = True
        pipeline.quiet_server_urls = True
        pipeline.large_server_webui = False
        if not pipeline.large_server_api_key:
            pipeline.large_server_api_key = uuid.uuid4().hex + uuid.uuid4().hex

        deprecated_vars = [
            "MORI_WEBUI_PRIMARY_SESSION",
            "MORI_WEBUI_NON_PRIMARY_POLICY",
            "MORI_WEBUI_SESSION_HEADER",
            "MORI_WEBUI_THREAD_HEADER",
            "MORI_WEBUI_PRIMARY_IDLE_TTL_SEC",
            "MORI_WEBUI_SESSION_DEBUG",
        ]
        for name in deprecated_vars:
            if os.environ.get(name) is not None:
                print(f"[Python][WARN] {name} is deprecated in local webui mode and will be ignored.")

        if (
            pipeline.large_server_port is not None
            and int(pipeline.large_server_port) == bridge_port
            and str(pipeline.large_server_host) == str(bridge_host)
        ):
            print(
                "[Python][WARN] MORI_LARGE_SERVER_PORT 与 MORI_WEBUI_BRIDGE_PORT 冲突，"
                "主模型端口改为自动分配。"
            )
            pipeline.large_server_port = None

    # 3) Restore persisted state, execute Lua pipeline, then optionally start local WebUI bridge.
    pipeline.unpack_state()

    lua_shutdown = None
    should_save_state = False
    try:
        print("[Python] Executing pipeline.lua...")
        with open("pipeline.lua", "r", encoding="utf-8") as f:
            lua.execute(f.read())

        lua_shutdown = lua.globals().mori_shutdown

        if run_mode == "webui":
            lua_handler = lua.globals().mori_handle_user_input
            if lua_handler is None:
                raise RuntimeError("Lua function mori_handle_user_input is not available.")
            if pipeline.llm_large is None:
                raise RuntimeError("Large model server is not loaded.")

            bridge = None

            def bridge_chat(user_text, on_piece=None, _ignored=None, read_only=False, uploads=None):
                uploads_lua = _py_to_lua_value(lua, uploads or [])
                assistant_text = str(lua_handler(user_text, on_piece, "", read_only, uploads_lua) or "")
                run_id = str(lua.globals().mori_last_run_id or "")
                trace = AIPipeline._coerce_lua_value(lua.globals().mori_last_trace_summary)
                if not isinstance(trace, dict):
                    trace = {}
                meta = AIPipeline._coerce_lua_value(lua.globals().mori_last_response_meta)
                if not isinstance(meta, dict):
                    meta = {}
                session_status = AIPipeline._coerce_lua_value(lua.globals().mori_session_status)
                if isinstance(session_status, dict) and bridge is not None:
                    bridge.last_session_status = session_status
                if bridge is not None:
                    bridge.last_response_meta = meta
                return assistant_text, run_id, trace, meta

            bridge = MoriLocalWebUIBridge(
                host=bridge_host,
                port=bridge_port,
                frontend_root=frontend_root,
                chat_handler=bridge_chat,
                model_name=pipeline.llm_large.model_name,
                agent_files_root=pipeline.agent_files_root,
            )
            print(f"[Python] Local no-build WebUI ready at http://{bridge_host}:{bridge_port}")
            print(f"[Python] Frontend root: {os.path.abspath(frontend_root)}")
            print(f"[Python] Uploaded files dir: {bridge._display_path(bridge.upload_download_root)}")
            print("[Python] Stream mode: SSE event protocol enabled.")
            print("[Python] /v1/chat/completions is deprecated in webui mode; use /mori/chat or /mori/chat/stream.")
            print("[Python] Press Ctrl+C to stop.")

            should_save_state = True
            try:
                bridge.serve_forever()
            except KeyboardInterrupt:
                print("\n[Python] KeyboardInterrupt received, stopping local WebUI bridge...")
    finally:
        # 4) Graceful teardown order: bridge -> Lua state flush -> model servers.
        if bridge is not None:
            try:
                bridge.shutdown()
            except Exception as e:
                print(f"[Python][WARN] Stop WebUI bridge failed: {e}")

        if should_save_state and lua_shutdown is not None:
            try:
                lua_shutdown()
            except Exception as e:
                print(f"[Python][WARN] Lua state save failed: {e}")

        pipeline.shutdown()


if __name__ == "__main__":
    main()
