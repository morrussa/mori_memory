import lupa.luajit21 as lupa
from lupa.luajit21 import LuaRuntime
import numpy as np
import zstandard as zstd
import errno
import os
import io
import tarfile
import shutil
import atexit
import json
import gzip
import hashlib
import socket
import subprocess
import time
import uuid
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

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
        host: str = "127.0.0.1",
        port: int = None,
        enable_webui: bool = False,
        enable_jinja: bool = True,
        api_key: str = "",
        log_ready_url: bool = True,
        startup_timeout: int = 600,
    ):
        self.server_bin = server_bin
        self.model_path = model_path
        self.ctx_size = int(ctx_size)
        self.embedding = bool(embedding)
        self.server_host = str(host or "127.0.0.1")
        self.request_host = "127.0.0.1" if self.server_host in ("0.0.0.0", "::") else self.server_host
        self.enable_webui = bool(enable_webui) and not self.embedding
        self.enable_jinja = bool(enable_jinja) and not self.embedding
        self.api_key = str(api_key or "").strip()
        self.log_ready_url = bool(log_ready_url)
        self.startup_timeout = int(startup_timeout)
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
        os.makedirs("logs", exist_ok=True)
        role = "embedding" if self.embedding else "chat"
        self.log_path = os.path.join("logs", f"llama_server_{role}_{self.port}.log")
        self._log_file = open(self.log_path, "a", encoding="utf-8")

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
            "--gpu-layers",
            "all",
        ]

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

        self.process = subprocess.Popen(
            cmd,
            stdout=self._log_file,
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
                raise RuntimeError(
                    f"llama-server exited early (code={self.process.returncode}) for model: {self.model_path}\n"
                    f"log tail:\n{log_tail}"
                )

            try:
                status, _ = self._raw_http("GET", "/health", timeout=2)
                if status == 200:
                    return
            except Exception:
                pass

            time.sleep(0.5)

        raise TimeoutError(
            f"Timed out waiting llama-server ({self.model_path}) on {self.base_url}. "
            f"Check logs: {self.log_path}"
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
    def __init__(self):
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
    def _coerce_tools(tools):
        raw = AIPipeline._coerce_lua_value(tools)
        if raw is None:
            return []
        if not isinstance(raw, list):
            return []

        out = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "function" and isinstance(item.get("function"), dict):
                fn = item["function"]
                name = str(fn.get("name", "") or "").strip()
                if not name:
                    continue
                out.append(
                    {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": str(fn.get("description", "") or ""),
                            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                        },
                    }
                )
                continue

            name = str(item.get("name", "") or "").strip()
            if not name:
                continue
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": str(item.get("description", "") or ""),
                        "parameters": item.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
        return out

    @staticmethod
    def _coerce_tool_choice(tool_choice):
        if tool_choice is None:
            return None
        raw = AIPipeline._coerce_lua_value(tool_choice)
        if isinstance(raw, str):
            s = raw.strip().lower()
            if s in {"auto", "none", "required"}:
                return s
            if s:
                return {"type": "function", "function": {"name": s}}
            return "auto"
        if isinstance(raw, dict):
            return raw
        return "auto"

    @staticmethod
    def _lua_escape_str(value: str) -> str:
        s = str(value or "")
        s = s.replace("\\", "\\\\")
        s = s.replace("\"", "\\\"")
        s = s.replace("\r", "\\r")
        s = s.replace("\n", "\\n")
        return s

    @staticmethod
    def _dict_to_lua_table_row(payload: dict) -> str:
        ordered_keys = [
            "act",
            "tool_call_id",
            "arguments_json",
            "arguments",
            "string",
            "query",
            "type",
            "types",
            "entity",
            "evidence",
            "confidence",
            "namespace",
            "key",
            "value",
        ]
        parts = []
        for key in ordered_keys:
            if key not in payload:
                continue
            val = payload[key]
            if val is None:
                continue
            if isinstance(val, bool):
                lit = "true" if val else "false"
            elif isinstance(val, (int, float)):
                lit = str(val)
            else:
                lit = '"' + AIPipeline._lua_escape_str(str(val)) + '"'
            parts.append(f"{key}={lit}")
        return "{" + ", ".join(parts) + "}"

    @staticmethod
    def _tool_calls_to_lua_rows(tool_calls):
        if not isinstance(tool_calls, list):
            return ""
        rows = []
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            if not isinstance(fn, dict):
                continue
            name = str(fn.get("name", "") or "").strip()
            if not name:
                continue
            args_raw = fn.get("arguments", "")
            args_obj = AIPipeline._parse_tool_arguments_relaxed(args_raw)
            args_text = str(args_raw or "").strip()

            payload = {"act": name}
            payload["tool_call_id"] = str(tc.get("id", "") or "")
            if isinstance(args_obj, dict):
                payload["arguments_json"] = json.dumps(args_obj, ensure_ascii=False)
                for key in ("string", "query", "type", "types", "entity", "evidence", "confidence", "namespace", "key", "value"):
                    if key in args_obj:
                        payload[key] = args_obj.get(key)
                if "input" in args_obj and (payload.get("query") is None) and (payload.get("string") is None):
                    payload["string"] = args_obj.get("input")
            elif args_text:
                payload["arguments_json"] = args_text
            if (
                isinstance(args_text, str)
                and args_text
                and payload.get("query") is None
                and payload.get("string") is None
                and payload.get("value") is None
            ):
                payload["string"] = args_text
            rows.append(AIPipeline._dict_to_lua_table_row(payload))
        return "\n".join(rows)

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
    def _parse_tool_arguments_relaxed(args_raw):
        if isinstance(args_raw, dict):
            return args_raw
        if not isinstance(args_raw, str):
            return {}

        text = args_raw.strip()
        if not text:
            return {}

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
        if isinstance(args, str):
            args = args.strip()
            if args == "":
                args = "{}"
        elif isinstance(args, dict):
            pass
        else:
            args = str(args)

        try:
            result = tool.call(args)
        except Exception as e:
            raise RuntimeError(f"qwen tool `{name}` call failed: {e}") from e
        return self._normalize_qwen_tool_result(result)

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

    def generate_chat_with_tools_sync(
        self,
        messages,
        params,
        tools,
        tool_choice="auto",
        parallel_tool_calls=True,
    ):
        """同步工具调用版本：返回 (text, lua_tool_rows)"""
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

        tools_payload = self._coerce_tools(tools)
        tool_choice_payload = self._coerce_tool_choice(tool_choice)
        parallel_payload = bool(parallel_tool_calls)

        output = self.llm_large.create_chat_completion(
            messages=messages_list,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop,
            seed=seed,
            tools=tools_payload,
            tool_choice=tool_choice_payload,
            parallel_tool_calls=parallel_payload,
        )
        text_result, tool_calls = self._extract_chat_text_and_tool_calls(output)
        lua_rows = self._tool_calls_to_lua_rows(tool_calls)
        return text_result, lua_rows

    def load_models(self, large_model_path: str, embedding_model_path: str):
        print("[Python] Loading models...")

        self.shutdown()
        if not os.path.isfile(self.llama_server_bin):
            raise FileNotFoundError(f"llama-server not found: {self.llama_server_bin}")

        # 1. 大模型（GPU）
        if large_model_path:
            if not os.path.exists(large_model_path):
                raise FileNotFoundError(f"Large model not found: {large_model_path}")
            print(f"[Python] Loading Large LLM on GPU: {large_model_path}")
            self.llm_large = LlamaCppServerClient(
                server_bin=self.llama_server_bin,
                model_path=large_model_path,
                ctx_size=30720,
                embedding=False,
                host=self.large_server_host,
                port=self.large_server_port,
                enable_webui=self.large_server_webui,
                enable_jinja=self.large_server_jinja,
                api_key=self.large_server_api_key,
                log_ready_url=not self.quiet_server_urls,
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
                log_ready_url=not self.quiet_server_urls,
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
        if not os.path.exists(zst_path):
            print("[Python] 未找到 state.zst，使用现有 raw 或全新启动")
            return

        v3_manifest = "memory/v3/manifest.txt"
        raw_files = [
            v3_manifest,
            "memory/history.txt",
            "memory/topic.bin",
            "memory/pending_cold.txt",
            "memory/notebook.txt",
            "memory/adaptive_state.txt",
        ]

        if not all(os.path.exists(f) for f in raw_files):
            print("[Python] 检测到归档状态，正在解压 state.zst...")
            self._do_decompress(zst_path)
            self._ensure_v3_state(zst_path)
            return

        zst_mtime = os.path.getmtime(zst_path)
        raw_mtime = max((os.path.getmtime(f) for f in raw_files), default=0)
        if zst_mtime > raw_mtime + 3:
            print("[Python] zst 比 raw 更新 → 强制解压覆盖")
            self._do_decompress(zst_path)
            self._ensure_v3_state(zst_path)
        else:
            print("[Python] raw 文件已是最新的，直接使用")
            self._ensure_v3_state(zst_path)

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
                "history.txt",
                "topic.bin",
                "pending_cold.txt",
                "notebook.txt",
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
            "notebook.txt",
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
            "notebook.txt",
            "adaptive_state.txt",
        ]:
            path = f"memory/{name}"
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"   已删除 {name}")
                except Exception as e:
                    print(f"   删除 {name} 失败: {e}")

        v3_root = "memory/v3"
        if os.path.isdir(v3_root):
            try:
                shutil.rmtree(v3_root)
                print("   已删除 v3/")
            except Exception as e:
                print(f"   删除 v3/ 失败: {e}")
        print("[Python] 归档完成！仅保留 state.zst")

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


class MoriWebUIChainBridge:
    HISTORY_HEADER = "HIST_V2"
    HISTORY_FIELD_SEP = "\x1F"
    SESSION_ID_FIELDS = ("conversation_id", "chat_id", "session_id")
    THREAD_ID_FIELD = "thread_id"
    HOP_BY_HOP_HEADERS = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "trailers",
        "transfer-encoding",
        "upgrade",
    }

    def __init__(
        self,
        host: str,
        port: int,
        upstream_base_url: str,
        chat_handler,
        model_name: str,
        upstream_api_key: str = "",
        primary_session_key: str = "",
        non_primary_policy: str = "readonly_upstream",
        session_header_name: str = "X-Mori-Session",
        thread_header_name: str = "X-Mori-Thread",
        primary_idle_ttl_sec: int = 1800,
        session_debug: bool = False,
        history_file_path: str = "memory/history.txt",
    ):
        self.host = str(host or "127.0.0.1")
        self.port = int(port)
        self.upstream_base_url = str(upstream_base_url).rstrip("/")
        self.chat_handler = chat_handler
        self.model_name = str(model_name or "mori-chain")
        self.upstream_api_key = str(upstream_api_key or "").strip()
        self.primary_session_key = str(primary_session_key or "").strip()
        policy = str(non_primary_policy or "readonly_upstream").strip().lower()
        if policy in {"readonly_upstream", "readonly_chain"}:
            policy = "readonly_chain"
        self.non_primary_policy = policy if policy in {"readonly_chain", "reject"} else "readonly_chain"
        self.session_header_name = str(session_header_name or "X-Mori-Session").strip() or "X-Mori-Session"
        self.thread_header_name = str(thread_header_name or "X-Mori-Thread").strip() or "X-Mori-Thread"
        self.primary_idle_ttl_sec = max(1, int(primary_idle_ttl_sec or 1800))
        self.session_debug = bool(session_debug)
        self.primary_conversation_name = "Mori"
        self.history_file_path = str(history_file_path or "memory/history.txt")
        self.webui_input_chunk_chars = max(256, _read_env_int("MORI_WEBUI_INPUT_CHUNK_CHARS", 4096))
        self.webui_user_text_max_chars = max(2048, _read_env_int("MORI_WEBUI_USER_TEXT_MAX_CHARS", 120000))
        self.webui_fingerprint_text_max_chars = max(
            256, _read_env_int("MORI_WEBUI_FINGERPRINT_TEXT_MAX_CHARS", 4096)
        )
        self._auto_primary_session = None
        self._primary_session_alias = None
        self._primary_last_seen_ts = 0.0
        self._fallback_warned = set()
        self._thread_fallback_warned = set()
        self._httpd = None

    @staticmethod
    def _is_client_disconnect_error(exc: Exception) -> bool:
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)):
            return True
        if isinstance(exc, OSError):
            return exc.errno in {errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED}
        return False

    @staticmethod
    def _iter_text_chunks(text: str, chunk_chars: int):
        value = str(text or "")
        if not value:
            return
        step = max(1, int(chunk_chars or 4096))
        for i in range(0, len(value), step):
            yield value[i : i + step]

    @classmethod
    def _iter_content_text_chunks(cls, content, chunk_chars: int = 4096):
        if isinstance(content, str):
            for chunk in cls._iter_text_chunks(content, chunk_chars):
                yield chunk
            return
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                text = ""
                if item.get("type") == "text":
                    text = str(item.get("text", ""))
                elif "text" in item and item.get("text") is not None:
                    text = str(item.get("text"))
                if not text:
                    continue
                for chunk in cls._iter_text_chunks(text, chunk_chars):
                    yield chunk
            return
        if content is None:
            return
        for chunk in cls._iter_text_chunks(str(content), chunk_chars):
            yield chunk

    @classmethod
    def _content_to_text_limited(cls, content, chunk_chars: int = 4096, max_chars=None):
        limit = None
        if max_chars is not None:
            limit = max(0, int(max_chars))
        out = io.StringIO()
        written = 0
        truncated = False
        for piece in cls._iter_content_text_chunks(content, chunk_chars=chunk_chars):
            if limit is not None and written >= limit:
                truncated = True
                break
            if limit is None:
                out.write(piece)
                written += len(piece)
                continue
            remaining = limit - written
            if len(piece) <= remaining:
                out.write(piece)
                written += len(piece)
                continue
            if remaining > 0:
                out.write(piece[:remaining])
                written += remaining
            truncated = True
            break
        return out.getvalue(), truncated

    @classmethod
    def _content_to_text(cls, content, chunk_chars: int = 4096, max_chars=None) -> str:
        text, _truncated = cls._content_to_text_limited(content, chunk_chars=chunk_chars, max_chars=max_chars)
        return text

    @classmethod
    def _extract_last_user_text(cls, messages, chunk_chars: int = 4096, max_chars=None) -> str:
        if not isinstance(messages, list):
            return ""
        for msg in reversed(messages):
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "user":
                continue
            text, truncated = cls._content_to_text_limited(
                msg.get("content"),
                chunk_chars=chunk_chars,
                max_chars=max_chars,
            )
            stripped = text.strip()
            if not stripped:
                continue
            if truncated:
                stripped = (
                    stripped
                    + "\n\n[Input truncated for this turn after chunked read to protect context limits. "
                    + "If needed, send the next chunk.]"
                )
            return stripped
        return ""

    @classmethod
    def _extract_first_role_text(cls, messages, role: str, chunk_chars: int = 4096, max_chars=None) -> str:
        if not isinstance(messages, list):
            return ""
        role_l = str(role or "").lower()
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != role_l:
                continue
            text = cls._content_to_text(
                msg.get("content"),
                chunk_chars=chunk_chars,
                max_chars=max_chars,
            )
            if text.strip():
                return text.strip()
        return ""

    def _debug_session_log(self, text: str):
        if self.session_debug:
            print(f"[WebUIBridge][Session] {text}")

    def _resolve_session_key(self, payload, handler: BaseHTTPRequestHandler):
        if isinstance(payload, dict):
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                for field in self.SESSION_ID_FIELDS:
                    raw = metadata.get(field)
                    if raw is None:
                        continue
                    value = str(raw).strip()
                    if value:
                        return value, f"metadata.{field}"

            for field in self.SESSION_ID_FIELDS:
                raw = payload.get(field)
                if raw is None:
                    continue
                value = str(raw).strip()
                if value:
                    return value, f"payload.{field}"

        header_value = handler.headers.get(self.session_header_name)
        if header_value and str(header_value).strip():
            return str(header_value).strip(), f"header.{self.session_header_name}"

        fixed = str(self.primary_session_key or "").strip()
        if fixed:
            return fixed, "fallback.primary_session"

        messages = payload.get("messages") if isinstance(payload, dict) else None
        first_system = self._extract_first_role_text(
            messages,
            "system",
            chunk_chars=self.webui_input_chunk_chars,
            max_chars=self.webui_fingerprint_text_max_chars,
        )
        first_user = self._extract_first_role_text(
            messages,
            "user",
            chunk_chars=self.webui_input_chunk_chars,
            max_chars=self.webui_fingerprint_text_max_chars,
        )
        user_agent = str(handler.headers.get("User-Agent", "") or "")
        client_ip = ""
        try:
            client_ip = str(handler.client_address[0] or "")
        except Exception:
            client_ip = ""
        key_src = "\x1f".join([first_system, first_user, client_ip, user_agent])
        digest = hashlib.sha1(key_src.encode("utf-8", errors="ignore")).hexdigest()
        session_key = f"fb-{digest}"
        if session_key not in self._fallback_warned:
            self._fallback_warned.add(session_key)
            print(f"[WebUIBridge][WARN] Session key missing, fallback fingerprint used: {session_key}")
        return session_key, "fallback.sha1"

    def _resolve_thread_key(self, payload, handler: BaseHTTPRequestHandler, session_key: str):
        if isinstance(payload, dict):
            metadata = payload.get("metadata")
            if isinstance(metadata, dict):
                raw = metadata.get(self.THREAD_ID_FIELD)
                if raw is not None:
                    value = str(raw).strip()
                    if value:
                        return value, f"metadata.{self.THREAD_ID_FIELD}"

            raw = payload.get(self.THREAD_ID_FIELD)
            if raw is not None:
                value = str(raw).strip()
                if value:
                    return value, f"payload.{self.THREAD_ID_FIELD}"

        header_value = handler.headers.get(self.thread_header_name)
        if header_value and str(header_value).strip():
            return str(header_value).strip(), f"header.{self.thread_header_name}"

        fallback_key = str(session_key or "").strip()
        if not fallback_key:
            fallback_key = "default"
        warn_key = f"{fallback_key}"
        if warn_key not in self._thread_fallback_warned:
            self._thread_fallback_warned.add(warn_key)
            print(f"[WebUIBridge][WARN] Thread key missing, fallback to session key: {fallback_key}")
        return fallback_key, "fallback.session_key"

    def _resolve_session_role(self, session_key: str):
        now = float(time.time())
        fixed = str(self.primary_session_key or "").strip()
        if fixed:
            if session_key == fixed:
                self._primary_session_alias = None
                role = "primary"
            elif self._primary_session_alias and session_key == self._primary_session_alias:
                role = "primary"
            elif self._primary_last_seen_ts <= 0 and not self._primary_session_alias:
                self._primary_session_alias = session_key
                role = "primary"
                print(
                    "[WebUIBridge][Session] Primary session bootstrap: "
                    f"{session_key} -> {fixed}"
                )
            else:
                role = "observer"
            if role == "primary":
                self._primary_last_seen_ts = now
            mode = "fixed_bootstrap" if self._primary_session_alias else "fixed"
            return role, mode, fixed, self._primary_last_seen_ts

        if (
            (not self._auto_primary_session)
            or self._primary_last_seen_ts <= 0
            or (now - self._primary_last_seen_ts) > float(self.primary_idle_ttl_sec)
        ):
            self._auto_primary_session = session_key
            self._primary_last_seen_ts = now

        role = "primary" if session_key == self._auto_primary_session else "observer"
        if role == "primary":
            self._primary_last_seen_ts = now
        return role, "auto", self._auto_primary_session, self._primary_last_seen_ts

    @staticmethod
    def _header_safe(value) -> str:
        return str(value or "").replace("\r", " ").replace("\n", " ").strip()

    def _build_session_response_headers(
        self,
        session_key: str,
        session_role: str,
        session_source: str,
        thread_key: str,
        thread_source: str,
    ):
        return [
            ("X-Mori-Session-Key", self._header_safe(session_key)),
            ("X-Mori-Session-Role", self._header_safe(session_role)),
            ("X-Mori-Session-Source", self._header_safe(session_source)),
            ("X-Mori-Thread-Key", self._header_safe(thread_key)),
            ("X-Mori-Thread-Source", self._header_safe(thread_source)),
        ]

    def _session_status_payload(self):
        mode = "fixed" if self.primary_session_key else "auto"
        primary = self.primary_session_key if mode == "fixed" else self._auto_primary_session
        last_seen = int(self._primary_last_seen_ts) if self._primary_last_seen_ts > 0 else None
        return {
            "primary_session": primary,
            "policy": self.non_primary_policy,
            "mode": mode,
            "ttl_sec": int(self.primary_idle_ttl_sec),
            "last_seen_ts": last_seen,
            "session_header": self.session_header_name,
            "thread_header": self.thread_header_name,
            "primary_alias": self._primary_session_alias,
        }

    @classmethod
    def _history_unescape_field(cls, value: str) -> str:
        s = str(value or "")
        out = []
        i = 0
        n = len(s)
        while i < n:
            ch = s[i]
            if ch != "\\":
                out.append(ch)
                i += 1
                continue

            if i + 1 >= n:
                out.append("\\")
                i += 1
                continue

            n1 = s[i + 1]
            if n1 == "\\":
                out.append("\\")
                i += 2
            elif n1 == "n":
                out.append("\n")
                i += 2
            elif n1 == "x" and i + 3 < n:
                hx = s[i + 2 : i + 4]
                if hx == "1F":
                    out.append(cls.HISTORY_FIELD_SEP)
                    i += 4
                elif hx == "1E":
                    out.append("\x1E")
                    i += 4
                else:
                    out.append("\\")
                    i += 1
            else:
                out.append("\\")
                i += 1
        return "".join(out)

    @classmethod
    def _parse_history_line(cls, line: str):
        raw = str(line or "")
        if cls.HISTORY_FIELD_SEP not in raw:
            return cls._history_unescape_field(raw), ""
        user_part, assistant_part = raw.split(cls.HISTORY_FIELD_SEP, 1)
        return cls._history_unescape_field(user_part), cls._history_unescape_field(assistant_part)

    def _load_history_pairs(self):
        path = self.history_file_path
        if not path or not os.path.exists(path):
            return []
        out = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                header = f.readline().rstrip("\r\n")
                if header != self.HISTORY_HEADER:
                    return []
                for line in f:
                    raw = line.rstrip("\r\n")
                    if raw == "":
                        continue
                    user_text, assistant_text = self._parse_history_line(raw)
                    out.append({"user": user_text, "assistant": assistant_text})
        except Exception as e:
            print(f"[WebUIBridge][WARN] Failed to read history for bootstrap: {e}")
            return []
        return out

    def _build_webui_bootstrap_payload(self):
        primary = str(self.primary_session_key or "").strip() or "mori"
        return {
            "primary_session": primary,
            "conversation_name": self.primary_conversation_name,
            "history": self._load_history_pairs(),
        }

    def _build_webui_bootstrap_script(self) -> str:
        payload_json = json.dumps(self._build_webui_bootstrap_payload(), ensure_ascii=False).replace("</", "<\\/")
        session_header_json = json.dumps(self.session_header_name, ensure_ascii=False).replace("</", "<\\/")
        thread_header_json = json.dumps(self.thread_header_name, ensure_ascii=False).replace("</", "<\\/")
        return (
            "<script data-mori-webui-bootstrap=\"1\">\n"
            "(function () {\n"
            "  if (window.__moriWebuiBootstrapInstalled) return;\n"
            "  window.__moriWebuiBootstrapInstalled = true;\n"
            "  const BOOT = "
            + payload_json
            + ";\n"
            "  const DB_NAME = 'LlamacppWebui';\n"
            "  const DB_VERSION = 1;\n"
            "  const PRIMARY_ID = String(BOOT.primary_session || 'mori').trim() || 'mori';\n"
            "  const PRIMARY_NAME = String(BOOT.conversation_name || 'Mori').trim() || 'Mori';\n"
            "  const HISTORY = Array.isArray(BOOT.history) ? BOOT.history : [];\n"
            "  const SESSION_HEADER = "
            + session_header_json
            + ";\n"
            "  const THREAD_HEADER = "
            + thread_header_json
            + ";\n"
            "  const RELOAD_KEY = '__mori_bootstrap_reload_once__';\n"
            "  const makeId = function () {\n"
            "    if (globalThis.crypto && typeof globalThis.crypto.randomUUID === 'function') {\n"
            "      return globalThis.crypto.randomUUID();\n"
            "    }\n"
            "    return 'mori-' + Math.random().toString(16).slice(2) + Date.now().toString(16);\n"
            "  };\n"
            "  const reqToPromise = function (req) {\n"
            "    return new Promise(function (resolve, reject) {\n"
            "      req.onsuccess = function () { resolve(req.result); };\n"
            "      req.onerror = function () { reject(req.error || new Error('IndexedDB request failed')); };\n"
            "    });\n"
            "  };\n"
            "  const txDone = function (tx) {\n"
            "    return new Promise(function (resolve, reject) {\n"
            "      tx.oncomplete = function () { resolve(); };\n"
            "      tx.onerror = function () { reject(tx.error || new Error('IndexedDB transaction failed')); };\n"
            "      tx.onabort = function () { reject(tx.error || new Error('IndexedDB transaction aborted')); };\n"
            "    });\n"
            "  };\n"
            "  const openDb = function () {\n"
            "    return new Promise(function (resolve, reject) {\n"
            "      const req = indexedDB.open(DB_NAME, DB_VERSION);\n"
            "      req.onupgradeneeded = function () {\n"
            "        const db = req.result;\n"
            "        const tx = req.transaction;\n"
            "        let convStore = null;\n"
            "        let msgStore = null;\n"
            "        if (!db.objectStoreNames.contains('conversations')) {\n"
            "          convStore = db.createObjectStore('conversations', { keyPath: 'id' });\n"
            "        } else if (tx) {\n"
            "          convStore = tx.objectStore('conversations');\n"
            "        }\n"
            "        if (convStore) {\n"
            "          if (!convStore.indexNames.contains('lastModified')) convStore.createIndex('lastModified', 'lastModified');\n"
            "          if (!convStore.indexNames.contains('currNode')) convStore.createIndex('currNode', 'currNode');\n"
            "          if (!convStore.indexNames.contains('name')) convStore.createIndex('name', 'name');\n"
            "        }\n"
            "        if (!db.objectStoreNames.contains('messages')) {\n"
            "          msgStore = db.createObjectStore('messages', { keyPath: 'id' });\n"
            "        } else if (tx) {\n"
            "          msgStore = tx.objectStore('messages');\n"
            "        }\n"
            "        if (msgStore) {\n"
            "          if (!msgStore.indexNames.contains('convId')) msgStore.createIndex('convId', 'convId');\n"
            "          if (!msgStore.indexNames.contains('type')) msgStore.createIndex('type', 'type');\n"
            "          if (!msgStore.indexNames.contains('role')) msgStore.createIndex('role', 'role');\n"
            "          if (!msgStore.indexNames.contains('timestamp')) msgStore.createIndex('timestamp', 'timestamp');\n"
            "          if (!msgStore.indexNames.contains('parent')) msgStore.createIndex('parent', 'parent');\n"
            "          if (!msgStore.indexNames.contains('children')) msgStore.createIndex('children', 'children');\n"
            "        }\n"
            "      };\n"
            "      req.onsuccess = function () { resolve(req.result); };\n"
            "      req.onerror = function () { reject(req.error || new Error('Open IndexedDB failed')); };\n"
            "    });\n"
            "  };\n"
            "  const hasRequiredStores = function (db) {\n"
            "    if (!db || !db.objectStoreNames) return false;\n"
            "    return db.objectStoreNames.contains('conversations') && db.objectStoreNames.contains('messages');\n"
            "  };\n"
            "  const deleteDb = function () {\n"
            "    return new Promise(function (resolve, reject) {\n"
            "      const req = indexedDB.deleteDatabase(DB_NAME);\n"
            "      req.onsuccess = function () { resolve(); };\n"
            "      req.onerror = function () { reject(req.error || new Error('Delete IndexedDB failed')); };\n"
            "      req.onblocked = function () { reject(new Error('Delete IndexedDB blocked by another tab')); };\n"
            "    });\n"
            "  };\n"
            "  const openDbWithSchema = async function () {\n"
            "    let db = await openDb();\n"
            "    if (hasRequiredStores(db)) return db;\n"
            "    try { db.close(); } catch (_e) {}\n"
            "    await deleteDb();\n"
            "    db = await openDb();\n"
            "    if (!hasRequiredStores(db)) {\n"
            "      try { db.close(); } catch (_e) {}\n"
            "      throw new Error('IndexedDB schema is invalid');\n"
            "    }\n"
            "    return db;\n"
            "  };\n"
            "  const sleep = function (ms) {\n"
            "    return new Promise(function (resolve) { setTimeout(resolve, ms); });\n"
            "  };\n"
            "  const waitForStores = async function (maxAttempts, intervalMs) {\n"
            "    let lastError = null;\n"
            "    for (let i = 0; i < maxAttempts; i += 1) {\n"
            "      try {\n"
            "        return await openDbWithSchema();\n"
            "      } catch (err) {\n"
            "        lastError = err;\n"
            "      }\n"
            "      await sleep(intervalMs);\n"
            "    }\n"
            "    if (lastError) throw lastError;\n"
            "    throw new Error('IndexedDB stores are not ready');\n"
            "  };\n"
            "  const extractActiveChatId = function () {\n"
            "    const hash = String(window.location.hash || '');\n"
            "    const m = hash.match(/^#\\/chat\\/([^/?#]+)/);\n"
            "    if (!m || !m[1]) return '';\n"
            "    try { return decodeURIComponent(m[1]); } catch (_e) { return m[1]; }\n"
            "  };\n"
            "  const patchFetch = function () {\n"
            "    if (window.__moriFetchPatched) return;\n"
            "    const originalFetch = (typeof window.fetch === 'function') ? window.fetch.bind(window) : null;\n"
            "    if (!originalFetch) return;\n"
            "    window.__moriFetchPatched = true;\n"
            "    window.fetch = function (input, init) {\n"
            "      try {\n"
            "        const rawUrl = (typeof input === 'string') ? input : ((input && input.url) ? input.url : '');\n"
            "        const url = new URL(rawUrl, window.location.href);\n"
            "        if (url.pathname.endsWith('/v1/chat/completions')) {\n"
            "          const activeId = extractActiveChatId() || PRIMARY_ID;\n"
            "          const nextInit = Object.assign({}, init || {});\n"
            "          const baseHeaders = (nextInit && nextInit.headers) || ((input instanceof Request) ? input.headers : undefined);\n"
            "          const headers = new Headers(baseHeaders || {});\n"
            "          headers.set(SESSION_HEADER, activeId);\n"
            "          headers.set(THREAD_HEADER, activeId);\n"
            "          nextInit.headers = headers;\n"
            "          if (typeof nextInit.body === 'string') {\n"
            "            try {\n"
            "              const payload = JSON.parse(nextInit.body);\n"
            "              if (payload && typeof payload === 'object') {\n"
            "                if (!payload.metadata || typeof payload.metadata !== 'object' || Array.isArray(payload.metadata)) {\n"
            "                  payload.metadata = {};\n"
            "                }\n"
            "                if (!payload.metadata.conversation_id) payload.metadata.conversation_id = activeId;\n"
            "                if (!payload.metadata.thread_id) payload.metadata.thread_id = activeId;\n"
            "                nextInit.body = JSON.stringify(payload);\n"
            "              }\n"
            "            } catch (_e) {}\n"
            "          }\n"
            "          return originalFetch(input, nextInit);\n"
            "        }\n"
            "      } catch (_e) {}\n"
            "      return originalFetch(input, init);\n"
            "    };\n"
            "  };\n"
            "  const normalizeHistory = function () {\n"
            "    const turns = [];\n"
            "    for (let i = 0; i < HISTORY.length; i += 1) {\n"
            "      const item = HISTORY[i] || {};\n"
            "      const user = typeof item.user === 'string' ? item.user : '';\n"
            "      const assistant = typeof item.assistant === 'string' ? item.assistant : '';\n"
            "      if (!user && !assistant) continue;\n"
            "      turns.push({ user: user, assistant: assistant, userId: makeId(), assistantId: makeId() });\n"
            "    }\n"
            "    return turns;\n"
            "  };\n"
            "  const createConversationFromHistory = async function () {\n"
            "    if (typeof indexedDB === 'undefined') return false;\n"
            "    const db = await waitForStores(30, 120);\n"
            "    try {\n"
            "      const tx = db.transaction(['conversations', 'messages'], 'readwrite');\n"
            "      const done = txDone(tx);\n"
            "      const convStore = tx.objectStore('conversations');\n"
            "      const msgStore = tx.objectStore('messages');\n"
            "      const byId = await reqToPromise(convStore.get(PRIMARY_ID));\n"
            "      if (byId) {\n"
            "        await done;\n"
            "        return false;\n"
            "      }\n"
            "      const allConversations = await reqToPromise(convStore.getAll());\n"
            "      if (Array.isArray(allConversations)) {\n"
            "        for (let i = 0; i < allConversations.length; i += 1) {\n"
            "          const conv = allConversations[i];\n"
            "          if (conv && String(conv.name || '').trim() === PRIMARY_NAME) {\n"
            "            await done;\n"
            "            return false;\n"
            "          }\n"
            "        }\n"
            "      }\n"
            "      let ts = Date.now();\n"
            "      if (!Number.isFinite(ts)) ts = 0;\n"
            "      const turns = normalizeHistory();\n"
            "      const conversation = { id: PRIMARY_ID, name: PRIMARY_NAME, lastModified: ts, currNode: '' };\n"
            "      await reqToPromise(convStore.add(conversation));\n"
            "      if (turns.length > 0) {\n"
            "        const rootId = makeId();\n"
            "        const firstUserId = turns[0].userId;\n"
            "        await reqToPromise(msgStore.add({\n"
            "          id: rootId,\n"
            "          convId: PRIMARY_ID,\n"
            "          type: 'root',\n"
            "          timestamp: ts,\n"
            "          role: 'system',\n"
            "          content: '',\n"
            "          parent: null,\n"
            "          toolCalls: '',\n"
            "          children: [firstUserId]\n"
            "        }));\n"
            "        for (let i = 0; i < turns.length; i += 1) {\n"
            "          const turn = turns[i];\n"
            "          const prevAssistantId = i > 0 ? turns[i - 1].assistantId : rootId;\n"
            "          const nextUserId = i + 1 < turns.length ? turns[i + 1].userId : null;\n"
            "          ts += 1;\n"
            "          await reqToPromise(msgStore.add({\n"
            "            id: turn.userId,\n"
            "            convId: PRIMARY_ID,\n"
            "            type: 'text',\n"
            "            timestamp: ts,\n"
            "            role: 'user',\n"
            "            content: String(turn.user || ''),\n"
            "            parent: prevAssistantId,\n"
            "            toolCalls: '',\n"
            "            children: [turn.assistantId]\n"
            "          }));\n"
            "          ts += 1;\n"
            "          await reqToPromise(msgStore.add({\n"
            "            id: turn.assistantId,\n"
            "            convId: PRIMARY_ID,\n"
            "            type: 'text',\n"
            "            timestamp: ts,\n"
            "            role: 'assistant',\n"
            "            content: String(turn.assistant || ''),\n"
            "            parent: turn.userId,\n"
            "            toolCalls: '',\n"
            "            children: nextUserId ? [nextUserId] : []\n"
            "          }));\n"
            "        }\n"
            "        conversation.currNode = turns[turns.length - 1].assistantId;\n"
            "        conversation.lastModified = ts;\n"
            "        await reqToPromise(convStore.put(conversation));\n"
            "      }\n"
            "      await done;\n"
            "      return true;\n"
            "    } finally {\n"
            "      db.close();\n"
            "    }\n"
            "  };\n"
            "  patchFetch();\n"
            "  createConversationFromHistory()\n"
            "    .then(function (created) {\n"
            "      if (!created) return;\n"
            "      if (!window.sessionStorage) return;\n"
            "      if (sessionStorage.getItem(RELOAD_KEY) === '1') return;\n"
            "      sessionStorage.setItem(RELOAD_KEY, '1');\n"
            "      window.location.reload();\n"
            "    })\n"
            "    .catch(function (err) {\n"
            "      console.warn('[MoriWebUI] bootstrap failed:', err);\n"
            "    });\n"
            "})();\n"
            "</script>"
        )

    def _inject_webui_bootstrap_html(self, body: bytes) -> bytes:
        if not body:
            return body
        html = body.decode("utf-8", errors="replace")
        if "__moriWebuiBootstrapInstalled" in html:
            return body
        script = self._build_webui_bootstrap_script()
        lower = html.lower()
        head_pos = lower.rfind("</head>")
        if head_pos >= 0:
            html = html[:head_pos] + script + "\n" + html[head_pos:]
        else:
            body_pos = lower.rfind("</body>")
            if body_pos >= 0:
                html = html[:body_pos] + script + "\n" + html[body_pos:]
            else:
                html = html + "\n" + script
        return html.encode("utf-8")

    def _inject_webui_bootstrap_body(self, body: bytes, content_encoding: str) -> bytes:
        enc = str(content_encoding or "").lower()
        if "gzip" not in enc:
            return self._inject_webui_bootstrap_html(body)

        try:
            decoded = gzip.decompress(body)
        except Exception as e:
            print(f"[WebUIBridge][WARN] Failed to decode gzip HTML for bootstrap injection: {e}")
            return body

        injected = self._inject_webui_bootstrap_html(decoded)
        try:
            return gzip.compress(injected)
        except Exception as e:
            print(f"[WebUIBridge][WARN] Failed to encode gzip HTML after bootstrap injection: {e}")
            return body

    def _send_json(self, handler: BaseHTTPRequestHandler, status: int, payload, extra_headers=None):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            handler.send_response(int(status))
            handler.send_header("Content-Type", "application/json; charset=utf-8")
            if extra_headers:
                for key, value in extra_headers:
                    handler.send_header(str(key), str(value))
            handler.send_header("Content-Length", str(len(body)))
            handler.end_headers()
            handler.wfile.write(body)
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return False
            raise
        return True

    def _send_oai_error(self, handler: BaseHTTPRequestHandler, status: int, message: str, extra_headers=None):
        err_type = "server_error" if int(status) >= 500 else "invalid_request_error"
        payload = {
            "error": {
                "code": int(status),
                "message": str(message),
                "type": err_type,
            }
        }
        return self._send_json(handler, status, payload, extra_headers=extra_headers)

    def _relay_response(self, handler: BaseHTTPRequestHandler, status: int, headers, body: bytes, extra_headers=None):
        try:
            handler.send_response(int(status))
            for key, value in headers:
                low = str(key).lower()
                if low in self.HOP_BY_HOP_HEADERS:
                    continue
                if low in {"content-length", "server", "date"}:
                    continue
                handler.send_header(key, value)
            if extra_headers:
                for key, value in extra_headers:
                    handler.send_header(str(key), str(value))
            handler.send_header("Content-Length", str(len(body or b"")))
            handler.end_headers()
            if body:
                handler.wfile.write(body)
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return False
            raise
        return True

    def _relay_stream_response(self, handler: BaseHTTPRequestHandler, status: int, headers, stream, extra_headers=None):
        try:
            handler.send_response(int(status))
            for key, value in headers:
                low = str(key).lower()
                if low in self.HOP_BY_HOP_HEADERS:
                    continue
                if low in {"content-length", "server", "date"}:
                    continue
                handler.send_header(key, value)
            if extra_headers:
                for key, value in extra_headers:
                    handler.send_header(str(key), str(value))
            handler.end_headers()

            while True:
                chunk = stream.read(4096)
                if not chunk:
                    break
                handler.wfile.write(chunk)
                handler.wfile.flush()
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return False
            raise
        return True

    def _proxy_request(
        self,
        handler: BaseHTTPRequestHandler,
        body: bytes,
        extra_response_headers=None,
        stream_response: bool = False,
        inject_webui_bootstrap: bool = False,
    ):
        url = f"{self.upstream_base_url}{handler.path}"
        data = body if handler.command in {"POST", "PUT", "PATCH", "DELETE"} else None
        req = urllib.request.Request(url, data=data, method=handler.command)

        for key, value in handler.headers.items():
            low = str(key).lower()
            if low in self.HOP_BY_HOP_HEADERS:
                continue
            if low in {"host", "content-length"}:
                continue
            req.add_header(key, value)
        if self.upstream_api_key:
            req.add_header("Authorization", f"Bearer {self.upstream_api_key}")

        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                content_type = str(resp.headers.get("Content-Type", "")).lower()
                should_stream = bool(stream_response) or content_type.startswith("text/event-stream")
                if should_stream:
                    self._relay_stream_response(
                        handler,
                        int(resp.status),
                        resp.getheaders(),
                        resp,
                        extra_headers=extra_response_headers,
                    )
                else:
                    resp_body = resp.read()
                    if inject_webui_bootstrap and "text/html" in content_type:
                        content_encoding = str(resp.headers.get("Content-Encoding", "")).lower()
                        resp_body = self._inject_webui_bootstrap_body(resp_body, content_encoding)
                    self._relay_response(
                        handler,
                        int(resp.status),
                        resp.getheaders(),
                        resp_body,
                        extra_headers=extra_response_headers,
                    )
                return
        except urllib.error.HTTPError as e:
            err_body = e.read()
            hdrs = e.headers.items() if e.headers else []
            self._relay_response(
                handler,
                int(e.code),
                hdrs,
                err_body,
                extra_headers=extra_response_headers,
            )
            return
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return
            self._send_oai_error(
                handler,
                502,
                f"Proxy upstream failed: {e}",
                extra_headers=extra_response_headers,
            )

    def _write_sse_data(self, handler: BaseHTTPRequestHandler, payload) -> bool:
        try:
            if payload is None:
                line = b"data: [DONE]\n\n"
            else:
                line = f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
            handler.wfile.write(line)
            handler.wfile.flush()
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return False
            raise
        return True

    @staticmethod
    def _normalize_webui_text(text: str) -> str:
        out = str(text or "")
        # WebUI markdown 解析较激进：将常见语法符号做实体转义，避免误渲染
        out = out.replace("&", "&amp;")
        out = out.replace("`", "&#96;")
        out = out.replace("*", "&#42;")
        out = out.replace("_", "&#95;")
        out = out.replace("[", "&#91;").replace("]", "&#93;")
        out = out.replace("(", "&#40;").replace(")", "&#41;")
        return out

    def _send_stream_completion(self, handler: BaseHTTPRequestHandler, model: str, chat_payload, extra_headers=None):
        created = int(time.time())
        req_id = f"chatcmpl-{uuid.uuid4().hex}"
        last_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
        }

        try:
            handler.send_response(200)
            handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
            handler.send_header("Cache-Control", "no-cache")
            handler.send_header("Connection", "close")
            if extra_headers:
                for key, value in extra_headers:
                    handler.send_header(str(key), str(value))
            handler.end_headers()
        except Exception as e:
            if self._is_client_disconnect_error(e):
                return
            raise

        # 先发 role chunk，兼容依赖 OpenAI 风格首块的客户端
        role_chunk = {
            "id": req_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }
        if not self._write_sse_data(handler, role_chunk):
            handler.close_connection = True
            return

        streamed = []
        carry = ""

        def emit_piece(piece: str):
            nonlocal carry
            text = carry + str(piece or "")
            if not text:
                return
            if text.endswith("`"):
                carry = "`"
                text = text[:-1]
            else:
                carry = ""
            text = self._normalize_webui_text(text)
            if not text:
                return
            chunk = {
                "id": req_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": text},
                        "finish_reason": None,
                    }
                ],
            }
            if not self._write_sse_data(handler, chunk):
                raise BrokenPipeError("client disconnected during stream")
            streamed.append(text)

        try:
            assistant_text = str(chat_payload(emit_piece) or "")
        except Exception as e:
            if not self._is_client_disconnect_error(e):
                print(f"[WebUIBridge][WARN] Stream chain failed: {e}")
            handler.close_connection = True
            return

        if carry:
            tail = self._normalize_webui_text(carry)
            carry = ""
            if tail:
                try:
                    emit_piece(tail)
                except Exception as e:
                    if not self._is_client_disconnect_error(e):
                        raise
                    handler.close_connection = True
                    return

        assistant_text = self._normalize_webui_text(assistant_text)
        joined = "".join(streamed)
        if assistant_text and assistant_text.startswith(joined):
            remain = assistant_text[len(joined):]
            if remain:
                try:
                    emit_piece(remain)
                except Exception as e:
                    if not self._is_client_disconnect_error(e):
                        raise
                    handler.close_connection = True
                    return
        elif assistant_text and not joined:
            try:
                emit_piece(assistant_text)
            except Exception as e:
                if not self._is_client_disconnect_error(e):
                    raise
                handler.close_connection = True
                return
        elif assistant_text and assistant_text != joined:
            print("[WebUIBridge][WARN] Stream text mismatch with final post-processed result.")

        if not self._write_sse_data(handler, last_chunk):
            handler.close_connection = True
            return
        self._write_sse_data(handler, None)
        handler.close_connection = True

    def _handle_chat_completion(self, handler: BaseHTTPRequestHandler, body: bytes):
        try:
            payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError as e:
            self._send_oai_error(handler, 400, f"Invalid JSON payload: {e}")
            return

        session_key, session_source = self._resolve_session_key(payload, handler)
        thread_key, thread_source = self._resolve_thread_key(payload, handler, session_key)
        session_role, session_mode, primary_session, _ = self._resolve_session_role(session_key)
        session_headers = self._build_session_response_headers(
            session_key=session_key,
            session_role=session_role,
            session_source=session_source,
            thread_key=thread_key,
            thread_source=thread_source,
        )
        self._debug_session_log(
            f"mode={session_mode} role={session_role} source={session_source} "
            f"session={session_key} thread={thread_key} primary={primary_session}"
        )

        if session_role != "primary":
            if self.non_primary_policy == "reject":
                self._send_oai_error(
                    handler,
                    409,
                    "This session is observer-only. Switch to the primary session to use Mori chain.",
                    extra_headers=session_headers,
                )
                return

        model = str(payload.get("model") or self.model_name)
        stream = bool(payload.get("stream"))
        read_only = session_role != "primary"
        user_text = self._extract_last_user_text(
            payload.get("messages"),
            chunk_chars=self.webui_input_chunk_chars,
            max_chars=self.webui_user_text_max_chars,
        )
        if not user_text:
            self._send_oai_error(
                handler,
                400,
                "No user message found in request messages.",
                extra_headers=session_headers,
            )
            return

        if stream:
            self._send_stream_completion(
                handler,
                model,
                lambda on_piece: self.chat_handler(user_text, on_piece, thread_key, read_only),
                extra_headers=session_headers,
            )
            return

        try:
            assistant_text = self._normalize_webui_text(
                str(self.chat_handler(user_text, None, thread_key, read_only) or "")
            )
        except Exception as e:
            self._send_oai_error(
                handler,
                500,
                f"Chain execution failed: {e}",
                extra_headers=session_headers,
            )
            return

        created = int(time.time())
        response = {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": assistant_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        self._send_json(handler, 200, response, extra_headers=session_headers)

    def _handle_session_status(self, handler: BaseHTTPRequestHandler):
        payload = self._session_status_payload()
        self._send_json(handler, 200, payload)

    def _make_handler(self):
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):
                print(f"[WebUIBridge] {self.address_string()} - {fmt % args}")

            def _read_body(self) -> bytes:
                content_length = self.headers.get("Content-Length")
                if not content_length:
                    return b""
                try:
                    n = int(content_length)
                except ValueError:
                    n = 0
                if n <= 0:
                    return b""
                return self.rfile.read(n)

            def _dispatch(self):
                path = self.path.split("?", 1)[0]
                if self.command == "OPTIONS":
                    self.send_response(204)
                    self.send_header("Allow", "GET,POST,PUT,PATCH,DELETE,OPTIONS")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return

                if self.command == "GET" and path == "/health":
                    bridge._send_json(self, 200, {"status": "ok"})
                    return

                if self.command == "GET" and path == "/mori/session/status":
                    bridge._handle_session_status(self)
                    return

                body = self._read_body()
                if self.command == "POST" and path == "/v1/chat/completions":
                    bridge._handle_chat_completion(self, body)
                    return

                inject_bootstrap = self.command == "GET" and path in {"/", "/index.html"}
                bridge._proxy_request(self, body, inject_webui_bootstrap=inject_bootstrap)

            def do_GET(self):
                self._dispatch()

            def do_POST(self):
                self._dispatch()

            def do_PUT(self):
                self._dispatch()

            def do_PATCH(self):
                self._dispatch()

            def do_DELETE(self):
                self._dispatch()

            def do_OPTIONS(self):
                self._dispatch()

        return Handler

    def serve_forever(self):
        handler = self._make_handler()
        self._httpd = HTTPServer((self.host, self.port), handler)
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


def main():
    run_mode = str(os.environ.get("MORI_RUN_MODE", "cli")).strip().lower()
    if run_mode not in {"cli", "webui"}:
        print(f"[Python][WARN] Unknown MORI_RUN_MODE={run_mode}, fallback to cli")
        run_mode = "cli"

    lua = LuaRuntime(unpack_returned_tuples=True)
    pipeline = AIPipeline()
    lua.globals()['py_pipeline'] = pipeline
    lua.globals()['MORI_RUN_MODE'] = run_mode

    bridge = None
    bridge_host = os.environ.get("MORI_WEBUI_BRIDGE_HOST", "127.0.0.1")
    bridge_port = _read_env_int("MORI_WEBUI_BRIDGE_PORT", 8080)
    bridge_primary_session = str(os.environ.get("MORI_WEBUI_PRIMARY_SESSION", "mori") or "mori").strip() or "mori"
    bridge_non_primary_policy = str(
        os.environ.get("MORI_WEBUI_NON_PRIMARY_POLICY", "readonly_upstream") or "readonly_upstream"
    ).strip().lower()
    bridge_session_header = str(os.environ.get("MORI_WEBUI_SESSION_HEADER", "X-Mori-Session") or "").strip()
    bridge_thread_header = str(os.environ.get("MORI_WEBUI_THREAD_HEADER", "X-Mori-Thread") or "").strip()
    bridge_primary_idle_ttl = _read_env_int("MORI_WEBUI_PRIMARY_IDLE_TTL_SEC", 1800)
    bridge_session_debug = _read_env_bool("MORI_WEBUI_SESSION_DEBUG", False)

    if run_mode == "webui":
        pipeline.suppress_large_webui_log = True
        pipeline.quiet_server_urls = True
        pipeline.large_server_webui = True
        if not pipeline.large_server_api_key:
            pipeline.large_server_api_key = uuid.uuid4().hex + uuid.uuid4().hex
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

            upstream = pipeline.llm_large.base_url
            bridge = MoriWebUIChainBridge(
                host=bridge_host,
                port=bridge_port,
                upstream_base_url=upstream,
                chat_handler=lambda user_text, on_piece=None, thread_id=None, read_only=False: lua_handler(
                    user_text, on_piece, thread_id, read_only
                ),
                model_name=pipeline.llm_large.model_name,
                upstream_api_key=pipeline.large_server_api_key,
                primary_session_key=bridge_primary_session,
                non_primary_policy=bridge_non_primary_policy,
                session_header_name=bridge_session_header,
                thread_header_name=bridge_thread_header,
                primary_idle_ttl_sec=bridge_primary_idle_ttl,
                session_debug=bridge_session_debug,
            )
            print(f"[Python] WebUI chain bridge ready at http://{bridge_host}:{bridge_port}")
            print("[Python] Upstream llama-server is internal-only in webui mode.")
            print("[Python] Press Ctrl+C to stop.")

            should_save_state = True
            try:
                bridge.serve_forever()
            except KeyboardInterrupt:
                print("\n[Python] KeyboardInterrupt received, stopping WebUI bridge...")
    finally:
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
