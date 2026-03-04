import lupa.luajit21 as lupa
from lupa.luajit21 import LuaRuntime
import numpy as np
import zstandard as zstd
import os
import io
import tarfile
import shutil
import atexit
import json
import socket
import subprocess
import time
import urllib.error
import urllib.request


class LlamaCppServerClient:
    def __init__(
        self,
        server_bin: str,
        model_path: str,
        ctx_size: int,
        embedding: bool = False,
        startup_timeout: int = 600,
    ):
        self.server_bin = server_bin
        self.model_path = model_path
        self.ctx_size = int(ctx_size)
        self.embedding = bool(embedding)
        self.startup_timeout = int(startup_timeout)
        self.model_name = os.path.basename(model_path) or "local-model"
        self.port = self._find_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.process = None
        self.log_path = None
        self._log_file = None
        self._start_server()

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])

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
            "127.0.0.1",
            "--port",
            str(self.port),
            "--ctx-size",
            str(self.ctx_size),
            "--gpu-layers",
            "all",
            "--no-webui",
            "--reasoning-format",
            "none",
        ]
        if self.embedding:
            cmd.append("--embeddings")

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
        print(f"[Python] llama-server ready ({role}) at {self.base_url}")

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
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
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

    def create_chat_completion(self, messages, max_tokens=128, temperature=0.7, stop=None, seed=None):
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
        return self._request_json(
            "POST",
            "/v1/chat/completions",
            payload=payload,
            timeout=3600,
        )

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
        self.llama_cpp_root = os.environ.get("LLAMA_CPP_ROOT", "/home/morusa/AI/llama-cpp")
        self.llama_server_bin = os.environ.get(
            "LLAMA_SERVER_BIN",
            os.path.join(self.llama_cpp_root, "build", "bin", "llama-server"),
        )
        atexit.register(self.shutdown)

    @staticmethod
    def _normalize_embedding_vec(embedding):
        emb_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tolist()

    @staticmethod
    def _extract_chat_text(output: dict) -> str:
        choices = output.get("choices") if isinstance(output, dict) else None
        if not choices:
            raise RuntimeError(f"Invalid chat completion response: {output}")

        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    texts.append(str(item.get("text", "")))
            return "".join(texts)
        return str(content)

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

    def generate_chat_sync(self, messages, params):
        """同步版本：直接返回生成文本（供原子事实提取使用）"""
        if not self.llm_large:
            raise RuntimeError("Large model not loaded")
        
        messages_list = []
        try:
            length = len(messages)
            if length > 0:
                for i in range(1, length + 1):
                    msg = messages[i]
                    if msg and 'role' in msg and 'content' in msg:
                        messages_list.append({
                            'role': str(msg['role']),
                            'content': str(msg['content'])
                        })
        except TypeError:
            if 'role' in messages and 'content' in messages:
                messages_list.append({
                    'role': str(messages['role']),
                    'content': str(messages['content'])
                })
        
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
                ctx_size=40960,
                embedding=False,
            )

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

    def generate_chat(self, messages, params, lua_callback):
        if not self.llm_large:
            raise RuntimeError("Large model not loaded")
    
        messages_list = []
        try:
            length = len(messages)
            if length > 0:
                for i in range(1, length + 1):
                    msg = messages[i]
                    if msg and 'role' in msg and 'content' in msg:
                        messages_list.append({
                            'role': str(msg['role']),
                            'content': str(msg['content'])
                        })
        except TypeError:
            if 'role' in messages and 'content' in messages:
                messages_list.append({
                    'role': str(messages['role']),
                    'content': str(messages['content'])
                })
    
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


def main():
    lua = LuaRuntime(unpack_returned_tuples=True)
    pipeline = AIPipeline()
    lua.globals()['py_pipeline'] = pipeline

    pipeline.unpack_state()

    try:
        print("[Python] Executing pipeline.lua...")
        with open("pipeline.lua", "r", encoding="utf-8") as f:
            lua.execute(f.read())
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
