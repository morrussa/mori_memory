import lupa.luajit21 as lupa
from lupa.luajit21 import LuaRuntime
from llama_cpp import Llama
import numpy as np
import zstandard as zstd
import os
import io
import tarfile


class AIPipeline:
    def __init__(self):
        self.llm_large = None   # GPU 大模型（生成用）
        self.llm_embed = None   # GGUF Embedding 模型

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
        
        emb_array = np.array(embedding, dtype=np.float32)
        norm = np.linalg.norm(emb_array)
        if norm > 0:
            emb_array = emb_array / norm
        return emb_array.tolist()

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
        return output['choices'][0]['message']['content']

    def load_models(self, large_model_path: str, embedding_model_path: str):
        print("[Python] Loading models...")
        
        # 1. 大模型（GPU）
        if large_model_path:
            print(f"[Python] Loading Large LLM on GPU: {large_model_path}")
            self.llm_large = Llama(
                model_path=large_model_path,
                n_gpu_layers=-1,
                n_ctx=4096,
                verbose=False
            )

        # 2. Embedding 模型（GGUF）
        if embedding_model_path:
            print(f"[Python] Loading GGUF Embedding model: {embedding_model_path}")
            self.llm_embed = Llama(
                model_path=embedding_model_path,
                embedding=True,
                logits_all=True,
                n_gpu_layers=0,
                n_ctx=8192,
                verbose=False
            )

        print("[Python] All models loaded (llama.cpp + GGUF).")

    def unpack_state(self):
        zst_path = "memory/state.zst"
        if not os.path.exists(zst_path):
            print("[Python] 未找到 state.zst，使用现有 raw 或全新启动")
            return

        raw_files = [
            "memory/memory.bin",
            "memory/clusters.bin",
            "memory/history.txt",
            "memory/topic.bin",
            "memory/pending_cold.txt"
        ]

        if not all(os.path.exists(f) for f in raw_files):
            print("[Python] 检测到归档状态，正在解压 state.zst...")
            self._do_decompress(zst_path)
            return

        zst_mtime = os.path.getmtime(zst_path)
        raw_mtime = max((os.path.getmtime(f) for f in raw_files), default=0)
        if zst_mtime > raw_mtime + 3:
            print("[Python] zst 比 raw 更新 → 强制解压覆盖")
            self._do_decompress(zst_path)
        else:
            print("[Python] raw 文件已是最新的，直接使用")

    def _do_decompress(self, zst_path):
        print("[Python] 正在解压 state.zst → raw files...")
        with open(zst_path, "rb") as f:
            compressed = f.read()
        dctx = zstd.ZstdDecompressor()
        tar_bytes = dctx.decompress(compressed)
        tar_io = io.BytesIO(tar_bytes)
        with tarfile.open(fileobj=tar_io, mode="r") as tar:
            tar.extractall("memory/")
        print(f"[Python] 解压完成（{len(compressed)/1024/1024:.1f} MB → raw）")

    def pack_state(self):
        print("[Python] 正在原子打包 state.zst...")
        tar_bytes = io.BytesIO()
        with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
            for name in [
                "memory.bin",
                "clusters.bin",
                "history.txt",
                "topic.bin",
                "pending_cold.txt"
            ]:
                path = f"memory/{name}"
                if os.path.exists(path):
                    tar.add(path, arcname=name)
        tar_bytes.seek(0)
        cctx = zstd.ZstdCompressor(level=5)
        compressed = cctx.compress(tar_bytes.read())
        temp = "memory/state.zst.tmp"
        with open(temp, "wb") as f:
            f.write(compressed)
        os.replace(temp, "memory/state.zst")
        print(f"[Python] 状态已打包完成（{len(compressed)/1024/1024:.1f} MB）")

    def cleanup_raw_files(self):
        print("[Python] 执行最终归档清理：删除所有 raw 文件...")
        for name in [
            "memory.bin",
            "clusters.bin",
            "history.txt",
            "topic.bin",
            "pending_cold.txt"
        ]:
            path = f"memory/{name}"
            if os.path.exists(path):
                try:
                    os.remove(path)
                    print(f"   已删除 {name}")
                except Exception as e:
                    print(f"   删除 {name} 失败: {e}")
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
    
        text_result = output['choices'][0]['message']['content']
        if lua_callback:
            lua_callback(text_result)
        return text_result


def main():
    lua = LuaRuntime(unpack_returned_tuples=True)
    pipeline = AIPipeline()
    lua.globals()['py_pipeline'] = pipeline

    pipeline.unpack_state()
    
    print("[Python] Executing pipeline.lua...")
    with open("pipeline.lua", "r", encoding="utf-8") as f:
        lua.execute(f.read())


if __name__ == "__main__":
    main()