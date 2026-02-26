#!/usr/bin/env python3
"""
将脚本所在目录下所有 .lua 文件复制为同名的 .txt 文件。
"""

import os
import glob
import shutil

# 获取脚本所在目录的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))

# 切换到该目录（方便使用相对路径）
os.chdir(script_dir)

# 查找所有 .lua 文件
lua_files = glob.glob("*.lua")

if not lua_files:
    print("当前目录下没有找到 .lua 文件。")
else:
    for lua_file in lua_files:
        # 生成对应的 .txt 文件名
        txt_file = os.path.splitext(lua_file)[0] + ".txt"
        try:
            shutil.copy2(lua_file, txt_file)  # copy2 保留文件元数据
            print(f"已复制: {lua_file} -> {txt_file}")
        except Exception as e:
            print(f"复制 {lua_file} 时出错: {e}")