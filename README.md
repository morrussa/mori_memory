# mori

*mori就是mori，memeto mori!*

这是heat_mem的重置版，因为我不喜欢python，就这么简单。

要求lupa支持luaJIT，因为向量计算余弦近似度我是用FFI强算的

# 可能有用的命令

先退出虚拟环境（如果当前还在里面）
deactivate

rm -rf .venv

python3 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip setuptools wheel

python -m pip install -r requirements.txt


CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120" \
CUDACXX=/usr/local/cuda-13.1/bin/nvcc \
CC=/usr/bin/gcc-14 \
CXX=/usr/bin/g++-14 \
FORCE_CMAKE=1 \
cmake -S . -B build $CMAKE_ARGS

--这个是我自己的cuda和gcc环境

git clone https://luajit.org/git/luajit-2.0.git luajit
cd luajit
make
cd ..
rm -rf build dist lupa.egg-info lupa/*.so lupa/*.c
pip install lupa --no-binary :all: --verbose --no-cache-dir --force-reinstall

qwen3.5支持pr：
