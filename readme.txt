# 先退出虚拟环境（如果当前还在里面）
deactivate

# 删除整个 venv 文件夹（最常用、最干净的方式）
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
pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir --verbose

git clone https://luajit.org/git/luajit-2.0.git luajit
cd luajit
make
cd ..
rm -rf build dist lupa.egg-info lupa/*.so lupa/*.c
pip install lupa --no-binary :all: --verbose --no-cache-dir --force-reinstall

gcc -shared -fPIC -O3 -march=native -o module/simdc_math.so simd_math.c -lm