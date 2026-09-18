#!/bin/bash
# Full measurement run on Linux: build, tests, kernel benchmark, model comparison, CUDA-graph decode.
# Usage: scripts/bench_all.sh [sm_86] [--docker]   (--docker: nvcc from the nvidia/cuda:12.6.3 container)
# Expects data/W.raw (scripts/make_wraw.py) and models/Qwen3-4B (scripts/download_models.py).
set -e
cd "$(dirname "$0")/.."
ARCH=${1:-sm_89}
mkdir -p build
if [ "$2" = "--docker" ]; then
    docker run --rm -u "$(id -u):$(id -g)" -v "$PWD:/w" -w /w nvidia/cuda:12.6.3-devel-ubuntu24.04 \
        nvcc -O3 -arch="$ARCH" --shared -Xcompiler -fPIC -o build/nwc_ops.so csrc/nwc_ops.cu
else
    ./build.sh "$ARCH"
fi
echo "=== tests"
python tests/test_k.py
python tests/test_gather.py
python tests/test_checkpoint.py
echo "=== kernels vs cuBLAS"
python scripts/kernbench.py --runs 30
echo "=== model: VRAM, GPU time per token, logits"
python scripts/compare_df11.py --mode nwc --fusion --tokens 64
echo "=== CUDA graph decode"
python scripts/graph_decode.py --mode nwc --fusion --tokens 256
