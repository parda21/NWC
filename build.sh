#!/bin/bash
# Linux build of the library (build/nwc_ops.so). Run from the project directory: ./build.sh [sm_86]
# nvcc either in PATH or as the pip wheel (nvidia-cuda-nvcc-cu12) in the active venv.
set -e
ARCH=${1:-sm_89}
cd "$(dirname "$0")"
mkdir -p build
if ! command -v nvcc >/dev/null 2>&1; then
    NV=$(python -c "import nvidia, os; print(os.path.dirname(nvidia.__path__[0]))" 2>/dev/null)/nvidia
    export PATH="$NV/cuda_nvcc/bin:$PATH"
    INC="-I$NV/cuda_runtime/include -I$NV/cuda_cccl/include"
    LIB="-L$NV/cuda_runtime/lib"
fi
nvcc -O3 -arch=$ARCH --shared -Xcompiler -fPIC ${INC:-} ${LIB:-} -o build/nwc_ops.so csrc/nwc_ops.cu "$@"
echo "built: build/nwc_ops.so ($ARCH)"
