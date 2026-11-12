#!/usr/bin/env bash
# Build the Arc B70 kernel libraries of this fork inside an environment that has the oneAPI DPC++/C++ compiler (icpx),
# oneDNN and the XPU build of PyTorch, e.g. the upstream image (docker/Dockerfile).
#
#   scripts/build_arc.sh [c|gdnchunk|ng|all]      (default: all)
#
# Toolchain of the production build: Intel oneAPI DPC++/C++ Compiler 2025.3.3 (2025.3.3.20260319), oneDNN 3.9.1,
# PyTorch 2.12.0+xpu, as shipped in intel/llm-scaler-vllm:0.26.0-b2 (the upstream base image). All libraries are
# JIT builds (-fsycl-targets=spir64): IGC finalizes the device code on first launch and caches it.
# See README-ARC.md ("Building") for which flags are recorded and which are reconstructed.
set -euo pipefail
cd "$(dirname "$0")/.."
WHAT=${1:-all}

# _C.so (torch library exl3xpu_C): scripts/build_ext.sh with the production compile-time options. build_ext.sh adds
# -DEXL3_DNNL (oneDNN int8 GEMM / SDPA paths) when oneDNN is installed.
EXL3_ARC_FLAGS="-DEXL3_ALL_CODEBOOKS -DEXL3_K5_UNROLL24=4 -DEXL3_K5_MB128 -DEXL3_FOLD -DEXL3_HADOUT_B=8 \
-DEXL3_DPAS40_SMALLGRF -DEXL3_DPAS40_NT=1 -DEXL3_I4_U4DPAS -DEXL3_I4_K2T -DEXL3_SPEC_EPIU"
build_c() {
  EXL3_FLAGS="${EXL3_FLAGS:-$EXL3_ARC_FLAGS}" EXL3_OUT="${EXL3_OUT:-exl3xpu/_C.so}" bash scripts/build_ext.sh
}

# setvars.sh is not nounset-safe: under `set -u` it aborts the script silently. Source it only when icpx is missing.
oneapi_env() {
  command -v icpx >/dev/null 2>&1 && return 0
  set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true; set -u
}

torch_flags() {
  T=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")
  TORCH_INC="-I$T/include -I$T/include/torch/csrc/api/include"
  TORCH_LIB="-L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu"
}

# gdnchunk.so (torch library gdnchunk): GDN prefill port (gdnchunk/, Apache-2.0). GDNC_FLAGS default = the options of
# the production build as documented when it was made (not re-derived from the binary). AOT=1 additionally compiles
# device code for the B70 (bmg-g31).
GDNC_ARC_FLAGS="-DGDNC_PF=1 -DGDNC_PF_DIST=2 -DGDNC_PF_QK256=1 -DGDNC_CONV2=1 -DGDNC_VNNI2D=1 -DGDNC_PLANPAR=1 -DGDNC_CONV_VEC=4"
build_gdnchunk() {
  oneapi_env
  torch_flags
  local targets=spir64; local aot=()
  if [ "${AOT:-0}" = 1 ]; then targets=spir64_gen; aot=(-Xs "-device bmg-g31"); fi
  icpx -fsycl -fsycl-targets=$targets "${aot[@]}" -O3 -ffast-math -fPIC -std=c++17 -shared \
    -fsycl-device-code-split=per_kernel -D_GLIBCXX_USE_CXX11_ABI=1 ${GDNC_FLAGS:-$GDNC_ARC_FLAGS} \
    -I gdnchunk/csrc $TORCH_INC -x c++ gdnchunk/csrc/gdn_chunk_ops.sycl -x none -o ${GDNC_OUT:-exl3xpu/gdnchunk.so} \
    $TORCH_LIB
  ls -la ${GDNC_OUT:-exl3xpu/gdnchunk.so}
}

# ng.so (torch library exl3ng): fused NaN guard. Built WITHOUT -ffast-math (isnan / isinf must survive).
build_ng() {
  oneapi_env
  torch_flags
  icpx -fsycl -fsycl-targets=spir64 -O3 -fPIC -std=c++17 -shared -D_GLIBCXX_USE_CXX11_ABI=1 \
    $TORCH_INC -x c++ ng/ng_ops.sycl -x none -o ${NG_OUT:-exl3xpu/ng.so} $TORCH_LIB -Wno-deprecated-declarations
  ls -la ${NG_OUT:-exl3xpu/ng.so}
}

case "$WHAT" in
  c) build_c ;;
  gdnchunk) build_gdnchunk ;;
  ng) build_ng ;;
  all) build_c
       build_gdnchunk
       build_ng
       ;;
  *) echo "usage: $0 [c|gdnchunk|ng|all] (what this version has)" >&2; exit 2 ;;
esac
