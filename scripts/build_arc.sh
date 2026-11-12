#!/usr/bin/env bash
# Build the Arc B70 kernel libraries of this fork inside an environment that has the oneAPI DPC++/C++ compiler (icpx),
# oneDNN and the XPU build of PyTorch, e.g. the upstream image (docker/Dockerfile).
#
#   scripts/build_arc.sh [c|all]      (default: all)
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

case "$WHAT" in
  c) build_c ;;
  all) build_c
       ;;
  *) echo "usage: $0 [c|gdnchunk|ng|all] (what this version has)" >&2; exit 2 ;;
esac
