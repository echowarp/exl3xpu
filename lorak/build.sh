#!/usr/bin/env bash
# Build lorak/llmt_lk.so (torch library llmt_lk) inside the exl3xpu-l0 image (icpx 2025.3.3, PyTorch XPU), at idle
# priority, container memory <= 4 GiB. Usage: lorak/build.sh [out.so]   (default lorak/llmt_lk.so)
set -euo pipefail
cd "$(dirname "$0")"
OUT=${1:-llmt_lk.so}
IMG=${IMG:-exl3xpu-l0:26.31-llmt-int8kv8}
chrt -i 0 nice -n 19 ionice -c 3 docker run --rm --memory 4g --memory-swap 4g --cpus 4 --cpu-shares 2 \
  -v "$PWD":/w -w /w --entrypoint bash "$IMG" -c '
set +u; source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1; set -u
T=$(/opt/venv/bin/python -c "import torch,os;print(os.path.dirname(torch.__file__))" 2>/dev/null)
nice -n 19 icpx -fsycl -fsycl-targets=spir64 -O3 -fPIC -std=c++17 -shared -D_GLIBCXX_USE_CXX11_ABI=1 ${LK_FLAGS:-} \
  -I$T/include -I$T/include/torch/csrc/api/include -x c++ lorak_ops.sycl -x none -o '"$OUT"' \
  -L$T/lib -Wl,-rpath,$T/lib -lc10 -ltorch -ltorch_cpu -lc10_xpu -ltorch_xpu -Wno-deprecated-declarations
ls -la '"$OUT"
