#!/usr/bin/env bash
# Build the exl3xpu _C.so of this branch (production flags, scripts/build_arc.sh c) inside an exl3xpu build image at idle
# priority, <= 4 GiB. Output: lorak/_C_lora.so (production exl3xpu/_C.so untouched).
set -euo pipefail
cd "$(dirname "$0")/.."
IMG=${IMG:?set IMG to an exl3xpu build image (docker/Dockerfile.arc)}
chrt -i 0 nice -n 19 ionice -c 3 docker run --rm --memory 4g --memory-swap 4g --cpus 4 --cpu-shares 2 --name lorak-build \
  -v "$PWD":/w -w /w --entrypoint bash "$IMG" -c '
export PATH=/opt/venv/bin:$PATH
EXL3_OUT=lorak/_C_lora.so nice -n 19 bash scripts/build_arc.sh c'
