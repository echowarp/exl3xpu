#!/usr/bin/env bash
# run.sh [check|bench|step|all]: test_lorak.py on the Arc in the exl3xpu-l0 image (engine must NOT be running on the
# Arc), <= 3 GiB host RAM. Output also in lorak/out/<what>-HHMM.txt.
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p out
WHAT=${1:-all}; IMG=${IMG:-exl3xpu-l0:26.31-llmt-int8kv8}
for w in $([ "$WHAT" = all ] && echo check bench step || echo "$WHAT"); do
  docker run --rm --memory 3g --name lorak-test --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
    -v "$PWD":/lk:ro --entrypoint /opt/venv/bin/python "$IMG" -I /lk/test_lorak.py "$w" 2>&1 \
    | grep -v "UserWarning\|warnings.warn" | tee out/$w-$(date +%H%M).txt
done
