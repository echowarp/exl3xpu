#!/usr/bin/env bash
# run_c.sh [check|bench|sums|sums-prod|all]: test_lora_c.py on the Arc (engine must NOT be running), real EXL3 weights
# (read-only model mount), <= 4 GiB host RAM. Output also in lorak/out/c-<what>-HHMM.txt.
set -uo pipefail
cd "$(dirname "$0")"; mkdir -p out
WHAT=${1:-all}; IMG=${IMG:-exl3xpu-l0:26.31-llmt-int8kv8}
PROD_C=/srv/exl3xpu-l0/patches/i4prod/_C.so
for w in $([ "$WHAT" = all ] && echo sums-prod sums check bench || echo "$WHAT"); do
  extra=(); arg=$w
  if [ "$w" = sums-prod ]; then extra=(-v $PROD_C:/prod_C.so:ro -e LK_C=/prod_C.so); arg=sums; fi
  docker run --rm --memory 4g --name lorak-test --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path:ro \
    -v /srv/models/turboderp-Qwen3.8-27B-exl3-4.00bpw:/models:ro -v "$PWD":/lk:ro "${extra[@]}" \
    --entrypoint /opt/venv/bin/python "$IMG" -I /lk/test_lora_c.py "$arg" 2>&1 \
    | grep -v "UserWarning\|warnings.warn" | tee out/c-$w-$(date +%H%M).txt
done
[ -f out/c-sums-prod-*.txt ] && true
