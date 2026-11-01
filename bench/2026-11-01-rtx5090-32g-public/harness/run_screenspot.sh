#!/bin/bash
# ScreenSpot-v2 click accuracy for each variant on a fresh isolated server (deterministic: one round).
set -u
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"
. ./variants.sh
for v in $VARIANTS; do
  name=${v%%:*}
  sh ./serve.sh "${v#*:}" || { echo "SERVE FAILED $name"; continue; }
  client python screenspot_eval.py --data /bench/screenspot --endpoint http://host.docker.internal:8011/v1 \
    --model qwen3.8-27b --out "/bench/results/$name-screenspot.json" > "results/$name-screenspot.log" 2>&1 \
    || echo "FAILED screenspot $name"
  tail -1 "results/$name-screenspot.log"
  docker rm -f velox-pubbench >/dev/null
done
echo SSDONE
