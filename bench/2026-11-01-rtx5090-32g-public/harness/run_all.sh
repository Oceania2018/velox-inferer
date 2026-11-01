#!/bin/bash
# Serving benchmark: 3 rounds x {nvfp4, v2, v3}; a fresh isolated server per run (no prefix-cache carry-over).
set -u
export MSYS_NO_PATHCONV=1
cd "$(dirname "$0")"
. ./variants.sh
bench() {  # label args...
  client python benchmark_serving.py \
    --backend openai-chat --base-url http://host.docker.internal:8011 --endpoint /v1/chat/completions \
    --model qwen3.8-27b --tokenizer /bench/tokenizer --seed 0 --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,90,99 --save-result --result-dir /bench/results --result-filename "$1.json" "${@:2}" \
    > "results/$1.log" 2>&1 || echo "FAILED $1"
}
for r in 1 2 3; do
  for v in $VARIANTS; do
    name=${v%%:*}
    sh ./serve.sh "${v#*:}" || { echo "SERVE FAILED $name"; continue; }
    bench "$name-sharegpt-c1-r$r" --dataset-name sharegpt --dataset-path /bench/sharegpt.json --num-prompts 200 --max-concurrency 1
    bench "$name-sharegpt-c6-r$r" --dataset-name sharegpt --dataset-path /bench/sharegpt.json --num-prompts 200 --max-concurrency 6
    bench "$name-random4k-c1-r$r" --dataset-name random --random-input-len 4096 --random-output-len 128 --num-prompts 50 --max-concurrency 1 --ignore-eos
    docker logs velox-pubbench > "results/$name-r$r-server.log" 2>&1
    docker rm -f velox-pubbench >/dev/null
    echo "done $name r$r $(date +%H:%M)"
  done
done
echo ALLDONE
