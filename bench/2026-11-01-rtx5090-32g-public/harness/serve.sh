#!/bin/sh
# usage: serve.sh ARTIFACT — isolated velox server on :8011 for one benchmark run.
#   MODELS    docker volume or host dir holding the artifacts, mounted at /models (default ninfer-models)
#   IMAGE     velox image built from the card's hardware branch (default velox:rtx-5090-32g)
#   TEMPLATE  host path of the chat template (default ./chat_template.jinja, the one the README runs used)
set -eu
export MSYS_NO_PATHCONV=1
TEMPLATE=${TEMPLATE:-"$(cd "$(dirname "$0")" && (pwd -W 2>/dev/null || pwd))/chat_template.jinja"}
docker rm -f velox-pubbench >/dev/null 2>&1 || true
docker run -d --name velox-pubbench --gpus all -p 8011:8011 -v "${MODELS:-ninfer-models}:/models:ro" \
  -v "$TEMPLATE:/templates/chat.jinja:ro" \
  "${IMAGE:-velox:rtx-5090-32g}" serve --artifact "/models/$1" --port 8011 --model-id qwen3.8-27b \
  --chat-template /templates/chat.jinja \
  --max-context 49152 --max-concurrency 6 --kv-format nvfp4 --prefill-chunk 2048 --host-context-mib 24576 \
  --host-state-dtype bf16 --device-state-dtype f32 --ctx-checkpoints 1 --force-tool-call on --vram-budget-mib 30720 \
  --disk-context-mib 0 >/dev/null
for i in $(seq 1 90); do curl -sf http://127.0.0.1:8011/v1/models >/dev/null && { echo "ready after $((i*5))s"; exit 0; }; sleep 5; done
echo "not ready"; docker logs velox-pubbench 2>&1 | tail -5; exit 1
