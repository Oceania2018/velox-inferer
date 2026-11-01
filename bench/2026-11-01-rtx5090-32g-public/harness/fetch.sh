#!/bin/bash
# usage: fetch.sh WORKDIR — copies this harness into WORKDIR and downloads the pinned public inputs.
# Needs curl, unzip and docker. About 2 GB of data; the model weights are fetched separately (README).
set -eu
W=${1:?usage: fetch.sh WORKDIR}
HF=https://huggingface.co
mkdir -p "$W/results" "$W/tokenizer" "$W/screenspot"
cp -r "$(dirname "$0")"/. "$W/"
cd "$W"
get() { [ -s "$2" ] || curl -fL --retry 5 -o "$2" "$1"; }

# vLLM's serving benchmark, unmodified (shim/ supplies the one vllm import it needs).
for f in benchmark_serving.py backend_request_func.py; do
  get https://raw.githubusercontent.com/vllm-project/vllm/v0.6.6.post1/benchmarks/$f $f
done
get $HF/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/192ab2185289094fc556ec8ce5ce1e8e587154ca/ShareGPT_V3_unfiltered_cleaned_split.json sharegpt.json
for f in config.json generation_config.json merges.txt tokenizer.json tokenizer_config.json vocab.json; do
  get $HF/Qwen/Qwen3.8-27B/resolve/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/$f tokenizer/$f
done
S=$HF/datasets/OS-Copilot/ScreenSpot-v2/resolve/5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c
for p in desktop mobile web; do get $S/screenspot_${p}_v2.json screenspot/screenspot_${p}_v2.json; done
get $S/screenspotv2_image.zip screenspot/images.zip
[ -d screenspot/screenspotv2_image ] || unzip -q screenspot/images.zip -d screenspot
docker build -t velox-pubbench:1 .
echo "ready: cd $W && bash run_all.sh && bash run_screenspot.sh && python summarize.py results"
