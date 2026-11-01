# Artifacts under MODELS, as named on Hugging Face (override to match local names).
VARIANTS="nvfp4:${NVFP4:-qwen3_8_27b_nvfp4_draftmlp-nvfp4.ninfer} v2:${V2:-qwen3_8_27b_iq3_xss_nvfp4_hybrid_v2.vi} v3:${V3:-qwen3_8_27b_iq3_xss_nvfp4_hybrid_v3.vi}"
# Clients run in velox-pubbench:1 and reach the server through the host.
B="$(pwd -W 2>/dev/null || pwd)"
client() { docker run --rm --add-host host.docker.internal:host-gateway -e PYTHONPATH=/bench/shim -v "$B:/bench" velox-pubbench:1 "$@"; }
