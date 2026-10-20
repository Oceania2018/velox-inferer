# Performance gate: no regressions

Two levels of measurement, both required before a kernel or engine change merges:

1. **Ops and synthetic stack**: `velox bench` on the target card. Compare the branch head with
   its previous head and with main's portable kernels (median of `--iters`).
2. **Real workload replay**: a recorded computer-use agent session replayed, request by
   request, through llm-inference-daemon against the ninfer baseline and against velox.

## Replay

A session directory holds `session.json` (per step: the exact chat request with images elided,
plus recorded metrics) and `screenshots/step-N.jpg`. `velox replay` restores each step's
screenshot into its request and sends the steps in order, so prefix-cache behaviour matches the
original run. Sessions contain business data. Replay talks only to the given endpoint, and
reports keep metrics plus an output hash, never prompt or output text. Reports go to
`target/replay/` (git-ignored).

Run both engines on the same port (8010) at different times behind the same daemon, so the
requests are byte-identical: the daemon applies the same image downscaling to `ninfer` and
`velox-inferer` backends.

```bash
# daemon model.servers.json: {"Port": 8010, "Framework": "ninfer", ...}
velox replay --session <dir> --endpoint http://127.0.0.1:8080 --label ninfer
# stop ninfer, start velox on 8010; daemon entry: "Framework": "velox-inferer"
velox replay --session <dir> --endpoint http://127.0.0.1:8080 --label velox
velox replay-compare --baseline target/replay/ninfer.json --candidate target/replay/velox.json
```

`replay-compare` fails if any median regresses by more than `--tolerance` percent (default 3):
client TTFT, client end-to-end latency, server prefill time (`timings.prompt_ms`) and server
decode speed (`timings.predicted_per_second`). It also reports how many outputs are identical;
the sessions run at temperature 0.

## Baseline (2026-10-19)

ninfer `master-13879011`, Qwen3.8-27B NVFP4 + DFlash2 (7 draft tokens), NVFP4 KV, vision on,
RTX 5090 32 GB, CUDA 13.1.2, through llm-inference-daemon (1 Mpx image budget). Session
`mvg18csx-6nesw9`, 137 steps, ~17–26k prompt tokens per step, mostly prefix-cached.
Source: `target/replay/ninfer-via-daemon.json` (not committed).

| Metric (137 steps) | p50 | p90 | mean | min / max |
|---|---:|---:|---:|---:|
| Client end-to-end latency (ms) | 733 | 1,079 | 761 | 317 / 1,439 |
| Server prefill, uncached tokens (ms) | 304 | 523 | 276 | 41 / 989 |
| Server decode (tok/s, DFlash2) | 305 | 381 | 310 | 216 / 416 |
| Recorded latency in the original remote run (ms) | 927 | 1,316 | 1,014 | 714 / 1,816 |

Prompt tokens averaged 20,266 (identical to the recording), 18,892 of them prefix-cached. TTFT
equals end-to-end latency because answers are tool calls, which arrive whole at the end. The
local run is faster than the recording because it has no network hop to a remote gateway.
Replay wall time: 105 s.
