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

## DSpine drafter (2026-11-08)

`--drafter dspine` (tencent/Qwen3.8-27B-DSpine-b16, [weights](weights.md#dspine-drafter-import))
against the production DFlash2 drafter on the Aiden replay session (95 steps, ~20k prompt tokens,
tool-call answers, temperature 0), one replay at a time, each from a cold server. RTX 5090 32 GB,
CUDA 13.1, `rtx-5090-32g` build with this change, `qwen3_8_27b_iq3_xss_nvfp4_hybrid_v3.vi`,
production flags (4 lanes, NVFP4 KV, 2048-token prefill chunks). Changes are medians against
the baseline report.

| Drafter | Device GB | Tokens / round | Decode tok/s | End-to-end |
|---|---:|---:|---:|---:|
| DFlash2, 7 drafts (baseline; rerun: 95/95 identical, ±0.2%) | 1.19 | 4.86 | 321.3 | 624.0 ms |
| DFlash2, 15 drafts (adaptive 7 / 15) | 1.19 | 4.91 | −1.0% | −1.9% |
| DSpine `--nvfp4 mlp`, adaptive 7 / 15, window 8192 | 1.62 | 6.51 | +10.3% | −3.8% |
| DSpine `--nvfp4 mlp`, adaptive, window 4096 | 1.62 | 6.42 | +14.3% | −6.0% |
| DSpine `--nvfp4 mlp`, adaptive, window 2048 | 1.62 | 5.82 | +6.8% | −7.7% |
| DSpine `--nvfp4 all`, adaptive, window 4096 | 1.06 | 6.29 | +15.5% | −5.4% |
| **DSpine `--nvfp4 all`, 15 drafts every round, window 4096** | **1.06** | **8.03** | **+27.4%** | **−14.6%** |
| DSpine `--nvfp4 all`, 15 drafts every round, window 8192 | 1.06 | 8.24 | +27.2% | −14.3% |
| DSpine BF16, 15 drafts every round (1 lane, acceptance only) | 3.54 | 8.22 | – | – |

- The 7 / 15 adaptation (`--dflash-short 7`, tuned for DFlash2) keeps DSpine on the short block
  most of the time; `--dflash-short 15` (no adaptation) is what reaches the acceptance an offline
  simulation of the reference predicted (8.07 tokens / round).
- NVFP4 costs no acceptance (8.24 vs 8.22 BF16). A 4096-row window is as fast as the trained 8192
  and halves every lane's drafter rings; 2048 loses too much acceptance.
- Prefill is unchanged (±1%). Tool-call quality matches the baseline (95/95 valid, 89–90 same
  tool, 42–44 / 50 points within 20 px). Outputs identical to the baseline: 30–46 / 95 steps; the
  verify width alone already changes greedy ties (DFlash2 at 15 drafts: 52 / 95), and a rerun of
  one configuration is 95 / 95.

Four concurrent replays of the same session (4 lanes decoding at once, 4 x 95 steps; replays that
advance in lockstep, so a harsher mix than production traffic):

| Drafter (4 lanes) | Wall | Aggregate tok/s | Tokens / round | Decode p50 per request | End-to-end p50 / p90 | Points within 20 px (per replay) |
|---|---:|---:|---:|---:|---:|---|
| DFlash2, 7 drafts (baseline) | 195 s | 208 | 4.87 | 102 | 1,327 / 2,643 ms | 43, 43, 45, 46 |
| DFlash2, 14 drafts every round | 189 s | 213 | 5.41 | 73 | 1,629 / 3,208 ms | 43, 44, 44, 42 |
| DSpine `--nvfp4 all`, 15 drafts every round, window 4096 | 175 s | 229 | 7.72 | 90 | 1,465 / 2,845 ms | 40, 41, 41, 40 |
| DSpine `--nvfp4 all`, `--dflash-short 7`, window 4096 | 174 s | 230 | 5.77 | 84 | 1,508 / 2,906 ms | 43, 43, 42, 45 |

- Batched rounds draft at most 14 per sequence (`BATCH_DRAFTS_MAX`): a replayed step cannot attend
  over 16+ tokens, which the 16-row windows of `--dflash-short 15` hit before that cap.
- DSpine finishes the batch 10% sooner, but each request waits on larger batched rounds: its
  proposal scores twice the rows over the full 248k-row head (DFlash2 uses the smaller draft
  vocabulary head), so per-request latency is ~10% worse when 4 lanes decode at once. That cost is
  the next thing to cut.
- All 380 tool calls are valid in every run. DFlash2 with the same 15-row batched windows keeps
  its hit rate, so the engine's batched verify is not the cause of the 15-draft DSpine run's lower
  point hits; the four lockstep replays produce near-identical outputs (one sample, not four), and
  the gap is within the spread single runs show.
