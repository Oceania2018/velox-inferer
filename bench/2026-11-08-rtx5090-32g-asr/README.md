# Qwen3-ASR-1.7B: 32 live streams on one RTX 5090 (2026-11-08)

At 32 concurrent two-channel call streams, the 99th-percentile time from the end of a sentence to
its final transcript is **0.646 s** after porting the laptop branch's Q8 kernel change (0.703 s
before). This is a third more streams than the 24-stream laptop run
([2026-11-07-rtx5090-24g-asr](../2026-11-07-rtx5090-24g-asr/README.md)), at a lower p99.

| | Before (`45691e2`) | After (`328d024`, merged as `3c2d3bb`) |
|---|---:|---:|
| Final latency p50 | 0.573 s | 0.555 s |
| p90 | 0.619 s | 0.587 s |
| p99 | 0.703 s | **0.646 s** |
| Max over 10 min | 1.084 s | 1.013 s |
| Sessions completed / errors | 32 / 0 | 32 / 0 |

## Setup

| | |
|---|---|
| GPU | RTX 5090, 32 GB, 170 SMs, driver 621.14 |
| Host | Intel Core Ultra 9 285K, 63.4 GB RAM; Windows 11 Pro + Docker Desktop 29.9.2 (WSL2), WSL VM capped at 48 GB |
| Decoder | `velox serve-asr` from branch `rtx-5090-32g`, CUDA 13.1; `--gguf Qwen3-ASR-1.7B-Q8_0.gguf --mmproj mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --kv-rows 4096` (8192 in one run) |
| Sidecar | `qwen3-asr` from qwen-llm-inferencer `main` `587e485` (silero VAD, 500 ms silence ends a sentence, final drafts, `ASR_DECODE_CONCURRENCY=32`, post-edit off) |
| Isolation | ASR alone on the GPU (27B stopped) |
| Load | [`../2026-11-07-rtx5090-24g-asr/harness/asr_load.py`](../2026-11-07-rtx5090-24g-asr/harness/asr_load.py), unchanged: 32 sessions × 2 channels = 64 live audio channels, real-time pace, 40 ms frames, 10 minutes, sessions staggered over 5 s, the same two 8 kHz stereo call recordings as the laptop run. About 3,670 finals per run. Each run follows a 60–90 s warm-up at the same load |
| Latency | From the wall time the sentence's last sample was sent to the arrival of its final, including the 500 ms the VAD waits for silence |

```bash
python asr_load.py --url ws://127.0.0.1:8080/transcribe/stream --streams 32 --seconds 600 \
  --out run.json a.wav b.wav
```

## Results

Seconds, 32 streams × 10 min, one run each. Raw summaries: [`results/summary.json`](results/summary.json).

| Build | KV rows | p50 | p90 | p99 | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|
| `45691e2` | 8192 | 0.572 | 0.620 | 0.703 | 0.896 | 1.054 |
| `45691e2` | 4096 | 0.573 | 0.620 | 0.701 | 0.982 | 1.108 |
| `45691e2`, `VELOX_ASR_PHASES=1` | 4096 | 0.573 | 0.619 | 0.703 | 0.901 | 1.084 |
| `328d024`, `VELOX_ASR_PHASES=1` | 4096 | **0.555** | **0.587** | **0.646** | 0.867 | 1.013 |

- **A 4096-row KV pool is enough for 32 streams.** p99 is the same as with 8192 rows, and the
  engine's periodic status line never showed a request waiting for KV. It saves 0.9 GB for the
  27B on the same card.
- **The phase timer costs nothing measurable** (p99 0.701 vs 0.703 s).

### Decode step time (`VELOX_ASR_PHASES=1`)

| Rows per step | Before, ms | After, ms | Laptop 24g (after its port), ms |
|---|---:|---:|---:|
| 1 | 2.15 | 2.12 | |
| 2–4 | 2.54 | 2.55 | |
| 5–8 | 2.71 | 2.75 | |
| 9–16 | 2.92 | 2.96 | |
| 17–23 | 4.17 | 3.05 | |
| 24–27 | 4.38 | **3.19** | 5.4–5.8 |
| 28–32 | 4.60 | 3.48 | |

The jump above 16 rows was the q8 skinny kernel re-reading every weight once per 16 rows. Steps of
17–32 rows are about 3% of all steps, so the end-to-end gain is in the tail.

### Where the engine's time goes (after)

Admission (encoder + prefill) is **56% of engine time** (267 s of 478 s over the run; 264 µs per
admitted row, encoder 151 s of it), decoding 44%. Before the port, over the first ~6 minutes,
admission was 68%. Draft acceptance on finals: 80–82%.

Most of a final's 0.65 s is fixed: 0.50 s of VAD silence plus up to ~0.07 s of VAD window and
frame alignment, which leaves about 0.1 s for the decoder at p99.

### q8 linear (`velox bench-linear --format q8`, µs, min of 3)

| Shape [n, k] | m=16 | m=17 | m=24 | m=32 | m=48 | m=63 |
|---|---|---|---|---|---|---|
| [4096, 2048] | 25 → 25 | 50 → 26 | 51 → 26 | 51 → 24 | 74 → 24 | 101 → 24 |
| [2048, 2048] | 26 → 25 | 62 → 26 | 47 → 26 | 55 → 25 | 95 → 22 | 94 → 23 |
| [12288, 2048] | 32 → 32 | 57 → 35 | 57 → 36 | 85 → 36 | 81 → 45 | 181 → 49 |
| [2048, 6144] | 23 → 22 | 56 → 23 | 48 → 26 | 48 → 25 | 83 → 31 | 99 → 28 |
| [151936, 2048] | 245 → 244 | 476 → 247 | 471 → 237 | 477 → 257 | 698 → 291 | 925 → 303 |

m ≤ 16 is unchanged within noise. Single runs on this card vary a lot; take the minimum of several.

## Accuracy

| Check | Result |
|---|---|
| `velox selftest` on `328d024` | 839 passed, 0 failed |
| 32 concurrent `/transcribe` calls (16 per recording, 992 segments), before vs after | 978 / 992 segments identical |
| Same, before vs before (two runs) | 976 / 992 identical |
| Majority text per distinct segment (62), before vs after | 62 / 62 identical |

Differences are top-2 near-ties that flip with batch composition, as on the laptop.

## 27B on the same branch

The Qwen3.8-27B hybrid v3 artifact has 8 q8_g32 objects (token embedding, vision merger, DFlash2
attention). The change only touches q8 linears at 17–63 rows, which a single CUA session almost
never produces (vision runs at ~1280 rows, drafts at ≤ 16, the embedding is a lookup). The replay
gate ([docs/performance.md](../../docs/performance.md)) passed with prefill and decode flat; the
reports stay out of the repository (business data) and are summarised on PR #14.

## Notes

- **First 32-stream run on this machine failed for reasons outside ASR.** About a minute in, the
  WSL GPU path (`dxgkio_*`) hung NVML: `nvidia-smi` blocked on the host and in WSL. An older
  llm-inference-daemon (before `646dcf2`) kept spawning `nvidia-smi` every 3 s and piled up 76
  blocked processes; at ~430 s the Docker VM stalled, DNS failed and all sessions closed. CUDA kept
  working throughout. After a driver restart and the daemon fix, every run completed, with
  `nvidia-smi` answering in ≤ 140 ms on every 10 s sample.
- Start the load only after the sidecar listens: a warm-up started right after recreating the
  containers lost 23 of 32 sessions to "connection refused".
- With the 27B serving CUA traffic, the two engines time-slice the GPU; these numbers are for ASR
  alone.

## Limits

- One 10-minute run per configuration; the laptop measured ±0.04 s run-to-run at p99.
- Two recordings only (8 kHz phone calls, mostly English with some Spanish).
- 48 and 64 streams not measured yet, so the card's ceiling is unknown.
