# Qwen3-ASR-1.7B: 24 live streams on one RTX 5090 Laptop (2026-11-07)

At 24 concurrent two-channel call streams, the 99th-percentile time from the end of a sentence to
its final transcript went from **0.934 s to 0.700 s**. The target was under 1.5 s. Transcripts are
unchanged: final drafts keep only the tokens one-token greedy decoding would have produced.

| | Before (`2b1f3e9`) | After (`78a45f8`) |
|---|---:|---:|
| Final latency p50 | 0.612 s | 0.562 s |
| p90 | 0.735 s | 0.613 s |
| p99 | 0.934 s | **0.700 s** |
| Max over 10 min | 1.111 s | 0.843 s |
| Segments identical to the previous build | | 84 / 84 |

## Setup

| | |
|---|---|
| GPU | RTX 5090 Laptop GPU, 24 GB, 82 SMs, driver 621.14 |
| Host | Intel Core Ultra 9 290HX Plus, 63.4 GB RAM; Windows 11 + Docker Desktop (WSL2), WSL VM capped at 48 GB |
| Decoder | `velox serve-asr`, image `velox-asr:rtx-5090-24g-78a45f8` built from branch `rtx-5090-24g` `78a45f8`, CUDA 13.1; `--gguf Qwen3-ASR-1.7B-Q8_0.gguf --mmproj mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --kv-rows 8192` |
| Sidecar | `qwen3-asr` from qwen-llm-inferencer `main` `399df2f` (silero VAD, a sentence ends after 500 ms of silence, `ASR_DECODE_CONCURRENCY=32`) |
| Isolation | The ASR pair had the GPU to itself (27B stopped) |
| Load | [`harness/asr_load.py`](harness/asr_load.py): 24 sessions × 2 channels = 48 live audio channels, real-time pace, 40 ms frames, 10 minutes, sessions staggered 5 s. Each session loops one of two 8 kHz stereo call recordings at a different offset. About 2,770 finals per run |
| Latency | From the wall time the sentence's last sample was sent to the arrival of its final. It includes the 500 ms the VAD waits for silence |
| Accuracy | Word error rate against the fp32 reference model (transformers) on the same VAD segments of both recordings, and segment-by-segment comparison between builds |

The recordings are customer calls and are not in the repository. Any stereo call audio works with
the harness; absolute numbers depend on how much speech it holds.

```bash
python harness/asr_load.py --url ws://qwen3-asr:8080/transcribe/stream \
  --streams 24 --seconds 600 --out run.json a.wav b.wav
```

Set `VELOX_ASR_PHASES=1` on `velox serve-asr` to print the engine-side `[phases]` breakdown
(admission vs decode time, decode steps by rows, draft tokens fed and accepted). The per-final
wait / admit / decode split quoted below came from a temporary instrumented build that is not
committed.

## Results

Seconds, 24 streams × 10 min, one run each. All runs completed with no errors. Raw summaries:
[`results/summary.json`](results/summary.json).

| Build | p50 | p90 | p99 | max | partials / stream |
|---|---:|---:|---:|---:|---:|
| Qwen3-ASR on 24g, split-K tuned (`2b1f3e9`) | 0.612 | 0.735 | 0.934 | 1.111 | 361 |
| + Q8 kernel reads weights once for up to 64 tokens (`54873ad`) | 0.596 | 0.715 | 0.915 | 1.153 | 375 |
| + final verifies its last partial as a draft, 64 rows per step | 0.562 | 0.618 | 0.736 | 0.871 | 392 |
| + at most 32 rows per step (final, `78a45f8`) | 0.562 | 0.613 | **0.700** | 0.843 | 393 |

Repeated runs of the same build differ by up to about ±0.04 s at p99, so the last step's gain is
within noise on its own. A same-binary A/B (64 vs 32 rows, instrumented build) agreed with it:
client p99 0.722 vs 0.704 s, max 0.965 vs 0.842 s; engine-side final time p99 157 vs 145 ms,
max 265 vs 214 ms. For comparison, 16 streams on the first build gave p99 0.871 s.

### What changed

- **Q8 small-batch kernel (`rtx-5090-24g` only).** Batches of 17–64 tokens used to re-read the
  weights once per 16 tokens; one pass now serves up to 64. Decode steps at batch 24–27 dropped
  from 7.5–8.4 ms to 5.4–5.8 ms; decoder throughput +13–16%.
- **Final drafts (main + sidecar).** A final usually repeats the last partial of the same sentence
  almost word for word. The sidecar sends that partial's raw text as `draft`; velox feeds up to 24
  draft tokens per sequence in the same decode pass and keeps only the prefix that matches its own
  greedy choice. 79% of draft tokens were accepted. A slow final now needs about 8 decode steps
  instead of about 50. See [docs/qwen3_asr.md](../../docs/qwen3_asr.md).
- **32-row step cap (main).** Verify steps of 33–64 rows cost 8–17 ms against 3–6 ms at 20 rows or
  fewer, and every sequence in the batch waits for them. Capping at 32 rows trims the tail.

## Accuracy

| Check | Result |
|---|---|
| Word error rate vs fp32 reference, recording 1 | 1.25% (5 / 399 words) |
| Word error rate vs fp32 reference, recording 2 | 3.36% (13 / 387 words) |
| Final build: no draft vs exact draft vs corrupted draft (a word changed, a word dropped) | 84 / 84 segments identical, both |
| Q8 kernel change vs previous build | 83 / 84 segments identical |

The one segment that differs after the kernel change is a sub-second fragment whose detected
language flipped. The same kind of flip already happens between different batch compositions
(top-2 logit gaps of 0.001–0.04).

## Where the remaining 0.70 s goes

| Part | Time |
|---|---:|
| VAD waits for 500 ms of silence | 0.50 s |
| VAD window and 40 ms audio frames | ~0.03–0.07 s |
| Decoder, slowest finals | ~0.15 s |

The decoder part splits roughly evenly between waiting for an admission already in progress
(about 50 ms), the final's own encode and prefill (about 45 ms for a 15 s sentence), and decoding,
which other requests' admissions stretch. Request handling outside the decoder costs 3–9 ms.

## Tried and not adopted

| Option | Result | Decision |
|---|---|---|
| NVFP4 weights (W4A16, community quantization) | Word errors 2.3% → 5.0% (18 → 39 of 786 words); identical segments 90% → 74%; not faster (16 streams: p99 0.855 vs 0.871 s); saves 0.6 GB | dropped |
| VAD silence 350 ms | p99 0.748 s; 14% more sentence breaks; recording 2 word errors 3.36% → 5.17%, including a cut sentence that changed meaning | dropped |
| VAD silence 250 ms | p99 0.641 s; 19% more breaks; recording 2 word errors 6.20%; short fragments transcribed in the wrong language | dropped |
| Fusing the activation conversion into the Q8 kernel | 5–19% faster in the kernel benchmark, no change in real decode steps (CUDA graphs already hide it) | reverted |
| Holding a new final up to 40 ms while another final decodes | p99 0.701 s, within run-to-run noise | dropped |
| Faster kernels for the output head and o-projection | With drafts a slow final takes about 8 steps, so these would save about 5–8 ms of p99 | not worth it |
| FunASR models | No Spanish in SenseVoice / Paraformer / Fun-ASR-Nano; MLT-Nano's Spanish unconfirmed and weaker English (FLEURS-en 4.90% vs 3.35%) | stay on Qwen3-ASR-1.7B |

## Other hardware branches

| Branch | Final drafts + 32-row cap | Q8 kernel change | Measured |
|---|---|---|---|
| `rtx-5090-24g` | yes | yes | this run |
| `rtx-5060-ti` | yes | no | not yet |
| `rtx-5090-32g` | yes | no | not yet |
| `rtx-4090` | yes | no | not yet |
| `rtx-pro-6000` | yes | no | not yet |

Transcripts are unchanged on every card, because only tokens the model would have chosen are kept.
The speed gain is unmeasured off the 24g: those branches run Q8 at 2–32 rows on a different,
untuned path, so verify steps may cost more there. Each needs one 24-stream run on its own card.

## Limits

- Two recordings only (8 kHz phone calls, mostly English with some Spanish). Spanish accuracy was
  not measured separately.
- One 10-minute run per configuration; p99 moves by up to ±0.04 s between identical runs.
- The 27B on `rtx-5090-24g` has not been replayed against these kernels yet; its vision tower and
  drafter share the split-K and Q8 kernels.
- Memory: one ASR instance uses about 5.4 GB (2.5 GB weights, 1.9 GB KV pool of 8,192 rows,
  ~0.3 GB workspaces, ~0.6 GB CUDA context and graphs). Peak KV use sampled during the run was about
  3,600 rows.
- One early 24-stream run on this laptop, before these changes, ended in a driver bugcheck (0x9F).
  The cause is unknown; every later run, including all runs here, completed cleanly.
