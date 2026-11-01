# Public-dataset benchmark: Qwen3.8-27B NVFP4 vs hybrid v2 vs v3 (RTX 5090, 2026-11-01)

The same velox server and flags, with three weight files. Every input is public and pinned, so
any RTX 5090 can rerun this (see [Reproduce](#reproduce)).

| Variant | File ([Hugging Face](https://huggingface.co/haiping/Qwen3.8-27B-velox)) | sha256 |
|---|---|---|
| nvfp4 (baseline) | `qwen3_8_27b_nvfp4_draftmlp-nvfp4.ninfer` from [haiping/Qwen3.8-27B-nvfp4-dflash2-NInfer](https://huggingface.co/haiping/Qwen3.8-27B-nvfp4-dflash2-NInfer) | `0ffe1b817ff6560c39fccdf5be3152a99b29f2bec893f749d6b6683be574a7f3` |
| v2 | `qwen3_8_27b_iq3_xss_nvfp4_hybrid_v2.vi` | `152ee4304efbc6b9531b92480f9b6f1f270e6b06a53fb18121de72eb4bb36bca` |
| v3 | `qwen3_8_27b_iq3_xss_nvfp4_hybrid_v3.vi` | `86e3782da7afaa9ce1b72eb95110118596fcca5b692a7a7b86384e248d759805` |

## Setup

| | |
|---|---|
| GPU | RTX 5090 32 GB desktop, driver 617.42 |
| Host | Intel Core Ultra 9 285K (24 cores), 63.4 GB RAM; Windows 11 + Docker Desktop (WSL2), WSL VM capped at 48 GB |
| Server | image `velox:rtx-5090-32g` built from branch `rtx-5090-32g` `5b627e3`, CUDA 13.1; flags in [`harness/serve.sh`](harness/serve.sh) (6 lanes, 49152 context, NVFP4 KV, DFlash2, 24 GiB pinned context store, NVMe tier off) |
| Isolation | A fresh server per variant and round, on its own port with no other traffic, so no prefix cache carries over |
| Serving client | vLLM [`benchmark_serving.py`](https://github.com/vllm-project/vllm/blob/v0.6.6.post1/benchmarks/benchmark_serving.py) v0.6.6.post1, unmodified, `--backend openai-chat --seed 0`, in the `velox-pubbench:1` image ([`harness/Dockerfile`](harness/Dockerfile)) |
| Grounding client | [`harness/screenspot_eval.py`](harness/screenspot_eval.py): one request per sample, temperature 0, `max_tokens` 48 |

| Test | Data | Load |
|---|---|---|
| ShareGPT | [`ShareGPT_V3_unfiltered_cleaned_split.json`](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) @ `192ab21` | 200 prompts at 1 and at 6 concurrent clients |
| Random 4K | vLLM `random` dataset, Qwen3.8-27B tokenizer @ `1d4bf0f` | 50 prompts of 4096 tokens in, up to 128 out, 1 client |
| ScreenSpot-v2 | [`OS-Copilot/ScreenSpot-v2`](https://huggingface.co/datasets/OS-Copilot/ScreenSpot-v2) @ `5efbb1f`, all 1272 samples | 1 client |

The serving tests ran 3 rounds; each round ran nvfp4, v2 and v3 in turn. ScreenSpot is
deterministic at temperature 0 and ran once per variant.

## Results

Each cell is the mean over the 3 rounds of that round's median (p90 where stated), its change
against nvfp4, and ± half the spread between rounds. Lower is better for latency and higher for
throughput. Full tables: [`results/summary.md`](results/summary.md).

| Test | Metric | nvfp4 | v2 | v3 |
|---|---|---:|---:|---:|
| ShareGPT, 1 client | TTFT p50 | 58.2 ms | 61.2 ms (+5.2%) | **53.3 ms (−8.4%)** |
| | TPOT p50 | 5.5 ms | 4.8 ms (−12.7%) | **4.5 ms (−17.9%)** |
| | End-to-end p50 | 814 ms | 722 ms (−11.3%) | **677 ms (−16.8%)** |
| ShareGPT, 6 clients | TTFT p50 | 118 ms | 124 ms (+5.3%) | **112 ms (−5.0%)** |
| | TPOT p50 | 12.2 ms | 14.3 ms (+17.4%) | **11.9 ms (−2.8%)** |
| | Output tokens/s | 445 ±12% | 334 ±21% (−24.8%) | **503 ±0.3% (+13.2%)** |
| Random 4K, 1 client | TTFT p50 (prefill) | 672 ms | 726 ms (+8.0%) | **640 ms (−4.7%)** |
| | TPOT p50 | 6.3 ms | 5.7 ms (−9.1%) | **5.3 ms (−15.9%)** |

The round-to-round spread of every median above is at most ±3%, except v2's TPOT at 6 clients
(±6.1%).

ScreenSpot-v2 click accuracy (%), a hit being a click inside the target box:

| Variant | All | Desktop text | Desktop icon | Mobile text | Mobile icon | Web text | Web icon | Parsed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| nvfp4 | **96.1** | 97.9 | 94.3 | 98.6 | 91.5 | 97.4 | 95.1 | 1272/1272 |
| v2 | 95.8 | 99.0 | 95.0 | 99.3 | 90.5 | 94.9 | 95.1 | 1271/1272 |
| v3 | 95.6 | 97.9 | 94.3 | 99.0 | 90.5 | 96.6 | 93.6 | 1271/1272 |

v3 lands 6 fewer clicks than nvfp4 out of 1272 (−0.5 points); v2 lands 3 fewer.

## Findings

- **v3 is faster than the baseline on every median**: single-client decode about 18% faster,
  prefill about 5% faster at 4K, and 13% more output throughput at 6 clients. Its grounding
  accuracy stays within half a point.
- **v2 decodes faster on one client but prefills slower**, and at 6 clients it is slower
  throughout. Its Q4/Q5 input projections are what v3 replaces with NVFP4.
- These agree in direction with the replay of a private agent session in the
  [model card](https://huggingface.co/haiping/Qwen3.8-27B-velox), where prefill matters more
  because every step sends a screenshot.

## Caveats

- **Connection stalls.** 14 of the 4,050 serving requests waited 5–70 s for their first token,
  spread over all three variants including the baseline. The waits (about 5, 12, 35 and 68 s)
  match TCP SYN retransmission backoff, and the ScreenSpot client saw the same thing as
  connection timeouts from Docker Desktop's port forwarding; the client retries those (5 retries
  were needed in total). We read them as host networking, not the server, but have not proven it.
  They inflate p99 and per-round output throughput (hence nvfp4's and v2's ±12–21% at 6 clients);
  medians are unaffected.
- **Output length differs per variant.** velox ignores `ignore_eos`, and ShareGPT requests stop
  at the model's own end of turn, so the variants generate different token counts. Compare TTFT
  and TPOT; output tokens/s and end-to-end time also reflect length.
- **ScreenSpot prompt.** The prompt is ours (in [`harness/screenspot_eval.py`](harness/screenspot_eval.py)),
  not an official one, and Qwen3.8 answers in 0–1000 normalized coordinates. Scores are for
  comparing these files with each other, not with published leaderboard numbers.
- **One machine.** All numbers come from one RTX 5090 under Windows + WSL2. In this run the
  ScreenSpot client ran with host Python; the committed scripts run it in `velox-pubbench:1`.

## Reproduce

Needs Docker with GPU support, about 2 GB for the data, and the three weight files (23–25 GB
each) in a Docker volume or directory.

```bash
# velox image from the card's hardware branch (see the model card for the build)
bash bench/2026-11-01-rtx5090-32g-public/harness/fetch.sh ~/pubbench   # data, vLLM client, client image
cd ~/pubbench
MODELS=/path/to/weights bash run_all.sh        # ~60 min: 3 rounds x 3 variants
MODELS=/path/to/weights bash run_screenspot.sh # ~30 min
python summarize.py results
```

`NVFP4`, `V2` and `V3` override the file names, and `IMAGE` the velox image. Raw results are in
[`results/`](results): per-run metrics with per-request lengths and TTFTs (generated text
removed), and per-sample ScreenSpot answers.
