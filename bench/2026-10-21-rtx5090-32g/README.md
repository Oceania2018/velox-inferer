# Response speed at 1, 2, 4 and 6 concurrent requests (RTX 5090, 2026-10-21)

## Setup

| | |
|---|---|
| Model | Qwen3.8-27B NVFP4 + DFlash2 drafter (7 drafts per block), NVFP4 KV cache, vision on |
| GPU | RTX 5090 32 GB desktop (32,607 MiB), driver 616.56 |
| Host | Intel Core Ultra 9 285K (24 cores), 63.4 GB RAM; Windows 11 + WSL2 (WDDM), WSL VM capped at 48 GB |
| Offload | Context store in pinned system RAM: 16 GiB (`--host-context-mib 16384`, the default). NVMe tier off (`--disk-context-mib 0`). Weights fully in VRAM |
| Software | velox main `15e5e24`, branch `rtx-5090-32g` `f2cd49b`, CUDA 13.1, `velox-dev` container |
| Server | `velox serve --max-context 32768 --kv-format nvfp4 --prefill-chunk 2048`: 6 lanes (`--max-concurrency 6`), paged KV on. Requests go through llm-inference-daemon, which adds at most 0.02 s |
| Client | [`scripts/speed-test.mjs`](../../scripts/speed-test.mjs), 5 rounds per row plus one discarded warm-up round |

The method is computer-autoplay's **Response speed test**, run by several clients at once. Each
request gets a fresh prompt of random common words that starts with a random nonce, so no prefix
cache can skip the prefill. Requests use temperature 0 and thinking off, and ask the model to
count far past the output cap so `max_tokens` ends every reply. In each round, `c` requests start
together and the round ends when all of them finish.

Charts of throughput, latency and decode rate are in [`index.html`](index.html). Open it in a
browser, since GitHub shows HTML as source. It is built with
`node scripts/bench-report.mjs bench/2026-10-21-rtx5090-32g`.

## Results

Medians per request. **Throughput** is completion tokens from all requests divided by the
summed round wall time (the server's capacity). The other columns are what one client sees.

| Concurrency | Input | Output | Decode (tok/s) | Prefill (s) | Prefill (tok/s) | Latency (s) | End-to-end (tok/s) | Throughput (tok/s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1K | 32 | 426 | 0.34 | 2,998 | 0.53 | 60.2 | 58 |
| 2 | 1K | 32 | 85 | 0.62 | 1,846 | 1.16 | 27.6 | 56 |
| 4 | 1K | 32 | 49 | 0.66 | 1,575 | 2.17 | 14.8 | 71 |
| 6 | 1K | 32 | 58 | 0.58 | 1,788 | 1.35 | 23.6 | 136 |
| 1 | 4K | 32 | 443 | 0.64 | 6,438 | 0.72 | 44.3 | 44 |
| 2 | 4K | 32 | 203 | 0.64 | 6,388 | 1.36 | 23.5 | 46 |
| 4 | 4K | 32 | 40 | 0.99 | 4,477 | 2.41 | 13.3 | 48 |
| 6 | 4K | 32 | 36 | 1.65 | 2,540 | 3.19 | 10.1 | 49 |
| 1 | 8K | 32 | 446 | 1.18 | 6,971 | 1.28 | 25.0 | 25 |
| 2 | 8K | 32 | 140 | 1.20 | 6,818 | 2.40 | 13.4 | 25 |
| 4 | 8K | 32 | 28 | 1.85 | 4,876 | 4.13 | 7.9 | 26 |
| 6 | 8K | 32 | 28 | 3.09 | 2,748 | 5.40 | 6.0 | 26 |
| 1 | 1K | 128 | 569 | 0.19 | 5,466 | 0.43 | 300.4 | 284 |
| 2 | 1K | 128 | 300 | 0.30 | 3,446 | 0.86 | 148.4 | 284 |
| 4 | 1K | 128 | 161 | 0.47 | 2,231 | 1.57 | 81.6 | 318 |
| 6 | 1K | 128 | 112 | 0.78 | 1,316 | 2.25 | 57.0 | 333 |
| 1 | 4K | 128 | 558 | 0.59 | 7,005 | 0.84 | 153.1 | 153 |
| 2 | 4K | 128 | 258 | 0.62 | 6,640 | 1.57 | 81.7 | 161 |
| 4 | 4K | 128 | 101 | 1.00 | 4,466 | 2.95 | 43.4 | 169 |
| 6 | 4K | 128 | 65 | 1.71 | 2,461 | 4.33 | 29.6 | 171 |
| 1 | 8K | 128 | 566 | 1.16 | 7,088 | 1.40 | 91.4 | 91 |
| 2 | 8K | 128 | 231 | 1.18 | 6,920 | 2.69 | 47.6 | 94 |
| 4 | 8K | 128 | 64 | 1.87 | 4,891 | 5.19 | 24.7 | 96 |
| 6 | 8K | 128 | 40 | 3.17 | 2,676 | 7.62 | 16.8 | 97 |

All 480 measured requests succeeded. Raw numbers are in [`results.json`](results.json). The 1–4
rows come from one run (VRAM peak 30.7 GB) and the 6 rows from a later run on the same server
(peak 31.9 GB). The slowest request at 6 took 7.9 s (8K/128).

## Reading the numbers

- **One request:** decode runs at 430–570 tok/s. A cold prompt prefills at about 7,000 tok/s
  (4K–8K tokens) and returns in 0.4–1.4 s.
- **Speculative decoding inflates decode here.** Counting is easy to draft, so the drafter's
  guesses are almost always accepted (8 tokens per round). Real agent steps reach about 270–300
  tok/s at one request.
- **Concurrent requests share one GPU.**
  - With short prompts, throughput rises with concurrency: 284 → 333 tok/s at 1K/128 from 1
    to 6 requests, and 58 → 136 at 1K/32.
  - With long prompts, throughput stays flat because prefill dominates.
  - Per-request latency grows accordingly: about 2× at 2 requests, 3–4× at 4, and 5–6× at 6.
- **Per-request decode falls further than throughput.** Under concurrency, one request's decode
  rounds also carry other requests' prefill chunks, and that waiting counts toward its decode
  time.
- **Short replies are dominated by prefill.** With 32 output tokens, extra requests mostly wait
  their turn for prefill.
- **VRAM headroom at 6:** the peak is 31.9 of 32.6 GB, close to the point where WDDM starts
  paging device memory to system RAM. CUDA graphs for new batch shapes and grow-only scratch
  buffers add memory under traffic. Bounding that growth is the next step.

## Runs not in the table

- **Before paged KV:** a run of the same build without paged attention caches allocated every
  lane's KV at full capacity (3.6 GB). That left too little VRAM headroom, and WDDM paged device
  memory to system RAM. Single-request decode fell to about 100 tok/s, and 4 × 8K/128 took 41 s
  per request. Paged KV backs only what each lane holds (0.4 GB at idle) and removed the
  collapse.
- **First 6-way run:** the first 6-way run, started right after the 1–4 run, had a few very slow
  rounds in its first configurations: 6 × 1K/32 throughput was 12 tok/s with normal median
  latency. Run again on the same server, the slow rounds were gone (the table above). The likely
  cause is one-time recording of new 6-request batch shapes; this was not confirmed.
