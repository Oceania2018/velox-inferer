# Benchmarks

Dated benchmark results, one directory per run: `YYYY-MM-DD-<card>/` with a `README.md` (setup,
method, results, notes) and the raw numbers. Newest first.

| Run | What |
|---|---|
| [2026-10-21-rtx5090-32g](2026-10-21-rtx5090-32g/README.md) | Response speed at 1, 2, 4 and 6 concurrent requests, Qwen3.8-27B NVFP4 |

## Response speed test

```bash
# velox serve (or any OpenAI-compatible endpoint that reports llama.cpp `timings`) running
node scripts/speed-test.mjs --endpoint http://127.0.0.1:8080/v1 --model qwen3.8-27b \
  --concurrency 1,2,4 --input 1024,4096,8192 --output 32,128 --rounds 5 --json results.json
```

The script follows computer-autoplay's Response speed test (fresh random prompt, temperature 0,
thinking off, output capped by `max_tokens`) with several clients at once.

Each run also has an `index.html` report with charts, made from its `results.json`. Add `title`,
`date` and `setup` to the JSON first, then run:

```bash
node scripts/bench-report.mjs bench/<run>
```

GitHub shows `.html` files as source. Open the report locally in a browser, or serve the folder
with GitHub Pages.

## Rules

- Record the hardware, driver, toolchain, commits and server flags with every run, including
  system RAM (and any VM cap) and the offload settings (context store size, NVMe tier).
- Report the whole matrix, including rows that got worse.
- Never commit results from recorded agent sessions: they hold business data. Replay reports stay
  in `target/replay/` (see [docs/performance.md](../docs/performance.md)).
