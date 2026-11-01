"""Summarise the public benchmark: mean over rounds of each run's metrics, change vs the first variant."""
import json
import statistics as st
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
variants = ["nvfp4", "v2", "v3"]
tests = [("sharegpt-c1", "ShareGPT, 1 client"), ("sharegpt-c6", "ShareGPT, 6 clients"),
         ("random4k-c1", "Random 4096 in / 128 out, 1 client")]
metrics = [("median_ttft_ms", "TTFT p50 (ms)", -1), ("p90_ttft_ms", "TTFT p90 (ms)", -1),
           ("median_tpot_ms", "TPOT p50 (ms)", -1), ("median_e2el_ms", "E2E p50 (ms)", -1),
           ("output_throughput", "Output tok/s", 1)]

out = {"tests": {}, "screenspot": {}}
for key, title in tests:
    print(f"\n### {title}\n")
    print("| Metric | " + " | ".join(variants) + " |")
    print("|---|" + "---:|" * len(variants))
    table = {}
    for v in variants:
        runs = [json.loads(p.read_text()) for p in sorted(root.glob(f"{v}-{key}-r*.json"))]
        table[v] = {m: [r[m] for r in runs] for m, _, _ in metrics}
        table[v]["runs"] = len(runs)
        table[v]["completed"] = [r["completed"] for r in runs]
    out["tests"][key] = table
    for m, label, _ in metrics:
        cells = []
        base = st.mean(table[variants[0]][m]) if table[variants[0]][m] else None
        for v in variants:
            vals = table[v][m]
            if not vals:
                cells.append("–")
                continue
            mean = st.mean(vals)
            spread = (max(vals) - min(vals)) / mean * 100 if mean else 0
            rel = "" if v == variants[0] or not base else f" ({100 * (mean / base - 1):+.1f}%)"
            cells.append(f"{mean:.1f}{rel} ±{spread / 2:.1f}%")
        print(f"| {label} | " + " | ".join(cells) + " |")
    print("| Runs (completed requests) | " + " | ".join(
        f"{table[v]['runs']} ({'/'.join(map(str, table[v]['completed']))})" for v in variants) + " |")

print("\n### ScreenSpot-v2 click accuracy (1272 samples, temperature 0)\n")
groups = ["all", "desktop/text", "desktop/icon", "mobile/text", "mobile/icon", "web/text", "web/icon"]
print("| Variant | " + " | ".join(groups) + " | parsed |")
print("|---|" + "---:|" * (len(groups) + 1))
for v in variants:
    p = root / f"{v}-screenspot.json"
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    s = d["scores"]
    out["screenspot"][v] = {g: s[g] for g in groups} | {"parsed": d["parsed"], "samples": d["samples"]}
    print(f"| {v} | " + " | ".join(f"{s[g]['accuracy']:.1f}" for g in groups) + f" | {d['parsed']}/{d['samples']} |")

(root / "summary.json").write_text(json.dumps(out, indent=1))
