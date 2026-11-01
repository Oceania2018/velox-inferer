"""ScreenSpot-v2 click accuracy against an OpenAI-compatible endpoint.

Each sample sends one screenshot and its instruction, asks for a single click point, and counts a
hit when the point falls inside the target box. Requests are sequential at temperature 0.
usage: screenspot_eval.py --data DIR --endpoint URL --model ID --out FILE [--limit N] [--coords auto|norm1000|pixel]
"""
import argparse
import base64
import io
import json
import re
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image

PROMPT = ("You are a GUI agent. Find the UI element for this instruction: \"{instruction}\".\n"
          "Answer with only its center point as JSON, e.g. {{\"point_2d\": [x, y]}}.")
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def ask(endpoint, model, png, instruction):
    body = {
        "model": model,
        "temperature": 0,
        "max_tokens": 48,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}},
            {"type": "text", "text": PROMPT.format(instruction=instruction)},
        ]}],
    }
    req = urllib.request.Request(endpoint.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    # Docker Desktop's port forward occasionally refuses a connection; retry those before giving up.
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.load(r)["choices"][0]["message"]["content"] or ""
        except urllib.error.URLError as e:
            if attempt == 4 or isinstance(e, urllib.error.HTTPError):
                raise
            print(f"retry after {e}", file=sys.stderr, flush=True)
            time.sleep(5 * (attempt + 1))


def point(text):
    # The first "[x, y" pair: the key "point_2d" holds a digit, and answers may wrap the object in a list.
    m = re.search(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", text)
    return (float(m.group(1)), float(m.group(2))) if m else None


def to_pixels(p, size, coords):
    w, h = size
    return (p[0] * w / 1000, p[1] * h / 1000) if coords == "norm1000" else p


def hit(p, bbox):
    x, y, w, h = bbox
    return p is not None and x <= p[0] <= x + w and y <= p[1] <= y + h


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True)
    a.add_argument("--endpoint", required=True)
    a.add_argument("--model", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--coords", default="norm1000", choices=["norm1000", "pixel", "auto"])
    args = a.parse_args()
    data = Path(args.data)
    samples = []
    for platform in ("desktop", "mobile", "web"):
        for s in json.loads((data / f"screenspot_{platform}_v2.json").read_text(encoding="utf-8")):
            samples.append({**s, "platform": platform})
    if args.limit:
        samples = samples[:: max(1, len(samples) // args.limit)][: args.limit]

    rows, t0 = [], time.time()
    for i, s in enumerate(samples):
        img_path = data / "screenspotv2_image" / s["img_filename"]
        png = img_path.read_bytes()
        size = Image.open(io.BytesIO(png)).size
        t = time.time()
        text = ask(args.endpoint, args.model, png, s["instruction"])
        p = point(text)
        rows.append({"img": s["img_filename"], "platform": s["platform"], "type": s["data_type"],
                     "bbox": s["bbox"], "size": size, "raw": p, "text": text[:120], "latency_s": round(time.time() - t, 3),
                     "parsed": p is not None})
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(samples)} {time.time() - t0:.0f}s", file=sys.stderr, flush=True)

    def score(coords):
        acc = defaultdict(lambda: [0, 0])
        for r in rows:
            ok = hit(to_pixels(r["raw"], r["size"], coords) if r["raw"] else None, r["bbox"])
            for key in ("all", r["platform"], f"{r['platform']}/{r['type']}"):
                acc[key][0] += ok
                acc[key][1] += 1
        return {k: {"hits": v[0], "n": v[1], "accuracy": round(100 * v[0] / v[1], 2)} for k, v in sorted(acc.items())}

    scores = {c: score(c) for c in ("norm1000", "pixel")}
    coords = args.coords if args.coords != "auto" else max(scores, key=lambda c: scores[c]["all"]["hits"])
    report = {"dataset": "OS-Copilot/ScreenSpot-v2@5efbb1f1b5463a575f2eb7bc30fe29e49c15f93c",
              "model": args.model, "coords": coords, "prompt": PROMPT, "samples": len(rows),
              "parsed": sum(r["parsed"] for r in rows), "scores": scores[coords], "all_conventions":
              {c: scores[c]["all"] for c in scores}, "wall_s": round(time.time() - t0, 1), "rows": rows}
    Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({"coords": coords, "parsed": report["parsed"], "samples": len(rows),
                      "conventions": report["all_conventions"], "by": {k: v["accuracy"] for k, v in scores[coords].items()}}))


if __name__ == "__main__":
    main()
