"""Realtime-stream load test for the qwen3-asr sidecar (WS /transcribe/stream).

N sessions each push a looping call recording at real-time pace for D seconds. Final latency is
measured from the wall time the segment's last sample was sent (offset_ms + duration_ms after the
session's first sample) to the time the final arrives, so it includes the sidecar's VAD silence.

Usage: python asr_load.py --url ws://qwen3-asr:8080/transcribe/stream --streams 24 --seconds 600 \
           --out /out/run.json WAV [WAV ...]
"""
import argparse
import asyncio
import json
import random
import time
import wave

import numpy as np
import websockets

RATE = 16000


def load(path: str, channels: int) -> np.ndarray:
    """int16 [samples, channels] at 16 kHz."""
    with wave.open(path, "rb") as w:
        src_ch, rate, width = w.getnchannels(), w.getframerate(), w.getsampwidth()
        assert width == 2, f"{path}: 16-bit PCM only"
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, src_ch).astype(np.float32)
    if channels == 1:
        pcm = pcm.mean(axis=1, keepdims=True)
    elif src_ch == 1:
        pcm = np.repeat(pcm, channels, axis=1)
    if rate != RATE:
        n = int(round(pcm.shape[0] * RATE / rate))
        t_src = np.arange(pcm.shape[0]) / rate
        t_dst = np.arange(n) / RATE
        pcm = np.stack([np.interp(t_dst, t_src, pcm[:, c]) for c in range(pcm.shape[1])], axis=1)
    return np.clip(np.round(pcm), -32768, 32767).astype(np.int16)


async def session(idx, args, audio, start_delay, results):
    await asyncio.sleep(start_delay)
    rec = {"stream": idx, "finals": [], "partials": 0, "warnings": [], "error": None, "done": False,
           "sender_lag_max_s": 0.0}
    results.append(rec)
    frame = int(RATE * args.frame_ms / 1000)
    offset = (idx * 7919 * frame) % audio.shape[0]        # each session starts elsewhere in the call
    try:
        async with websockets.connect(args.url, max_size=None, ping_interval=20, ping_timeout=60) as ws:
            await ws.send(json.dumps({"type": "start", "sample_rate": RATE, "channels": audio.shape[1],
                                      "language": args.language or None}))
            msg = json.loads(await ws.recv())
            if msg.get("type") != "ready":
                rec["error"] = msg
                return
            t0 = None

            async def reader():
                async for raw in ws:
                    m = json.loads(raw)
                    kind = m.get("type")
                    now = time.monotonic()
                    if kind == "final":
                        end_s = (m["offset_ms"] + m["duration_ms"]) / 1000
                        rec["finals"].append({"latency": now - (t0 + end_s), "channel": m.get("channel"),
                                              "audio_s": m["duration_ms"] / 1000, "chars": len(m.get("text") or ""),
                                              "at": now - t0, "text": m.get("text") if args.keep_text else None})
                    elif kind == "partial":
                        rec["partials"] += 1
                    elif kind == "warning":
                        rec["warnings"].append(m.get("message"))
                    elif kind == "error":
                        rec["error"] = m
                    elif kind == "done":
                        rec["done"] = True
                        return

            reader_task = asyncio.create_task(reader())
            total = int(args.seconds * RATE)
            sent = 0
            t0 = time.monotonic()
            while sent < total:
                due = t0 + sent / RATE
                lag = time.monotonic() - due
                if lag < 0:
                    await asyncio.sleep(-lag)
                else:
                    rec["sender_lag_max_s"] = max(rec["sender_lag_max_s"], lag)
                n = min(frame, total - sent)
                idxs = (offset + sent + np.arange(n)) % audio.shape[0]
                await ws.send(audio[idxs].tobytes())
                sent += n
            await ws.send(json.dumps({"type": "stop"}))
            await asyncio.wait_for(reader_task, timeout=120)
    except Exception as e:  # noqa: BLE001 — a load test records every failure and keeps going
        rec["error"] = repr(e)


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--url", default="ws://qwen3-asr:8080/transcribe/stream")
    ap.add_argument("--streams", type=int, default=24)
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--channels", type=int, default=2)
    ap.add_argument("--stagger", type=float, default=5.0)
    ap.add_argument("--frame-ms", type=int, default=40)
    ap.add_argument("--language", default="")
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    audios = [load(p, args.channels) for p in args.wavs]
    rng = random.Random(1)
    results = []
    t_start = time.time()
    await asyncio.gather(*(session(i, args, audios[i % len(audios)], rng.uniform(0, args.stagger), results)
                           for i in range(args.streams)))
    lat = [f["latency"] for r in results for f in r["finals"]]
    summary = {
        "streams": args.streams, "seconds": args.seconds, "channels": args.channels,
        "completed": sum(1 for r in results if r["done"] and not r["error"]),
        "errors": [r["error"] for r in results if r["error"]],
        "warnings": sum(len(r["warnings"]) for r in results),
        "finals": len(lat),
        "partials_per_stream": sum(r["partials"] for r in results) / max(1, len(results)),
        "p50": pct(lat, 50), "p90": pct(lat, 90), "p99": pct(lat, 99), "max": max(lat) if lat else None,
        "sender_lag_max_s": max(r["sender_lag_max_s"] for r in results),
        "wall_s": time.time() - t_start,
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "streams": results}, f)


if __name__ == "__main__":
    asyncio.run(main())
