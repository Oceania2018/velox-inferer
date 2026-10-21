# velox-inferer

A vendor-neutral Rust inference framework for consumer GPUs. Ops are defined once by the
`Backend` contract and verified against a CPU oracle; each GPU vendor plugs in a backend, and each
supported card gets its own hardware branch for tuned kernels.

| Backend | Vendors | Status |
|---|---|---|
| CPU | any | working; the numerical oracle |
| CUDA | NVIDIA | working |
| HIP | AMD | reserved: device profiles and planning work, no executing backend yet ([docs/adding-amd.md](docs/adding-amd.md)) |

- **main** owns architecture and everything shareable: the `Backend` op contract, device profile
  catalog, VRAM planner, CPU reference backend (the numerical oracle), portable GPU kernels,
  weight formats (`.vi`, `.ninfer` import), OpenAI-compatible server, conformance suite and the
  replay performance gate.
- **hardware branches** own implementation for their cards and change only
  `crates/velox-variant/`. They merge main without conflicts. See
  [docs/branching.md](docs/branching.md).

| Branch | Cards |
|---|---|
| `main` | any GPU with a backend, portable kernels |
| `rtx-5090-32g` | RTX 5090 (32 GB desktop) |
| `rtx-5090-24g` | RTX 5090 Laptop GPU (24 GB) |
| `rtx-5060-ti` | RTX 5060 Ti 16 GB |
| `rtx-4090` | RTX 4090 (24 GB) |

## Status

Serving Qwen3.8-27B NVFP4 on the RTX 5090 through llm-inference-daemon
(`"Framework": "velox-inferer"`), with the OpenAI-compatible API, vision, DFlash2 speculative
decoding and up to 6 concurrent requests. Prefixes are reused from a context store in system RAM
(optional NVMe tier), and attention caches are paged so lanes share identical prefixes. Every op
is checked against the CPU oracle (`velox selftest`). velox replaces ninfer on a card only when
the replay gate ([docs/performance.md](docs/performance.md)) shows no regression.

## Performance

RTX 5090 32 GB, Qwen3.8-27B NVFP4 + DFlash2. The host has 63.4 GB RAM (WSL capped at 48 GB),
with a 16 GiB context store in system RAM and the NVMe tier off. Measured with
computer-autoplay's response speed test run by several clients at once. Every request gets a
fresh prompt, so nothing is prefix-cached. Latency and decode are medians per request;
throughput is all requests' output tokens per second of wall time.

| Requests at once | Input / output tokens | Latency | Decode (tok/s) | Throughput (tok/s) |
|---:|---:|---:|---:|---:|
| 1 | 1K / 128 | 0.43 s | 569 | 284 |
| 2 | 1K / 128 | 0.86 s | 300 | 284 |
| 4 | 1K / 128 | 1.57 s | 161 | 318 |
| 6 | 1K / 128 | 2.25 s | 112 | 333 |
| 1 | 8K / 128 | 1.40 s | 566 | 91 |
| 2 | 8K / 128 | 2.69 s | 231 | 94 |
| 4 | 8K / 128 | 5.19 s | 64 | 96 |
| 6 | 8K / 128 | 7.62 s | 40 | 97 |

All 6 lanes run without errors. At 6 requests, VRAM peaks at 31.9 of 32.6 GB, which leaves
little headroom. A cold 8K prompt prefills at about 7,000 tok/s. The test prompt asks the model to count, which
the drafter predicts almost perfectly, so these decode rates are an upper bound. Real agent steps
decode at about 270–300 tok/s. For the full matrix, method and notes, see
[bench/2026-10-21-rtx5090-32g](bench/2026-10-21-rtx5090-32g/README.md); all runs are listed in
[bench/](bench/README.md).

## Build

Requirements: Rust 1.80+. Without a GPU feature everything builds and runs on the CPU backend.
For NVIDIA GPUs, enable the `cuda` feature and install the CUDA toolkit (12.8+ for `sm_120`;
13.1 validated).

```bash
cargo build --release --features cuda      # or: scripts/dev.sh cargo build --release --features cuda
velox devices                              # which profile and branch fit this machine
velox selftest                             # conformance vs CPU oracle on the GPU
velox plan --model qwen3-32b --weight-bits 4.5 --kv-bits 8
velox bench --layers 4
velox inspect --artifact model.ninfer
velox serve --artifact model.vi --port 8010
```

`scripts/dev.sh` runs any command in the `velox-dev` container (CUDA 13.1 + Rust,
`docker/dev.Dockerfile`) with the GPU attached.

## Documentation

- [Architecture](docs/architecture.md): crates, the three layers of hardware specificity, roadmap
- [Hardware branches](docs/branching.md): the branch rule and the main-first feature workflow
- [Weight files](docs/weights.md): `.vi` and `.ninfer`
- [Performance gate](docs/performance.md): benchmarks and session replay
- [Benchmark results](bench/README.md): dated runs
- [Adding AMD](docs/adding-amd.md)
