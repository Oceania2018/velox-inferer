# velox-inferer

A Rust + CUDA inference framework for consumer GPUs. Each supported card gets its own hardware
branch, and AMD support is planned.

- **main** owns architecture and everything shareable: the `Backend` op contract, device profile
  catalog, VRAM planner, CPU reference backend (the numerical oracle), portable CUDA kernels,
  weight formats (`.vi`, `.ninfer` import), OpenAI-compatible server, conformance suite and the
  replay performance gate.
- **hardware branches** own implementation for their cards and change only
  `crates/velox-variant/`. They merge main without conflicts. See
  [docs/branching.md](docs/branching.md).

| Branch | Cards |
|---|---|
| `main` | any supported GPU, portable kernels |
| `rtx-5090-32g` | RTX 5090 (32 GB desktop) |
| `rtx-5090-laptop-24g` | RTX 5090 Laptop GPU (24 GB) |
| `rtx-5060-ti` | RTX 5060 Ti 16 GB / 8 GB |
| `rtx-4090` | RTX 4090 (24 GB) |

## Status

Working end to end on the RTX 5090: device detection and profiles, VRAM planning, an f32
decoder stack on portable CUDA kernels verified against the CPU oracle, `.ninfer` v3 / `.vi` v1
read, validate and convert (validated on the 23 GB production Qwen3.8-27B artifact), the
OpenAI-compatible API, and llm-inference-daemon integration (`"Framework": "velox-inferer"`).

Not yet: quantised weight kernels, the Qwen3.5/3.8 architecture, tokenizer, vision, prefix
cache and speculative decoding. Until then `velox serve` answers chat requests with 501. The
roadmap is in [docs/architecture.md](docs/architecture.md); velox replaces ninfer on a card only
when the replay gate ([docs/performance.md](docs/performance.md)) shows no regression.

## Build

Requirements: Rust 1.80+, and for GPUs the CUDA toolkit (12.8+ for `sm_120`; 13.1 validated).
Without the `cuda` feature everything builds and runs on the CPU backend.

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
- [Adding AMD](docs/adding-amd.md)
