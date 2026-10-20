# Architecture

```
velox-cli ──► velox-conformance ──► velox-cpu (oracle)
    │                │
    └──► velox-variant (per branch) ──► velox-cuda (backend, portable kernels) ──► velox-core
                                                        ▲
                                         velox-build (nvcc helper, build-time only)
```

| Crate | Owns | Changes on |
|---|---|---|
| `velox-core` | `Backend` op contract, shape checks, device profile catalog, VRAM planner, `VariantInfo`, reference decoder stack | main |
| `velox-cpu` | Reference backend: the numerical oracle | main |
| `velox-cuda` | CUDA runtime shim, buffers, portable kernels, `CudaKernels` override trait, `CudaBackend` | main |
| `velox-build` | `build.rs` helper that compiles `.cu` with nvcc | main |
| `velox-variant` | The build's hardware variant: tuned kernels and per-SKU policy | **hardware branches** |
| `velox-conformance` | Op and end-to-end cases compared against the oracle | main |
| `velox-weights` | `.vi` / `.ninfer` reader, validator and converter | main |
| `velox-serve` | OpenAI-compatible HTTP API over a `ChatEngine` | main |
| `velox-cli` | `velox` binary: devices, plan, selftest, bench, inspect, convert, serve, replay | main |

## Three layers of hardware specificity

1. **Profile** (data, main): what a card is. SKU, arch, SMs, VRAM, bandwidth, default reserve,
   tuned branch. Detected from (arch, SM count, nearest VRAM).
2. **Backend** (code, main): how to talk to a vendor runtime. `CudaBackend` today, `HipBackend`
   later. Validates every call with `velox_core::check` so all backends reject the same input.
3. **Variant** (code, branch): how to be fast on specific cards. A `CudaKernels` implementation
   whose methods default to the portable kernels, plus `plan_policy`. A branch covering several
   SKUs (the RTX 5060 Ti 16 GB and 8 GB) picks policy per detected profile at runtime.

## Correctness model

The CPU backend is the oracle: plain loops with f64 accumulation. `velox selftest` runs every op
on awkward shapes plus a multi-step prefill/decode through a decoder stack, and compares against
the oracle with a relative-error metric. Tuned kernels must fall back (usually to `portable::*`)
for shapes they do not cover; the conformance shapes are chosen to hit those fallbacks.

### Low-precision activations

Tensor-core `linear` kernels round their inputs to FP8 or NVFP4. That is a model decision, not a
kernel one: each weight view carries the artifact's `ActPolicy` (`A16Only`, `AllowA8`, or
`AllowA4 { divisor }` from `uses[].activation_policy` and its `activation_input_divisor`). A
backend reports what it actually does through `Backend::linear_act(view, m)`, which returns
`Exact`, `Fp8Row` or `Nvfp4 { divisor }` and never exceeds the policy. The conformance suite
applies the same rounding on the host (`velox_core::quant`, bit-exact with the device: RNE,
saturating, same operation order) and compares against the exact product of the rounded input.
A low-precision kernel is therefore still held to the f64 oracle, and rounding a view whose
policy forbids it fails the suite. Portable kernels on main stay exact.

## Current scope and next steps

The engine is f32 end to end with synthetic weights. In dependency order, the work to run
production checkpoints is:

1. Weight loading: `.vi` format and `.ninfer` v3 import ([weights.md](weights.md)).
2. Quantised weights (BF16, FP8, NVFP4, groupwise int) as new matmul ops.
3. The Qwen3.5/3.8 hybrid architecture (gated delta net layers alongside attention), embeddings,
   LM head, sampling and the tokenizer.
4. Prefix/context cache, vision encoder, MTP speculative decode.
5. An OpenAI-compatible server, required by the replay gate ([performance.md](performance.md)).
