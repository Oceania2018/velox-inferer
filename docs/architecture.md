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
   SKUs picks policy per detected profile at runtime.

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

Attention follows the same pattern: `Backend::attention_precision(shape)` returns `Exact` or
`Fp8QkF16Pv` (Q and K rounded per head vector to E4M3 for the scores, V and the softmax weights
to f16 for the value product, as ninfer does). The oracle rounds Q, K and V identically
(`quant::round_attention_inputs`); the unemulated f16 softmax weights set a 2e-3 tolerance.

### KV cache formats

`AttnShape::kv` names the storage format of the attention caches (`velox_core::kv::KvFormat`:
`f32`, `bf16`, `fp8`, `nvfp4`, following ninfer's `--kv-dtype`). `kv_store` encodes K/V rows
bit-exactly as `kv::encode_row` specifies (fp8: Hadamard-rotated K, one f16-valued scale per row;
nvfp4: rotated K and V, E4M3 scale per 16 values). A cache *represents* `H · dequant(codes)` for
rotated rows, and attention is exact softmax attention over the represented rows; kernels rotate
the query and the output instead of the rows. The oracle decodes caches to f32 and runs the f32
reference, so every format is checked against the same f64 attention. Models choose the format
per session (`velox serve --kv-format`); the default stays `f32`.

### Recurrent-state checkpoint formats

Context-store entries keep the GDN `S` matrices in `velox_core::StateFormat` (`f32`, `bf16`,
`nvfp4`, following ninfer's `--host-state-dtype`; `velox serve --host-state-dtype`, default
`bf16`); lanes always run on f32 state. A `Checkpoint` records its format: `state_pack` encodes
when a snapshot is written (straight into pinned system RAM, the store's storage) and
`state_unpack` decodes on restore, both bit-exact with `state::pack` / `state::unpack`. A 27B
state is 157 MB in f32, 81 MB in bf16 and 28 MB in nvfp4 (per-row f32 divisor, E4M3 scale per
16 values). `velox generate --state-quant bf16,nvfp4` measures what a format costs a resumed
generation against the exact state.

## Dense and MoE models

`Qwen3_5ForCausalLM` (Qwen3.6/3.8-27B) and `Qwen3_5MoeForCausalLM` (Qwen3.6-35B-A3B) share the
hybrid GDN/attention stack and differ only in each block's FFN, so they are one model
(`velox_models::qwen3_5`) with one FFN extension point rather than two models:

- `TextConfig::ffn` is `FfnConfig::Dense { intermediate }` or `FfnConfig::Moe(MoeConfig)`; the
  two field sets are mutually exclusive, as ninfer's converter writes them.
- `Layer::ffn` is `Ffn::Dense(DenseFfn)` or `Ffn::Moe(MoeFfn)`. Routed experts load as
  `ExpertBank`s: one parent in its stored encoding, expert `e` at a constant row stride, so a
  kernel addresses any expert without gathering or repacking.
- `Qwen35Text::mlp` is the single dispatch point; chunked prefill, verify, batched verify and MTP
  all reach the FFN through it. The engine, sessions, KV cache, context store and speculative
  decoding see no difference between the two.

MoE execution is not implemented yet: a MoE block returns `Error::Unsupported`, and `velox
serve` answers 501 for the architecture. It lands like any op, on main with a CPU reference,
portable kernels, `CudaKernels` defaults and conformance cases (planned contract in
[qwen3_5-text.md](qwen3_5-text.md#sparse-moe-qwen3_5moeforcausallm)); branches then tune the
expert GEMMs. Models whose experts do not fit in VRAM (expert offload to system RAM) are out of
scope for this design; 35B-A3B fits a 32 GB card.

## Current scope and next steps

The engine is f32 end to end with synthetic weights. In dependency order, the work to run
production checkpoints is:

1. Weight loading: `.vi` format and `.ninfer` v3 import ([weights.md](weights.md)).
2. Quantised weights (BF16, FP8, NVFP4, groupwise int) as new matmul ops.
3. The Qwen3.5/3.8 hybrid architecture (gated delta net layers alongside attention), embeddings,
   LM head, sampling and the tokenizer.
4. Prefix/context cache, vision encoder, MTP speculative decode.
5. An OpenAI-compatible server, required by the replay gate ([performance.md](performance.md)).
