# AGENTS.md

## Product and architecture

velox-inferer is a Rust (host) + CUDA (kernels) inference framework for consumer GPUs. Read
[architecture](docs/architecture.md) and [branching](docs/branching.md) before structural changes.

- main owns abstractions and everything shareable. Hardware branches (`rtx-5090-32g`,
  `rtx-5090-24g`, `rtx-5060-ti`, `rtx-4090`) change **only**
  `crates/velox-variant/{src,kernels}`; `scripts/check-branch-scope.sh` enforces it.
- New ops land on main with a CPU reference, a portable kernel, a `CudaKernels` method that
  defaults to the portable kernel, and conformance cases. Branches then override.
- Hardware branches add no dependencies. Dependencies are added on main only.
- AMD is reserved (`GpuArch::Hip`, `BackendKind::Hip`); keep portable kernels hipify-clean.

## Verification

- Correctness: `velox selftest` (every op and an end-to-end stack against the CPU oracle) on the
  target card. Matching another GPU kernel is not evidence; the oracle is.
- Performance: no regressions. Use `velox bench` for kernels and `velox replay` +
  `velox replay-compare` through llm-inference-daemon for end-to-end
  ([performance](docs/performance.md)). Report baseline, hardware, toolchain, commands and the
  worst changes, including small or unexplained regressions.
- Replay sessions hold confidential business data: never commit them, their screenshots, or
  reports containing prompt/output text.

## Environment

Use `scripts/dev.sh <cmd>` to run in the `velox-dev` container (CUDA 13.1, Rust stable, GPU).
Commits use Conventional Commit subjects.
