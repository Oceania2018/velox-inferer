# Hardware branches

velox-inferer ships one build per hardware family. Users pick the branch that matches their GPU;
`velox devices` names it for each detected card.

| Branch | Profiles | Arch | Notes |
|---|---|---|---|
| `main` | any | portable (`sm_89`, `sm_120` + PTX) | Generic variant: correct everywhere, tuned nowhere. All feature work lands here. |
| `rtx-5090-32g` | `rtx-5090` (32 GB desktop) | `sm_120a` | Primary implementation branch (170 SMs). |
| `rtx-5090-laptop-24g` | `rtx-5090-laptop` (24 GB) | `sm_120a` | 82 SMs, larger VRAM reserve. |
| `rtx-5060-ti` | `rtx-5060-ti-16g`, `rtx-5060-ti-8g` | `sm_120a` | Small-SM (36) shapes. |
| `rtx-4090` | `rtx-4090` | `sm_89` | Ada: no TMA, no FP4. |

AMD cards are already in the profile catalog; they need the HIP backend first
([adding-amd.md](adding-amd.md)), then an AMD branch on the same model.

## The one rule: a hardware branch only changes `crates/velox-variant/`

Everything else (core contracts, profiles, planner, CPU oracle, portable kernels, CLI,
conformance suite, weight formats) lives on main and reaches branches by merge. Because main does
not edit `crates/velox-variant/` and branches edit nothing else, `git merge main` into a branch does
not conflict. `scripts/check-branch-scope.sh` enforces this and runs in CI on every hardware branch.

Per-card branches drift when they carry changes outside one owned directory; ninfer's
`ninfer-4090` branch reached ~9k lines of divergence before it was folded back. Keeping the
branch surface to one crate is what makes the multi-branch model sustainable.

The variant crate's `Cargo.toml` and `build.rs` are identical on every branch. A branch adds:

- `src/`: `INFO`, `plan_policy`, `cuda_kernels` (the contract in `crates/velox-variant/src/lib.rs`);
- `kernels/*.cu` and `kernels/gencode.txt` (nvcc `-gencode` flags), compiled automatically.

No branch adds dependencies, so `Cargo.lock` is identical across branches too.

## Adding a feature (main first, branches second)

1. **On main**, in one change:
   - add the op to `velox_core::Op` and a method to the `Backend` trait;
   - implement it in `velox-cpu` (the oracle: simple, f64 accumulation);
   - add the portable CUDA kernel in `crates/velox-cuda/kernels/portable.cu`, its wrapper in
     `portable.rs`, and a `CudaKernels` method **with a default body that calls the portable
     wrapper**;
   - add conformance cases (include awkward shapes);
   - `velox selftest` passes on main.
2. **On each hardware branch**: `git merge main`. It builds and passes immediately: the new op runs
   on the portable kernel through the default method. `velox variant` lists it as
   "portable fallback".
3. **Fill in the implementation**: override the method in the branch's kernel set, add the op to
   `native_ops`, then:
   - `velox selftest` passes (correctness against the oracle);
   - `velox bench` and the replay gate show no regression against the previous branch head
     ([performance.md](performance.md)).

A trait method without a default would break every branch at once; only do that for a deliberate
contract change, and then update every branch in the same release.

## Choosing a branch as a user

```bash
velox devices            # e.g. "profile rtx-5090-laptop  tuned branch: rtx-5090-laptop-24g"
git checkout rtx-5090-32g
cargo build --release --features cuda
velox plan --model qwen3-32b --weight-bits 4.5 --kv-bits 8
```

Within a branch the SKU is detected at startup (SM count and VRAM), or forced with
`--profile` / `VELOX_PROFILE`. Running a branch on a card of another architecture is refused
with a message naming the right branch.
