# Hardware branches

velox-inferer ships one build per hardware family. Users pick the branch that matches their GPU;
`velox devices` names it for each detected card.

| Branch | Profiles | Arch | Notes |
|---|---|---|---|
| `main` | any | portable (`sm_89`, `sm_120` + PTX) | Generic variant: correct everywhere, tuned nowhere. All feature work lands here. |
| `rtx-5090-32g` | `rtx-5090` (32 GB desktop) | `sm_120a` | Primary implementation branch (170 SMs). |
| `rtx-5090-24g` | `rtx-5090-laptop` (24 GB) | `sm_120a` | 82 SMs, larger VRAM reserve. |
| `rtx-5060-ti` | `rtx-5060-ti-16g`, `rtx-5060-ti-8g` | `sm_120a` | Small-SM (36) shapes. |
| `rtx-4090` | `rtx-4090` | `sm_89` | Ada: no TMA, no FP4. |

AMD cards are already in the profile catalog; they need the HIP backend first
([adding-amd.md](adding-amd.md)), then an AMD branch on the same model.

## The one rule: disjoint ownership, so merges never conflict

Main and the hardware branches own disjoint files:

| Owner | Files |
|---|---|
| hardware branch | `crates/velox-variant/src/`, `crates/velox-variant/kernels/` |
| main | everything else, including the variant crate's `Cargo.toml` and `build.rs`, and `Cargo.lock` |

Main never edits `crates/velox-variant/{src,kernels}` after the commit that introduced them
(`7c7f18c`), and a branch never edits anything else. Since no file is changed on both sides,
`git merge main` into a branch cannot conflict. Enforcement:

- `scripts/check-main-scope.sh` fails if main has touched the branch-owned directories;
- `scripts/check-branch-scope.sh` fails if a branch has touched anything else;
- `.githooks/pre-commit` runs the matching check on every commit
  (enable once per clone: `git config core.hooksPath .githooks`);
- `.github/workflows/scope.yml` runs it in CI, and on branches also test-merges main.

If a conflict ever appears anyway, **main wins**. `scripts/sync-branches.sh` merges main into every
hardware branch with `-X theirs` and resets any out-of-scope file on the branch to main's version.

### Evolving the contract without touching branch files

Because main cannot edit `crates/velox-variant/src`, every extension point a branch implements
lives in a main-owned crate and is additive:

- New kernel hooks are `CudaKernels` methods **with a default body** (the portable kernel).
  `impl CudaKernels for PortableKernels {}` in velox-cuda is the compile-time check that every
  method has one.
- Existing signatures that branches call or implement (`CudaKernels`, `velox_cuda::portable`,
  `status`, `CudaBuffer`, `velox_core` types) are not changed in place. Add the new form next to
  the old one, move branches over, then retire the old one in a later release.
- Per-branch knobs belong in methods with defaults (e.g. `linear_act`, `attention_precision`),
  not in new fields of the `VariantInfo` literal or new free functions in the variant crate.

A hardware branch therefore never needs a design that disagrees with main: anything shared is
proposed on main first, and the branch only overrides.

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
velox devices            # e.g. "profile rtx-5090-laptop  tuned branch: rtx-5090-24g"
git checkout rtx-5090-32g
cargo build --release --features cuda
velox plan --model qwen3-32b --weight-bits 4.5 --kv-bits 8
```

Within a branch the SKU is detected at startup (SM count and VRAM), or forced with
`--profile` / `VELOX_PROFILE`. Running a branch on a card of another architecture is refused
with a message naming the right branch.
