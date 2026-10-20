# Adding AMD (HIP) support

The architecture already reserves AMD. `Vendor::Amd`, `GpuArch::Hip { gfx }` and
`BackendKind::Hip` exist, the RX 7900 XTX is in the catalog, and `velox plan --profile rx-7900-xtx`
works today. What's missing is a backend that executes.

## Steps

1. **`crates/velox-hip`** (on main), mirroring `velox-cuda`:
   - a C ABI shim over the HIP runtime (`hipGetDeviceProperties`, `hipMalloc`, ...);
   - portable kernels generated from `crates/velox-cuda/kernels/portable.cu` with `hipify-perl`.
     The portable kernels stay hipify-clean: no inline PTX, no arch-specific intrinsics, lane
     math through `VELOX_WARP`. RDNA runs wave32 so 32 holds; CDNA (wave64) needs
     `VELOX_WARP` 64 and full-width shuffle masks;
   - `HipKernels` trait with portable defaults and `HipBackend: Backend`;
   - a `velox-build` helper for `hipcc --offload-arch=gfx1100`.
2. **Variant contract**: add `#[cfg(feature = "hip")] pub fn hip_kernels(..)` to the variant
   contract, defaulting to the portable HIP kernels on main. This is a contract change, so every
   branch gets the default in the same release.
3. **CLI**: select `--backend hip` / auto-detect AMD devices; conformance runs unchanged because
   it is generic over `Backend`.
4. **Branch** `rx-7900` (or one per RDNA generation), following [branching.md](branching.md).
