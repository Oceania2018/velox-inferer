# Weight files: `.vi` and `.ninfer`

velox reads two containers through one reader (`crates/velox-weights`):

| Container | Magic (entry / part) | Role |
|---|---|---|
| `.vi` v1 | `VELOXVI\x01` / `VELOXPT\x01` | velox's native format |
| `.ninfer` v3 | `NINFER\0\x03` / `NINPRT\0\x03` | read directly for compatibility |

## `.vi` v1

`.vi` v1 deliberately uses ninfer v3's directory model, so existing artifacts convert without
re-quantising:

- 32-byte header (magic, little-endian `json_bytes`, 16-byte `artifact_id`), JSON directory,
  payload from `align_up(32 + json_bytes, 4096)`; parts carry a 32-byte header and payload at 4096;
- root fields `components`, `objects`, `bindings`, `uses`, `files`, optional `metadata`,
  `provenance`, plus the `.vi`-only `velox` record
  (`format_version`, `source_container`, `source_artifact_id`);
- objects, format/layout names (`bf16`, `nvfp4`, `fp8_e4m3fn_row_bf16`, `q4_g64_fp16`, ...),
  binding parts and uses have the same meaning as in ninfer's v3 spec
  (`docs/maintainer/artifact-container.md` in ninfer).

Later `.vi` versions may add velox-specific layouts (for example per-branch pre-swizzled
weights). Such layouts are format additions on main; a branch's kernels decide which layouts they
consume.

## Commands

```bash
velox inspect --artifact qwen3_8_27b_nvfp4.ninfer      # validate structure, summarise formats
velox convert --input qwen3_8_27b_nvfp4.ninfer --output qwen3_8_27b_nvfp4.vi
velox convert --input model.vi --output model.ninfer --to ninfer
```

`inspect` checks framing, the file table and part headers, object order/alignment/overlap, and
every binding, use, resource and component reference. `convert` copies the payload unchanged,
writes to temporary files, and renames parts before the entry, so a visible entry always has its
parts. It never overwrites an existing output.

## What the production artifact needs

`velox inspect` on `qwen3_8_27b_nvfp4_draftmlp-nvfp4.ninfer` (23.27 GB, text + vision + MTP +
DFlash2): 1,246 objects, 1,513 bindings, 844 uses. Bytes by format:

| Format / layout | Objects | GB |
|---|---:|---:|
| `fp8_e4m3fn_row_bf16` / `row_scale_v1` | 146 | 11.90 |
| `nvfp4` / `block_scale_k16_m128x4_v1` | 117 | 8.92 |
| `q8_g32_fp16` / `row_split_k128_v1` | 23 | 1.39 |
| `q4_g64_fp16` / `row_split_k128_v1` | 55 | 0.49 |
| `bf16` / `contiguous_le_v1` | 579 | 0.45 |
| `q5_g64_fp16`, `q6_g64_fp16`, `int32`, `fp32`, resources | | 0.13 |

These are the weight matmul kernels velox must implement to run that model natively.
