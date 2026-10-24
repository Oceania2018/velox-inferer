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

## GGUF import (`ggml_block_v1`)

llama.cpp GGUFs of Qwen3.5 / 3.8 (`general.architecture = "qwen35"`) import into `.vi`:

```bash
velox import-gguf --gguf Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf \
  --donor qwen3_8_27b_nvfp4_draftmlp-nvfp4.ninfer --output qwen3_8_27b_gsq_iq3s_dflash2.vi
```

A GGUF carries the text weights only in velox's terms, so a donor artifact of the same model
supplies the text config, tokenizer and chat template, the vision tower, the DFlash2 drafter and
its draft vocabulary head. The donor's text weights and MTP head are dropped.

Quantized tensors keep their GGML blocks byte for byte as format `ggml_<type>` with layout
`ggml_block_v1` (a row is `K / block` blocks; `velox_core::ggml`): 256-value super-blocks `q2_k`,
`q4_k`, `q6_k`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_xs`, and the
smaller blocks `q2_0` (64 values), `iq4_nl` and `q8_0` (32 values). The reference decoders follow
ggml's `dequantize_row_*` and are checked against gguf-py (`q2_0`, which gguf-py lacks, against a
port of the C reference). GGUF `F32` and `BF16` tensors import as `fp32` / `bf16`; `F16` is not
supported (velox has no f16 weight format). The importer only moves whole rows, which is
exact for every type, to undo llama.cpp's conversion rewrites:

| llama.cpp stores | `.vi` stores |
|---|---|
| `1 + w` for every `*norm.weight` except the GDN norm | `w` (velox applies `1 + w`) |
| `ssm_a = -exp(A_log)` | `gdn/a_log` |
| `ssm_conv1d` as `[channels][width]` | `[width][channels]` |
| GDN value heads tiled (`j * k_heads + g`) | grouped (`g * v_per_k + j`) for qkv value rows, conv channels, a/b, a_log, dt_bias |
| `attn_q` with each head's query and gate rows interleaved | query rows, then gate rows |

`attn_gate` (z) rows and `ssm_out` columns stay tiled, because columns cannot move inside a block.
The text config therefore gets `"velox_gdn_value_heads": "tiled"`, and the model transposes the
recurrence output into that order (`Backend::transpose_heads`) before the gated norm. GGML matrices
get `AllowA8` uses (llama.cpp runs them on 8-bit activations), and bf16/fp32 ones `A16Only`.

`Qwen3.8-27B-GSQ-RCO-IQ3_S.gguf` imports to 14.21 GB (11.76 GB of text weights, 11 encodings;
2.45 GB from the donor) and loads 12.12 GB to the device.
