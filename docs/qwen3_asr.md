# Qwen3-ASR contract

How 16 kHz audio becomes text in `velox_models::qwen3_asr`. The reference is the official
`qwen-asr` 0.0.6 transformers code (`modeling_qwen3_asr.py`) with one correction noted under
[Encoder](#encoder-aut). Weights are read straight from the llama.cpp GGUF pair
(`Qwen3-ASR-1.7B-Q8_0.gguf` + `mmproj-Qwen3-ASR-1.7B-Q8_0.gguf`); matrices keep their GGML
blocks.

## Geometry (1.7B)

| | |
|---|---|
| Mel | 128 bins, 16 kHz, n_fft 400, hop 160: 100 frames per second |
| Encoder | 3 x Conv2d(3x3, stride 2, pad 1, 480 ch) -> Linear 7680->1024; 24 pre-LN blocks, d 1024, 16 heads x 64, FFN 4096 (GELU erf); LN; Linear 1024->1024, GELU, Linear 1024->2048 |
| Chunking | `n_window` 50: conv runs per 100-frame chunk (1 s -> 13 tokens). `n_window_infer` 800: attention windows of 8 chunks = 104 tokens |
| Decoder | Qwen3 dense: 28 layers, hidden 2048, 16 q heads / 8 kv heads x 128, q/k RMS norm per head, RoPE theta 1e6 on all 128 dims (rotate-half), SwiGLU 6144, RMS eps 1e-6, untied `output.weight` |
| Tokens | `<|im_start|>` 151644, `<|im_end|>` 151645 (stop), `<|audio_start|>` 151669, `<|audio_end|>` 151670, `<|audio_pad|>` 151676, `<asr_text>` 151704 |

`n_window` / `n_window_infer` are not in the GGUF; they are fixed by the checkpoint
(`config.json` of both 0.6B and 1.7B).

## Log-mel (WhisperFeatureExtractor)

```text
x      = pcm_i16 / 32768                                   # no dither, no normalisation
X      = STFT(x, n_fft 400, hop 160, periodic Hann, center, reflect pad 200)   # frames = len/160 + 1
P      = |X|^2, frames 0 .. len/160 - 1 (the last frame is dropped)
M      = filters[128, 201] . P                            # Slaney mel scale, Slaney area norm, 0..8000 Hz
L      = log10(max(M, 1e-10))
L      = max(L, max(L) - 8)                               # over the whole clip
mel    = (L + 4) / 4                                      # [128, frames]
```

Filters: FFT bin frequencies `linspace(0, 8000, 201)`; 130 edge frequencies equally spaced on the
Slaney mel scale (linear `f / (200/3)` below 1 kHz, `15 + ln(f/1000) / (ln 6.4 / 27)` above),
triangles `max(0, min(-(e_i - f)/(e_{i+1} - e_i), (e_{i+2} - f)/(e_{i+2} - e_{i+1})))`, each
scaled by `2 / (e_{i+2} - e_i)`.

## Encoder (AuT)

```text
chunks     = mel split along time into 100-frame chunks; the last holds frames % 100 (or 100)
pad_to     = 100 if frames >= 100 else frames             # pad_sequence pads to the longest chunk
for each chunk (zero-padded on the right to pad_to frames), image [1, 128 freq, pad_to time]:
  h = gelu(conv1(img)); h = gelu(conv2(h)); h = gelu(conv3(h))      # [480, 16, ceil^3(pad_to)]
  e = conv_out(h permuted to [time, 480 * 16], channel-major: c * 16 + f)   # [t, 1024], no bias
  e += sinusoid[0 .. t]                                    # positions restart in every chunk
  keep the first valid(len) rows, valid(L) = ceil(ceil(ceil(L/2)/2)/2)
x          = concatenation of the kept rows                # n = 13 * (frames / 100) + valid(frames % 100)
windows    = consecutive blocks of 104 rows over x, the last one shorter
for each block:  x += attn(LN1(x)) within its window; x += fc2(gelu(fc1(LN2(x))))   # LN eps 1e-5
out        = proj2(gelu(proj1(LN_post(x))))                # [n, 2048]
```

- **Zero padding is not equivalent to "conv with pad 1".** Padded frames go through conv bias and
  GELU, so conv2 / conv3 rows at the valid edge read non-zero values there. A tail chunk must be
  padded exactly as above to match the reference.
- `sinusoid[t] = [sin(t * w_j) for j < 512] ++ [cos(t * w_j) for j < 512]`,
  `w_j = exp(-j * ln(10000) / 511)`. The GGUF's `a.position_embd.weight` holds the same table.
- Attention is bidirectional **within a window only** (scale 1/8). The transformers eager/SDPA
  path in `qwen-asr` 0.0.6 passes no mask and attends across windows; its FA2 / vLLM path uses
  `cu_seqlens` and is windowed. velox follows the windowed (production) behaviour; the two agree
  for clips up to 8 s.
- Windows are independent, and a chunk's conv output depends only on that chunk, so the rows of a
  completed window never change as audio is appended: a streaming session may keep them.

## Prompt

```text
<|im_start|>system\n{context}<|im_end|>\n
<|im_start|>user\n<|audio_start|>{<|audio_pad|> x n}<|audio_end|><|im_end|>\n
<|im_start|>assistant\n[language {Language}<asr_text>]
```

The system turn is always present (empty context included). The bracketed prefix forces the
language; without it the model writes `language X<asr_text>` itself. Embeddings of the
`<|audio_pad|>` rows are replaced by the encoder output in order. Positions are `0 .. len - 1`
(the checkpoint's MRoPE sections all see the same position for audio-only input, which is plain
RoPE). Decoding is greedy until `<|im_end|>`.

## Verification

`crates/velox-models/tests/qwen3_asr_golden.rs` compares mel, the encoder input after conv,
encoder output, first-step logits and greedy tokens against dumps of the reference run with the
GGUF's dequantized weights (two clips from real calls: 9.37 s over two windows with a partial
tail chunk, and 6.5 s in one window). The dumps are produced by `scripts/qwen3_asr_golden.py` and
are not committed (call audio).

## Serving (`velox serve-asr`)

llama-server's `/v1/chat/completions` as the qwen3-asr sidecar calls it: a WAV in an
`input_audio` part, an optional system message (the bias context) and an optional final
assistant message (`language X<asr_text>`, continued by the model). The content returned is that
prefix plus the transcription, as llama-server returns it. The prompt is the processor's template
above: the system turn is written even when empty. llama-server drops it without a system message
(the GGUF's template); on 52 call segments that alone moved 24% of words away from the reference.

- Request threads decode the WAV (16-bit PCM, mixed to mono, resampled to 16 kHz), compute the
  log-mel and tokenize. One engine thread owns the device.
- Q8_0 matrices are relaid as `q8_g32_fp16` (same codes and per-32 scales; lossless), which the
  5090 variant runs on its tuned kernels. Activations may be int8 / bf16 there (`AllowA8`);
  `VELOX_ASR_A16` keeps them exact, `VELOX_ASR_GGML` keeps the GGML blocks.
- Decoding sequences share one F32 KV pool (`--kv-rows`, 229 KB per row for 1.7B): each request
  reserves its prompt plus `min(max_tokens, 6 x seconds + 32)` output rows, first fit; a request
  that does not fit yet waits. A request prefills in a reused scratch session (prefill KV kept
  contiguous for the oracle-checked path), its rows are copied into its pool range, then it joins
  the batch.
- Every step advances all active requests at once (`step_batch`), recorded as a static graph per
  (batch size, KV bucket). q | k | v and gate | up are stacked by rows at load (row views keep the
  session path), so a layer is 9 kernels: the q|k|v GEMV, `qk_norm_rope_kv` (per-head norms,
  rotary from a cos / sin table `rope_cos_sin` computes once per pass, K / V straight into the
  pool), `attention_ranges`, o, `add_rms_norm`, the gate|up GEMV, `swiglu`, down, and the
  residual add fused into the next layer's norm. (Before: 17 per layer, and `rope_positions`
  recomputing its angles in f64 twice per layer; 3.0 -> 2.5 ms per batch-1 step.)
- Between steps, up to 8 waiting requests are admitted together: one `encode_many` pass (chunks
  of all clips through the conv stack, rows through the encoder, windows inside each clip) and
  one `prefill_many` pass straight into the pool (prefill row `j` of a request is an
  `attention_ranges` query over its rows `0 ..= j`), within 512 prompt rows. A longer prompt, or a
  clip under one second, is admitted alone. The encoder's windowed attention is one
  `attention_ranges` call per layer (each token's range is its window).
- On the 5090 the row-split GEMM can split K when the grid is short of the card (prefill groups of
  ~250 rows on 1024- and 2048-row weights); slices sum in order, so results stay deterministic.
  Opt-in with `VELOX_SPLITK=1` (the qwen3-asr-velox service sets it): other models on that kernel
  (the 27B's vision tower and drafter) keep the unsplit kernel unchanged.
- `VELOX_ASR_PHASES=1` prints, whenever the engine goes idle, wall time admitting (encode +
  prefill) vs decoding by batch size. At 16 requests admission was 75% of the time before the
  split-K and encoder changes.
- Tried and dropped: an int8 (dp4a, Int8G32) kernel for q8_g32 at 2-16 tokens. End to end it was
  ~10% slower at 16 requests than the variant's bf16 skinny kernel and moved 3-6 of 52 segments
  off the reference.

### Results (RTX 5090 shared with the 27B, 2026-11-05)

52 segments (497 s) cut from two real calls, identical requests to each server, Q8_0 weights;
paired runs (each server restarted, two rounds each, alternating):

| | conc 1 | conc 4 | conc 16 |
|---|---|---|---|
| llama.cpp b11347 (`-np 1`) | 43.8 / 48.5x realtime, p50 0.22 / 0.19 s | 109.0 / 126.8x | 127.6 / 127.8x |
| velox serve-asr | 87.9x, p50 0.11 s | 161.0 / 164.1x | 257.5 / 271.5x |
| velox, + split-K, encoder ranges, fused decode | 110.5 / 112.1x, p50 0.086 s | 225.5x | 354-367x |

The first run after a restart is slower at conc 1 (graph recordings; 57-70x). Through the
qwen3-asr sidecar (`ASR_DECODE_CONCURRENCY=16`), 20 concurrent realtime streams of distinct audio
for 9 minutes: 20 / 20 complete, final latency p50 0.88 s, p90 1.71 s, p99 4.4 s (llama.cpp behind
the stock sidecar: 1 / 20 complete, the rest dropped on keepalive timeouts, p90 17.7 s). With the
fused build, 32 streams (3 minutes, staggered over 5 s): 32 / 32 complete, 1060 finals, p50
0.76 s, p90 1.34 s, p99 2.27 s, no drift.

Against the reference (transformers, fp32, official prompt) on the same segments: velox 50 / 52
identical, 0.12% of words differ (int8 and exact activations alike); llama.cpp 22 / 52, 12.5%.
Segments that flip between velox builds or batch compositions (18, 23, 27, 36 of the set) are
near-ties: top-2 logit gaps of 0.001-0.04 at the deciding step (36: "." vs "," after "ten", 0.001),
below f32 noise; the f64 golden check is unchanged across builds (enc 4-5e-3, logits ~3e-3).

Deployment note (Windows / WDDM): the 27B sizes itself device-wide at startup, so start
serve-asr first; started after the 27B it oversubscribes VRAM, the driver pages to system RAM and
serve-asr runs ~3x slower without any error.
