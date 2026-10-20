# Qwen3.5 / Qwen3.8 vision contract

How an image becomes text-model input. Companion to [qwen3_5-text.md](qwen3_5-text.md).

## Preprocessing (still image)

Fixed geometry: patch 16, temporal patch 2, merge 2 (factor 32), 1536 features per patch.
`min_pixels` / `max_pixels` are `preprocessor_config.json` `size.shortest_edge` / `size.longest_edge`
(areas). Mean = std = 0.5, rescale 1/255, bicubic.

```text
reject if max(H,W)/min(H,W) > 200
h = rint(H/32)*32; w = rint(W/32)*32                      # rint: ties to even
if h*w > max_px: b = sqrt(H*W/max_px); h = max(32, floor(H/b/32)*32); w = max(32, floor(W/b/32)*32)
elif h*w < min_px: b = sqrt(min_px/(H*W)); h = ceil(H*b/32)*32; w = ceil(W*b/32)*32
grid = (t=1, gh=h/16, gw=w/16)
```

Resize is Pillow-style antialiased separable bicubic (Keys, a = -0.5): horizontal pass then
vertical; per pass `scale = in/out`, `support = 2*max(scale,1)`, `center = scale*(dst+0.5)`, taps
`int(center-support+0.5) .. int(center+support+0.5)` clamped, weight
`cubic((j - center + 0.5) * (scale >= 1 ? 1/scale : 1))` normalised; each pass rounds half-to-even
to uint8. Pixel value `v/127.5 - 1`. The frame is used twice (temporal 2).

Patch vector `idx = c*512 + t*256 + y*16 + x`. Patch order is merge-group raster:
`for by, bx (merged grid) for my, mx in 0..2: patch (2by+my, 2bx+mx)`.

## Tower (27 blocks, hidden 1152, 16 heads x 72, MLP 4304)

```text
x = W_pe[1152,1536] p + b_pe
x += bilinear(pos_table[48x48,1152]) at (y*47/(gh-1), x*47/(gw-1))   # align-corners; 0 if a side is 1
block: h = LayerNorm(x; eps 1e-6)
       q,k,v = W_qkv h + b (rows Q | K | V, head-major)
       q,k = rope2d(q,k): pair i in 0..36 rotates (x[i], x[i+36]) by pos_axis * 10000^(-2(i%18)/36),
             axis = y for i < 18, x otherwise (raw patch coordinates)
       x += W_o softmax(q k^T / sqrt(72)) v + b_o          # non-causal, whole image
       x += W_fc2 gelu_tanh(W_fc1 LayerNorm(x) + b1) + b2
merger: n = LayerNorm(x); token j = concat(n[4j..4j+4]) (4608);
        out = W2[5120,4608] gelu_erf(W1[4608,4608] token + b1) + b2
```

No deepstack. Weights: patch embedding Q6, attention/MLP Q4/Q5, merger Q8 (groupwise
`row_split_k128_v1`); norms, biases and the position table are direct.

## Prompt integration

`<|vision_start|><|image_pad|><|vision_end|>` with the pad expanded to `(gh/2)*(gw/2)` tokens whose
embeddings are replaced, in order, by the merger rows. MRoPE positions are `[3, T]` (t, h, w):
text tokens advance all axes together; image token (y, x) of the merged grid gets
`(c, c+y, c+x)` and afterwards `c += max(gh/2, gw/2)`. Decode positions are
`cache_pos + rope_delta`, `rope_delta = max_position + 1 - T`.

Default per-item limit 16,384 merged tokens.
