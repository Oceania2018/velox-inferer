# Qwen3.5 / Qwen3.8 text model contract

The mathematics velox implements for `Qwen3_5ForCausalLM` (Qwen3.6/3.8-27B Dense), as stored in
`.ninfer` v3 / `.vi` v1 artifacts. Layouts are row-major: `[T, D]` means token-major rows.
Derived from the public model definition and ninfer's documented op contracts; velox's kernels
are independent implementations validated against the CPU oracle.

## Geometry (27B)

H = 5120, 64 layers. Layer `i` is full attention when `i % 4 == 3` (16 layers) and Gated DeltaNet
(GDN) otherwise (48 layers); the config's `layer_types` is authoritative. Attention: 24 query
heads, 4 KV heads, head_dim 256. GDN: 16 key heads, 48 value heads, Dk = Dv = 128, conv width 4.
FFN 17408. Vocabulary rows 248,320 (sampling restricted to the tokenizer's ids). eps = 1e-6.

## Norms

```text
offset_rmsnorm(x, w) = x * rsqrt(mean(x^2) + eps) * (1 + w)   # input, post-attention, final, q/k norms
plain_rmsnorm(x, w)  = x * rsqrt(mean(x^2) + eps) * w         # GDN output norm only
```

q/k norms (`[256]`) apply per head; the GDN norm (`[128]`) applies per value head.

## Block

```text
h = offset_rmsnorm(x, input_norm);        x += mixer(h)
h = offset_rmsnorm(x, post_attention_norm); x += down(silu(gate(h)) * up(h))
```

## Gated attention

```text
q = query(h) [T,24,256]   g = gate(h) [T,24,256]   k = key(h) [T,4,256]   v = value(h) [T,4,256]
q = offset_rmsnorm_per_head(q, query_norm);  k = offset_rmsnorm_per_head(k, key_norm)
q, k = partial_rope(q, k)                  # gate is neither normed nor rotated
a[t,h] = causal softmax(q[t,h] . k[s,h/6] / sqrt(256)) . v[s,h/6]
x += output(a * sigmoid(g))
```

The KV cache holds normalised+rotated K and raw V.

## Partial interleaved MRoPE

Rotary dims R = head_dim * 0.25 = 64 (32 pairs); dims [64, 256) pass through. NeoX rotate-half
inside the rotary slice, theta = 1e7:

```text
phi_i = pos[axis(i), t] * theta^(-2i/64),  axis(i) = i % 3   (sections [11,11,10])
y[i]    = x[i] cos(phi_i) - x[i+32] sin(phi_i)
y[i+32] = x[i+32] cos(phi_i) + x[i] sin(phi_i)
```

For text all three axes equal the token position, which is ordinary partial RoPE. With images,
an image of merged grid gh x gw gives token (y, x) positions (c, c+y, c+x), then c += max(gh, gw);
decode positions add `rope_delta = max_pos + 1 - T` to the cache index.

## Gated DeltaNet

```text
qkv = in_proj(h) [T, 10240] = q 2048 | k 2048 | v 6144;  z = z(h) [T, 6144];  a, b = a(h), b(h) [T, 48]
u = [conv_state (3 oldest->newest); qkv]
c[t, ch] = silu(sum_{j=0..3} w[j, ch] * u[t + j, ch])   # w is [4, 10240]; j = 3 is the current token
conv_state = last 3 rows of u
for value head h (q/k head h / 3), per token:
  g = -exp(A_log[h]) * softplus(a + dt_bias[h]);  alpha = exp(g);  beta = sigmoid(b)
  q, k = l2norm(q), l2norm(k)       # x / sqrt(sum(x^2) + 1e-6)
  S = alpha * S                     # S[h] is [Dv, Dk], FP32, zero at sequence start
  S += outer(beta * (v - S k), k)
  o = (S q) / sqrt(128)
y = out_proj(plain_rmsnorm_per_head(o, gdn_norm) * silu(z))
```

## Sparse MoE (`Qwen3_5MoeForCausalLM`)

Same stack, attention, GDN, norms and RoPE; only the block FFN changes. Every block is MoE.
Geometry of Qwen3.6-35B-A3B (from ninfer's model reference): H = 2048, 40 layers (10 attention,
30 GDN; GDN 16 key / 32 value heads), attention 16 query / 2 KV heads of 256; 256 routed experts,
top-8, expert width 512; one shared expert of width 512. The config carries `num_experts`,
`num_experts_per_tok`, `moe_intermediate_size`, `shared_expert_intermediate_size` and no
`intermediate_size`.

```text
p      = softmax(router(h))                 # [E], router is [E, H]
(w, e) = top_k(p, K)                         # ties to the lower expert id
w      = w / sum(w)                          # always renormalised over the selection
routed = sum_j w_j * down[e_j](silu(gate[e_j](h)) * up[e_j](h))
y      = routed + sigmoid(shared_score . h) * shared_down(silu(shared_gate(h)) * shared_up(h))
```

The shared expert always runs and is outside the top-k. Bindings and the layout ninfer's
converter packs them into:

| Binding (`text/layers/i/` + ...) | Shape | Stored as |
|---|---|---|
| `moe/router`, `moe/shared_score` | `[E, H]`, `[1, H]` | one parent `[E + 1, H]` (BF16) |
| `moe/experts/e/gate`, `.../up` | `[I, H]` each | one parent `[E * 2I, H]`: gate at row `2eI`, up at `(2e + 1)I` (Q4 G64) |
| `moe/experts/e/down` | `[H, I]` | one parent `[E * H, I]`, expert at row `eH` (Q5/Q6 G64) |
| `moe/shared/{gate,up,down}` | `[I_s, H]`, `[I_s, H]`, `[H, I_s]` | as a Dense MLP |

The loader accepts any layout in which each role's experts sit at a constant row stride of one
parent (`ExpertBank`), and refuses others. MTP (`Qwen3_5MoeMTP`) uses the same MoE block.

Planned op contract (not implemented; lands with CPU reference, portable kernels and conformance):

- `moe_route(logits [T, row_len], experts, k) -> ids [T, k], weights [T, k]` on the device: the
  softmax / top-k / renormalisation above over the first `experts` columns (the router and
  shared-score rows run as one linear, so `row_len = E + 1`). Routing stays on the device so
  recorded graph steps can replay it.
- `moe_experts(x, ids, weights, gate, up, down banks) : y += routed`, with an `ActQuant` report
  like `linear_act` for tuned low-precision kernels. Separate from routing so conformance can
  feed both sides identical routes (near-tie logits may otherwise select different experts).
- `sigmoid_scale_rows`: the shared expert's per-token scalar gate.

## Embedding, head, precision

The embedding row is the dequantised table row (FP8: code x row scale); the head is
`output_head(offset_rmsnorm(x, final_norm))`, untied. ninfer rounds activations to BF16 at every op
output (residual stream included) and keeps g, beta and GDN state in FP32. velox's portable path
computes in FP32 throughout, so results agree closely but not bitwise.

## Frontend

HF `tokenizer.json` (byte-level BPE), the artifact's Jinja `chat_template.jinja`
(`enable_thinking=false` renders `<think>\n\n</think>\n\n` after the assistant header), XML tool
calls `<tool_call><function=NAME><parameter=P>VALUE</parameter></function></tool_call>`, stop on
`generation_config.json` `eos_token_id`.
