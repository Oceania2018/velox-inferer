# Qwen4-Exp text model contract (Qwen3.8-Flash-Next)

The mathematics velox implements for `qwen4_exp` (`Qwen4ExpForConditionalGeneration`,
Qwen3.8-Flash-Next), as llama.cpp GGUFs store it (`general.architecture = "qwen4exp"`). Derived
from the public model definition (transformers `modeling_qwen4_exp.py`) and llama.cpp's
`src/models/qwen4exp.cpp`, which is the reference for the GGUF layout because the GSQ-RCO
quantizations were evaluated with it. velox's kernels are independent implementations validated
against the CPU oracle. Layouts are row-major; `[T, D]` is token-major.

## Geometry (Flash-Next)

H = 2560, 48 layers. Layer `i` is QSA attention when `(i + 1) % 4 == 0` (12 layers) and Gated
DeltaNet otherwise (36 layers). Attention: 24 query heads, 2 KV heads, head_dim 256, partial
interleaved MRoPE (64 rotary dims, theta 1e7). GDN: 16 key heads, 48 value heads, Dk = Dv = 128,
conv width 4. MoE in every layer: 512 experts, top-10, expert width 640, one shared expert of
width 640. Hyper-connections: S = 4 residual streams, low rank 320. PLE on layer 1. Vocabulary
248,320. eps = 1e-6. The GGUF has no MTP block; one can be added as a sidecar ([MTP](#mtp-drafting)).

## Residual streams (hyper-connections)

The residual is `R [T, S, H]`, initialised to S copies of the token embedding. Every mixer and FFN
reads a mix of the streams and writes back into all of them:

```text
hc_pre(R; g, W_down [r, S H], W_up [S H, r], W_inj [S, S H]):
  xn[s]  = plain_rmsnorm(R[s]) * g[s]                 # one norm per stream, g is [S, H]
  lo     = silu(W_down xn / S)                         # [r]
  mixed  = (1/S) sum_s xn[s] * sigmoid((W_up lo)[s])   # [H], the block input
  inj    = W_inj xn                                    # [S] (absent for the head mixer)
hc_post(R, y, inj):
  R[s]  += y * 2 sigmoid(inj[s] / S)
```

```text
R = repeat(embed(token), S)
for each layer:
  if PLE layer: R += ple(R, ids)
  m, inj = hc_pre(R; attn_*);  R = hc_post(R, mixer(m), inj)       # GDN or QSA attention
  m, inj = hc_pre(R; ffn_*);   R = hc_post(R, moe(m), inj)
logits = output(hc_pre(R; output_hc_*).mixed)                     # no final norm: the mixer is it
```

velox folds the `1/S` scales into `W_down` and `W_inj` at load (exact for bf16: S is a power of
two), so the device computes `silu(W_down' xn)` and `2 sigmoid(W_inj' xn)`.

Ops: `rms_norm_groups` (the per-stream norm), `silu`, `hc_mix` (the sigmoid-weighted stream mean),
`hc_combine` (the scatter back into the streams).

## Gated DeltaNet

As Qwen3.5 ([qwen3_5-text.md](qwen3_5-text.md#gated-deltanet)), except the output gate is a
sigmoid: `y = out_proj(plain_rmsnorm_per_head(o, gdn_norm) * sigmoid(z))`.

## QSA attention (`indexed_attention`)

Gated attention as Qwen3.5 (query and gate rows, per-head q/k norms, partial rope, sigmoid output
gate, KV cache of normalised+rotated K and raw V), but each query attends only to the tokens a
lightweight indexer selects. With ratio `c = 4`, budget `B = 2048`, 4 index heads of `Di = 128`:

```text
k_raw[p]   = W_ik h[p]                                   # cached per position, no norm or rope
pooled[b]  = rope(plain_rmsnorm(mean(k_raw[c b .. c b + c])) * k_norm, pos = c b)   # block b
q_i[t]     = rope(plain_rmsnorm_per_head(W_iq h[t]) * q_norm, pos = p)              # [4, Di]
# token t at position p sees positions 0..=p; nc = (p + 1) / c complete blocks
score[t,b] = sum_h relu(q_i[t,h] . pooled[b]) / sqrt(Di)  for b < nc
selected   = members of the top min(B / c, nc) blocks by score (ties to the lower block)
             + the tail c nc ..= p
a[t]       = softmax attention of q[t] over the selected positions only
```

The index rope is the same partial rotate-half rope (64 of the 128 dims, theta 1e7) at the block's
first position. When `nc <= B / c` (`p + 1 < B + c`) every position is selected and QSA is exact
dense causal attention, so only long contexts take the sparse path.

Ops: `qsa_pool` (pooled keys of completed blocks into a per-layer pooled cache), `qsa_select`
(scores, top blocks and tail as position lists), `attention_sparse` (attention over the listed
positions of a KV cache in any [`KvFormat`]).

## Sparse MoE

```text
p      = softmax(router h)                  # [512]
(w, e) = top_k(p, 10)                        # ties to the lower expert id
w      = w / sum(w)
y      = sum_j w_j down_e(silu(gate_e h) * up_e h) + sigmoid(shared_score . h) shared(h)
```

The GGUF's `ffn_gate_inp` `[512, H]` and `ffn_gate_inp_shexp` `[H]` (both bf16) are loaded as one
parent `[513, H]`, so one linear yields the router logits and the shared expert's gate.

Ops: `moe_route` (softmax, top-k and renormalisation on the device, optionally counting each
expert's selections), `moe_experts` (the routed sum over an [`ExpertBank`]),
`sigmoid_scale_rows` (the shared gate).

### Expert offload

A layer's routed experts (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) are 24,576 experts of
about 1.4 MiB in the IQ2_XS GGUF (35.5 GB), so they do not fit a 16 GB card. An `ExpertBank` keeps
every expert's record (gate rows, up rows, down rows, each GGML-encoded as in the GGUF) in system
RAM that the device reads directly (pinned and mapped: `Backend::alloc_host`), plus a pool of
device slots holding copies of some experts and a map `expert -> slot or -1`. Kernels read each
selected expert from its slot when it has one and from system RAM otherwise; both are exact copies,
so residency never changes results, only speed.

The model decides residency (`velox_models::qwen4_exp::ExpertCache`): `moe_route` counts
selections per expert; between steps the cache promotes the most selected non-resident experts
into the least selected slots. A swap is three stream-ordered writes (the evicted expert's map
entry to -1, the record into its slot, the promoted expert's map entry), so steps already queued
read either the old or the new placement consistently.

Measured on the RTX 5060 Ti (PCIe 5.0 x8, WSL2): 36 GB of pinned mapped memory allocates, kernels
stream it at ~26 GB/s with 16-byte loads (one thread block per 1.5 MiB expert already saturates
it), the same as `cudaMemcpyAsync`.

## PLE (per-layer n-gram embedding), layer 1

```text
ctx[t]   = (tok[t], tok[t-1], tok[t-2]) with every token at or before an EOS (and before the
           sequence start) replaced by EOS; tok[t] itself is never replaced
mix_n    = (ctx0 m0) ^ (ctx1 m1) ^ ... ^ (ctx_{n-1} m_{n-1})       # u64 wrapping, n = 2, 3
row[h]   = mix_n % vocab[h] + offset[h]   for the 8 heads of each n   # 16 heads
e        = concat_h table[row[h]]                                      # [16 x 160] = [H]
key      = W_key e [S H];  value = W_value e [H]
kn[s]    = plain_rmsnorm(key[s]) * norm_key[s];  qn[s] = plain_rmsnorm(R[s]) * norm_query[s]
g[s]     = (kn[s] . qn[s]) / sqrt(H);  gate[s] = sigmoid(sign(g) sqrt(max(|g|, 1e-6)))
gv[s]    = value * gate[s]
c        = silu(dilated causal depthwise conv(plain_rmsnorm_groups(gv) * norm_conv;
                                              width 4, dilation 3))  # over tokens, per channel
R       += gv + c
```

`m`, `vocab`, `offset` come from the GGUF metadata (`qwen4exp.ple.*`). The table
(`per_layer_token_embd`, IQ4_NL `[320,001,536, 160]`, 28.8 GB, the second GGUF shard) stays in the
mapped file: the host hashes, reads the 16 rows of each token, decodes them and uploads `e`. The
conv state holds the last 9 normalised rows; the n-gram context needs the previous two tokens.
Image positions hash as the image token id.

Ops: `ple_gate` (the per-stream gate and the gated values), `conv1d_dilated_silu`.

## MTP drafting

The published checkpoint has one MTP layer (`mtp.*`). velox reads it from a sidecar GGUF named as
llama.cpp names layer `L = 48` (`blk.48.nextn.{fc_e,fc_h,enorm,hnorm,hc_head_*}`,
`blk.48.{hc_attn_*,hc_ffn_*,attn_*,indexer.*,ffn_*}`; norms stored as 1 + w, routed experts
IQ4_NL). `velox generate --mtp <sidecar>` loads it with the model's shards.

```text
entry p (h_p = the main residual [S, H] after the last layer at position p, x = token p + 1):
  R[s] = fc_h(rmsnorm(h_p[s]) * hnorm[s]) + fc_e(rmsnorm(embed(x)) * enorm)
  one QSA layer (own KV / index caches, entry p at position p) and one MoE layer, as the stack
  draft(p + 2) = argmax output(hc_pre(R; nextn.hc_head_*).mixed)
```

Every kept position gets its entry: prompt chunks (shifted by one token) right after their main
step, then each round's committed tokens. A round verifies `[next, draft]` as one two-row window
that leaves the recurrent states (GDN conv/SSM, PLE conv history) untouched and records the rows
`commit` replays (`gdn_replay`) for the kept prefix; KV and index rows past the kept prefix are
rewritten before they are read. Greedy output is identical to plain decoding. The MTP layer's
experts are read from system RAM (no device slots).

## GGUF mapping

Weights load directly from the GGUF shards (no `.vi` import: the experts and the PLE table are
used in place). The tokenizer, chat template and generation config come from the Hugging Face
model directory (`tokenizer.json`, `chat_template.jinja`, `generation_config.json`).

llama.cpp's converter rewrites tensors the same way as for Qwen3.5 (see
[weights.md](weights.md#gguf-import-ggml_block_v1)): every norm the model applies as `1 + w`
(including the PLE and indexer norms) is stored as `1 + w`, so velox applies all GGUF norms as
plain norms with the stored weight; `ssm_a = -exp(A_log)`; `ssm_conv1d` and `ple_conv1d` are
`[channels][width]`; GDN value heads are tiled; `attn_q` interleaves each head's query and gate.
The loader undoes the GDN and attention orders by moving whole rows, as the Qwen3.5 import does.
