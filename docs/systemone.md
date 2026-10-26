# System One (`POST /v1/systemone`)

ninfer's decision endpoint (Ollama's System One spec plus ninfer's `rank` extension), served by
`velox serve` with the same contract, so llm-inference-daemon and its callers can use either
engine. ninfer's implementation is the reference: `src/serve/systemone.cpp` (requests, prompts,
answers), `systemone_http.cpp` (HTTP) and `ProgramImpl::score_branches` (rank).

## Contract

```
{ "model": "...", "state": "<text>" | {...} | [...], "images": ["<base64>", ...],
  "system": "...", "keep_alive": ..., "questions": { "<name>": {"type": ..., "instructions": ..., ...} } }
```

| type | criteria | assistant prefill | scored tokens | answer |
|---|---|---|---|---|
| `choice` | object, 2-26 options | `Answer:` | `" A"` .. `" Z"` | `choice`, `probabilities`, `confidence` (1 - H/ln N) |
| `noul` | optional `{true, false}` descriptions | `Answer:` | `" true" " True" " false" " False"` | `noul` = P(first two) |
| `score` | array, 2-10 levels | `Answer: (` | `"0"` .. `"9"` | expected `score`, `legend`, `probabilities`, `confidence` |
| `rank` | object, 1-256 candidates; `threshold` (0.5), `limit`, `scale` (2-10 levels) | `[` | after each branch `"<id>] "`: the level digits | `similarities` (expected level / (L - 1), most similar first), `ranking`, `legend`, `threshold` |

Candidate probabilities are renormalized among the scored tokens. Every question is one scoring
request: system, user (images first, then the state and question), and the assistant prefill,
rendered as a continuation of that final assistant message with thinking off
(`ChatTemplate::render_continuation`, the template's `continue_final_message`). Deploy with
ninfer's `qwen3_8.jinja` (`--chat-template`): the artifact's embedded template drops the empty
think block in this case and the model ends the turn instead of answering. A request's questions
go to the engine together (`ChatEngine::score`), which feeds their shared prefix (system, state,
images) once; see below. Response: `{"model", "answers": {name: {"type", ...}}, "usage": {"input_tokens",
"cached_input_tokens", "output_tokens"}}` (`output_tokens`: 1 per question, the candidate count for
`rank`). Errors are `{"error": "<message>"}`. An empty body answers 400, which is how the daemon's
probe tells a build with the endpoint from one without (404).

## Engine path

`ChatEngine::score` takes a request's questions: each a chat request, its candidate texts (one
token each) and its branches (1-5 tokens). The engine renders and tokenizes them, then:

1. **Exact repeats** (the same rendered prompt, image bytes, candidates and branches) are answered
   from the last 256 answers without decoding the images or touching the GPU.
2. **Plans jobs** (`score_plan`), each admitted like a chat request (prefix reuse and the context
   store work as usual):
   - questions without branches that share the start of their prompts are one job: the shared
     prefix (system, state, images) is prefilled once, then each question's rest (text, ending in
     its assistant prefill) is a *window* scored from that state;
   - a single question without branches is prefilled whole, and its candidates are read off the
     prompt's last row (`Qwen35Text::last_logprobs`): no extra pass;
   - a `rank` question prefills its prompt but the last token, and every window is that token
     followed by one branch.
3. **Schedules them ahead of chat prefills** (they finish right after their prompt, while a chat
   prompt is followed by its decode), without the draft model, without context-store roots or
   prompt ends (an exact repeat comes from step 1, and their storage would be pinned again on
   the engine thread before the next request), and on lanes whose last request was a scoring one
   when nothing is reused, so chat lanes keep their device state.

`Qwen35Text::score_windows` feeds the windows of a job, padded to the longest (tokens after a
window's last one do not change its row), as many per forward pass as `--score-branch-rows` x 6
rows hold (100 rank windows, as ninfer), so the weights are read once per pass. Each op takes all
windows of a pass in one call, and the pass runs eagerly (about 15 launches per layer):

- `conv1d_silu_windows` / `gated_delta_net_windows`: every window starts from the session's
  recurrent state, which they only read (scratch outputs, no state written);
- attention rotates every window at the positions after the prompt, stores window `i`'s keys and
  values at cache rows `pos + i * len` (past the prompt; left as after a rejected verify), decodes
  the cache rows to f32 (`kv_load`, as long prefill chunks do) and runs `attention_windows`: each
  row sees the prompt and its own window. The rtx-5090 variant runs the flash prefill schedule over
  f16 operands for it (`AttnPrecision::F16`; the prefill kernel's E4M3 scores move the candidate
  distributions a window is read for); the portable kernel is exact f32;
- one set of GDN rows serves every layer (nothing is replayed), about `rows * 2 * gdn_qkv` floats
  (49 MB for 600 rows on Qwen3.8-27B);
- each window's last row goes through the final norm and the head (16 rows at a time) and
  `log_softmax_gather` reads the candidates' log-probabilities over the public vocabulary.

`hf::tests::score_windows_match_each_window_fed_alone` checks the batched pass against each window
fed alone on the CPU oracle (an f32 and a bf16 cache, windows longer than the old 6-token limit),
and that the session state is untouched; `velox selftest` covers the three window ops.

## Verification

- `velox selftest` covers `log_softmax_gather`, `conv1d_silu_windows`, `gated_delta_net_windows`
  and `attention_windows` against the oracle (on their own seeds, so the cases after them keep
  their data).
- `velox bench-attn --window 5 --pos0 1440 --tokens 500` times `attention_windows`.
- Parity with ninfer: compare the prompt token counts (`usage.input_tokens`) first; a difference
  means the template or tokenization diverged. Probabilities differ by engine numerics, and
  velox's own prefill varies between runs (GEMM reduction order, prefixes restored from bf16
  context-store states): two runs of the same build moved a rank similarity by up to 0.11 and an
  expected score by 0.2. Compare precision choices over many requests, not one.

Measured 2026-10-26 on the RTX 5090 32 GB (rtx-5090-32g + main, production flags, Qwen3.8-27B
NVFP4, NVFP4 KV) against ninfer `master-13879011`, synthetic requests. Alone:

| Request | velox | ninfer |
|---|---:|---:|
| prompt tokens (4 requests) | 2289 / 139 / 289 / 1540 | identical |
| `rank` 100 candidates (pass 74 ms) | 222-236 ms | 260 ms |
| `rank` 10 candidates | 82 ms | 100-117 ms |
| 3 questions with an image | 269 ms | 854 ms |
| 2 text questions | 77 ms | 168 ms |
| exact repeat of a request | 2-4 ms | 15-25 ms |

Beside a chat replay (session `mvg18csx-6nesw9` through the daemon), a unique System One request
every 2 s alternating rank (100, 30, 60 candidates) and 3 questions with an image; velox pooled
over two runs, p50 / p90 / mean in ms:

| Request | velox | ninfer |
|---|---:|---:|
| 3 questions with an image | 290 / 548 / 344 | 971 / 1067 / 950 |
| `rank` 100 | 261 / 515 / 310 | 327 / 540 / 374 |
| `rank` 60 | 387 / 468 / 333 | 317 / 633 / 337 |
| `rank` 30 | 142 / 338 / 195 | 284 / 704 / 323 |
| chat replay TTFT / decode, velox vs ninfer | -6.9 to -8.2% / -0.9 to +1.1% | |

`rank` 60 is bimodal on velox: about 170 ms, or about 400 ms when it arrives during a chat prefill
chunk (up to 2048 tokens, ~300 ms at 18k context), which runs to its end. Capping chat chunks at
1024 tokens while System One traffic is active cut that wait but cost chat prefill 12% at these
context lengths (each chunk decodes the whole cache to f32 in every attention layer), so it was
not kept. Pure chat (A/B/A/B against the same build without this path): PASS, 137/137 outputs
identical, worst change TTFT +0.4% (baseline runs differ by 1.6%).

Open: a scoring request still waits for the chat prefill chunk in flight; the rtx-5090 variant's
`attention_windows` is the only tuned window kernel (others use the exact portable one).
