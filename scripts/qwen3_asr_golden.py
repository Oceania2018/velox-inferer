"""Golden data for velox's Qwen3-ASR: HF reference modeling code run with the GGUF's own
(dequantized Q8_0) weights, so velox and the reference see identical parameters.

Writes to OUT_DIR:
  pcm.i16         16 kHz mono int16 input (the test feeds exactly these samples)
  mel.f32         [128, frames] log-mel (WhisperFeatureExtractor)
  conv.f32        [tokens, 1024] encoder input after conv_out + positions
  enc.f32         [tokens, 2048] audio embeddings (after proj2)
  prompt.u32      prompt token ids (audio pads included)
  logits0.f32     [vocab] logits of the first generated position
  tokens.u32      greedy generated ids (until <|im_end|>)
  meta.json       shapes, text
  layers.f32      [24, tokens, 1024] each encoder block's output

Usage: QWEN3_ASR_REF=<dir> python -I qwen3_asr_golden.py TEXT_GGUF MMPROJ_GGUF WAV START_S DUR_S OUT_DIR

QWEN3_ASR_REF holds the reference pieces (none are redistributed here):
  tb/                 qwen-asr 0.0.6 `core/transformers_backend/` (configuration_, modeling_,
                      processing_qwen3_asr.py) plus an empty __init__.py
  config-1.7b.json    Qwen/Qwen3-ASR-1.7B config.json
  tok/                vocab.json, merges.txt, tokenizer_config.json of the same repo
Needs torch 2.11 (CPU), transformers 4.57.6, gguf, numpy. WAV may be stereo / any rate: it is
mixed to mono and linearly resampled to 16 kHz, and the exact samples used are written to pcm.i16.
"""
import json, os, sys, wave
import numpy as np
import torch
import gguf
from gguf.quants import dequantize

HERE = os.environ["QWEN3_ASR_REF"]
sys.path.insert(0, HERE)
from tb.configuration_qwen3_asr import Qwen3ASRConfig  # noqa: E402
from tb.modeling_qwen3_asr import Qwen3ASRForConditionalGeneration  # noqa: E402
from transformers import WhisperFeatureExtractor, Qwen2TokenizerFast, Qwen2Tokenizer  # noqa: E402

TEXT, MMPROJ, WAV, START, DUR, OUT = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
os.makedirs(OUT, exist_ok=True)
torch.manual_seed(0)


def tensors(path):
    r = gguf.GGUFReader(path)
    out = {}
    for t in r.tensors:
        a = dequantize(t.data, t.tensor_type).astype(np.float32)
        shape = [int(x) for x in reversed(t.shape)]  # numpy order
        out[t.name] = a.reshape(shape)
    return out


def hf_state(text, mm, layers_a, layers_t):
    s = {}
    A, T = "thinker.audio_tower.", "thinker.model."
    for i in (1, 2, 3):
        s[f"{A}conv2d{i}.weight"] = mm[f"a.conv2d.{i}.weight"]
        s[f"{A}conv2d{i}.bias"] = mm[f"a.conv2d.{i}.bias"].reshape(-1)
    s[A + "conv_out.weight"] = mm["a.conv_out.weight"]
    for n in range(layers_a):
        p, g = f"{A}layers.{n}.", f"a.blk.{n}."
        for hf, gg in [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                       ("self_attn.out_proj", "attn_out"), ("fc1", "ffn_up"), ("fc2", "ffn_down"),
                       ("self_attn_layer_norm", "ln1"), ("final_layer_norm", "ln2")]:
            s[p + hf + ".weight"] = mm[g + gg + ".weight"]
            s[p + hf + ".bias"] = mm[g + gg + ".bias"]
    s[A + "ln_post.weight"], s[A + "ln_post.bias"] = mm["a.post_ln.weight"], mm["a.post_ln.bias"]
    s[A + "proj1.weight"], s[A + "proj1.bias"] = mm["mm.a.mlp.1.weight"], mm["mm.a.mlp.1.bias"]
    s[A + "proj2.weight"], s[A + "proj2.bias"] = mm["mm.a.mlp.2.weight"], mm["mm.a.mlp.2.bias"]
    s[T + "embed_tokens.weight"] = text["token_embd.weight"]
    for n in range(layers_t):
        p, g = f"{T}layers.{n}.", f"blk.{n}."
        for hf, gg in [("self_attn.q_proj", "attn_q"), ("self_attn.k_proj", "attn_k"), ("self_attn.v_proj", "attn_v"),
                       ("self_attn.o_proj", "attn_output"), ("self_attn.q_norm", "attn_q_norm"),
                       ("self_attn.k_norm", "attn_k_norm"), ("mlp.gate_proj", "ffn_gate"), ("mlp.up_proj", "ffn_up"),
                       ("mlp.down_proj", "ffn_down"), ("input_layernorm", "attn_norm"),
                       ("post_attention_layernorm", "ffn_norm")]:
            s[p + hf + ".weight"] = text[g + gg + ".weight"]
    s[T + "norm.weight"] = text["output_norm.weight"]
    s["thinker.lm_head.weight"] = text["output.weight"]
    return {k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in s.items()}


def load_pcm(path, start, dur):
    w = wave.open(path)
    sr, nch = w.getframerate(), w.getnchannels()
    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, nch).astype(np.float64).mean(1)
    t_new = np.arange(int(round(dur * 16000))) / 16000 + start
    y = np.interp(t_new, np.arange(len(x)) / sr, x)
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


cfg_json = json.load(open(os.path.join(HERE, "config-1.7b.json")))
cfg = Qwen3ASRConfig(**cfg_json)
cfg.thinker_config.audio_config._attn_implementation = "eager"
cfg.thinker_config.text_config._attn_implementation = "eager"
tc = cfg.thinker_config
# config.json says bfloat16 and transformers creates the parameters in it: widen BEFORE loading,
# or load_state_dict rounds every weight to bf16 (that alone is ~2e-4 per encoder block).
model = Qwen3ASRForConditionalGeneration(cfg).eval().double()
state = hf_state(tensors(TEXT), tensors(MMPROJ), tc.audio_config.encoder_layers, tc.text_config.num_hidden_layers)
missing, unexpected = model.load_state_dict(state, strict=False)
missing = [m for m in missing if "rotary" not in m and "positional" not in m]
assert not missing and not unexpected, (missing, unexpected)
assert all(p.dtype == torch.float64 for p in model.parameters())

pcm = load_pcm(WAV, START, DUR)
pcm.tofile(os.path.join(OUT, "pcm.i16"))
fe = WhisperFeatureExtractor(feature_size=128, sampling_rate=16000, hop_length=160, n_fft=400, chunk_length=30)
feats = fe(pcm.astype(np.float32) / 32768.0, sampling_rate=16000, padding=True, truncation=False,
           return_attention_mask=True, return_tensors="pt")
mel = feats["input_features"][0].double()  # [128, frames]
frames = int(feats["attention_mask"][0].sum())
mel = mel[:, :frames].contiguous()
mel.numpy().astype(np.float32).tofile(os.path.join(OUT, "mel.f32"))

tower = model.thinker.audio_tower
conv_out = {}
def _grab(m, a):
    conv_out.setdefault("x", a[0].detach().clone())  # returns None: inputs untouched


tower.layers[0].register_forward_pre_hook(_grab)
layer_out = []
for _l in tower.layers:
    _l.register_forward_hook(lambda m, a, o: layer_out.append(o[0].detach().clone()))

# The eager path ignores cu_seqlens (only FA2 varlen reads it), i.e. attends across 8 s windows.
# Production (vLLM / FA2) is block-diagonal per window; give eager the same mask.
import tb.modeling_qwen3_asr as _m  # noqa: E402
_orig_layer_fwd = _m.Qwen3ASRAudioEncoderLayer.forward


def _windowed(self, hidden_states, cu_seqlens, attention_mask=None, **kw):
    if attention_mask is None:
        n = hidden_states.shape[0]
        attention_mask = torch.full((1, 1, n, n), torch.finfo(hidden_states.dtype).min, dtype=hidden_states.dtype)
        for i in range(1, len(cu_seqlens)):
            a, b = int(cu_seqlens[i - 1]), int(cu_seqlens[i])
            attention_mask[..., a:b, a:b] = 0
    return _orig_layer_fwd(self, hidden_states, cu_seqlens, attention_mask=attention_mask, **kw)


_m.Qwen3ASRAudioEncoderLayer.forward = _windowed
with torch.no_grad():
    enc = tower(mel, feature_lens=torch.tensor([frames])).last_hidden_state
conv_out["x"].numpy().astype(np.float32).tofile(os.path.join(OUT, "conv.f32"))
enc.numpy().astype(np.float32).tofile(os.path.join(OUT, "enc.f32"))
torch.stack(layer_out).numpy().astype(np.float32).tofile(os.path.join(OUT, "layers.f32"))  # [layers, n, d]
n_audio = enc.shape[0]

tok_dir = os.path.join(HERE, "tok")
tok = Qwen2Tokenizer(os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt"))
# vocab.json + merges.txt carry no added tokens; register them with their real ids.
from transformers import AddedToken  # noqa: E402
_added = json.load(open(os.path.join(tok_dir, "tokenizer_config.json")))["added_tokens_decoder"]
for _id, _t in sorted(_added.items(), key=lambda kv: int(kv[0])):
    tok.add_tokens([AddedToken(_t["content"], special=_t.get("special", False), normalized=False)],
                   special_tokens=_t.get("special", False))
    assert tok.convert_tokens_to_ids(_t["content"]) == int(_id), (_t["content"], _id)
ids = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
IM_S, IM_E, A_S, A_P, A_E = 151644, 151645, tc.audio_start_token_id, tc.audio_token_id, tc.audio_end_token_id
prompt = ([IM_S] + ids("system\n") + [IM_E] + ids("\n") + [IM_S] + ids("user\n") + [A_S] + [A_P] * n_audio + [A_E]
          + [IM_E] + ids("\n") + [IM_S] + ids("assistant\n") + ids("language English") + ids("<asr_text>"))
asr_text = tok.convert_tokens_to_ids("<asr_text>")
assert asr_text == 151704 and prompt[-1] == asr_text, prompt[-6:]
np.array(prompt, dtype=np.uint32).tofile(os.path.join(OUT, "prompt.u32"))

with torch.no_grad():
    input_ids = torch.tensor([prompt])
    embeds = model.thinker.model.embed_tokens(input_ids)
    mask = input_ids == A_P
    embeds[mask] = enc
    out = model.thinker.model(inputs_embeds=embeds, use_cache=True)
    logits = model.thinker.lm_head(out.last_hidden_state[:, -1])
    logits[0].numpy().astype(np.float32).tofile(os.path.join(OUT, "logits0.f32"))
    past, gen = out.past_key_values, []
    nxt = int(logits.argmax(-1))
    pos = len(prompt)
    while nxt != IM_E and len(gen) < 256:
        gen.append(nxt)
        e = model.thinker.model.embed_tokens(torch.tensor([[nxt]]))
        o = model.thinker.model(inputs_embeds=e, past_key_values=past, use_cache=True,
                                cache_position=torch.tensor([pos]))
        past, pos = o.past_key_values, pos + 1
        nxt = int(model.thinker.lm_head(o.last_hidden_state[:, -1]).argmax(-1))
np.array(gen, dtype=np.uint32).tofile(os.path.join(OUT, "tokens.u32"))
text = tok.decode(gen)
meta = {"frames": frames, "audio_tokens": n_audio, "prompt_len": len(prompt), "generated": len(gen),
        "asr_text_id": asr_text, "text": text, "start_s": START, "dur_s": DUR}
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), ensure_ascii=False, indent=1)
print(json.dumps(meta, ensure_ascii=False))
