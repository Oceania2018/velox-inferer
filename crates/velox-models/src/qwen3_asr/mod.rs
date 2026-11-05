//! Qwen3-ASR: Whisper-style log-mel, the AuT audio encoder and a Qwen3 dense decoder, loaded
//! straight from the llama.cpp GGUF pair. The mathematics is specified in docs/qwen3_asr.md.
//!
//! This is the single-request path that the oracle checks; batched serving builds on the same
//! pieces.

pub mod config;
pub mod mel;
pub mod weights;

use std::path::Path;

use velox_core::backend::{
    AttnShape, FullAttnShape, GeluKind, Im2colShape, KvStoreShape, NormKind, PoolShape, RopeLayout,
    RopePosShape, RopeShape,
};
use velox_core::{Backend, Error, KvFormat, Result, GRAPH_KV_BUCKET, GRAPH_STATIC};

use config::EncConfig;
pub use config::{AsrConfig, ASR_TEXT, AUDIO_END, AUDIO_PAD, AUDIO_START, IM_END, IM_START};
use mel::MelFrontend;
use weights::{DecWeights, EncWeights, Files, Linear};

pub struct Qwen3Asr<B: Backend> {
    pub cfg: AsrConfig,
    enc: EncWeights<B>,
    dec: DecWeights<B>,
    mel: MelFrontend,
    /// [`chunk_positions`] for [`POS_CHUNKS`] chunks, on the device.
    pos: B::Buffer,
    pos_rows: usize,
    /// The decoder workspace every request uses: static decode graphs replay on its buffers.
    dec_ws: std::sync::Mutex<DecWorkspace<B>>,
    pub bytes: u64,
}

/// Intermediate results of [`Qwen3Asr::encode_traced`], for verification.
pub struct EncodeTrace {
    pub mel: Vec<f32>,
    pub frames: usize,
    /// Encoder input: conv_out rows plus positions, `[tokens, d]`.
    pub conv: Vec<f32>,
    /// Each block's output, `[tokens, d]`.
    pub layers: Vec<Vec<f32>>,
}

/// Sinusoidal position table rows `0 .. rows` of width `d`.
fn sinusoids(rows: usize, d: usize) -> Vec<f32> {
    let half_d = d / 2;
    let inc = (10_000f64).ln() / (half_d as f64 - 1.0);
    let mut out = vec![0f32; rows * d];
    for t in 0..rows {
        for j in 0..half_d {
            let a = t as f64 * (-(j as f64) * inc).exp();
            out[t * d + j] = a.sin() as f32;
            out[t * d + half_d + j] = a.cos() as f32;
        }
    }
    out
}

/// Conv chunks per device pass: bounds the im2col scratch (conv2's patches are ~28 MB per chunk).
const CONV_GROUP: usize = 4;

/// Chunks the device position table covers; longer clips build theirs per call.
const POS_CHUNKS: usize = 120;

/// Encoder positions for `chunks` full chunks: the per-chunk table repeated, so any clip's
/// positions (a tail chunk included) are a prefix of it.
fn chunk_positions(chunks: usize, per_chunk: usize, d: usize) -> Vec<f32> {
    let table = sinusoids(per_chunk, d);
    (0..chunks).flat_map(|_| table.iter().copied()).collect()
}

impl<B: Backend> Qwen3Asr<B> {
    pub fn load(b: &B, text: &Path, mmproj: &Path) -> Result<Self> {
        let files = Files::open(text, mmproj)?;
        let cfg = AsrConfig::from_gguf(&files.text, &files.mmproj)?;
        let (enc, eb) = weights::load_encoder(b, &files, &cfg)?;
        let (dec, db) = weights::load_decoder(b, &files, &cfg)?;
        let per = EncConfig::chunk_tokens(cfg.enc.chunk_frames);
        let pos = b.upload(&chunk_positions(POS_CHUNKS, per, cfg.enc.d))?;
        let dec_ws = std::sync::Mutex::new(DecWorkspace::new(b, &cfg)?);
        Ok(Qwen3Asr {
            dec_ws,
            pos,
            pos_rows: POS_CHUNKS * per,
            cfg,
            enc,
            dec,
            mel: MelFrontend::new(),
            bytes: eb + db,
        })
    }

    fn linear(b: &B, x: &B::Buffer, l: &Linear<B>, y: &mut B::Buffer, m: usize) -> Result<()> {
        b.linear(x, &l.w.view(), y, m)?;
        if let Some(bias) = &l.bias {
            b.add_bias(y, bias, m, l.w.rows)?;
        }
        Ok(())
    }

    /// Mel -> encoder input rows (conv stack, conv_out, per-chunk positions), `[tokens, d]`.
    /// Everything after the host-built mel image runs on the device, `CONV_GROUP` chunks per pass.
    fn conv_stack(&self, b: &B, mel: &[f32], frames: usize) -> Result<(B::Buffer, usize)> {
        let e = &self.cfg.enc;
        let chunk = e.chunk_frames;
        let chunks = frames.div_ceil(chunk);
        let pad_to = if frames >= chunk { chunk } else { frames };
        // NHWC [chunks, mels, pad_to, 1], zero padded on the right (docs: tail chunks).
        let mut img = vec![0f32; chunks * e.mels * pad_to];
        for c in 0..chunks {
            let (t0, len) = (c * chunk, (frames - c * chunk).min(chunk));
            for m in 0..e.mels {
                let src = &mel[m * frames + t0..][..len];
                img[(c * e.mels + m) * pad_to..][..len].copy_from_slice(src);
            }
        }
        let ch = e.conv_ch;
        let shapes = |g: usize| {
            let s1 = Im2colShape {
                images: g,
                h: e.mels,
                w: pad_to,
                c: 1,
            };
            let s2 = Im2colShape {
                images: g,
                h: s1.out_h(),
                w: s1.out_w(),
                c: ch,
            };
            let s3 = Im2colShape {
                images: g,
                h: s2.out_h(),
                w: s2.out_w(),
                c: ch,
            };
            [s1, s2, s3]
        };
        let group = CONV_GROUP.min(chunks);
        let full = shapes(group);
        let act = |s: &Im2colShape| s.images * s.out_h() * s.out_w() * ch;
        let cols_len = full.iter().map(Im2colShape::out_len).max().unwrap_or(0);
        let mut cols = b.alloc(cols_len)?;
        let mut acts = [
            b.alloc(act(&full[0]))?,
            b.alloc(act(&full[1]))?,
            b.alloc(act(&full[2]))?,
        ];
        let mut tr = b.alloc(act(&full[2]))?;
        let (fq, t) = (full[2].out_h(), full[2].out_w());
        debug_assert_eq!(fq, e.conv_freq());
        let feat = ch * fq;
        let tokens = e.tokens(frames);
        let mut rows = b.alloc(tokens * feat)?;
        let mut at = 0;
        let image = e.mels * pad_to;
        for g0 in (0..chunks).step_by(group) {
            let gn = group.min(chunks - g0);
            let sh = shapes(gn);
            let x0 = b.upload(&img[g0 * image..(g0 + gn) * image])?;
            for (stage, s) in sh.iter().enumerate() {
                let src = if stage == 0 { &x0 } else { &acts[stage - 1] };
                b.im2col_3x3s2(src, &mut cols, *s)?;
                let (wm, bias) = &self.enc.conv[stage];
                let m = s.images * s.out_h() * s.out_w();
                b.linear(&cols, &wm.view(), &mut acts[stage], m)?;
                b.add_bias_gelu(&mut acts[stage], bias, m, ch, GeluKind::Erf)?;
            }
            // [gn, fq, t, ch] -> [gn, t, fq, ch]: each time step's features f * ch + c.
            b.transpose_heads(&acts[2], &mut tr, gn, fq, t, ch)?;
            for c in g0..g0 + gn {
                let valid = EncConfig::chunk_tokens((frames - c * chunk).min(chunk));
                b.copy(&tr, (c - g0) * t * feat, &mut rows, at * feat, valid * feat)?;
                at += valid;
            }
        }
        debug_assert_eq!(at, tokens);
        let mut out = b.alloc(tokens * e.d)?;
        b.linear(&rows, &self.enc.conv_out.view(), &mut out, tokens)?;
        // Positions restart in every chunk: a prefix of the repeated table.
        if tokens <= self.pos_rows {
            b.add(&mut out, &self.pos, tokens * e.d)?;
        } else {
            let per = EncConfig::chunk_tokens(chunk);
            let pos = b.upload(&chunk_positions(chunks, per, e.d))?;
            b.add(&mut out, &pos, tokens * e.d)?;
        }
        Ok((out, tokens))
    }

    /// Audio embeddings `[tokens, hidden]` for 16 kHz mono PCM.
    pub fn encode(&self, b: &B, pcm: &[i16]) -> Result<(B::Buffer, usize)> {
        let (out, tokens, _) = self.encode_traced(b, pcm, false)?;
        Ok((out, tokens))
    }

    pub fn encode_traced(
        &self,
        b: &B,
        pcm: &[i16],
        trace: bool,
    ) -> Result<(B::Buffer, usize, Option<EncodeTrace>)> {
        let frames = MelFrontend::frames(pcm.len());
        if frames == 0 {
            return Err(Error::Shape("audio shorter than one mel frame".into()));
        }
        let mel = self.mel.compute(pcm);
        self.encode_mel_traced(b, mel, frames, trace)
    }

    /// The host mel frontend (a server computes features on its request threads).
    pub fn mel(&self) -> &MelFrontend {
        &self.mel
    }

    /// Audio embeddings `[tokens, hidden]` from `[N_MELS, frames]` log-mel.
    pub fn encode_mel(&self, b: &B, mel: Vec<f32>, frames: usize) -> Result<(B::Buffer, usize)> {
        let (out, tokens, _) = self.encode_mel_traced(b, mel, frames, false)?;
        Ok((out, tokens))
    }

    fn encode_mel_traced(
        &self,
        b: &B,
        mel: Vec<f32>,
        frames: usize,
        trace: bool,
    ) -> Result<(B::Buffer, usize, Option<EncodeTrace>)> {
        let e = &self.cfg.enc;
        if frames == 0 || mel.len() != e.mels * frames {
            return Err(Error::Shape(format!(
                "mel of {} values for {frames} frames",
                mel.len()
            )));
        }
        let (mut x, n) = self.conv_stack(b, &mel, frames)?;
        let mut tr = if trace {
            let mut conv = vec![0f32; n * e.d];
            b.synchronize()?;
            b.download(&x, &mut conv)?;
            Some(EncodeTrace {
                mel,
                frames,
                conv,
                layers: Vec::new(),
            })
        } else {
            None
        };
        let (d, hd) = (e.d, e.head_dim());
        let win = e.window_tokens(frames);
        let mut hbuf = b.alloc(n * d)?;
        let (mut q, mut k, mut v) = (b.alloc(n * d)?, b.alloc(n * d)?, b.alloc(n * d)?);
        let mut attn = b.alloc(n * d)?;
        let mut o = b.alloc(n * d)?;
        let mut f = b.alloc(n * e.ffn)?;
        let wmax = win.min(n);
        let (mut qw, mut kw, mut vw, mut aw) = (
            b.alloc(wmax * d)?,
            b.alloc(wmax * d)?,
            b.alloc(wmax * d)?,
            b.alloc(wmax * d)?,
        );
        for l in &self.enc.layers {
            b.layer_norm(&x, &l.ln1.0, &l.ln1.1, &mut hbuf, n, d, e.ln_eps)?;
            Self::linear(b, &hbuf, &l.q, &mut q, n)?;
            Self::linear(b, &hbuf, &l.k, &mut k, n)?;
            Self::linear(b, &hbuf, &l.v, &mut v, n)?;
            let mut s = 0;
            while s < n {
                let len = win.min(n - s);
                b.copy(&q, s * d, &mut qw, 0, len * d)?;
                b.copy(&k, s * d, &mut kw, 0, len * d)?;
                b.copy(&v, s * d, &mut vw, 0, len * d)?;
                b.attention_full(
                    &qw,
                    &kw,
                    &vw,
                    &mut aw,
                    FullAttnShape {
                        tokens: len,
                        heads: e.heads,
                        head_dim: hd,
                    },
                )?;
                b.copy(&aw, 0, &mut attn, s * d, len * d)?;
                s += len;
            }
            Self::linear(b, &attn, &l.out, &mut o, n)?;
            b.add(&mut x, &o, n * d)?;
            b.layer_norm(&x, &l.ln2.0, &l.ln2.1, &mut hbuf, n, d, e.ln_eps)?;
            Self::linear(b, &hbuf, &l.fc1, &mut f, n)?;
            b.gelu(&mut f, n * e.ffn, GeluKind::Erf)?;
            Self::linear(b, &f, &l.fc2, &mut o, n)?;
            b.add(&mut x, &o, n * d)?;
            if let Some(t) = tr.as_mut() {
                let mut host = vec![0f32; n * d];
                b.synchronize()?;
                b.download(&x, &mut host)?;
                t.layers.push(host);
            }
        }
        let (lw, lb) = &self.enc.ln_post;
        b.layer_norm(&x, lw, lb, &mut hbuf, n, d, e.ln_eps)?;
        Self::linear(b, &hbuf, &self.enc.proj1, &mut o, n)?;
        b.gelu(&mut o, n * d, GeluKind::Erf)?;
        let mut out = b.alloc(n * self.cfg.dec.hidden)?;
        Self::linear(b, &o, &self.enc.proj2, &mut out, n)?;
        Ok((out, n, tr))
    }

    /// Prompt ids for one clip of `audio_tokens` encoder rows (docs/qwen3_asr.md#prompt).
    /// `text` tokenizes plain text; `language` forces the output language.
    pub fn prompt(
        audio_tokens: usize,
        context: &[u32],
        language: Option<&[u32]>,
        text: impl Fn(&str) -> Vec<u32>,
    ) -> Vec<u32> {
        let mut p = vec![IM_START];
        p.extend(text("system\n"));
        p.extend_from_slice(context);
        p.push(IM_END);
        p.extend(text("\n"));
        p.push(IM_START);
        p.extend(text("user\n"));
        p.push(AUDIO_START);
        p.extend(std::iter::repeat_n(AUDIO_PAD, audio_tokens));
        p.push(AUDIO_END);
        p.push(IM_END);
        p.extend(text("\n"));
        p.push(IM_START);
        p.extend(text("assistant\n"));
        if let Some(lang) = language {
            p.extend_from_slice(lang);
            p.push(ASR_TEXT);
        }
        p
    }
}

/// Rows one decoder pass handles; longer prompts prefill in chunks of this many.
pub const PREFILL_ROWS: usize = 512;

const GRAPH_DECODE: u64 = 1 << 16;

/// A decoder session: its KV cache (F32, `cap` positions) and position.
pub struct Session<B: Backend> {
    k_cache: Vec<B::Buffer>,
    v_cache: Vec<B::Buffer>,
    pub pos: usize,
    cap: usize,
}

impl<B: Backend> Session<B> {
    /// Positions the cache holds.
    pub fn capacity(&self) -> usize {
        self.cap
    }

    /// Starts over at position 0 (the cache is overwritten as it fills again).
    pub fn reset(&mut self) {
        self.pos = 0;
    }
}

/// Decoder activations, reused by every pass: freeing device memory synchronizes the device,
/// and a static graph replays on the same buffers.
pub struct DecWorkspace<B: Backend> {
    rows: usize,
    x: B::Buffer,
    hb: B::Buffer,
    q: B::Buffer,
    qn: B::Buffer,
    k: B::Buffer,
    kn: B::Buffer,
    v: B::Buffer,
    attn: B::Buffer,
    o: B::Buffer,
    g: B::Buffer,
    u: B::Buffer,
    last: B::Buffer,
    normed: B::Buffer,
    pub logits: B::Buffer,
    /// The latest greedy id (u32 bits), which the next decode step embeds from.
    next_id: B::Buffer,
}

impl<B: Backend> DecWorkspace<B> {
    pub fn new(b: &B, cfg: &AsrConfig) -> Result<Self> {
        let t = &cfg.dec;
        let n = PREFILL_ROWS;
        let (h, qd, kvd) = (t.hidden, t.heads * t.head_dim, t.kv_heads * t.head_dim);
        Ok(DecWorkspace {
            rows: n,
            x: b.alloc(n * h)?,
            hb: b.alloc(n * h)?,
            q: b.alloc(n * qd)?,
            qn: b.alloc(n * qd)?,
            k: b.alloc(n * kvd)?,
            kn: b.alloc(n * kvd)?,
            v: b.alloc(n * kvd)?,
            attn: b.alloc(n * qd)?,
            o: b.alloc(n * h)?,
            g: b.alloc(n * t.ffn)?,
            u: b.alloc(n * t.ffn)?,
            last: b.alloc(h)?,
            normed: b.alloc(h)?,
            logits: b.alloc(t.vocab)?,
            next_id: b.alloc(1)?,
        })
    }
}

/// Where a pass's input ids come from.
enum Ids<'a> {
    Host(&'a [u32]),
    /// One id from the workspace's device slot.
    Slot,
}

impl<B: Backend> Qwen3Asr<B> {
    pub fn session(&self, b: &B, cap: usize) -> Result<Session<B>> {
        let t = &self.cfg.dec;
        let row = t.kv_heads * t.head_dim;
        let mut k_cache = Vec::with_capacity(t.layers);
        let mut v_cache = Vec::with_capacity(t.layers);
        for _ in 0..t.layers {
            k_cache.push(b.alloc(cap * row)?);
            v_cache.push(b.alloc(cap * row)?);
        }
        Ok(Session {
            k_cache,
            v_cache,
            pos: 0,
            cap,
        })
    }

    /// One decoder pass of `n` rows at the session's position, ending with the last row's logits
    /// in `ws.logits` and their greedy id in `ws.next_id`. `audio` rows replace the embeddings of
    /// `<|audio_pad|>` ids, the second value being the encoder row of the first pad in `ids`.
    fn pass(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut DecWorkspace<B>,
        ids: Ids<'_>,
        audio: Option<(&B::Buffer, usize)>,
    ) -> Result<()> {
        let t = &self.cfg.dec;
        let n = match ids {
            Ids::Host(v) => v.len(),
            Ids::Slot => 1,
        };
        if n == 0 || n > ws.rows || s.pos + n > s.cap {
            return Err(Error::Shape(format!(
                "decoder pass of {n} rows at {} (workspace {}, capacity {})",
                s.pos, ws.rows, s.cap
            )));
        }
        let (h, hd) = (t.hidden, t.head_dim);
        match ids {
            Ids::Host(v) => {
                b.embed(v, &self.dec.embed.view(), &mut ws.x)?;
                if let Some((audio, mut src)) = audio {
                    let mut i = 0;
                    while i < n {
                        if v[i] != AUDIO_PAD {
                            i += 1;
                            continue;
                        }
                        let start = i;
                        while i < n && v[i] == AUDIO_PAD {
                            i += 1;
                        }
                        let len = i - start;
                        b.copy(audio, src * h, &mut ws.x, start * h, len * h)?;
                        src += len;
                    }
                }
            }
            Ids::Slot => b.embed_from(&ws.next_id, 0, 1, &self.dec.embed.view(), &mut ws.x)?,
        }
        let eps = t.rms_eps;
        let pos0 = s.pos;
        let rope = |heads| RopeShape {
            tokens: n,
            heads,
            head_dim: hd,
            rotary_dim: hd,
            pos0,
            theta: t.rope_theta,
        };
        for (li, l) in self.dec.layers.iter().enumerate() {
            b.rms_norm(&ws.x, &l.attn_norm, &mut ws.hb, n, h, eps, NormKind::Plain)?;
            b.linear(&ws.hb, &l.q.view(), &mut ws.q, n)?;
            b.linear(&ws.hb, &l.k.view(), &mut ws.k, n)?;
            b.linear(&ws.hb, &l.v.view(), &mut ws.v, n)?;
            b.rms_norm_groups(&ws.q, &l.q_norm, &mut ws.qn, n, t.heads, hd, eps)?;
            b.rms_norm_groups(&ws.k, &l.k_norm, &mut ws.kn, n, t.kv_heads, hd, eps)?;
            b.rope(&mut ws.qn, rope(t.heads))?;
            b.rope(&mut ws.kn, rope(t.kv_heads))?;
            b.kv_store(
                &ws.kn,
                &ws.v,
                &mut s.k_cache[li],
                &mut s.v_cache[li],
                KvStoreShape {
                    tokens: n,
                    kv_heads: t.kv_heads,
                    head_dim: hd,
                    pos0,
                    format: KvFormat::F32,
                },
            )?;
            b.attention(
                &ws.qn,
                &s.k_cache[li],
                &s.v_cache[li],
                &mut ws.attn,
                AttnShape {
                    tokens: n,
                    heads: t.heads,
                    kv_heads: t.kv_heads,
                    head_dim: hd,
                    pos0,
                    kv: KvFormat::F32,
                },
            )?;
            b.linear(&ws.attn, &l.o.view(), &mut ws.o, n)?;
            b.add_rms_norm(
                &mut ws.x,
                &ws.o,
                &l.ffn_norm,
                &mut ws.hb,
                n,
                h,
                eps,
                NormKind::Plain,
            )?;
            b.linear(&ws.hb, &l.gate.view(), &mut ws.g, n)?;
            b.linear(&ws.hb, &l.up.view(), &mut ws.u, n)?;
            b.silu_mul(&mut ws.g, &ws.u, n * t.ffn)?;
            b.linear(&ws.g, &l.down.view(), &mut ws.o, n)?;
            b.add(&mut ws.x, &ws.o, n * h)?;
        }
        s.pos += n;
        b.copy(&ws.x, (n - 1) * h, &mut ws.last, 0, h)?;
        b.rms_norm(
            &ws.last,
            &self.dec.norm,
            &mut ws.normed,
            1,
            h,
            eps,
            NormKind::Plain,
        )?;
        b.linear(&ws.normed, &self.dec.lm_head.view(), &mut ws.logits, 1)?;
        b.argmax_to(&ws.logits, t.vocab, &mut ws.next_id, 0)
    }

    /// Prefills `ids` (chunks of [`PREFILL_ROWS`]); the greedy id is left in `ws.next_id`.
    pub fn prefill(
        &self,
        b: &B,
        s: &mut Session<B>,
        ws: &mut DecWorkspace<B>,
        ids: &[u32],
        audio: Option<&B::Buffer>,
    ) -> Result<()> {
        let mut src = 0;
        for part in ids.chunks(ws.rows) {
            self.pass(b, s, ws, Ids::Host(part), audio.map(|a| (a, src)))?;
            src += part.iter().filter(|&&id| id == AUDIO_PAD).count();
        }
        Ok(())
    }

    /// One greedy decode step from the id in `ws.next_id`, recorded as a static graph (one
    /// recording per KV bucket serves every session and position).
    pub fn step(&self, b: &B, s: &mut Session<B>, ws: &mut DecWorkspace<B>) -> Result<()> {
        let kv = s.pos + 1;
        let key =
            GRAPH_DECODE | GRAPH_STATIC | ((kv.div_ceil(GRAPH_KV_BUCKET) as u64) & 0xffff) << 24;
        b.graph(key, &mut || self.pass(b, s, ws, Ids::Slot, None))
    }

    /// Greedy transcription: prompt prefill, then one token per step until `<|im_end|>` or
    /// `max_new` tokens. Returns the generated ids (the stop token excluded) and the prefill
    /// logits.
    pub fn generate(
        &self,
        b: &B,
        prompt: &[u32],
        audio: &B::Buffer,
        max_new: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let (out, first, _) = self.generate_timed(b, prompt, audio, max_new)?;
        Ok((out, first))
    }

    /// [`Self::generate`], also returning the prefill's wall time. Steps are pipelined: the next
    /// step is queued before the current id is read back, so the device never waits for the
    /// host (one step past `<|im_end|>` is computed and discarded).
    pub fn generate_timed(
        &self,
        b: &B,
        prompt: &[u32],
        audio: &B::Buffer,
        max_new: usize,
    ) -> Result<(Vec<u32>, Vec<f32>, std::time::Duration)> {
        let t0 = std::time::Instant::now();
        let mut s = self.session(b, prompt.len() + max_new + 2)?;
        let mut guard = self
            .dec_ws
            .lock()
            .map_err(|_| Error::Backend("decoder workspace poisoned".into()))?;
        let ws = &mut *guard;
        self.prefill(b, &mut s, ws, prompt, Some(audio))?;
        let mut first = vec![0f32; self.cfg.dec.vocab];
        b.synchronize()?;
        b.download(&ws.logits, &mut first)?;
        let prefill = t0.elapsed();
        let mut out = Vec::new();
        b.fetch_u32_begin(&ws.next_id, 0, 0)?;
        let mut slot = 0;
        loop {
            let pending = slot;
            let more = out.len() < max_new;
            if more {
                self.step(b, &mut s, ws)?;
                slot = (slot + 1) % 8;
                b.fetch_u32_begin(&ws.next_id, 0, slot)?;
            }
            let id = b.fetch_u32(&ws.next_id, 0, pending)?;
            if id == IM_END || !more {
                break;
            }
            out.push(id);
        }
        b.synchronize()?;
        Ok((out, first, prefill))
    }
}

/// Sequences one batched decode step serves at most.
pub const MAX_BATCH: usize = 32;

/// Positions the prefill scratch session holds (prompts are encoder rows plus ~20 tokens: 13 per
/// second of audio, so 2048 covers about two and a half minutes).
pub const PREFILL_CAP: usize = 2048;

const GRAPH_DECODE_BATCH: u64 = 2 << 16;

/// One F32 KV pool for all serving sequences: per layer `[rows, kv_heads, head_dim]` K and V, and
/// first-fit ranges of rows handed to sequences.
pub struct KvPool<B: Backend> {
    k: Vec<B::Buffer>,
    v: Vec<B::Buffer>,
    rows: usize,
    /// Free ranges `(start, len)`, sorted by start, coalesced.
    free: Vec<(usize, usize)>,
}

/// A sequence living in a [`KvPool`]: rows `base .. base + cap`, `pos` of them filled.
pub struct PoolSeq {
    base: usize,
    cap: usize,
    pub pos: usize,
}

impl PoolSeq {
    pub fn capacity(&self) -> usize {
        self.cap
    }
}

impl<B: Backend> KvPool<B> {
    pub fn new(b: &B, cfg: &AsrConfig, rows: usize) -> Result<Self> {
        let t = &cfg.dec;
        let row = t.kv_heads * t.head_dim;
        let mut k = Vec::with_capacity(t.layers);
        let mut v = Vec::with_capacity(t.layers);
        for _ in 0..t.layers {
            k.push(b.alloc(rows * row)?);
            v.push(b.alloc(rows * row)?);
        }
        Ok(KvPool {
            k,
            v,
            rows,
            free: vec![(0, rows)],
        })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Rows still free.
    pub fn free_rows(&self) -> usize {
        self.free.iter().map(|r| r.1).sum()
    }

    /// A range of `cap` rows, or `None` when no free range is that long.
    pub fn alloc(&mut self, cap: usize) -> Option<PoolSeq> {
        let i = self.free.iter().position(|r| r.1 >= cap)?;
        let (start, len) = self.free[i];
        if len == cap {
            self.free.remove(i);
        } else {
            self.free[i] = (start + cap, len - cap);
        }
        Some(PoolSeq {
            base: start,
            cap,
            pos: 0,
        })
    }

    pub fn release(&mut self, s: PoolSeq) {
        let i = self.free.partition_point(|r| r.0 < s.base);
        self.free.insert(i, (s.base, s.cap));
        // Coalesce with the neighbours.
        if i + 1 < self.free.len() && self.free[i].0 + self.free[i].1 == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == self.free[i].0 {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
    }
}

/// Buffers of [`Qwen3Asr::step_batch`] beyond the decoder workspace's.
pub struct BatchWorkspace<B: Backend> {
    /// `[MAX_BATCH, hidden]` final-norm rows, `[MAX_BATCH, vocab]` logits, one logits row, ids.
    normed: B::Buffer,
    logits: B::Buffer,
    row: B::Buffer,
    ids: B::Buffer,
    /// The prefill session every admission reuses (its rows are then copied into the pool).
    prefill: Session<B>,
}

impl<B: Backend> BatchWorkspace<B> {
    pub fn new(b: &B, m: &Qwen3Asr<B>) -> Result<Self> {
        let t = &m.cfg.dec;
        Ok(BatchWorkspace {
            normed: b.alloc(MAX_BATCH * t.hidden)?,
            logits: b.alloc(MAX_BATCH * t.vocab)?,
            row: b.alloc(t.vocab)?,
            ids: b.alloc(MAX_BATCH)?,
            prefill: m.session(b, PREFILL_CAP)?,
        })
    }
}

impl<B: Backend> Qwen3Asr<B> {
    /// Prefills `prompt` (audio rows from `audio`) into the scratch session, copies its KV rows
    /// into `seq`'s range of the pool and returns the first greedy id.
    #[allow(clippy::too_many_arguments)]
    pub fn start_pooled(
        &self,
        b: &B,
        pool: &mut KvPool<B>,
        seq: &mut PoolSeq,
        ws: &mut DecWorkspace<B>,
        bw: &mut BatchWorkspace<B>,
        prompt: &[u32],
        audio: &B::Buffer,
    ) -> Result<u32> {
        let n = prompt.len();
        if n > PREFILL_CAP || n >= seq.cap {
            return Err(Error::Shape(format!(
                "prompt of {n} tokens (prefill cap {PREFILL_CAP}, sequence cap {})",
                seq.cap
            )));
        }
        let s = &mut bw.prefill;
        s.reset();
        self.prefill(b, s, ws, prompt, Some(audio))?;
        let row = self.cfg.dec.kv_heads * self.cfg.dec.head_dim;
        for li in 0..self.cfg.dec.layers {
            b.copy(&s.k_cache[li], 0, &mut pool.k[li], seq.base * row, n * row)?;
            b.copy(&s.v_cache[li], 0, &mut pool.v[li], seq.base * row, n * row)?;
        }
        seq.pos = n;
        b.fetch_u32_begin(&ws.next_id, 0, 0)?;
        b.fetch_u32(&ws.next_id, 0, 0)
    }

    /// One greedy decode step for every sequence in `seqs` at once: `ids[i]` is sequence `i`'s
    /// last token. Projections and the MLP run on all rows together (each weight is read once per
    /// step); the sequences store and attend in the pool with one call per layer. Returns each
    /// sequence's next id. Recorded as a static graph per (batch size, KV bucket): ids, positions
    /// and ranges are per-call values.
    pub fn step_batch(
        &self,
        b: &B,
        pool: &mut KvPool<B>,
        seqs: &mut [&mut PoolSeq],
        ids: &[u32],
        ws: &mut DecWorkspace<B>,
        bw: &mut BatchWorkspace<B>,
    ) -> Result<Vec<u32>> {
        let n = seqs.len();
        if n == 0 || n != ids.len() || n > MAX_BATCH || n > ws.rows {
            return Err(Error::Shape(format!(
                "step_batch: {n} sequences, {} ids (max {MAX_BATCH})",
                ids.len()
            )));
        }
        if let Some(s) = seqs.iter().find(|s| s.pos + 1 > s.cap) {
            return Err(Error::Shape(format!(
                "sequence at {} past capacity {}",
                s.pos, s.cap
            )));
        }
        let positions: Vec<u32> = seqs.iter().map(|s| s.pos as u32).collect();
        let dst: Vec<u32> = seqs.iter().map(|s| (s.base + s.pos) as u32).collect();
        let ranges: Vec<(u32, u32)> = seqs
            .iter()
            .map(|s| (s.base as u32, (s.pos + 1) as u32))
            .collect();
        let kv = seqs.iter().map(|s| s.pos + 1).max().unwrap_or(1);
        let key = GRAPH_DECODE_BATCH
            | GRAPH_STATIC
            | (n as u64) << 8
            | ((kv.div_ceil(GRAPH_KV_BUCKET) as u64) & 0xffff) << 24;
        b.graph(key, &mut || {
            self.batch_pass(b, pool, ids, &positions, &dst, &ranges, ws, bw)
        })?;
        for s in seqs.iter_mut() {
            s.pos += 1;
        }
        let mut host = vec![0f32; n];
        b.synchronize()?;
        b.download(&bw.ids, &mut host)?;
        Ok(host[..n].iter().map(|v| v.to_bits()).collect())
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_pass(
        &self,
        b: &B,
        pool: &mut KvPool<B>,
        ids: &[u32],
        positions: &[u32],
        dst: &[u32],
        ranges: &[(u32, u32)],
        ws: &mut DecWorkspace<B>,
        bw: &mut BatchWorkspace<B>,
    ) -> Result<()> {
        let t = &self.cfg.dec;
        let n = ids.len();
        let (h, hd) = (t.hidden, t.head_dim);
        let eps = t.rms_eps;
        let rope = |heads| RopePosShape {
            tokens: n,
            heads,
            head_dim: hd,
            rotary_dim: hd,
            theta: t.rope_theta,
            axes: 1,
            layout: RopeLayout::Interleaved,
        };
        let shape = PoolShape {
            rows: n,
            heads: t.heads,
            kv_heads: t.kv_heads,
            head_dim: hd,
            pool_rows: pool.rows,
        };
        b.embed(ids, &self.dec.embed.view(), &mut ws.x)?;
        for (li, l) in self.dec.layers.iter().enumerate() {
            b.rms_norm(&ws.x, &l.attn_norm, &mut ws.hb, n, h, eps, NormKind::Plain)?;
            b.linear(&ws.hb, &l.q.view(), &mut ws.q, n)?;
            b.linear(&ws.hb, &l.k.view(), &mut ws.k, n)?;
            b.linear(&ws.hb, &l.v.view(), &mut ws.v, n)?;
            b.rms_norm_groups(&ws.q, &l.q_norm, &mut ws.qn, n, t.heads, hd, eps)?;
            b.rms_norm_groups(&ws.k, &l.k_norm, &mut ws.kn, n, t.kv_heads, hd, eps)?;
            b.rope_positions(&mut ws.qn, positions, rope(t.heads))?;
            b.rope_positions(&mut ws.kn, positions, rope(t.kv_heads))?;
            b.kv_scatter(&ws.kn, &ws.v, &mut pool.k[li], &mut pool.v[li], dst, shape)?;
            b.attention_ranges(
                &ws.qn,
                &pool.k[li],
                &pool.v[li],
                &mut ws.attn,
                ranges,
                shape,
            )?;
            b.linear(&ws.attn, &l.o.view(), &mut ws.o, n)?;
            b.add_rms_norm(
                &mut ws.x,
                &ws.o,
                &l.ffn_norm,
                &mut ws.hb,
                n,
                h,
                eps,
                NormKind::Plain,
            )?;
            b.linear(&ws.hb, &l.gate.view(), &mut ws.g, n)?;
            b.linear(&ws.hb, &l.up.view(), &mut ws.u, n)?;
            b.silu_mul(&mut ws.g, &ws.u, n * t.ffn)?;
            b.linear(&ws.g, &l.down.view(), &mut ws.o, n)?;
            b.add(&mut ws.x, &ws.o, n * h)?;
        }
        b.rms_norm(
            &ws.x,
            &self.dec.norm,
            &mut bw.normed,
            n,
            h,
            eps,
            NormKind::Plain,
        )?;
        b.linear(&bw.normed, &self.dec.lm_head.view(), &mut bw.logits, n)?;
        for i in 0..n {
            b.copy(&bw.logits, i * t.vocab, &mut bw.row, 0, t.vocab)?;
            b.argmax_to(&bw.row, t.vocab, &mut bw.ids, i)?;
        }
        Ok(())
    }

    /// The model's decoder workspace (static graphs replay on its buffers).
    pub fn decoder_workspace(&self) -> Result<std::sync::MutexGuard<'_, DecWorkspace<B>>> {
        self.dec_ws
            .lock()
            .map_err(|_| Error::Backend("decoder workspace poisoned".into()))
    }
}

/// Index of the largest value (lowest index on ties, NaN never chosen).
pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] || x[best].is_nan() {
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_positions_restart_every_chunk() {
        let (per, d) = (13, 8);
        let p = chunk_positions(3, per, d);
        assert_eq!(p.len(), 3 * per * d);
        assert_eq!(&p[..per * d], &p[per * d..2 * per * d]);
        // Row 0 of each chunk is sin(0) = 0 and cos(0) = 1.
        assert_eq!(
            &p[per * d..per * d + d],
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
        );
    }
}
