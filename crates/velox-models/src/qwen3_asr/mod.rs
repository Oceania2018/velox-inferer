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
    AttnShape, GeluKind, Im2colShape, KvStoreShape, NormKind, PoolShape, QkvRopeShape, RopeShape,
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
    /// The encoder's scratch, reused by every encode (see [`EncWorkspace`]).
    enc_ws: std::sync::Mutex<EncWorkspace<B>>,
    pub bytes: u64,
}

/// A device buffer that grows on demand and is otherwise kept: `cudaFree` synchronizes the
/// device and `cudaMalloc` + memset are not cheap either, so per-call scratch costs far more
/// than the kernels it serves on short clips.
struct Grow<B: Backend>(Option<(B::Buffer, usize)>);

impl<B: Backend> Default for Grow<B> {
    fn default() -> Self {
        Grow(None)
    }
}

impl<B: Backend> Grow<B> {
    /// The buffer, at least `len` long. Contents are whatever the last user left: every caller
    /// writes the range it reads.
    fn get(&mut self, b: &B, len: usize) -> Result<&mut B::Buffer> {
        if self.0.as_ref().is_none_or(|(_, cap)| *cap < len) {
            // A quarter of headroom, so a slowly growing clip length does not reallocate every time.
            let cap = len + len / 4;
            self.0 = None; // free the old one first: the new one may need its memory
            self.0 = Some((b.alloc(cap)?, cap));
        }
        Ok(&mut self.0.as_mut().expect("allocated above").0)
    }

    /// The buffer itself, at least `len` long, for a caller that keeps it past this call;
    /// [`Grow::give`] hands it back.
    fn take(&mut self, b: &B, len: usize) -> Result<(B::Buffer, usize)> {
        match self.0.take() {
            Some((buf, cap)) if cap >= len => Ok((buf, cap)),
            old => {
                drop(old);
                let cap = len + len / 4;
                Ok((b.alloc(cap)?, cap))
            }
        }
    }

    /// Keeps `buf` (of `cap` slots) for the next [`Grow::take`] unless a larger one is held.
    fn give(&mut self, buf: B::Buffer, cap: usize) {
        if self.0.as_ref().is_none_or(|(_, have)| *have < cap) {
            self.0 = Some((buf, cap));
        }
    }
}

/// Encoder scratch reused across encodes. Measured on an RTX 5060 Ti (Windows / WDDM) with 20
/// realtime call streams: admission (encode + prefill) took 195 ms per group of ~90 rows while
/// the GPU was busy about a third of the time; the rest was the ~16 allocations, frees and
/// uploads per encode. The returned embeddings are still allocated per call (the caller keeps them).
pub struct EncWorkspace<B: Backend> {
    img: Grow<B>,
    cols: Grow<B>,
    acts: [Grow<B>; 3],
    tr: Grow<B>,
    rows: Grow<B>,
    x: Grow<B>,
    pos: Grow<B>,
    hbuf: Grow<B>,
    q: Grow<B>,
    k: Grow<B>,
    v: Grow<B>,
    attn: Grow<B>,
    o: Grow<B>,
    f: Grow<B>,
    /// The returned embeddings: a server hands them back with [`Qwen3Asr::recycle_audio`] once
    /// prefilled. Allocating them per call was the last allocation on the request path, and the
    /// one that hurt: past ~1 MB (8 s of audio) a WDDM `cudaMalloc` took 127 ms on average and up
    /// to 0.5 s on a near-full RTX 5060 Ti, so a 16 s clip took 211 ms against 45 ms for 8 s.
    out: Grow<B>,
    /// Capacity of the embeddings last handed out (what [`Qwen3Asr::recycle_audio`] gives back).
    out_cap: usize,
}

impl<B: Backend> Default for EncWorkspace<B> {
    fn default() -> Self {
        EncWorkspace {
            img: Grow::default(),
            cols: Grow::default(),
            acts: Default::default(),
            tr: Grow::default(),
            rows: Grow::default(),
            x: Grow::default(),
            pos: Grow::default(),
            hbuf: Grow::default(),
            q: Grow::default(),
            k: Grow::default(),
            v: Grow::default(),
            attn: Grow::default(),
            o: Grow::default(),
            f: Grow::default(),
            out: Grow::default(),
            out_cap: 0,
        }
    }
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
            enc_ws: std::sync::Mutex::new(EncWorkspace::default()),
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

    /// Mel -> encoder input rows (conv stack, conv_out, per-chunk positions) of several clips,
    /// concatenated `[sum tokens, d]`, and each clip's token count. Every clip pads its tail chunk
    /// to the same width (docs: tail chunks), so clips batch when all have at least one full
    /// chunk; a shorter clip must come alone. Everything after the host-built mel image runs on the
    /// device, `CONV_GROUP` chunks per pass.
    fn conv_stack(
        &self,
        b: &B,
        clips: &[(&[f32], usize)],
        ws: &mut EncWorkspace<B>,
    ) -> Result<Vec<usize>> {
        let e = &self.cfg.enc;
        let chunk = e.chunk_frames;
        let pad_to = match clips {
            [(_, frames)] if *frames < chunk => *frames,
            _ if clips.iter().all(|c| c.1 >= chunk) => chunk,
            _ => {
                return Err(Error::Shape(
                    "a clip shorter than one chunk must be encoded alone".into(),
                ))
            }
        };
        // NHWC [chunks, mels, pad_to, 1] over all clips, and each chunk's valid rows.
        let mut img = Vec::new();
        let mut valid = Vec::new();
        for &(mel, frames) in clips {
            for c in 0..frames.div_ceil(chunk) {
                let (t0, len) = (c * chunk, (frames - c * chunk).min(chunk));
                let base = img.len();
                img.resize(base + e.mels * pad_to, 0.0);
                for m in 0..e.mels {
                    let src = &mel[m * frames + t0..][..len];
                    img[base + m * pad_to..][..len].copy_from_slice(src);
                }
                valid.push(EncConfig::chunk_tokens(len));
            }
        }
        let chunks = valid.len();
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
        let cols = ws.cols.get(b, cols_len)?;
        let [a0, a1, a2] = &mut ws.acts;
        let acts = [
            a0.get(b, act(&full[0]))?,
            a1.get(b, act(&full[1]))?,
            a2.get(b, act(&full[2]))?,
        ];
        let tr = ws.tr.get(b, act(&full[2]))?;
        let (fq, t) = (full[2].out_h(), full[2].out_w());
        debug_assert_eq!(fq, e.conv_freq());
        let feat = ch * fq;
        let tokens: usize = valid.iter().sum();
        let rows = ws.rows.get(b, tokens * feat)?;
        let mut at = 0;
        let image = e.mels * pad_to;
        // Each group's images go into the same scratch: write_at is stream-ordered (work queued
        // before it reads the old values) and does not allocate.
        let x0 = ws.img.get(b, group * image)?;
        for g0 in (0..chunks).step_by(group) {
            let gn = group.min(chunks - g0);
            let sh = shapes(gn);
            b.write_at(x0, 0, &img[g0 * image..(g0 + gn) * image])?;
            for (stage, s) in sh.iter().enumerate() {
                let src: &B::Buffer = if stage == 0 { x0 } else { acts[stage - 1] };
                b.im2col_3x3s2(src, cols, *s)?;
                let (wm, bias) = &self.enc.conv[stage];
                let m = s.images * s.out_h() * s.out_w();
                b.linear(cols, &wm.view(), acts[stage], m)?;
                b.add_bias_gelu(acts[stage], bias, m, ch, GeluKind::Erf)?;
            }
            // [gn, fq, t, ch] -> [gn, t, fq, ch]: each time step's features f * ch + c.
            b.transpose_heads(acts[2], tr, gn, fq, t, ch)?;
            for c in g0..g0 + gn {
                b.copy(tr, (c - g0) * t * feat, rows, at * feat, valid[c] * feat)?;
                at += valid[c];
            }
        }
        debug_assert_eq!(at, tokens);
        let out = ws.x.get(b, tokens * e.d)?;
        b.linear(rows, &self.enc.conv_out.view(), out, tokens)?;
        // Positions restart in every chunk: each clip's are a prefix of the repeated table.
        let counts: Vec<usize> = clips.iter().map(|c| e.tokens(c.1)).collect();
        if let [n] = counts[..] {
            if n <= self.pos_rows {
                b.add(out, &self.pos, n * e.d)?;
                return Ok(counts);
            }
        }
        let per = EncConfig::chunk_tokens(chunk);
        let pos: Vec<f32> = clips
            .iter()
            .zip(&counts)
            .flat_map(|(c, &n)| {
                chunk_positions(c.1.div_ceil(chunk), per, e.d)
                    .into_iter()
                    .take(n * e.d)
            })
            .collect();
        let pb = ws.pos.get(b, pos.len())?;
        b.write_at(pb, 0, &pos)?;
        b.add(out, pb, tokens * e.d)?;
        Ok(counts)
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
        let (out, counts, tr) = self.encode_clips(b, &[(&mel, frames)], trace)?;
        let tr = tr.map(|t| EncodeTrace { mel, frames, ..t });
        Ok((out, counts[0], tr))
    }

    /// The host mel frontend (a server computes features on its request threads).
    pub fn mel(&self) -> &MelFrontend {
        &self.mel
    }

    /// Audio embeddings `[tokens, hidden]` from `[N_MELS, frames]` log-mel.
    pub fn encode_mel(&self, b: &B, mel: Vec<f32>, frames: usize) -> Result<(B::Buffer, usize)> {
        let (out, counts, _) = self.encode_clips(b, &[(&mel, frames)], false)?;
        Ok((out, counts[0]))
    }

    /// Hands back embeddings from [`Self::encode_many`] once nothing still reads them (after the
    /// prefill that consumed them), so the next encode reuses the buffer instead of allocating.
    pub fn recycle_audio(&self, audio: B::Buffer) {
        if let Ok(mut ws) = self.enc_ws.lock() {
            let cap = ws.out_cap;
            ws.out.give(audio, cap);
        }
    }

    /// Audio embeddings of several clips in one pass, `[sum tokens, hidden]` in clip order, and
    /// each clip's token count. Clips shorter than one second (one conv chunk) must come alone.
    /// Attention windows never cross clips, so the rows equal each clip encoded on its own.
    pub fn encode_many(&self, b: &B, clips: &[(&[f32], usize)]) -> Result<(B::Buffer, Vec<usize>)> {
        let (out, counts, _) = self.encode_clips(b, clips, false)?;
        Ok((out, counts))
    }

    fn encode_clips(
        &self,
        b: &B,
        clips: &[(&[f32], usize)],
        trace: bool,
    ) -> Result<(B::Buffer, Vec<usize>, Option<EncodeTrace>)> {
        let e = &self.cfg.enc;
        if clips.is_empty() {
            return Err(Error::Shape("no clips to encode".into()));
        }
        for &(mel, frames) in clips {
            if frames == 0 || mel.len() != e.mels * frames {
                return Err(Error::Shape(format!(
                    "mel of {} values for {frames} frames",
                    mel.len()
                )));
            }
        }
        let mut guard = self
            .enc_ws
            .lock()
            .map_err(|_| Error::Backend("encoder workspace poisoned".into()))?;
        let ws = &mut *guard;
        let counts = self.conv_stack(b, clips, ws)?;
        let n: usize = counts.iter().sum();
        let (d, hd) = (e.d, e.head_dim());
        let x = ws.x.get(b, n * d)?;
        let mut tr = if trace {
            let mut conv = vec![0f32; n * e.d];
            b.synchronize()?;
            b.download(x, &mut conv)?;
            Some(EncodeTrace {
                mel: Vec::new(),
                frames: 0,
                conv,
                layers: Vec::new(),
            })
        } else {
            None
        };
        // Attention windows: consecutive blocks inside each clip.
        let mut windows = Vec::new();
        let mut at = 0;
        for (&(_, frames), &cn) in clips.iter().zip(&counts) {
            let win = e.window_tokens(frames);
            let mut s = 0;
            while s < cn {
                let len = win.min(cn - s);
                windows.push((at + s, len));
                s += len;
            }
            at += cn;
        }
        let hbuf = ws.hbuf.get(b, n * d)?;
        let (q, k, v) = (
            ws.q.get(b, n * d)?,
            ws.k.get(b, n * d)?,
            ws.v.get(b, n * d)?,
        );
        let attn = ws.attn.get(b, n * d)?;
        let o = ws.o.get(b, n * d)?;
        let f = ws.f.get(b, n * e.ffn)?;
        // Every token attends to its whole window: one ranges call per layer instead of five
        // launches (three copies in, attention, one copy out) per window.
        let ranges: Vec<(u32, u32)> = windows
            .iter()
            .flat_map(|&(s, len)| std::iter::repeat_n((s as u32, len as u32), len))
            .collect();
        let shape = PoolShape {
            rows: n,
            heads: e.heads,
            kv_heads: e.heads,
            head_dim: hd,
            pool_rows: n,
        };
        for l in &self.enc.layers {
            b.layer_norm(x, &l.ln1.0, &l.ln1.1, hbuf, n, d, e.ln_eps)?;
            Self::linear(b, hbuf, &l.q, q, n)?;
            Self::linear(b, hbuf, &l.k, k, n)?;
            Self::linear(b, hbuf, &l.v, v, n)?;
            b.attention_ranges(q, k, v, attn, &ranges, shape)?;
            Self::linear(b, attn, &l.out, o, n)?;
            b.add(x, o, n * d)?;
            b.layer_norm(x, &l.ln2.0, &l.ln2.1, hbuf, n, d, e.ln_eps)?;
            Self::linear(b, hbuf, &l.fc1, f, n)?;
            b.gelu(f, n * e.ffn, GeluKind::Erf)?;
            Self::linear(b, f, &l.fc2, o, n)?;
            b.add(x, o, n * d)?;
            if let Some(t) = tr.as_mut() {
                let mut host = vec![0f32; n * d];
                b.synchronize()?;
                b.download(x, &mut host)?;
                t.layers.push(host);
            }
        }
        let (lw, lb) = &self.enc.ln_post;
        b.layer_norm(x, lw, lb, hbuf, n, d, e.ln_eps)?;
        Self::linear(b, hbuf, &self.enc.proj1, o, n)?;
        b.gelu(o, n * d, GeluKind::Erf)?;
        let (mut out, cap) = ws.out.take(b, n * self.cfg.dec.hidden)?;
        ws.out_cap = cap;
        Self::linear(b, o, &self.enc.proj2, &mut out, n)?;
        Ok((out, counts, tr))
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
    /// Pooled passes (`batch_pass`): the fused `q | k | v` and `gate | up` projections, and the
    /// rotary cos / sin of each row's position (one table per pass, read by every layer).
    qkv: B::Buffer,
    gu: B::Buffer,
    rope: B::Buffer,
    last: B::Buffer,
    normed: B::Buffer,
    pub logits: B::Buffer,
    /// The latest greedy id (u32 bits), which the next decode step embeds from.
    next_id: B::Buffer,
    /// Each step's id copied out of `next_id` (8 slots), so a queued step cannot overwrite the one
    /// still being read back on backends whose fetch reads the buffer itself (the CPU oracle).
    id_history: B::Buffer,
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
            qkv: b.alloc(n * (qd + 2 * kvd))?,
            gu: b.alloc(n * 2 * t.ffn)?,
            rope: b.alloc(n * t.head_dim)?,
            last: b.alloc(h)?,
            normed: b.alloc(h)?,
            logits: b.alloc(t.vocab)?,
            next_id: b.alloc(1)?,
            id_history: b.alloc(8)?,
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
            b.linear(&ws.hb, &l.q(), &mut ws.q, n)?;
            b.linear(&ws.hb, &l.k(), &mut ws.k, n)?;
            b.linear(&ws.hb, &l.v(), &mut ws.v, n)?;
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
            b.linear(&ws.hb, &l.gate(), &mut ws.g, n)?;
            b.linear(&ws.hb, &l.up(), &mut ws.u, n)?;
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
        b.copy(&ws.next_id, 0, &mut ws.id_history, 0, 1)?;
        b.fetch_u32_begin(&ws.id_history, 0, 0)?;
        let mut slot = 0;
        loop {
            let pending = slot;
            let more = out.len() < max_new;
            if more {
                self.step(b, &mut s, ws)?;
                slot = (slot + 1) % 8;
                b.copy(&ws.next_id, 0, &mut ws.id_history, slot, 1)?;
                b.fetch_u32_begin(&ws.id_history, slot, slot)?;
            }
            let id = b.fetch_u32(&ws.id_history, pending, pending)?;
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

/// Positions the prefill scratch session holds: a prompt is its encoder rows (13 per second of
/// audio) plus ~20 tokens, so 1024 covers ~77 s (the sidecar cuts segments at 60 s). F32 KV: 229 KB
/// per position.
pub const PREFILL_CAP: usize = 1024;

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
            self.batch_pass(b, pool, ids, &positions, &dst, &ranges, None, None, ws, bw)
        })?;
        for s in seqs.iter_mut() {
            s.pos += 1;
        }
        let mut host = vec![0f32; n];
        b.synchronize()?;
        b.download(&bw.ids, &mut host)?;
        Ok(host[..n].iter().map(|v| v.to_bits()).collect())
    }

    /// One decoder pass of `ids.len()` rows over the pool: row `r` is position `positions[r]` of a
    /// sequence, stored at pool row `dst[r]`, attending to `ranges[r]`. `audio` replaces the
    /// embeddings of rows `dst_row .. dst_row + len` by encoder rows `src_row ..`. Logits and
    /// greedy ids are computed for the rows in `last` (all rows when `None`), in that order.
    #[allow(clippy::too_many_arguments)]
    fn batch_pass(
        &self,
        b: &B,
        pool: &mut KvPool<B>,
        ids: &[u32],
        positions: &[u32],
        dst: &[u32],
        ranges: &[(u32, u32)],
        audio: Option<(&B::Buffer, &[(usize, usize, usize)])>,
        last: Option<&[usize]>,
        ws: &mut DecWorkspace<B>,
        bw: &mut BatchWorkspace<B>,
    ) -> Result<()> {
        let t = &self.cfg.dec;
        let n = ids.len();
        let (h, hd) = (t.hidden, t.head_dim);
        let eps = t.rms_eps;
        let shape = PoolShape {
            rows: n,
            heads: t.heads,
            kv_heads: t.kv_heads,
            head_dim: hd,
            pool_rows: pool.rows,
        };
        let qkv_shape = QkvRopeShape {
            rows: n,
            heads: t.heads,
            kv_heads: t.kv_heads,
            head_dim: hd,
            pool_rows: pool.rows,
            eps,
        };
        let (qkv_cols, gu_cols) = (qkv_shape.row(), 2 * t.ffn);
        b.embed(ids, &self.dec.embed.view(), &mut ws.x)?;
        if let Some((enc, copies)) = audio {
            for &(src, dst_row, len) in copies {
                b.copy(enc, src * h, &mut ws.x, dst_row * h, len * h)?;
            }
        }
        // Every layer rotates with the same positions: one cos / sin table per pass.
        b.rope_cos_sin(positions, hd, t.rope_theta, &mut ws.rope)?;
        let layers = &self.dec.layers;
        b.rms_norm(
            &ws.x,
            &layers[0].attn_norm,
            &mut ws.hb,
            n,
            h,
            eps,
            NormKind::Plain,
        )?;
        for (li, l) in layers.iter().enumerate() {
            // One GEMV for q | k | v, then one kernel: per-head norms, rotary, K / V into the pool.
            b.linear(&ws.hb, &l.qkv.view(), &mut ws.qkv, n)?;
            debug_assert_eq!(l.qkv.rows, qkv_cols);
            b.qk_norm_rope_kv(
                &ws.qkv,
                &l.q_norm,
                &l.k_norm,
                &ws.rope,
                &mut ws.qn,
                &mut pool.k[li],
                &mut pool.v[li],
                dst,
                qkv_shape,
            )?;
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
            // One GEMV for gate | up, then SwiGLU over its halves.
            b.linear(&ws.hb, &l.gate_up.view(), &mut ws.gu, n)?;
            b.swiglu(&ws.gu, n, gu_cols, 0, t.ffn, t.ffn, &mut ws.g)?;
            b.linear(&ws.g, &l.down.view(), &mut ws.o, n)?;
            // The residual add fused with the next norm (the next layer's, or the final one when
            // every row needs logits).
            match (layers.get(li + 1), last) {
                (Some(next), _) => b.add_rms_norm(
                    &mut ws.x,
                    &ws.o,
                    &next.attn_norm,
                    &mut ws.hb,
                    n,
                    h,
                    eps,
                    NormKind::Plain,
                )?,
                (None, None) => b.add_rms_norm(
                    &mut ws.x,
                    &ws.o,
                    &self.dec.norm,
                    &mut bw.normed,
                    n,
                    h,
                    eps,
                    NormKind::Plain,
                )?,
                (None, Some(_)) => b.add(&mut ws.x, &ws.o, n * h)?,
            }
        }
        let heads = match last {
            None => n,
            Some(rows) => {
                // Gather the rows that need logits into hb, then norm them.
                for (i, &r) in rows.iter().enumerate() {
                    b.copy(&ws.x, r * h, &mut ws.hb, i * h, h)?;
                }
                let m = rows.len();
                b.rms_norm(
                    &ws.hb,
                    &self.dec.norm,
                    &mut bw.normed,
                    m,
                    h,
                    eps,
                    NormKind::Plain,
                )?;
                m
            }
        };
        b.linear(&bw.normed, &self.dec.lm_head.view(), &mut bw.logits, heads)?;
        for i in 0..heads {
            b.copy(&bw.logits, i * t.vocab, &mut bw.row, 0, t.vocab)?;
            b.argmax_to(&bw.row, t.vocab, &mut bw.ids, i)?;
        }
        Ok(())
    }

    /// Prefills several requests in one pass straight into their pool ranges (prefill row `j` of a
    /// request attends to its rows `0 ..= j`, as a decode row does) and returns each one's first
    /// greedy id. `reqs[i]` is (sequence, prompt with pads expanded, encoder row of its first pad
    /// in `audio`); all prompts together must fit the workspace ([`PREFILL_ROWS`]).
    pub fn prefill_many(
        &self,
        b: &B,
        pool: &mut KvPool<B>,
        reqs: &mut [(&mut PoolSeq, &[u32], usize)],
        audio: &B::Buffer,
        ws: &mut DecWorkspace<B>,
        bw: &mut BatchWorkspace<B>,
    ) -> Result<Vec<u32>> {
        let rows: usize = reqs.iter().map(|r| r.1.len()).sum();
        if reqs.is_empty() || reqs.len() > MAX_BATCH || rows > ws.rows {
            return Err(Error::Shape(format!(
                "prefill_many: {} requests, {rows} rows (workspace {})",
                reqs.len(),
                ws.rows
            )));
        }
        let (mut ids, mut positions, mut dst, mut ranges) = (
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
        );
        let (mut copies, mut last) = (Vec::new(), Vec::new());
        for (seq, prompt, audio_row0) in reqs.iter() {
            if seq.pos + prompt.len() >= seq.cap {
                return Err(Error::Shape(format!(
                    "prompt of {} tokens in a sequence of {} rows",
                    prompt.len(),
                    seq.cap
                )));
            }
            let row0 = ids.len();
            if let Some(first) = prompt.iter().position(|&id| id == AUDIO_PAD) {
                let pads = prompt[first..]
                    .iter()
                    .take_while(|&&id| id == AUDIO_PAD)
                    .count();
                copies.push((*audio_row0, row0 + first, pads));
            }
            for (j, &id) in prompt.iter().enumerate() {
                let p = seq.pos + j;
                ids.push(id);
                positions.push(p as u32);
                dst.push((seq.base + p) as u32);
                ranges.push((seq.base as u32, (p + 1) as u32));
            }
            last.push(ids.len() - 1);
        }
        self.batch_pass(
            b,
            pool,
            &ids,
            &positions,
            &dst,
            &ranges,
            Some((audio, &copies)),
            Some(&last),
            ws,
            bw,
        )?;
        for (seq, prompt, _) in reqs.iter_mut() {
            seq.pos += prompt.len();
        }
        let mut host = vec![0f32; reqs.len()];
        b.synchronize()?;
        b.download(&bw.ids, &mut host)?;
        Ok(host.iter().map(|v| v.to_bits()).collect())
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
