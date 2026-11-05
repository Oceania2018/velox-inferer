//! Qwen3-ASR: Whisper-style log-mel, the AuT audio encoder and a Qwen3 dense decoder, loaded
//! straight from the llama.cpp GGUF pair. The mathematics is specified in docs/qwen3_asr.md.
//!
//! This is the single-request path that the oracle checks; batched serving builds on the same
//! pieces.

pub mod config;
pub mod mel;
pub mod weights;

use std::path::Path;

use velox_core::backend::{AttnShape, FullAttnShape, GeluKind, KvStoreShape, NormKind, RopeShape};
use velox_core::{Backend, Error, KvFormat, Result};

use config::{half, EncConfig};
pub use config::{AsrConfig, ASR_TEXT, AUDIO_END, AUDIO_PAD, AUDIO_START, IM_END, IM_START};
use mel::MelFrontend;
use weights::{DecWeights, EncWeights, Files, Linear};

pub struct Qwen3Asr<B: Backend> {
    pub cfg: AsrConfig,
    enc: EncWeights<B>,
    dec: DecWeights<B>,
    mel: MelFrontend,
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

/// im2col of a 3x3, stride 2, pad 1 convolution: `img` is `[images, cin, h, w]`; returns
/// `[images * ho * wo, cin * 9]` with column `ci * 9 + ky * 3 + kx` (PyTorch's weight order).
fn im2col(img: &[f32], images: usize, cin: usize, h: usize, w: usize) -> Vec<f32> {
    let (ho, wo) = (half(h), half(w));
    let k = cin * 9;
    let mut out = vec![0f32; images * ho * wo * k];
    for i in 0..images {
        for oy in 0..ho {
            for ox in 0..wo {
                let row = &mut out[((i * ho + oy) * wo + ox) * k..][..k];
                for ci in 0..cin {
                    let plane = &img[(i * cin + ci) * h * w..][..h * w];
                    for ky in 0..3 {
                        let y = (2 * oy + ky) as isize - 1;
                        if y < 0 || y >= h as isize {
                            continue;
                        }
                        for kx in 0..3 {
                            let x = (2 * ox + kx) as isize - 1;
                            if x >= 0 && x < w as isize {
                                row[ci * 9 + ky * 3 + kx] = plane[y as usize * w + x as usize];
                            }
                        }
                    }
                }
            }
        }
    }
    out
}

/// Exact (erf) GELU on the host, evaluated in f64: the conv stack runs here until it has a device op.
fn gelu_erf(x: f32) -> f32 {
    let x = f64::from(x);
    (0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2))) as f32
}

/// erf with ~1e-15 error (series below 3, continued-fraction tail above), host-side.
fn erf(x: f64) -> f64 {
    let a = x.abs();
    let r = if a < 3.0 {
        // Maclaurin series.
        let mut sum = a;
        let mut term = a;
        let a2 = a * a;
        let mut n = 0.0;
        loop {
            n += 1.0;
            term *= -a2 / n;
            let add = term / (2.0 * n + 1.0);
            sum += add;
            // `<=` so erf(0), whose terms are all zero, terminates.
            if add.abs() <= 1e-17 * sum.abs() {
                break;
            }
        }
        sum * 2.0 / std::f64::consts::PI.sqrt()
    } else {
        // erfc continued fraction (Lentz).
        let a2 = a * a;
        let mut f = a;
        let (mut c, mut d) = (a, 0.0);
        for n in 1..200 {
            let an = n as f64 / 2.0;
            d = a + an * d;
            d = if d == 0.0 { 1e-300 } else { 1.0 / d };
            c = a + an / c;
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - (-a2).exp() / (f * std::f64::consts::PI.sqrt())
    };
    if x < 0.0 {
        -r
    } else {
        r
    }
}

impl<B: Backend> Qwen3Asr<B> {
    pub fn load(b: &B, text: &Path, mmproj: &Path) -> Result<Self> {
        let files = Files::open(text, mmproj)?;
        let cfg = AsrConfig::from_gguf(&files.text, &files.mmproj)?;
        let (enc, eb) = weights::load_encoder(b, &files, &cfg)?;
        let (dec, db) = weights::load_decoder(b, &files, &cfg)?;
        Ok(Qwen3Asr {
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

    /// One conv stage over `images` images `[cin, h, w]` (host), with bias and GELU; returns
    /// `[images, cout, ho, wo]` on the host.
    #[allow(clippy::too_many_arguments)]
    fn conv(
        &self,
        b: &B,
        stage: usize,
        img: &[f32],
        images: usize,
        cin: usize,
        h: usize,
        w: usize,
    ) -> Result<Vec<f32>> {
        let (wm, bias) = &self.enc.conv[stage];
        let (ho, wo) = (half(h), half(w));
        let cout = wm.rows;
        let rows = images * ho * wo;
        let cols = b.upload(&im2col(img, images, cin, h, w))?;
        let mut y = b.alloc(rows * cout)?;
        b.linear(&cols, &wm.view(), &mut y, rows)?;
        let mut host = vec![0f32; rows * cout];
        b.synchronize()?;
        b.download(&y, &mut host)?;
        // [images, ho, wo, cout] -> [images, cout, ho, wo], with bias and GELU.
        let mut out = vec![0f32; images * cout * ho * wo];
        for i in 0..images {
            for p in 0..ho * wo {
                for c in 0..cout {
                    out[(i * cout + c) * ho * wo + p] =
                        gelu_erf(host[(i * ho * wo + p) * cout + c] + bias[c]);
                }
            }
        }
        Ok(out)
    }

    /// Mel -> encoder input rows (conv stack, conv_out, per-chunk positions), `[tokens, d]`.
    fn conv_stack(&self, b: &B, mel: &[f32], frames: usize) -> Result<(B::Buffer, usize)> {
        let e = &self.cfg.enc;
        let chunk = e.chunk_frames;
        let chunks = frames.div_ceil(chunk);
        let pad_to = if frames >= chunk { chunk } else { frames };
        // [chunks, 1, mels, pad_to], zero padded on the right.
        let mut img = vec![0f32; chunks * e.mels * pad_to];
        for c in 0..chunks {
            let (t0, len) = (c * chunk, (frames - c * chunk).min(chunk));
            for m in 0..e.mels {
                let src = &mel[m * frames + t0..][..len];
                img[(c * e.mels + m) * pad_to..][..len].copy_from_slice(src);
            }
        }
        let (mut h, mut w) = (e.mels, pad_to);
        let mut x = self.conv(b, 0, &img, chunks, 1, h, w)?;
        for stage in 1..3 {
            x = self.conv(
                b,
                stage,
                &x,
                chunks,
                e.conv_ch,
                h.div_ceil(2),
                w.div_ceil(2),
            )?;
            h = h.div_ceil(2);
            w = w.div_ceil(2);
        }
        let (fq, t) = (half(h), half(w));
        debug_assert_eq!(fq, e.conv_freq());
        // Keep the valid rows of each chunk, channel-major features c * fq + f.
        let feat = e.conv_ch * fq;
        let tokens = e.tokens(frames);
        let mut rows = vec![0f32; tokens * feat];
        let mut at = 0;
        for c in 0..chunks {
            let len = (frames - c * chunk).min(chunk);
            for tt in 0..EncConfig::chunk_tokens(len) {
                let row = &mut rows[at * feat..][..feat];
                for ch in 0..e.conv_ch {
                    for f in 0..fq {
                        row[ch * fq + f] = x[((c * e.conv_ch + ch) * fq + f) * t + tt];
                    }
                }
                at += 1;
            }
        }
        debug_assert_eq!(at, tokens);
        let cols = b.upload(&rows)?;
        let mut out = b.alloc(tokens * e.d)?;
        b.linear(&cols, &self.enc.conv_out.view(), &mut out, tokens)?;
        // Positions restart in every chunk.
        let table = sinusoids(EncConfig::chunk_tokens(chunk), e.d);
        let mut pos = vec![0f32; tokens * e.d];
        let mut at = 0;
        for c in 0..chunks {
            let len = (frames - c * chunk).min(chunk);
            for tt in 0..EncConfig::chunk_tokens(len) {
                pos[at * e.d..][..e.d].copy_from_slice(&table[tt * e.d..][..e.d]);
                at += 1;
            }
        }
        b.add(&mut out, &b.upload(&pos)?, tokens * e.d)?;
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
        let e = &self.cfg.enc;
        let frames = MelFrontend::frames(pcm.len());
        if frames == 0 {
            return Err(Error::Shape("audio shorter than one mel frame".into()));
        }
        let mel = self.mel.compute(pcm);
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

/// A single-request decoder session: an F32 KV cache sized for `cap` positions.
pub struct Session<B: Backend> {
    k_cache: Vec<B::Buffer>,
    v_cache: Vec<B::Buffer>,
    pub pos: usize,
    cap: usize,
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

    /// Runs `ids` through the decoder at the session's position (audio rows from `audio`
    /// replacing the embeddings of `<|audio_pad|>` ids, in order) and returns the logits of the
    /// last position.
    pub fn forward(
        &self,
        b: &B,
        s: &mut Session<B>,
        ids: &[u32],
        audio: Option<&B::Buffer>,
    ) -> Result<Vec<f32>> {
        let t = &self.cfg.dec;
        let n = ids.len();
        if n == 0 || s.pos + n > s.cap {
            return Err(Error::Shape(format!(
                "decoder step of {n} at {} past capacity {}",
                s.pos, s.cap
            )));
        }
        let (h, hd) = (t.hidden, t.head_dim);
        let (qd, kvd) = (t.heads * hd, t.kv_heads * hd);
        let mut x = b.alloc(n * h)?;
        b.embed(ids, &self.dec.embed.view(), &mut x)?;
        if let Some(audio) = audio {
            // Pads are contiguous in the prompt; copy runs in order.
            let mut src = 0;
            let mut i = 0;
            while i < n {
                if ids[i] != AUDIO_PAD {
                    i += 1;
                    continue;
                }
                let start = i;
                while i < n && ids[i] == AUDIO_PAD {
                    i += 1;
                }
                let len = i - start;
                b.copy(audio, src * h, &mut x, start * h, len * h)?;
                src += len;
            }
        }
        let mut hb = b.alloc(n * h)?;
        let (mut q, mut qn) = (b.alloc(n * qd)?, b.alloc(n * qd)?);
        let (mut k, mut kn) = (b.alloc(n * kvd)?, b.alloc(n * kvd)?);
        let mut v = b.alloc(n * kvd)?;
        let mut attn = b.alloc(n * qd)?;
        let mut o = b.alloc(n * h)?;
        let (mut g, mut u) = (b.alloc(n * t.ffn)?, b.alloc(n * t.ffn)?);
        let eps = t.rms_eps;
        let rope = |heads| RopeShape {
            tokens: n,
            heads,
            head_dim: hd,
            rotary_dim: hd,
            pos0: s.pos,
            theta: t.rope_theta,
        };
        let pos0 = s.pos;
        for (li, l) in self.dec.layers.iter().enumerate() {
            b.rms_norm(&x, &l.attn_norm, &mut hb, n, h, eps, NormKind::Plain)?;
            b.linear(&hb, &l.q.view(), &mut q, n)?;
            b.linear(&hb, &l.k.view(), &mut k, n)?;
            b.linear(&hb, &l.v.view(), &mut v, n)?;
            b.rms_norm_groups(&q, &l.q_norm, &mut qn, n, t.heads, hd, eps)?;
            b.rms_norm_groups(&k, &l.k_norm, &mut kn, n, t.kv_heads, hd, eps)?;
            b.rope(&mut qn, rope(t.heads))?;
            b.rope(&mut kn, rope(t.kv_heads))?;
            b.kv_store(
                &kn,
                &v,
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
                &qn,
                &s.k_cache[li],
                &s.v_cache[li],
                &mut attn,
                AttnShape {
                    tokens: n,
                    heads: t.heads,
                    kv_heads: t.kv_heads,
                    head_dim: hd,
                    pos0,
                    kv: KvFormat::F32,
                },
            )?;
            b.linear(&attn, &l.o.view(), &mut o, n)?;
            b.add_rms_norm(&mut x, &o, &l.ffn_norm, &mut hb, n, h, eps, NormKind::Plain)?;
            b.linear(&hb, &l.gate.view(), &mut g, n)?;
            b.linear(&hb, &l.up.view(), &mut u, n)?;
            b.silu_mul(&mut g, &u, n * t.ffn)?;
            b.linear(&g, &l.down.view(), &mut o, n)?;
            b.add(&mut x, &o, n * h)?;
        }
        s.pos += n;
        let mut last = b.alloc(h)?;
        b.copy(&x, (n - 1) * h, &mut last, 0, h)?;
        let mut normed = b.alloc(h)?;
        b.rms_norm(
            &last,
            &self.dec.norm,
            &mut normed,
            1,
            h,
            eps,
            NormKind::Plain,
        )?;
        let mut logits = b.alloc(t.vocab)?;
        b.linear(&normed, &self.dec.lm_head.view(), &mut logits, 1)?;
        let mut host = vec![0f32; t.vocab];
        b.synchronize()?;
        b.download(&logits, &mut host)?;
        Ok(host)
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
        let mut s = self.session(b, prompt.len() + max_new + 1)?;
        let first = self.forward(b, &mut s, prompt, Some(audio))?;
        let mut out = Vec::new();
        let mut next = argmax(&first);
        while next != IM_END && out.len() < max_new {
            out.push(next);
            next = argmax(&self.forward(b, &mut s, &[next], None)?);
        }
        Ok((out, first))
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
    fn erf_matches_known_values() {
        for (x, want) in [
            (0.0, 0.0),
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.5, 0.999_593_047_982_555),
            (3.5, 0.999_999_256_901_627_7),
            (-1.0, -0.842_700_792_949_714_9),
        ] {
            assert!((erf(x) - want).abs() < 1e-13, "erf({x}) = {}", erf(x));
        }
    }

    #[test]
    fn im2col_matches_direct_convolution() {
        let (cin, h, w) = (2, 5, 4);
        let img: Vec<f32> = (0..cin * h * w).map(|i| (i as f32 * 0.37).sin()).collect();
        let cols = im2col(&img, 1, cin, h, w);
        let wts: Vec<f32> = (0..cin * 9).map(|i| (i as f32 * 0.11).cos()).collect();
        let (ho, wo) = (half(h), half(w));
        for oy in 0..ho {
            for ox in 0..wo {
                let mut direct = 0f32;
                for ci in 0..cin {
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let (y, x) = ((2 * oy + ky) as isize - 1, (2 * ox + kx) as isize - 1);
                            if (0..h as isize).contains(&y) && (0..w as isize).contains(&x) {
                                direct += img[(ci * h + y as usize) * w + x as usize]
                                    * wts[ci * 9 + ky * 3 + kx];
                            }
                        }
                    }
                }
                let row = &cols[(oy * wo + ox) * cin * 9..][..cin * 9];
                let via: f32 = row.iter().zip(&wts).map(|(a, b)| a * b).sum();
                assert!((direct - via).abs() < 1e-5);
            }
        }
    }
}
