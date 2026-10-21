//! DFlash2 block drafter (`DFlash2DraftModel`): proposes several tokens with one masked-block
//! forward conditioned on the target's residual features (ninfer `docs/maintainer/dflash.md`).
//!
//! ```text
//! context  c_t = plain_rmsnorm(W_feature concat(r_t^l for l in target_layers), context_norm)
//!          per layer: K_ctx(t) = rope(plain_head_rmsnorm(W_k c_t, key_norm), t), V_ctx(t) = W_v c_t
//! block    ids [anchor, MASK x K] at cache positions F .. F + K, each layer:
//!            n = plain_rmsnorm(x, input_norm); (prep, d) = conv.prepare(n)
//!            q, k, v = W_qkv prep; q, k = rope(plain_head_rmsnorm(.), positions)
//!            a = attention over the context window and the whole block (non-causal)
//!            x += conv.finish(W_o a, d); and the same around the MLP
//! proposal per mask column: unary top-16 of the proposal head, then a left-to-right walk over
//!          E_i[p, c] = u_i[c] + sum_r pred[p_tok, r] g_i[r] succ[c_tok, r], g_i = W_hidden h_i
//! ```
//!
//! Context K/V live in per-layer rings of `window` rows (position `p` at row `p % window`); the
//! drafter's live interval is `[lo, end)`. Rewinding (prefix reuse) only shrinks it.

use serde_json::Value;
use velox_core::backend::{Backend, BlockAttnShape, DynConvShape, NormKind, RopeShape, WeightView};
use velox_core::format::WeightFormat;
use velox_core::{Error, Result, GRAPH_STATIC};
use velox_weights::{Artifact, ObjectKind};

use super::weights::{fuse, Loader, WDesc};
use super::{Qwen35Text, Workspace};

/// Graph keys (`Backend::graph`) of drafting steps; low bits carry the shape.
const GRAPH_CONTEXT: u64 = 8 << 16;
const GRAPH_BLOCK: u64 = 9 << 16;

static NEXT_DRAFTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Graph-key id of drafting steps that bind workspace `dw` and rings `st` (several sequences
/// can share one workspace).
fn graph_id<B: Backend>(dw: &DFlashWorkspace<B>, st: &DFlashState<B>) -> u64 {
    dw.id.wrapping_mul(97) ^ st.id.wrapping_mul(0x9e37)
}
/// Context rows up to this many (decode commits) materialize as a recorded step.
const GRAPH_ROWS: usize = 16;

#[derive(Clone, Debug)]
pub struct DFlashConfig {
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate: usize,
    pub eps: f32,
    pub theta: f32,
    pub window: usize,
    /// Target blocks whose residual outputs form the conditioning features, in order.
    pub target_layers: Vec<usize>,
    pub mask_token: u32,
    pub conv_group: usize,
    pub rank: usize,
    pub top_k: usize,
}

fn field(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(Value::as_u64)
        .map(|x| x as usize)
        .ok_or_else(|| Error::Config(format!("dflash2 config: '{key}' must be an integer")))
}

impl DFlashConfig {
    pub fn from_json(v: &Value) -> Result<Self> {
        let arch = v["architectures"][0].as_str().unwrap_or("");
        if arch != "DFlash2DraftModel" {
            return Err(Error::Unsupported(format!("draft architecture '{arch}'")));
        }
        let d = &v["dflash_config"];
        let layers = field(v, "num_hidden_layers")?;
        let types = v["layer_types"].as_array().map_or(0, |t| {
            t.iter()
                .filter(|x| x.as_str() == Some("sliding_attention"))
                .count()
        });
        if types != layers || field(d, "conv_kernel_size")? != 2 {
            return Err(Error::Unsupported(
                "dflash2: only sliding-attention layers with two conv taps".into(),
            ));
        }
        let target_layers = d["target_layer_ids"]
            .as_array()
            .ok_or_else(|| Error::Config("dflash2: target_layer_ids missing".into()))?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::Config("dflash2: bad target_layer_ids".into()))?;
        Ok(DFlashConfig {
            layers,
            heads: field(v, "num_attention_heads")?,
            kv_heads: field(v, "num_key_value_heads")?,
            head_dim: field(v, "head_dim")?,
            intermediate: field(v, "intermediate_size")?,
            eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            theta: v["rope_parameters"]["rope_theta"].as_f64().unwrap_or(1e7) as f32,
            window: field(v, "sliding_window")?,
            target_layers,
            mask_token: field(d, "mask_token_id")? as u32,
            conv_group: field(d, "conv_group_size")?,
            rank: field(d, "selector_rank")?,
            top_k: field(d, "selector_top_k")?,
        })
    }

    fn q_dim(&self) -> usize {
        self.heads * self.head_dim
    }

    fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }
}

struct Conv {
    /// `[2 sides, 2 taps, hidden]`.
    base: usize,
    /// `[4 * hidden / group, hidden]`.
    proj: WDesc,
}

struct DLayer {
    input_norm: usize,
    post_norm: usize,
    qkv: WDesc,
    q_norm: usize,
    k_norm: usize,
    output: WDesc,
    context_key: WDesc,
    context_value: WDesc,
    gate: WDesc,
    up: WDesc,
    gate_up: Option<WDesc>,
    down: WDesc,
    attn_conv: Conv,
    mlp_conv: Conv,
}

/// DFlash2 weights (`dflash2/`); the token embedding and proposal head are the target's.
pub struct DFlashModel<B: Backend> {
    pub cfg: DFlashConfig,
    hidden: usize,
    parents: Vec<B::Buffer>,
    params: Vec<B::Buffer>,
    feature_projection: WDesc,
    context_norm: usize,
    final_norm: usize,
    layers: Vec<DLayer>,
    hidden_projection: WDesc,
    /// Host BF16 codebooks `[vocab, rank]`.
    pred: Vec<u16>,
    succ: Vec<u16>,
    pub bytes: u64,
}

/// Cache positions per [`DraftContext`] page.
pub const DRAFT_PAGE_ROWS: usize = 256;

/// Drafter context rows `lo..hi` (every layer's K and V) kept outside the rings, so a prefix
/// restored into another sequence drafts with the context the drafter had there. Stored in pages
/// of [`DRAFT_PAGE_ROWS`] positions, each one buffer of [`DFlashModel::context_page_len`] slots
/// laid out `[layer][K | V][row][kv_dim]`, so storage recycles through a pool of equal pages.
pub struct DraftContext<B: Backend> {
    pub lo: usize,
    pub hi: usize,
    pages: Vec<B::Buffer>,
}

impl<B: Backend> DraftContext<B> {
    /// Storage size in bytes.
    pub fn bytes(&self, b: &B) -> usize {
        self.pages.iter().map(|x| b.len(x) * 4).sum()
    }

    /// Storage buffers in a fixed order (for serialization).
    pub fn buffers(&self) -> impl Iterator<Item = &B::Buffer> {
        self.pages.iter()
    }

    /// Mutable [`Self::buffers`], same order.
    pub fn buffers_mut(&mut self) -> impl Iterator<Item = &mut B::Buffer> {
        self.pages.iter_mut()
    }

    /// The pages, for reuse.
    pub fn into_pages(self) -> Vec<B::Buffer> {
        self.pages
    }
}

/// Context rings of one sequence and their live interval `[lo, end)`.
pub struct DFlashState<B: Backend> {
    /// Distinguishes graph recordings that bind these rings (with a shared workspace).
    id: u64,
    k: Vec<B::Buffer>,
    v: Vec<B::Buffer>,
    pub lo: usize,
    pub end: usize,
}

/// Activations: context materialization of up to `ctx_rows` rows, and blocks of up to `block`.
pub struct DFlashWorkspace<B: Backend> {
    /// Distinguishes this drafter's graph recordings (they bind its buffers and context state).
    id: u64,
    ctx_rows: usize,
    block: usize,
    ctx_raw: B::Buffer,
    /// Feature rows staged from an offset of the target's capture (batched verify).
    feat: B::Buffer,
    ctx: B::Buffer,
    kr: B::Buffer,
    kn: B::Buffer,
    vr: B::Buffer,
    x: B::Buffer,
    n: B::Buffer,
    prep: B::Buffer,
    z: B::Buffer,
    delta: B::Buffer,
    qkv: B::Buffer,
    q: B::Buffer,
    k: B::Buffer,
    v: B::Buffer,
    qn: B::Buffer,
    bkn: B::Buffer,
    attn: B::Buffer,
    gu: B::Buffer,
    mg: B::Buffer,
    mu: B::Buffer,
    tail: B::Buffer,
    hid: B::Buffer,
    logits: B::Buffer,
    g: B::Buffer,
}

/// One block's proposal: per mask column, the unary candidates (global id, score) and the
/// projected selector hidden (`rank` values).
pub struct BlockCandidates {
    pub cands: Vec<Vec<(u32, f32)>>,
    pub g: Vec<f32>,
}

impl<B: Backend> DFlashModel<B> {
    /// Loads `dflash2/` when the artifact has it.
    pub fn load(b: &B, art: &Artifact, hidden: usize) -> Result<Option<Self>> {
        let Some(comp) = art.components().get("dflash2") else {
            return Ok(None);
        };
        let cfg = DFlashConfig::from_json(&comp["config"])?;
        let mut l = Loader::new(b, art);
        let h = hidden;
        let feature_projection = l.matrix("dflash2/feature_projection")?;
        if feature_projection.rows != h || feature_projection.k != cfg.target_layers.len() * h {
            return Err(Error::Config("dflash2/feature_projection shape".into()));
        }
        let context_norm = l.param("dflash2/context_norm", h)?;
        let final_norm = l.param("dflash2/final_norm", h)?;
        let groups = h / cfg.conv_group;
        let mut layers = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            let p = |s: &str| format!("dflash2/layers/{i}/{s}");
            let mut conv = |name: &str| -> Result<Conv> {
                let proj = l.matrix(&p(&format!("{name}/kernel_projection")))?;
                if proj.rows != 4 * groups || proj.k != h {
                    return Err(Error::Config(format!("{name}: kernel_projection shape")));
                }
                Ok(Conv {
                    base: l.param(&p(&format!("{name}/base_kernel")), 4 * h)?,
                    proj,
                })
            };
            let attn_conv = conv("attention_conv")?;
            let mlp_conv = conv("mlp_conv")?;
            let qkv = l.rows_of(&[
                &p("attention/query"),
                &p("attention/key"),
                &p("attention/value"),
            ])?;
            if qkv.rows != cfg.q_dim() + 2 * cfg.kv_dim() {
                return Err(Error::Config("dflash2 attention shape".into()));
            }
            let gate = l.matrix(&p("mlp/gate"))?;
            let up = l.matrix(&p("mlp/up"))?;
            layers.push(DLayer {
                input_norm: l.param(&p("input_norm"), h)?,
                post_norm: l.param(&p("post_attention_norm"), h)?,
                qkv,
                q_norm: l.param(&p("attention/query_norm"), cfg.head_dim)?,
                k_norm: l.param(&p("attention/key_norm"), cfg.head_dim)?,
                output: l.matrix(&p("attention/output"))?,
                context_key: l.matrix(&p("attention/context_key"))?,
                context_value: l.matrix(&p("attention/context_value"))?,
                gate_up: fuse(&[&gate, &up]),
                gate,
                up,
                down: l.matrix(&p("mlp/down"))?,
                attn_conv,
                mlp_conv,
            });
        }
        let hidden_projection = l.matrix("dflash2/candidate_selector/hidden_projection")?;
        let codebook = |l: &Loader<'_, B>, name: &str| -> Result<Vec<u16>> {
            let (id, range) = l.binding(name)?;
            let obj = art
                .object(&id)
                .ok_or_else(|| Error::Config(format!("missing object {id}")))?;
            let ok = matches!(&obj.kind, ObjectKind::Tensor { shape, format, layout }
                if shape.len() == 2 && shape[1] as usize == cfg.rank
                    && WeightFormat::from_names(format, layout).ok() == Some(WeightFormat::Bf16));
            if !ok || range.is_some() {
                return Err(Error::Config(format!(
                    "{name}: expected bf16 [vocab, rank]"
                )));
            }
            let raw = art
                .read_object(&id)
                .map_err(|e| Error::Config(e.to_string()))?;
            Ok(raw
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect())
        };
        let pred = codebook(&l, "dflash2/candidate_selector/predecessor_codebook")?;
        let succ = codebook(&l, "dflash2/candidate_selector/successor_codebook")?;
        let (parents, params, bytes) = l.finish();
        Ok(Some(DFlashModel {
            cfg,
            hidden: h,
            parents,
            params,
            feature_projection,
            context_norm,
            final_norm,
            layers,
            hidden_projection,
            pred,
            succ,
            bytes,
        }))
    }

    fn view(&self, d: &WDesc) -> WeightView<'_, B::Buffer> {
        WeightView {
            buf: &self.parents[d.parent],
            format: d.format,
            parent_rows: d.n,
            k: d.k,
            row_begin: d.row_begin,
            rows: d.rows,
            act: d.act,
        }
    }

    pub fn state(&self, b: &B) -> Result<DFlashState<B>> {
        let n = self.cfg.window * self.cfg.kv_dim();
        let alloc =
            || -> Result<Vec<B::Buffer>> { (0..self.cfg.layers).map(|_| b.alloc(n)).collect() };
        Ok(DFlashState {
            id: NEXT_DRAFTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            k: alloc()?,
            v: alloc()?,
            lo: 0,
            end: 0,
        })
    }

    pub fn workspace(
        &self,
        b: &B,
        target: &Qwen35Text<B>,
        ctx_rows: usize,
        block: usize,
    ) -> Result<DFlashWorkspace<B>> {
        let c = &self.cfg;
        let (h, r, w) = (self.hidden, ctx_rows, block);
        let tw = &target.w;
        let head_rows = tw.proposal.as_ref().map_or(tw.head.rows, |p| p.head.rows);
        Ok(DFlashWorkspace {
            id: NEXT_DRAFTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            ctx_rows,
            block,
            ctx_raw: b.alloc(r * h)?,
            feat: b.alloc(super::VERIFY_MAX * self.feature_projection.k)?,
            ctx: b.alloc(r * h)?,
            kr: b.alloc(r * c.kv_dim())?,
            kn: b.alloc(r * c.kv_dim())?,
            vr: b.alloc(r * c.kv_dim())?,
            x: b.alloc(w * h)?,
            n: b.alloc(w * h)?,
            prep: b.alloc(w * h)?,
            z: b.alloc(w * h)?,
            delta: b.alloc(w * 4 * (h / c.conv_group))?,
            qkv: b.alloc(w * (c.q_dim() + 2 * c.kv_dim()))?,
            q: b.alloc(w * c.q_dim())?,
            k: b.alloc(w * c.kv_dim())?,
            v: b.alloc(w * c.kv_dim())?,
            qn: b.alloc(w * c.q_dim())?,
            bkn: b.alloc(w * c.kv_dim())?,
            attn: b.alloc(w * c.q_dim())?,
            gu: b.alloc(w * 2 * c.intermediate)?,
            mg: b.alloc(w * c.intermediate)?,
            mu: b.alloc(w * c.intermediate)?,
            tail: b.alloc(w * h)?,
            hid: b.alloc(w * h)?,
            logits: b.alloc(w * head_rows)?,
            g: b.alloc(w * c.rank)?,
        })
    }

    /// f32 slots of one [`DraftContext`] page.
    pub fn context_page_len(&self) -> usize {
        2 * self.cfg.layers * DRAFT_PAGE_ROWS * self.cfg.kv_dim()
    }

    /// The live context rows of `st` below `hi` as `(lo, hi)`, if any.
    pub fn context_rows(&self, st: &DFlashState<B>, hi: usize) -> Option<(usize, usize)> {
        let lo = st.lo.max(st.end.saturating_sub(self.cfg.window));
        let hi = hi.min(st.end);
        (lo < hi).then_some((lo, hi))
    }

    /// Storage for context rows `lo..hi` with pages of [`Self::context_page_len`] slots from
    /// `page`; filled by [`Self::save_context`] or through [`DraftContext::buffers_mut`].
    pub fn blank_context(
        &self,
        b: &B,
        lo: usize,
        hi: usize,
        page: &mut dyn FnMut() -> Result<B::Buffer>,
    ) -> Result<DraftContext<B>> {
        if lo > hi || hi - lo > self.cfg.window {
            return Err(Error::Shape(format!(
                "dflash2 context rows {lo}..{hi} exceed the window {}",
                self.cfg.window
            )));
        }
        let mut pages = Vec::new();
        for _ in 0..(hi - lo).div_ceil(DRAFT_PAGE_ROWS) {
            let p = page()?;
            if b.len(&p) < self.context_page_len() {
                return Err(Error::Config("draft context page too small".into()));
            }
            pages.push(p);
        }
        Ok(DraftContext { lo, hi, pages })
    }

    /// Ring runs covering rows `lo..hi`: `(row, ring slot, rows)`, each contiguous in the ring.
    fn ring_runs(&self, lo: usize, hi: usize) -> Vec<(usize, usize, usize)> {
        let w = self.cfg.window;
        let mut runs = Vec::new();
        let mut r = lo;
        while r < hi {
            let slot = r % w;
            let n = (hi - r).min(w - slot);
            runs.push((r, slot, n));
            r += n;
        }
        runs
    }

    /// Copies context rows `lo..hi` (live in `st`, see [`Self::context_rows`]) out of the rings.
    pub fn save_context(
        &self,
        b: &B,
        st: &DFlashState<B>,
        lo: usize,
        hi: usize,
        page: &mut dyn FnMut() -> Result<B::Buffer>,
    ) -> Result<DraftContext<B>> {
        if self
            .context_rows(st, hi)
            .is_none_or(|(l, h)| lo < l || hi > h)
        {
            return Err(Error::Shape(format!(
                "dflash2 context rows {lo}..{hi} are not live"
            )));
        }
        let mut ctx = self.blank_context(b, lo, hi, page)?;
        let kvd = self.cfg.kv_dim();
        for (p, buf) in ctx.pages.iter_mut().enumerate() {
            let r0 = lo + p * DRAFT_PAGE_ROWS;
            let r1 = (r0 + DRAFT_PAGE_ROWS).min(hi);
            for (row, slot, n) in self.ring_runs(r0, r1) {
                for li in 0..self.cfg.layers {
                    let at = |kind: usize| ((2 * li + kind) * DRAFT_PAGE_ROWS + row - r0) * kvd;
                    b.copy(&st.k[li], slot * kvd, buf, at(0), n * kvd)?;
                    b.copy(&st.v[li], slot * kvd, buf, at(1), n * kvd)?;
                }
            }
        }
        Ok(ctx)
    }

    /// Replaces the context of `st` with `ctx`: the live interval becomes `ctx.lo .. ctx.hi`.
    pub fn load_context(
        &self,
        b: &B,
        st: &mut DFlashState<B>,
        ctx: &DraftContext<B>,
    ) -> Result<()> {
        if ctx.hi - ctx.lo > self.cfg.window {
            return Err(Error::Shape("dflash2 context exceeds the window".into()));
        }
        let kvd = self.cfg.kv_dim();
        for (p, buf) in ctx.pages.iter().enumerate() {
            let r0 = ctx.lo + p * DRAFT_PAGE_ROWS;
            let r1 = (r0 + DRAFT_PAGE_ROWS).min(ctx.hi);
            for (row, slot, n) in self.ring_runs(r0, r1) {
                for li in 0..self.cfg.layers {
                    let at = |kind: usize| ((2 * li + kind) * DRAFT_PAGE_ROWS + row - r0) * kvd;
                    b.copy(buf, at(0), &mut st.k[li], slot * kvd, n * kvd)?;
                    b.copy(buf, at(1), &mut st.v[li], slot * kvd, n * kvd)?;
                }
            }
        }
        st.lo = ctx.lo;
        st.end = ctx.hi;
        Ok(())
    }

    /// Forgets context at and after `pos` (the target rewound to `pos`).
    pub fn rewind(&self, st: &mut DFlashState<B>, pos: usize) {
        if pos < st.end {
            st.lo = st.lo.max(st.end.saturating_sub(self.cfg.window)).min(pos);
            st.end = pos;
        }
    }

    /// Materializes context rows for cache positions `pos0 .. pos0 + n` from the features the
    /// target captured in `ws` (rows `feat_row .. feat_row + n` of its latest chunk or verify;
    /// a batched verify leaves each sequence's rows after the earlier sequences').
    #[allow(clippy::too_many_arguments)]
    pub fn materialize(
        &self,
        b: &B,
        st: &mut DFlashState<B>,
        dw: &mut DFlashWorkspace<B>,
        ws: &Workspace<B>,
        feat_row: usize,
        pos0: usize,
        n: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        if n == 0 {
            return Ok(());
        }
        if n > dw.ctx_rows || n > c.window {
            return Err(Error::Shape(format!(
                "dflash2 context: {n} rows (workspace {}, window {})",
                dw.ctx_rows, c.window
            )));
        }
        let features = ws
            .features
            .as_ref()
            .ok_or_else(|| Error::Config("target workspace captures no features".into()))?;
        if pos0 < st.end {
            self.rewind(st, pos0);
        } else if pos0 > st.end {
            st.lo = pos0;
        }
        let (h, kvd) = (self.hidden, c.kv_dim());
        let slot = pos0 % c.window;
        let first = n.min(c.window - slot);
        let id = graph_id(dw, st);
        // Rows past the first stage through `feat` (the projection reads from row 0).
        let cols = self.feature_projection.k;
        if feat_row > 0 {
            if n > super::VERIFY_MAX {
                return Err(Error::Shape(format!("dflash2 context: {n} offset rows")));
            }
            b.copy(features, feat_row * cols, &mut dw.feat, 0, n * cols)?;
        }
        let src: &B::Buffer = if feat_row > 0 { &dw.feat } else { features };
        let mut run = || -> Result<()> {
            b.linear(
                src,
                &self.view(&self.feature_projection),
                &mut dw.ctx_raw,
                n,
            )?;
            b.rms_norm(
                &dw.ctx_raw,
                &self.params[self.context_norm],
                &mut dw.ctx,
                n,
                h,
                c.eps,
                NormKind::Plain,
            )?;
            for (li, l) in self.layers.iter().enumerate() {
                b.linear(&dw.ctx, &self.view(&l.context_key), &mut dw.kr, n)?;
                b.linear(&dw.ctx, &self.view(&l.context_value), &mut dw.vr, n)?;
                b.rms_norm(
                    &dw.kr,
                    &self.params[l.k_norm],
                    &mut dw.kn,
                    n * c.kv_heads,
                    c.head_dim,
                    c.eps,
                    NormKind::Plain,
                )?;
                b.rope(
                    &mut dw.kn,
                    RopeShape {
                        tokens: n,
                        heads: c.kv_heads,
                        head_dim: c.head_dim,
                        rotary_dim: c.head_dim,
                        pos0,
                        theta: c.theta,
                    },
                )?;
                b.copy(&dw.kn, 0, &mut st.k[li], slot * kvd, first * kvd)?;
                b.copy(&dw.vr, 0, &mut st.v[li], slot * kvd, first * kvd)?;
                if first < n {
                    b.copy(&dw.kn, first * kvd, &mut st.k[li], 0, (n - first) * kvd)?;
                    b.copy(&dw.vr, first * kvd, &mut st.v[li], 0, (n - first) * kvd)?;
                }
            }
            Ok(())
        };
        if n <= GRAPH_ROWS {
            // Static unless the rows wrap around the ring (copy lengths then depend on the position).
            let wrap = first < n;
            let key = GRAPH_CONTEXT
                | n as u64
                | u64::from(wrap) << 8
                | u64::from(feat_row > 0) << 9
                | (id & 0xffff) << 40
                | if wrap { 0 } else { GRAPH_STATIC };
            b.graph(key, &mut run)?;
        } else {
            run()?;
        }
        st.end = pos0 + n;
        st.lo = st.lo.max(st.end.saturating_sub(c.window));
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// One masked block at cache positions `st.end ..`: `anchor` then `drafts` masks. Returns the
    /// unary candidates of every mask column and their projected selector hiddens.
    pub fn propose(
        &self,
        b: &B,
        target: &Qwen35Text<B>,
        st: &DFlashState<B>,
        dw: &mut DFlashWorkspace<B>,
        anchor: u32,
        drafts: usize,
        limit: usize,
    ) -> Result<BlockCandidates> {
        let c = &self.cfg;
        let (h, w) = (self.hidden, drafts + 1);
        if drafts == 0 || w > dw.block || w > c.window {
            return Err(Error::Shape(format!("dflash2 block of {w}")));
        }
        let tw = &target.w;
        let mut ids = vec![c.mask_token; w];
        ids[0] = anchor;
        let (head, map) = match &tw.proposal {
            Some(p) => (tw.view(&p.head), Some(&p.token_ids)),
            None => (tw.view(&tw.head), None),
        };
        let (qd, kvd) = (c.q_dim(), c.kv_dim());
        let k = drafts;
        // The block forward through the selector projection is one recorded step per width.
        // Static: positions are staged and block attention spans a full window of grid.
        let key = GRAPH_BLOCK | w as u64 | (graph_id(dw, st) & 0xffff) << 40 | GRAPH_STATIC;
        b.graph(key, &mut || {
            b.embed(&ids, &tw.view(&tw.embed), &mut dw.x)?;
            for (li, l) in self.layers.iter().enumerate() {
                // Attention branch.
                self.prepare(b, dw, l.input_norm, &l.attn_conv, w)?;
                b.linear(&dw.prep, &self.view(&l.qkv), &mut dw.qkv, w)?;
                b.split_cols(
                    &dw.qkv,
                    w,
                    qd + 2 * kvd,
                    &mut [
                        (&mut dw.q, 0, qd),
                        (&mut dw.k, qd, kvd),
                        (&mut dw.v, qd + kvd, kvd),
                    ],
                )?;
                b.rms_norm(
                    &dw.q,
                    &self.params[l.q_norm],
                    &mut dw.qn,
                    w * c.heads,
                    c.head_dim,
                    c.eps,
                    NormKind::Plain,
                )?;
                b.rms_norm(
                    &dw.k,
                    &self.params[l.k_norm],
                    &mut dw.bkn,
                    w * c.kv_heads,
                    c.head_dim,
                    c.eps,
                    NormKind::Plain,
                )?;
                let rope = |heads| RopeShape {
                    tokens: w,
                    heads,
                    head_dim: c.head_dim,
                    rotary_dim: c.head_dim,
                    pos0: st.end,
                    theta: c.theta,
                };
                b.rope(&mut dw.qn, rope(c.heads))?;
                b.rope(&mut dw.bkn, rope(c.kv_heads))?;
                b.attention_block(
                    &dw.qn,
                    &st.k[li],
                    &st.v[li],
                    &dw.bkn,
                    &dw.v,
                    &mut dw.attn,
                    BlockAttnShape {
                        tokens: w,
                        heads: c.heads,
                        kv_heads: c.kv_heads,
                        head_dim: c.head_dim,
                        pos0: st.end,
                        ctx_lo: st.lo,
                        window: c.window,
                        ring: c.window,
                    },
                )?;
                b.linear(&dw.attn, &self.view(&l.output), &mut dw.z, w)?;
                self.finish(b, dw, &l.attn_conv, w)?;
                // MLP branch.
                self.prepare(b, dw, l.post_norm, &l.mlp_conv, w)?;
                match &l.gate_up {
                    Some(f) => {
                        b.linear(&dw.prep, &self.view(f), &mut dw.gu, w)?;
                        let (g, u) = (l.gate.row_begin - f.row_begin, l.up.row_begin - f.row_begin);
                        b.swiglu(&dw.gu, w, f.rows, g, u, c.intermediate, &mut dw.mg)?;
                    }
                    None => {
                        b.linear(&dw.prep, &self.view(&l.gate), &mut dw.mg, w)?;
                        b.linear(&dw.prep, &self.view(&l.up), &mut dw.mu, w)?;
                        b.silu_mul(&mut dw.mg, &dw.mu, w * c.intermediate)?;
                    }
                }
                b.linear(&dw.mg, &self.view(&l.down), &mut dw.z, w)?;
                self.finish(b, dw, &l.mlp_conv, w)?;
            }
            // Mask columns 1 .. w: final norm, proposal head, unary top-k, selector projection.
            b.copy(&dw.x, h, &mut dw.tail, 0, k * h)?;
            b.rms_norm(
                &dw.tail,
                &self.params[self.final_norm],
                &mut dw.hid,
                k,
                h,
                c.eps,
                NormKind::Plain,
            )?;
            b.linear(&dw.hid, &head, &mut dw.logits, k)?;
            b.linear(&dw.hid, &self.view(&self.hidden_projection), &mut dw.g, k)
        })?;
        let n = head.rows;
        let fetch = c.top_k + if map.is_some() { 8 } else { 0 };
        let rows = b.top_k(
            &dw.logits,
            k,
            n,
            if map.is_some() { n } else { limit.min(n) },
            fetch,
        )?;
        let cands = rows
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|(i, v)| (map.map_or(i, |m| m[i as usize]), v))
                    .filter(|&(id, _)| (id as usize) < limit)
                    .take(c.top_k)
                    .collect()
            })
            .collect();
        let mut g = vec![0.0; k * c.rank];
        b.download(&dw.g, &mut g)?;
        Ok(BlockCandidates { cands, g })
    }

    /// `prep = conv side 0 of plain_rmsnorm(x, norm)`, keeping the side-1 delta in `dw.delta`.
    fn prepare(
        &self,
        b: &B,
        dw: &mut DFlashWorkspace<B>,
        norm: usize,
        conv: &Conv,
        w: usize,
    ) -> Result<()> {
        let c = &self.cfg;
        let h = self.hidden;
        b.rms_norm(
            &dw.x,
            &self.params[norm],
            &mut dw.n,
            w,
            h,
            c.eps,
            NormKind::Plain,
        )?;
        b.linear(&dw.n, &self.view(&conv.proj), &mut dw.delta, w)?;
        b.dyn_conv(
            &dw.n,
            &self.params[conv.base],
            &dw.delta,
            &mut dw.prep,
            DynConvShape {
                rows: w,
                channels: h,
                group: c.conv_group,
                side: 0,
                accumulate: false,
            },
        )
    }

    /// `x += conv side 1 of dw.z` with the delta of the matching `prepare`.
    fn finish(&self, b: &B, dw: &mut DFlashWorkspace<B>, conv: &Conv, w: usize) -> Result<()> {
        b.dyn_conv(
            &dw.z,
            &self.params[conv.base],
            &dw.delta,
            &mut dw.x,
            DynConvShape {
                rows: w,
                channels: self.hidden,
                group: self.cfg.conv_group,
                side: 1,
                accumulate: true,
            },
        )
    }

    /// The conditional candidate walk: per mask column `i`, scores
    /// `E[c] = u[c] + sum_r pred[prev, r] g_i[r] succ[cand_c, r]` with `prev` the anchor, then each
    /// selected draft. Greedy (`temperature <= 0`) takes the lowest-rank maximum; otherwise draws
    /// from softmax(E / temperature) with `uniform` in [0, 1). Returns the drafts and, per draft,
    /// the proposal distribution it was drawn from (for verification).
    #[allow(clippy::type_complexity)]
    pub fn select(
        &self,
        bc: &BlockCandidates,
        anchor: u32,
        temperature: f32,
        uniform: &mut dyn FnMut() -> f32,
    ) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
        let r = self.cfg.rank;
        let bf = |v: u16| f32::from_bits(u32::from(v) << 16);
        let mut prev = anchor;
        let (mut drafts, mut qs) = (Vec::new(), Vec::new());
        for (i, cands) in bc.cands.iter().enumerate() {
            if cands.is_empty() {
                break;
            }
            let g = &bc.g[i * r..(i + 1) * r];
            let p = &self.pred[prev as usize * r..][..r];
            let pg: Vec<f32> = p.iter().zip(g).map(|(&a, &b)| bf(a) * b).collect();
            let e: Vec<f32> = cands
                .iter()
                .map(|&(id, u)| {
                    let s = &self.succ[id as usize * r..][..r];
                    u + pg.iter().zip(s).map(|(&a, &b)| a * bf(b)).sum::<f32>()
                })
                .collect();
            let (pick, q) = if temperature <= 0.0 {
                let mut best = 0;
                for (j, &v) in e.iter().enumerate() {
                    if v > e[best] {
                        best = j;
                    }
                }
                (best, vec![(cands[best].0, 1.0)])
            } else {
                let m = e.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let w: Vec<f32> = e.iter().map(|&v| ((v - m) / temperature).exp()).collect();
                let total: f32 = w.iter().sum();
                let q: Vec<(u32, f32)> = cands
                    .iter()
                    .zip(&w)
                    .map(|(&(id, _), &x)| (id, x / total))
                    .collect();
                let mut u = uniform() * total;
                let mut pick = w.len() - 1;
                for (j, &x) in w.iter().enumerate() {
                    u -= x;
                    if u <= 0.0 {
                        pick = j;
                        break;
                    }
                }
                (pick, q)
            };
            prev = cands[pick].0;
            drafts.push(prev);
            qs.push(q);
        }
        (drafts, qs)
    }
}
