//! DSpine block drafter (Tencent/DSpine, arXiv 2609.36173) on the DFlash machinery: the same
//! context (feature projection, context norm, each layer's own key / value rows) and rings, a
//! different block and proposal.
//!
//! ```text
//! block    ids [anchor, MASK x K] at cache positions F .. F + K; per layer, a plain pre-norm Qwen3
//!          layer whose block attention is causal (row t sees context and block rows 0 ..= t),
//!          then the code-bridge write on the residual stream u:
//!            code_t = sqrt(C) normalize(R_l (u_t / rms(u_t)))     (row 0: the anchor's codebook row)
//!            (q | s | v)_t = [G_q; G_p; A] code_t
//!            m_t = sigmoid(q_t + s_{t-1} + b) * v_{t-1},  m_0 = 0
//!            x_t = u_t + rms(u_t) W_l m_t                         (t > 0)
//! proposal per mask column t: the top-K ids of head(final_norm(x_t)); for the drafted
//!          predecessor s (the anchor for the first column), each candidate c scores
//!            head_c . final_norm(u_t)  +  m_t(s) . TG[c]
//!            m_t(s) = sigmoid(q_t + TSV_g[s] + b) * TSV_v[s]
//!          with u_t the last layer's pre-write row, TSV[s] = [G_p; A] sqrt(C) codebook[s] and
//!          TG[c] = W_L^T (final_norm_w * head_c): the reference's greedy serial last-write
//!          refinement (`RefinementTail.greedy`), divided by rms(u_t) so it reads on a logit scale.
//! ```
//!
//! Only the written rows go through the head (the draft vocabulary head unless
//! `set_full_head`): x_t = u_t + rms(u_t) W_L m_t, so the pre-write score of a candidate follows
//! from its logit, the two rows' rms and m_t . TG[c] on the host.
//!
//! Row 0's bridge state only feeds row 1 through `(s | v)_0 = TSV[anchor]`, so the codebook stays
//! on the host: it builds TSV once, and TSV / TG live in system RAM as BF16.

use std::sync::{Mutex, OnceLock};

use serde_json::Value;
use velox_core::backend::{
    Backend, BlockAttnShape, Copy2dShape, NormKind, PrevGateShape, RmsResidualShape, RopeShape,
};
use velox_core::format::{bf16_to_f32, f32_to_bf16, WeightFormat};
use velox_core::{Error, Result, GRAPH_STATIC};
use velox_weights::{Artifact, ObjectKind};

use super::{
    field, host_rows, BlockCandidates, DFlashConfig, DFlashModel, DFlashState, DFlashWorkspace,
    DLayer, DraftKind,
};
use crate::qwen3_5::weights::{fuse, Loader, WDesc};
use crate::qwen3_5::Qwen35Text;

const GRAPH_BLOCK_SPINE: u64 = 11 << 16;

/// The code bridge's weights and host tables.
pub(crate) struct Bridge<B: Backend> {
    ones_h: B::Buffer,
    ones_c: B::Buffer,
    /// Per layer `[code, hidden]`.
    read: Vec<WDesc>,
    /// `[3 rank, code]`: `G_q | G_p | A`.
    packed: WDesc,
    /// Its `G_p | A` rows.
    pv: WDesc,
    gate_bias: usize,
    gate_bias_host: Vec<f32>,
    /// Per layer `[hidden, rank]`.
    write: Vec<WDesc>,
    /// Inputs of the tables, held until the target's head is known: the BF16 codebook `[vocab,
    /// code]`, the last layer's write `[hidden, rank]` and the final norm.
    pending: Mutex<Option<(Vec<u16>, Vec<u16>, Vec<f32>)>>,
    tables: OnceLock<Tables>,
}

/// BF16 host tables: `tsv` `[vocab, 2 rank]` (`G_p e_s | A e_s`), `tg` `[vocab, rank]`.
struct Tables {
    tsv: Vec<u16>,
    tg: Vec<u16>,
}

/// DSpine's block activations (inside [`DFlashWorkspace`]).
pub(crate) struct Scratch<B: Backend> {
    proj: B::Buffer,
    codes: B::Buffer,
    packed: B::Buffer,
    msg: B::Buffer,
    rms: B::Buffer,
    /// Each block's `(s | v)` for its anchor row.
    anchor_sv: B::Buffer,
    /// rms of the written rows that reach the head.
    rms_post: B::Buffer,
    /// Host-mapped `rms(u)` and last-layer messages of every block row, then `rms_post`.
    aux_host: B::Buffer,
}

impl<B: Backend> Scratch<B> {
    pub(crate) fn new(b: &B, c: &DFlashConfig, rows: usize) -> Result<Self> {
        Ok(Scratch {
            proj: b.alloc(rows * c.code_size)?,
            codes: b.alloc(rows * c.code_size)?,
            packed: b.alloc(rows * 3 * c.rank)?,
            msg: b.alloc(rows * c.rank)?,
            rms: b.alloc(rows)?,
            anchor_sv: b.alloc(rows * 2 * c.rank)?,
            rms_post: b.alloc(rows)?,
            aux_host: b.alloc_host(rows * (2 + c.rank))?,
        })
    }
}

fn bad(what: &str) -> Error {
    Error::Unsupported(format!("dspine: {what}"))
}

/// The config of an HF `DFlashDraftModel` with a DSpine code bridge, checked against what velox
/// runs: terminal-disabled independent writes after every layer, causal blocks, the serial exact
/// last-write refinement and no convolutions.
pub(super) fn config(v: &Value) -> Result<DFlashConfig> {
    let d = &v["dflash_config"];
    let layers = field(v, "num_hidden_layers")?;
    let sliding = v["layer_types"].as_array().map_or(0, |t| {
        t.iter()
            .filter(|x| x.as_str() == Some("sliding_attention"))
            .count()
    });
    if sliding != layers {
        return Err(bad("only sliding-attention layers"));
    }
    if d["draft_block_causal"].as_bool() != Some(true) {
        return Err(bad("the draft block must be causal (draft_block_causal)"));
    }
    if d["selected_prefix_terminal"].as_bool() != Some(false)
        || d["selected_prefix_feature_write_mode"].as_str() != Some("independent")
        || d["selected_prefix_feature_writes"].as_bool() == Some(false)
    {
        return Err(bad("terminal-disabled independent writes only"));
    }
    if d["conv_kernel_size"].as_u64().unwrap_or(0) != 0
        || d["selected_prefix_terminal_residual_rank"]
            .as_u64()
            .unwrap_or(0)
            != 0
        || d.get("selected_prefix_feature_reliability")
            .is_some_and(|x| !x.is_null())
        || d["selected_prefix_feature_next_token_rows"].as_bool() == Some(true)
    {
        return Err(bad(
            "convolutions, terminal residuals, reliability probes or next-token rows",
        ));
    }
    let writes: Vec<u64> = d["selected_prefix_feature_layers"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_u64).collect())
        .unwrap_or_default();
    if writes != (1..=layers as u64).collect::<Vec<_>>() {
        return Err(bad("a bridge write after every layer"));
    }
    let second = &v["code_bridge_second_pass"];
    if second["mode"].as_str() != Some("serial_table")
        || second["first_exact"].as_u64() != Some(layers as u64 - 1)
    {
        return Err(bad(
            "the serial last-write refinement (code_bridge_second_pass)",
        ));
    }
    let refine = &d["selected_prefix_last_write_refine"];
    if refine["kind"].as_str() != Some("exact_sender") {
        return Err(bad("exact_sender last-write refinement"));
    }
    let top_k = field(refine, "head_topk")?;
    if field(second, "head_topk")? != top_k {
        return Err(bad("refinement widths disagree"));
    }
    let bridge = &d["selected_prefix_code_bridge"];
    let target_layers = d["target_layer_ids"]
        .as_array()
        .ok_or_else(|| Error::Config("dspine: target_layer_ids missing".into()))?
        .iter()
        .map(|x| x.as_u64().map(|x| x as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| Error::Config("dspine: bad target_layer_ids".into()))?;
    let theta = v["rope_theta"]
        .as_f64()
        .or_else(|| v["rope_parameters"]["rope_theta"].as_f64())
        .unwrap_or(1e7) as f32;
    Ok(DFlashConfig {
        kind: DraftKind::DSpine,
        layers,
        heads: field(v, "num_attention_heads")?,
        kv_heads: field(v, "num_key_value_heads")?,
        head_dim: field(v, "head_dim")?,
        intermediate: field(v, "intermediate_size")?,
        eps: v["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        theta,
        window: field(v, "sliding_window")?,
        target_layers,
        mask_token: field(d, "mask_token_id")? as u32,
        conv_group: 0,
        rank: field(bridge, "rank")?,
        top_k,
        code_size: field(bridge, "code_size")?,
        max_block: field(v, "block_size")?,
    })
}

/// A BF16 matrix binding (whole object or a row range) read to the host.
fn host_bf16(
    l: &Loader<'_, impl Backend>,
    art: &Artifact,
    name: &str,
) -> Result<(Vec<u16>, usize)> {
    let (id, range) = l.binding(name)?;
    let obj = art
        .object(&id)
        .ok_or_else(|| Error::Config(format!("missing object {id}")))?;
    let k = match &obj.kind {
        ObjectKind::Tensor {
            shape,
            format,
            layout,
        } if shape.len() == 2
            && WeightFormat::from_names(format, layout).ok() == Some(WeightFormat::Bf16) =>
        {
            shape[1] as usize
        }
        _ => return Err(Error::Config(format!("{name}: expected a bf16 matrix"))),
    };
    let raw = art
        .read_object(&id)
        .map_err(|e| Error::Config(e.to_string()))?;
    let all: Vec<u16> = raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect();
    Ok(match range {
        None => (all, k),
        Some((a, e)) => (all[a as usize..e as usize].to_vec(), k),
    })
}

pub(super) fn load<B: Backend>(
    b: &B,
    art: &Artifact,
    hidden: usize,
    cfg: DFlashConfig,
) -> Result<DFlashModel<B>> {
    let mut l = Loader::new(b, art);
    let h = hidden;
    let (code, rank) = (cfg.code_size, cfg.rank);
    let feature_projection = l.matrix("dspine/feature_projection")?;
    if feature_projection.rows != h || feature_projection.k != cfg.target_layers.len() * h {
        return Err(Error::Config("dspine/feature_projection shape".into()));
    }
    let context_norm = l.param("dspine/context_norm", h)?;
    let final_norm = l.param("dspine/final_norm", h)?;
    let final_norm_host = l.host_param("dspine/final_norm", h)?;
    let mut layers = Vec::with_capacity(cfg.layers);
    let (mut read, mut write) = (Vec::new(), Vec::new());
    for i in 0..cfg.layers {
        let p = |s: &str| format!("dspine/layers/{i}/{s}");
        let qkv = l.rows_of(&[
            &p("attention/query"),
            &p("attention/key"),
            &p("attention/value"),
        ])?;
        if qkv.rows != cfg.q_dim() + 2 * cfg.kv_dim() {
            return Err(Error::Config("dspine attention shape".into()));
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
            attn_conv: None,
            mlp_conv: None,
        });
        let r = l.matrix(&format!("dspine/bridge/read/{i}"))?;
        let w = l.matrix(&format!("dspine/bridge/write/{i}"))?;
        if (r.rows, r.k) != (code, h) || (w.rows, w.k) != (h, rank) {
            return Err(Error::Config(format!("dspine bridge layer {i} shape")));
        }
        read.push(r);
        write.push(w);
    }
    let (g, pg, a) = (
        "dspine/bridge/gate",
        "dspine/bridge/predecessor_gate",
        "dspine/bridge/prev",
    );
    let packed = l.rows_of(&[g, pg, a])?;
    let pv = l.rows_of(&[pg, a])?;
    if (packed.rows, packed.k) != (3 * rank, code) {
        return Err(Error::Config("dspine bridge projection shape".into()));
    }
    let gate_bias = l.param("dspine/bridge/gate_bias", rank)?;
    let gate_bias_host = l.host_param("dspine/bridge/gate_bias", rank)?;
    let (codebook, ck) = host_bf16(&l, art, "dspine/bridge/codebook")?;
    let (write_last, wk) = host_bf16(&l, art, &format!("dspine/bridge/write/{}", cfg.layers - 1))?;
    if ck != code || codebook.len() % code != 0 || wk != rank || write_last.len() != h * rank {
        return Err(Error::Config("dspine codebook / write shape".into()));
    }
    let (parents, params, bytes) = l.finish();
    let bridge = Bridge {
        ones_h: b.upload(&vec![1.0; h])?,
        ones_c: b.upload(&vec![1.0; code])?,
        read,
        packed,
        pv,
        gate_bias,
        gate_bias_host,
        write,
        pending: Mutex::new(Some((codebook, write_last, final_norm_host))),
        tables: OnceLock::new(),
    };
    Ok(DFlashModel {
        cfg,
        hidden: h,
        parents,
        params,
        feature_projection,
        context_norm,
        final_norm,
        layers,
        hidden_projection: None,
        pred: Vec::new(),
        succ: Vec::new(),
        bridge: Some(bridge),
        full_head: false,
        bytes,
    })
}

impl<B: Backend> DFlashModel<B> {
    /// Builds the host tables once the target (whose output head they read) is known.
    pub(super) fn dspine_tables(&self, b: &B, target: &Qwen35Text<B>) -> Result<()> {
        let br = self.bridge.as_ref().expect("a DSpine drafter");
        if br.tables.get().is_some() {
            return Ok(());
        }
        let Some((codebook, write_last, norm)) = br
            .pending
            .lock()
            .map_err(|_| Error::Backend("dspine tables poisoned".into()))?
            .take()
        else {
            return Ok(());
        };
        let (h, r, c) = (self.hidden, self.cfg.rank, self.cfg.code_size);
        let tw = &target.w;
        let vocab = tw.head.rows;
        if codebook.len() != vocab * c || tw.head.k != h {
            return Err(Error::Config(format!(
                "dspine codebook has {} rows, the target's head {vocab}",
                codebook.len() / c
            )));
        }
        let round = |v: &[f32]| v.iter().map(|&x| f32_to_bf16(x)).collect::<Vec<u16>>();
        // TSV = sqrt(C) codebook . [G_p; A]^T, in chunks of rows.
        const CHUNK: usize = 2048;
        let sq = (c as f32).sqrt();
        let mut act = b.alloc(CHUNK * c)?;
        let mut out = b.alloc(CHUNK * 2 * r)?;
        let mut tsv = Vec::with_capacity(vocab * 2 * r);
        for v0 in (0..vocab).step_by(CHUNK) {
            let n = CHUNK.min(vocab - v0);
            let rows: Vec<f32> = codebook[v0 * c..(v0 + n) * c]
                .iter()
                .map(|&x| sq * bf16_to_f32(x))
                .collect();
            b.write_at(&mut act, 0, &rows)?;
            b.linear(&act, &self.view(&br.pv), &mut out, n)?;
            b.synchronize()?;
            let mut host = vec![0.0; n * 2 * r];
            b.download(&out, &mut host)?;
            tsv.extend(round(&host));
        }
        drop(codebook);
        // TG^T = (W_L^T * final_norm_w) . head^T: the rank rows through the target's head.
        const RG: usize = 16;
        let mut a = b.alloc(RG * h)?;
        let mut logits = b.alloc(RG * vocab)?;
        let mut tg = vec![0u16; vocab * r];
        let head = tw.view(&tw.head);
        for r0 in (0..r).step_by(RG) {
            let n = RG.min(r - r0);
            let mut rows = vec![0.0f32; n * h];
            for i in 0..n {
                for j in 0..h {
                    rows[i * h + j] = bf16_to_f32(write_last[j * r + r0 + i]) * norm[j];
                }
            }
            b.write_at(&mut a, 0, &rows)?;
            b.linear(&a, &head, &mut logits, n)?;
            b.synchronize()?;
            let mut host = vec![0.0; n * vocab];
            b.download(&logits, &mut host)?;
            for i in 0..n {
                for v in 0..vocab {
                    tg[v * r + r0 + i] = f32_to_bf16(host[i * vocab + v]);
                }
            }
        }
        let _ = br.tables.set(Tables { tsv, tg });
        Ok(())
    }

    /// One causal block per sequence: sequence `i` drafts `drafts` tokens after `anchors[i]` over
    /// its context `states[i]`. Returns per mask column the top-K candidates with their pre-write
    /// scores, and in `g` the column's `q` (`rank` values) for [`Self::dspine_select`].
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dspine_propose(
        &self,
        b: &B,
        target: &Qwen35Text<B>,
        states: &[&DFlashState<B>],
        dw: &mut DFlashWorkspace<B>,
        anchors: &[u32],
        drafts: usize,
        limit: usize,
    ) -> Result<Vec<BlockCandidates>> {
        let c = &self.cfg;
        let br = self.bridge.as_ref().expect("a DSpine drafter");
        let tables = br
            .tables
            .get()
            .ok_or_else(|| Error::Backend("dspine tables not built (workspace first)".into()))?;
        let n = states.len();
        let (h, w) = (self.hidden, drafts + 1);
        let (rows, k) = (n * w, drafts);
        let m = n * k;
        let (r, code) = (c.rank, c.code_size);
        if n == 0
            || anchors.len() != n
            || drafts == 0
            || w > c.max_block
            || w > c.window
            || rows > dw.block
            || w > dw.stage_rows
        {
            return Err(Error::Shape(format!(
                "dspine batch of {n} blocks of {w} (workspace {} rows, trained block {})",
                dw.block, c.max_block
            )));
        }
        let tw = &target.w;
        let (head, map) = self.proposal_head(target);
        let nh = head.rows;
        let head = tw.view(head);
        let mut ids = vec![c.mask_token; rows];
        let mut sv = Vec::with_capacity(n * 2 * r);
        for (i, &a) in anchors.iter().enumerate() {
            ids[i * w] = a;
            let row = tables
                .tsv
                .get(a as usize * 2 * r..(a as usize + 1) * 2 * r)
                .ok_or_else(|| Error::Shape(format!("dspine anchor {a} past the vocabulary")))?;
            sv.extend(row.iter().map(|&x| bf16_to_f32(x)));
        }
        let (qd, kvd) = (c.q_dim(), c.kv_dim());
        let last = self.layers.len() - 1;
        let sp = dw.spine.as_mut().expect("a DSpine workspace");
        // One recorded step per batch size and width. Static: positions, context rings and the
        // anchors' bridge rows are staged, so the recording serves whatever sequences it holds.
        let key = GRAPH_BLOCK_SPINE
            | w as u64
            | (n as u64) << 8
            | (dw.id.wrapping_mul(97) & 0xffff) << 40
            | GRAPH_STATIC;
        b.graph(key, &mut || {
            b.embed(&ids, &tw.view(&tw.embed), &mut dw.x)?;
            b.write_at(&mut sp.anchor_sv, 0, &sv)?;
            for (li, l) in self.layers.iter().enumerate() {
                // Attention.
                b.rms_norm(
                    &dw.x,
                    &self.params[l.input_norm],
                    &mut dw.n,
                    rows,
                    h,
                    c.eps,
                    NormKind::Plain,
                )?;
                b.linear(&dw.n, &self.view(&l.qkv), &mut dw.qkv, rows)?;
                b.split_cols(
                    &dw.qkv,
                    rows,
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
                    rows * c.heads,
                    c.head_dim,
                    c.eps,
                    NormKind::Plain,
                )?;
                b.rms_norm(
                    &dw.k,
                    &self.params[l.k_norm],
                    &mut dw.bkn,
                    rows * c.kv_heads,
                    c.head_dim,
                    c.eps,
                    NormKind::Plain,
                )?;
                for (i, st) in states.iter().enumerate() {
                    let o = i * w;
                    b.copy(&dw.qn, o * qd, &mut dw.sq, 0, w * qd)?;
                    b.copy(&dw.bkn, o * kvd, &mut dw.sk, 0, w * kvd)?;
                    b.copy(&dw.v, o * kvd, &mut dw.sv, 0, w * kvd)?;
                    let rope = |heads| RopeShape {
                        tokens: w,
                        heads,
                        head_dim: c.head_dim,
                        rotary_dim: c.head_dim,
                        pos0: st.end,
                        theta: c.theta,
                    };
                    b.rope(&mut dw.sq, rope(c.heads))?;
                    b.rope(&mut dw.sk, rope(c.kv_heads))?;
                    b.attention_block(
                        &dw.sq,
                        &st.k[li],
                        &st.v[li],
                        &dw.sk,
                        &dw.sv,
                        &mut dw.sattn,
                        BlockAttnShape {
                            tokens: w,
                            heads: c.heads,
                            kv_heads: c.kv_heads,
                            head_dim: c.head_dim,
                            pos0: st.end,
                            ctx_lo: st.lo,
                            window: c.window,
                            ring: c.window,
                            causal: true,
                        },
                    )?;
                    b.copy(&dw.sattn, 0, &mut dw.attn, o * qd, w * qd)?;
                }
                b.linear(&dw.attn, &self.view(&l.output), &mut dw.z, rows)?;
                b.add(&mut dw.x, &dw.z, rows * h)?;
                // MLP.
                b.rms_norm(
                    &dw.x,
                    &self.params[l.post_norm],
                    &mut dw.n,
                    rows,
                    h,
                    c.eps,
                    NormKind::Plain,
                )?;
                match &l.gate_up {
                    Some(f) => {
                        b.linear(&dw.n, &self.view(f), &mut dw.gu, rows)?;
                        let (g, u) = (l.gate.row_begin - f.row_begin, l.up.row_begin - f.row_begin);
                        b.swiglu(&dw.gu, rows, f.rows, g, u, c.intermediate, &mut dw.mg)?;
                    }
                    None => {
                        b.linear(&dw.n, &self.view(&l.gate), &mut dw.mg, rows)?;
                        b.linear(&dw.n, &self.view(&l.up), &mut dw.mu, rows)?;
                        b.silu_mul(&mut dw.mg, &dw.mu, rows * c.intermediate)?;
                    }
                }
                b.linear(&dw.mg, &self.view(&l.down), &mut dw.z, rows)?;
                b.add(&mut dw.x, &dw.z, rows * h)?;
                // Code bridge: read the row's code, message from its predecessor, write.
                b.rms_norm(
                    &dw.x,
                    &br.ones_h,
                    &mut dw.n,
                    rows,
                    h,
                    c.eps,
                    NormKind::Plain,
                )?;
                b.linear(&dw.n, &self.view(&br.read[li]), &mut sp.proj, rows)?;
                b.rms_norm(
                    &sp.proj,
                    &br.ones_c,
                    &mut sp.codes,
                    rows,
                    code,
                    1e-12,
                    NormKind::Plain,
                )?;
                b.linear(&sp.codes, &self.view(&br.packed), &mut sp.packed, rows)?;
                for i in 0..n {
                    b.copy(
                        &sp.anchor_sv,
                        i * 2 * r,
                        &mut sp.packed,
                        i * w * 3 * r + r,
                        2 * r,
                    )?;
                }
                b.prev_gate(
                    &sp.packed,
                    &self.params[br.gate_bias],
                    &mut sp.msg,
                    PrevGateShape {
                        rows,
                        rank: r,
                        block: w,
                    },
                )?;
                b.linear(&sp.msg, &self.view(&br.write[li]), &mut dw.z, rows)?;
                if li == last {
                    // Each column's q.
                    for i in 0..n {
                        b.copy2d(
                            &sp.packed,
                            &mut dw.g,
                            Copy2dShape {
                                rows: k,
                                cols: r,
                                src_offset: (i * w + 1) * 3 * r,
                                src_stride: 3 * r,
                                dst_offset: i * k * r,
                                dst_stride: r,
                            },
                        )?;
                    }
                }
                b.rms_residual_add(
                    &mut dw.x,
                    &dw.z,
                    &mut sp.rms,
                    RmsResidualShape {
                        rows,
                        cols: h,
                        block: w,
                    },
                )?;
            }
            for i in 0..n {
                b.copy(&dw.x, (i * w + 1) * h, &mut dw.tail, i * k * h, k * h)?;
            }
            // rms of the written rows: with blocks of one row every row is a block's first, so
            // `rms_residual_add` only measures.
            b.rms_residual_add(
                &mut dw.tail,
                &dw.z,
                &mut sp.rms_post,
                RmsResidualShape {
                    rows: m,
                    cols: h,
                    block: 1,
                },
            )?;
            b.rms_norm(
                &dw.tail,
                &self.params[self.final_norm],
                &mut dw.hid,
                m,
                h,
                c.eps,
                NormKind::Plain,
            )?;
            b.linear(&dw.hid, &head, &mut dw.logits, m)?;
            b.copy(&dw.g, 0, &mut dw.g_host, 0, m * r)?;
            // What recovers the pre-write scores: rms(u) and the last layer's messages of every
            // block row, the written rows' rms.
            b.copy(&sp.rms, 0, &mut sp.aux_host, 0, rows)?;
            b.copy(&sp.msg, 0, &mut sp.aux_host, rows, rows * r)?;
            b.copy(&sp.rms_post, 0, &mut sp.aux_host, rows + rows * r, m)
        })?;
        let fetch = c.top_k + if map.is_some() { 8 } else { 0 };
        let top = b.top_k(
            &dw.logits,
            m,
            nh,
            if map.is_some() { nh } else { limit.min(nh) },
            fetch,
        )?;
        let g = host_rows(b, &dw.g_host, m * r)?;
        let aux = host_rows(b, &sp.aux_host, rows + rows * r + m)?;
        let (rms_u, msg, rms_x) = (
            &aux[..rows],
            &aux[rows..rows + rows * r],
            &aux[rows + rows * r..],
        );
        let eps = f64::from(c.eps);
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let cands = (0..k)
                .map(|t| {
                    let (row, j) = (i * w + 1 + t, i * k + t);
                    // The head reads the final norm of x = u + rms(u) W m, so with
                    // TG[c] = W^T (final_norm_w * head_c):
                    //   head_c . final_norm(u) = (logit_c rms_eps(x) - rms(u) m . TG[c]) / rms_eps(u)
                    // (the written row's logit, the parallel pass's message m).
                    let ru = f64::from(rms_u[row]);
                    let (sx, su) = (
                        (f64::from(rms_x[j]).powi(2) + eps).sqrt(),
                        (ru * ru + eps).sqrt(),
                    );
                    let mp = &msg[row * r..(row + 1) * r];
                    top[j]
                        .iter()
                        .map(|&(x, v)| (map.map_or(x, |ids| ids[x as usize]), v))
                        .filter(|&(id, _)| (id as usize) < limit)
                        .take(c.top_k)
                        .map(|(id, logit)| {
                            let tg = &tables.tg[id as usize * r..][..r];
                            let dot: f64 = mp
                                .iter()
                                .zip(tg)
                                .map(|(&a, &b)| f64::from(a) * f64::from(bf16_to_f32(b)))
                                .sum();
                            (id, ((f64::from(logit) * sx - ru * dot) / su) as f32)
                        })
                        .collect()
                })
                .collect();
            out.push(BlockCandidates {
                cands,
                g: g[i * k * r..(i + 1) * k * r].to_vec(),
            });
        }
        Ok(out)
    }

    /// The serial walk: column `i` scores its candidates against the draft before it (the anchor
    /// for the first), then draws as [`Self::select`] does.
    #[allow(clippy::type_complexity)]
    pub(super) fn dspine_select(
        &self,
        br: &Bridge<B>,
        bc: &BlockCandidates,
        anchor: u32,
        proposal: &mut dyn FnMut(&[(u32, f32)], &[u32]) -> Vec<(u32, f32)>,
        uniform: &mut dyn FnMut() -> f32,
    ) -> (Vec<u32>, Vec<Vec<(u32, f32)>>) {
        let r = self.cfg.rank;
        let t = br
            .tables
            .get()
            .expect("dspine tables are built with the workspace");
        let bf = bf16_to_f32;
        let mut prev = anchor;
        let mut msg = vec![0.0f32; r];
        let (mut drafts, mut qs): (Vec<u32>, Vec<Vec<(u32, f32)>>) = (Vec::new(), Vec::new());
        for (i, cands) in bc.cands.iter().enumerate() {
            if cands.is_empty() {
                break;
            }
            let q = &bc.g[i * r..(i + 1) * r];
            let sv = &t.tsv[prev as usize * 2 * r..][..2 * r];
            for (j, mj) in msg.iter_mut().enumerate() {
                let x = q[j] + bf(sv[j]) + br.gate_bias_host[j];
                *mj = bf(sv[r + j]) / (1.0 + (-x).exp());
            }
            let scored: Vec<(u32, f32)> = cands
                .iter()
                .map(|&(id, base)| {
                    let tg = &t.tg[id as usize * r..][..r];
                    (
                        id,
                        base + msg.iter().zip(tg).map(|(&a, &b)| a * bf(b)).sum::<f32>(),
                    )
                })
                .collect();
            let q = proposal(&scored, &drafts);
            if q.is_empty() {
                break;
            }
            let mut pick = q.len() - 1;
            if q.len() > 1 {
                let mut u = uniform();
                for (j, &(_, x)) in q.iter().enumerate() {
                    u -= x;
                    if u <= 0.0 {
                        pick = j;
                        break;
                    }
                }
            }
            prev = q[pick].0;
            drafts.push(prev);
            qs.push(q);
        }
        (drafts, qs)
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The fields of tencent/Qwen3.8-27B-DSpine-b16's config.json that velox reads.
    fn b16() -> Value {
        json!({
            "architectures": ["DFlashDraftModel"],
            "num_hidden_layers": 5,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "head_dim": 128,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000000.0,
            "sliding_window": 8192,
            "block_size": 16,
            "vocab_size": 248320,
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "code_bridge_second_pass": { "mode": "serial_table", "first_exact": 4, "head_topk": 16 },
            "dflash_config": {
                "target_layer_ids": [1, 16, 31, 46, 61],
                "mask_token_id": 248070,
                "draft_block_causal": true,
                "selected_prefix_terminal": false,
                "selected_prefix_feature_write_mode": "independent",
                "selected_prefix_feature_layers": [1, 2, 3, 4, 5],
                "selected_prefix_code_bridge": { "code_size": 1024, "eps": 1e-6, "kappa": 10.0, "rank": 512 },
                "selected_prefix_last_write_refine": { "kind": "exact_sender", "head_topk": 16, "version": 1, "weight": 1.0 }
            }
        })
    }

    #[test]
    fn b16_config_parses_as_dspine() {
        let c = DFlashConfig::from_json(&b16()).unwrap();
        assert_eq!(c.kind, DraftKind::DSpine);
        assert_eq!((c.layers, c.heads, c.kv_heads, c.head_dim), (5, 32, 8, 128));
        assert_eq!(
            (c.code_size, c.rank, c.top_k, c.max_block),
            (1024, 512, 16, 16)
        );
        assert_eq!((c.window, c.mask_token, c.conv_group), (8192, 248070, 0));
        assert_eq!(c.target_layers, vec![1, 16, 31, 46, 61]);
        assert_eq!(c.theta, 1e7);
    }

    #[test]
    fn unsupported_bridges_are_refused() {
        let cases: [(&str, Value); 4] = [
            ("/dflash_config/draft_block_causal", json!(false)),
            ("/dflash_config/selected_prefix_terminal", json!(true)),
            ("/code_bridge_second_pass/first_exact", json!(2)),
            (
                "/dflash_config/selected_prefix_feature_layers",
                json!([1, 2, 5]),
            ),
        ];
        for (path, value) in cases {
            let mut v = b16();
            *v.pointer_mut(path).unwrap() = value;
            assert!(DFlashConfig::from_json(&v).is_err(), "{path} accepted");
        }
    }
}
