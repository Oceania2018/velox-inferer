//! Qwen3.5 vision tower and merger: image patches -> text-width embeddings.
//! The mathematics is specified in docs/qwen3_5-vision.md.

use std::sync::Mutex;

use serde_json::Value;
use velox_core::backend::{Backend, FullAttnShape, GeluKind, RopeLayout, RopePosShape, WeightView};
use velox_core::{Error, Result};
use velox_weights::Artifact;

use super::weights::{Loader, WDesc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub heads: usize,
    pub patch: usize,
    pub temporal: usize,
    pub merge: usize,
    /// Side of the learned square position table (`sqrt(num_position_embeddings)`).
    pub pos_side: usize,
    /// Output width (the text hidden size).
    pub out_hidden: usize,
}

impl VisionConfig {
    pub fn from_json(v: &Value, out_hidden: usize) -> Result<Self> {
        let f = |k: &str| {
            v.get(k)
                .and_then(Value::as_u64)
                .filter(|&x| x > 0)
                .map(|x| x as usize)
                .ok_or_else(|| {
                    Error::Config(format!("vision config: '{k}' must be a positive integer"))
                })
        };
        let positions = f("num_position_embeddings")?;
        let pos_side = (positions as f64).sqrt().round() as usize;
        if pos_side * pos_side != positions {
            return Err(Error::Config(format!(
                "num_position_embeddings {positions} is not a square"
            )));
        }
        let cfg = VisionConfig {
            depth: f("depth")?,
            hidden: f("hidden_size")?,
            intermediate: f("intermediate_size")?,
            heads: f("num_heads")?,
            patch: f("patch_size")?,
            temporal: f("temporal_patch_size")?,
            merge: f("spatial_merge_size")?,
            pos_side,
            out_hidden,
        };
        if cfg.hidden % cfg.heads != 0 || cfg.merge != 2 {
            return Err(Error::Unsupported(format!("vision geometry {cfg:?}")));
        }
        Ok(cfg)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden / self.heads
    }

    /// Features per patch: 3 channels x temporal x patch x patch.
    pub fn patch_features(&self) -> usize {
        3 * self.temporal * self.patch * self.patch
    }
}

struct VLayer {
    norm1: (usize, usize),
    norm2: (usize, usize),
    q: (WDesc, usize),
    k: (WDesc, usize),
    v: (WDesc, usize),
    out: (WDesc, usize),
    fc1: (WDesc, usize),
    fc2: (WDesc, usize),
}

pub struct Qwen35Vision<B: Backend> {
    pub cfg: VisionConfig,
    parents: Vec<B::Buffer>,
    params: Vec<B::Buffer>,
    /// `[pos_side * pos_side, hidden]`, interpolated on the host per image grid.
    pos_table: Vec<f32>,
    patch: (WDesc, usize),
    layers: Vec<VLayer>,
    merger_norm: (usize, usize),
    merger_fc1: (WDesc, usize),
    merger_fc2: (WDesc, usize),
    pub bytes: u64,
    /// Activations of the latest image grid, reused while the grid repeats (agent screenshots):
    /// fresh buffers per layer would allocate and free, and a free synchronizes the device.
    ws: Mutex<Option<VisionWs<B>>>,
}

struct VisionWs<B: Backend> {
    grid: (usize, usize),
    /// Patches the buffers hold: the largest grid seen (the engine's warm-up encodes the largest
    /// its pixel budget allows), so smaller images reuse them and nothing is reallocated.
    cap: usize,
    /// Interpolated position embeddings of the grid.
    pos: B::Buffer,
    x: B::Buffer,
    h: B::Buffer,
    attn: B::Buffer,
    q: B::Buffer,
    k: B::Buffer,
    v: B::Buffer,
    o: B::Buffer,
    f: B::Buffer,
    m: B::Buffer,
}

impl<B: Backend> Qwen35Vision<B> {
    pub fn load(b: &B, art: &Artifact, text_hidden: usize) -> Result<Self> {
        let cv = art
            .components()
            .get("vision")
            .ok_or_else(|| Error::Unsupported("artifact has no vision component".into()))?;
        let cfg = VisionConfig::from_json(&cv["config"], text_hidden)?;
        let (h, i) = (cfg.hidden, cfg.intermediate);
        let mut l = Loader::new(b, art);
        let lin =
            |l: &mut Loader<'_, B>, w: &str, bias: &str, rows: usize| -> Result<(WDesc, usize)> {
                let d = l.matrix(w)?;
                if d.rows != rows {
                    return Err(Error::Config(format!(
                        "{w}: {} rows, expected {rows}",
                        d.rows
                    )));
                }
                Ok((d, l.param(bias, rows)?))
            };
        let patch = lin(
            &mut l,
            "vision/patch_embedding",
            "vision/patch_embedding_bias",
            h,
        )?;
        let pos_table =
            l.host_param("vision/position_embedding", cfg.pos_side * cfg.pos_side * h)?;
        let mut layers = Vec::with_capacity(cfg.depth);
        for n in 0..cfg.depth {
            let p = |s: &str| format!("vision/layers/{n}/{s}");
            layers.push(VLayer {
                norm1: (
                    l.param(&p("norm1_weight"), h)?,
                    l.param(&p("norm1_bias"), h)?,
                ),
                norm2: (
                    l.param(&p("norm2_weight"), h)?,
                    l.param(&p("norm2_bias"), h)?,
                ),
                q: lin(&mut l, &p("attention/query"), &p("attention/query_bias"), h)?,
                k: lin(&mut l, &p("attention/key"), &p("attention/key_bias"), h)?,
                v: lin(&mut l, &p("attention/value"), &p("attention/value_bias"), h)?,
                out: lin(
                    &mut l,
                    &p("attention/output"),
                    &p("attention/output_bias"),
                    h,
                )?,
                fc1: lin(&mut l, &p("mlp/fc1"), &p("mlp/fc1_bias"), i)?,
                fc2: lin(&mut l, &p("mlp/fc2"), &p("mlp/fc2_bias"), h)?,
            });
        }
        let merged = h * cfg.merge * cfg.merge;
        let merger_norm = (
            l.param("vision/merger/norm_weight", h)?,
            l.param("vision/merger/norm_bias", h)?,
        );
        let merger_fc1 = lin(
            &mut l,
            "vision/merger/fc1",
            "vision/merger/fc1_bias",
            merged,
        )?;
        let merger_fc2 = lin(
            &mut l,
            "vision/merger/fc2",
            "vision/merger/fc2_bias",
            text_hidden,
        )?;
        let (parents, params, bytes) = l.finish();
        Ok(Qwen35Vision {
            cfg,
            parents,
            params,
            pos_table,
            patch,
            layers,
            merger_norm,
            merger_fc1,
            merger_fc2,
            bytes,
            ws: Mutex::new(None),
        })
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

    /// `y = x . W^T + bias` for `m` rows.
    fn linear_bias(
        &self,
        b: &B,
        x: &B::Buffer,
        w: &(WDesc, usize),
        y: &mut B::Buffer,
        m: usize,
    ) -> Result<()> {
        b.linear(x, &self.view(&w.0), y, m)?;
        b.add_bias(y, &self.params[w.1], m, w.0.rows)
    }

    /// Raw patch coordinates `(y, x)` of a `gh x gw` grid in merge-group order.
    pub fn patch_coords(gh: usize, gw: usize) -> Vec<(usize, usize)> {
        let mut v = Vec::with_capacity(gh * gw);
        for by in 0..gh / 2 {
            for bx in 0..gw / 2 {
                for my in 0..2 {
                    for mx in 0..2 {
                        v.push((2 * by + my, 2 * bx + mx));
                    }
                }
            }
        }
        v
    }

    /// Bilinear (align-corners) interpolation of the position table at each patch coordinate.
    fn position_embeddings(&self, coords: &[(usize, usize)], gh: usize, gw: usize) -> Vec<f32> {
        let (h, side) = (self.cfg.hidden, self.cfg.pos_side);
        let last = (side - 1) as f32;
        let scale = |n: usize| if n > 1 { last / (n - 1) as f32 } else { 0.0 };
        let (sy, sx) = (scale(gh), scale(gw));
        let mut out = vec![0f32; coords.len() * h];
        for (p, &(y, x)) in coords.iter().enumerate() {
            let (yf, xf) = (y as f32 * sy, x as f32 * sx);
            let (y0, x0) = (yf as usize, xf as usize);
            let (y1, x1) = ((y0 + 1).min(side - 1), (x0 + 1).min(side - 1));
            let (wy, wx) = (yf - y0 as f32, xf - x0 as f32);
            let corners = [
                ((1.0 - wy) * (1.0 - wx), y0 * side + x0),
                ((1.0 - wy) * wx, y0 * side + x1),
                (wy * (1.0 - wx), y1 * side + x0),
                (wy * wx, y1 * side + x1),
            ];
            let row = &mut out[p * h..(p + 1) * h];
            for (w, idx) in corners {
                for (o, &t) in row.iter_mut().zip(&self.pos_table[idx * h..(idx + 1) * h]) {
                    *o += w * t;
                }
            }
        }
        out
    }

    /// Encodes one image: `patches` is `[gh * gw, patch_features]` in merge-group order. Returns
    /// `[gh * gw / 4, out_hidden]` embeddings, one per merged image token.
    pub fn encode(&self, b: &B, patches: &[f32], gh: usize, gw: usize) -> Result<B::Buffer> {
        let c = &self.cfg;
        let n = gh * gw;
        if gh % 2 != 0 || gw % 2 != 0 || patches.len() != n * c.patch_features() {
            return Err(Error::Shape(format!(
                "vision input: grid {gh}x{gw}, {} features",
                patches.len()
            )));
        }
        let (h, d) = (c.hidden, c.head_dim());
        let coords = Self::patch_coords(gh, gw);
        let positions: Vec<u32> = coords
            .iter()
            .map(|p| p.0 as u32)
            .chain(coords.iter().map(|p| p.1 as u32))
            .collect();
        let rope = RopePosShape {
            tokens: n,
            heads: c.heads,
            head_dim: d,
            rotary_dim: d,
            theta: 10_000.0,
            axes: 2,
            layout: RopeLayout::Blocked,
        };

        let mut guard = self
            .ws
            .lock()
            .map_err(|_| Error::Backend("vision workspace poisoned".into()))?;
        if guard
            .as_ref()
            .is_some_and(|w| w.grid != (gh, gw) && w.cap >= n)
        {
            // Same buffers, this grid's position embeddings (the old ones freed first).
            let ws = guard.as_mut().expect("checked");
            ws.grid = (gh, gw);
            ws.pos = b.alloc(1)?;
            ws.pos = b.upload(&self.position_embeddings(&coords, gh, gw))?;
        }
        if guard.as_ref().is_none_or(|w| w.cap < n) {
            *guard = None; // free the smaller buffers first
            let inter = c.intermediate.max(self.merger_fc1.0.rows);
            *guard = Some(VisionWs {
                grid: (gh, gw),
                cap: n,
                pos: b.upload(&self.position_embeddings(&coords, gh, gw))?,
                x: b.alloc(n * h)?,
                h: b.alloc(n * h)?,
                attn: b.alloc(n * h)?,
                q: b.alloc(n * h)?,
                k: b.alloc(n * h)?,
                v: b.alloc(n * h)?,
                o: b.alloc(n * h)?,
                f: b.alloc(n * inter)?,
                m: b.alloc(n * inter)?,
            });
        }
        let ws = guard.as_mut().expect("allocated above");
        let VisionWs {
            pos,
            x,
            h: hbuf,
            attn,
            q,
            k,
            v,
            o,
            f,
            m,
            ..
        } = ws;
        let input = b.upload(patches)?;
        self.linear_bias(b, &input, &self.patch, x, n)?;
        b.add(x, pos, n * h)?;

        for l in &self.layers {
            b.layer_norm(
                x,
                &self.params[l.norm1.0],
                &self.params[l.norm1.1],
                hbuf,
                n,
                h,
                1e-6,
            )?;
            self.linear_bias(b, hbuf, &l.q, q, n)?;
            self.linear_bias(b, hbuf, &l.k, k, n)?;
            self.linear_bias(b, hbuf, &l.v, v, n)?;
            b.rope_positions(q, &positions, rope)?;
            b.rope_positions(k, &positions, rope)?;
            b.attention_full(
                q,
                k,
                v,
                attn,
                FullAttnShape {
                    tokens: n,
                    heads: c.heads,
                    head_dim: d,
                },
            )?;
            self.linear_bias(b, attn, &l.out, o, n)?;
            b.add(x, o, n * h)?;
            b.layer_norm(
                x,
                &self.params[l.norm2.0],
                &self.params[l.norm2.1],
                hbuf,
                n,
                h,
                1e-6,
            )?;
            self.linear_bias(b, hbuf, &l.fc1, f, n)?;
            b.gelu(f, n * c.intermediate, GeluKind::Tanh)?;
            self.linear_bias(b, f, &l.fc2, o, n)?;
            b.add(x, o, n * h)?;
        }
        // Merger: patches are merge-group ordered, so [n, h] is already [n/4, 4h].
        b.layer_norm(
            x,
            &self.params[self.merger_norm.0],
            &self.params[self.merger_norm.1],
            hbuf,
            n,
            h,
            1e-6,
        )?;
        let tokens = n / 4;
        self.linear_bias(b, hbuf, &self.merger_fc1, m, tokens)?;
        b.gelu(m, tokens * self.merger_fc1.0.rows, GeluKind::Erf)?;
        let mut out = b.alloc(tokens * self.merger_fc2.0.rows)?;
        self.linear_bias(b, m, &self.merger_fc2, &mut out, tokens)?;
        Ok(out)
    }
}
