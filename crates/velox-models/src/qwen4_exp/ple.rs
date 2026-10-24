//! PLE n-gram hashing and table rows on the host (docs/qwen4_exp.md#ple-per-layer-n-gram-embedding-layer-1).
//!
//! The table (`per_layer_token_embd`, 28.8 GB in the IQ2_XS GGUF) stays in the file: a token needs
//! one row per hashed head, read positionally and decoded here.

use velox_core::format::decode_rows;
use velox_core::ggml::GgmlType;
use velox_core::{Error, Result, WeightFormat};
use velox_weights::gguf::Gguf;

use super::config::PleConfig;
use super::load::Shards;

pub struct PleTable {
    file: Gguf,
    offset: u64,
    rows: u64,
    row_bytes: usize,
    format: WeightFormat,
    cfg: PleConfig,
}

impl PleTable {
    pub fn open(shards: &Shards, cfg: &PleConfig) -> Result<Self> {
        let (g, t) = shards.tensor("per_layer_token_embd.weight")?;
        let [k, rows] = t.dims[..] else {
            return Err(Error::Config("per_layer_token_embd is not a matrix".into()));
        };
        if k as usize != cfg.row_dim {
            return Err(Error::Config(format!(
                "per_layer_token_embd rows of {k}, config implies {}",
                cfg.row_dim
            )));
        }
        let ty = GgmlType::from_gguf_id(t.ggml_type)
            .ok_or_else(|| Error::Unsupported(format!("PLE table of ggml type {}", t.ggml_type)))?;
        let need = (0..cfg.heads())
            .map(|h| cfg.offsets[h] + cfg.vocab_sizes[h])
            .max()
            .unwrap_or(0);
        if need > rows {
            return Err(Error::Config(format!(
                "PLE table of {rows} rows, hash heads reach {need}"
            )));
        }
        // A second handle on the shard: reads here are positional and independent.
        let file = Gguf::open(&g.path).map_err(|e| Error::Config(e.to_string()))?;
        Ok(PleTable {
            file,
            offset: t.offset,
            rows,
            row_bytes: ty.row_bytes(k as usize),
            format: WeightFormat::Ggml(ty),
            cfg: cfg.clone(),
        })
    }

    /// Table rows of every hashed head of each token in `ids` (see [`hash_rows`]).
    pub fn rows(&self, prev: &[u32], ids: &[u32]) -> Vec<u64> {
        hash_rows(&self.cfg, prev, ids)
    }

    /// The decoded embedding `[ids.len(), heads * row_dim]` of `rows` (from [`Self::rows`]),
    /// read with `threads` parallel readers.
    pub fn embed(&self, rows: &[u64], threads: usize) -> Result<Vec<f32>> {
        let d = self.cfg.row_dim;
        let mut out = vec![0f32; rows.len() * d];
        let per = rows.len().div_ceil(threads.max(1)).max(1);
        std::thread::scope(|scope| -> Result<()> {
            let jobs: Vec<_> = rows
                .chunks(per)
                .zip(out.chunks_mut(per * d))
                .map(|(rs, dst)| scope.spawn(move || self.read_rows(rs, dst)))
                .collect();
            for j in jobs {
                j.join()
                    .map_err(|_| Error::Backend("PLE reader panicked".into()))??;
            }
            Ok(())
        })?;
        Ok(out)
    }

    fn read_rows(&self, rows: &[u64], dst: &mut [f32]) -> Result<()> {
        let d = self.cfg.row_dim;
        let mut raw = vec![0u8; self.row_bytes];
        for (&r, out) in rows.iter().zip(dst.chunks_exact_mut(d)) {
            if r >= self.rows {
                return Err(Error::Shape(format!("PLE row {r} of {}", self.rows)));
            }
            self.file
                .read_at(self.offset + r * self.row_bytes as u64, &mut raw)
                .map_err(|e| Error::Config(e.to_string()))?;
            out.copy_from_slice(&decode_rows(self.format, &raw, 1, d, 0, 1)?);
        }
        Ok(())
    }
}

/// Table rows of every hashed head of each token in `ids`, whose sequence so far ends with `prev`
/// (oldest first; shorter at the sequence start): `[ids.len(), heads]`.
pub fn hash_rows(c: &PleConfig, prev: &[u32], ids: &[u32]) -> Vec<u64> {
    let (heads, n) = (c.heads(), c.ngram);
    let mut out = Vec::with_capacity(ids.len() * heads);
    let mut ctx = vec![0u64; n];
    for t in 0..ids.len() {
        // Predecessor s back; at or before an EOS, and before the sequence, it reads as EOS.
        ctx[0] = u64::from(ids[t]);
        let mut cut = false;
        for (s, slot) in ctx.iter_mut().enumerate().skip(1) {
            let tok = if cut {
                None
            } else if s <= t {
                Some(ids[t - s])
            } else {
                prev.len().checked_sub(s - t).map(|i| prev[i])
            };
            cut = tok.is_none_or(|v| v == c.eos);
            *slot = u64::from(tok.unwrap_or(c.eos));
        }
        for order in 2..=n {
            let mut mixed = ctx[0].wrapping_mul(c.multipliers[0]);
            for (v, m) in ctx[1..order].iter().zip(&c.multipliers[1..order]) {
                mixed ^= v.wrapping_mul(*m);
            }
            let base = (order - 2) * c.heads_per_ngram;
            for h in base..base + c.heads_per_ngram {
                out.push(mixed % c.vocab_sizes[h] + c.offsets[h]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> PleConfig {
        PleConfig {
            layer: 1,
            ngram: 3,
            heads_per_ngram: 2,
            conv_width: 4,
            dilation: 3,
            eos: 9,
            image_token: 9,
            row_dim: 4,
            multipliers: vec![3, 5, 7],
            offsets: vec![0, 11, 24, 41],
            vocab_sizes: vec![11, 13, 17, 19],
        }
    }

    /// The reference hash of transformers' `Qwen4ExpTextNGramEmbedding` for one token.
    fn reference(c: &PleConfig, ctx: [u64; 3]) -> Vec<u64> {
        let mut out = Vec::new();
        for order in 2..=3 {
            let mut m = ctx[0] * c.multipliers[0];
            for (v, mul) in ctx[1..order].iter().zip(&c.multipliers[1..order]) {
                m ^= v * mul;
            }
            let base = (order - 2) * c.heads_per_ngram;
            for h in base..base + c.heads_per_ngram {
                out.push(m % c.vocab_sizes[h] + c.offsets[h]);
            }
        }
        out
    }

    #[test]
    fn hashes_follow_eos_cuts_and_history() {
        let c = cfg();
        // Sequence start: missing predecessors read as EOS.
        assert_eq!(hash_rows(&c, &[], &[4]), reference(&c, [4, 9, 9]));
        // History from an earlier chunk.
        assert_eq!(hash_rows(&c, &[2, 3], &[4]), reference(&c, [4, 3, 2]));
        // An EOS predecessor cuts everything before it; the token itself is never replaced.
        let r = hash_rows(&c, &[5], &[9, 6, 7]);
        assert_eq!(&r[..4], &reference(&c, [9, 5, 9])[..]);
        assert_eq!(&r[4..8], &reference(&c, [6, 9, 9])[..]);
        assert_eq!(&r[8..], &reference(&c, [7, 6, 9])[..]);
    }
}
