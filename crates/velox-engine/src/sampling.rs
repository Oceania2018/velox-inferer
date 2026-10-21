//! Token sampling over device-selected candidates.
//!
//! The model returns the [`CANDIDATES`] best (id, logit) pairs per step (`Backend::top_k`), so the
//! host never sees the vocabulary. Order of operations: presence penalty, then greedy (temperature
//! 0) or temperature, top-k, top-p. A penalty only lowers logits, so the true top-k (k <= 20)
//! after the penalty is always inside the 64 candidates unless more than 44 of them were already
//! generated.
//!
//! Defaults when the request omits a parameter are Qwen's presets, the same as ninfer's:
//! non-thinking temperature 0.7, top_k 20, top_p 0.8, presence_penalty 1.5; thinking 1.0, 20,
//! 0.95, 0.

use std::collections::HashSet;

use serde_json::Value;
use velox_core::rng::Rng;

/// Candidates fetched from the device per sampled token.
pub const CANDIDATES: usize = 64;
/// Largest honoured `top_k` (ninfer's limit too).
pub const MAX_TOP_K: usize = 20;

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    presence_penalty: f32,
    /// Tokens generated so far (presence penalty applies to the output).
    seen: HashSet<u32>,
    rng: Rng,
}

impl Sampler {
    pub fn from_request(req: &Value, thinking: bool) -> Self {
        let f = |k: &str, d: f32| req.get(k).and_then(Value::as_f64).map_or(d, |v| v as f32);
        let (t, p, pp) = if thinking {
            (1.0, 0.95, 0.0)
        } else {
            (0.7, 0.8, 1.5)
        };
        let seed = req.get("seed").and_then(Value::as_u64).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| d.as_nanos() as u64)
        });
        let top_k = req
            .get("top_k")
            .and_then(Value::as_i64)
            .filter(|&k| k > 0)
            .map_or(MAX_TOP_K, |k| (k as usize).min(MAX_TOP_K));
        Sampler {
            temperature: f("temperature", t).max(0.0),
            top_k,
            top_p: f("top_p", p).clamp(0.0, 1.0),
            presence_penalty: f("presence_penalty", pp),
            seen: HashSet::new(),
            rng: Rng::new(seed),
        }
    }

    pub fn greedy() -> Self {
        Sampler {
            temperature: 0.0,
            top_k: 1,
            top_p: 1.0,
            presence_penalty: 0.0,
            seen: HashSet::new(),
            rng: Rng::new(1),
        }
    }

    /// Records a generated token for the presence penalty.
    pub fn observe(&mut self, token: u32) {
        if self.presence_penalty != 0.0 {
            self.seen.insert(token);
        }
    }

    /// Picks the next token from candidates sorted by logit (descending).
    pub fn sample(&mut self, cands: &[(u32, f32)]) -> u32 {
        let d = self.distribution(cands);
        self.draw(&d)
    }

    /// Speculative sampling with a deterministic draft: accepts `draft` with probability p(draft)
    /// under this step's distribution. On rejection returns the token drawn from p with `draft`
    /// removed (the residual max(p - q, 0) for a one-hot q), so the output distribution is exactly
    /// that of [`Self::sample`]. Greedy accepts iff the draft is the argmax.
    pub fn verify_draft(&mut self, cands: &[(u32, f32)], draft: u32) -> Result<(), u32> {
        self.verify_draft_q(cands, draft, &[(draft, 1.0)])
    }

    /// Speculative sampling with a draft drawn from the proposal distribution `q` (id,
    /// probability): accepts with probability min(1, p(draft) / q(draft)); on rejection returns a
    /// token drawn from the residual max(p - q, 0), so the output follows this step's distribution
    /// exactly. Greedy accepts iff the draft is the argmax.
    pub fn verify_draft_q(
        &mut self,
        cands: &[(u32, f32)],
        draft: u32,
        q: &[(u32, f32)],
    ) -> Result<(), u32> {
        let d = self.distribution(cands);
        let prob = |dist: &[(u32, f64)], id: u32| {
            dist.iter()
                .find(|&&(t, _)| t == id)
                .map_or(0.0, |&(_, p)| p)
        };
        let qf: Vec<(u32, f64)> = q.iter().map(|&(t, p)| (t, f64::from(p))).collect();
        let (p, qd) = (prob(&d, draft), prob(&qf, draft).max(1e-12));
        if p > 0.0 && (p >= qd || f64::from(self.rng.uniform() * 0.5 + 0.5) * qd <= p) {
            return Ok(());
        }
        let rest: Vec<(u32, f64)> = d
            .iter()
            .map(|&(id, pv)| (id, (pv - prob(&qf, id)).max(0.0)))
            .filter(|&(_, r)| r > 0.0)
            .collect();
        if rest.is_empty() {
            // Only reachable through rounding (p <= q everywhere means p == q): draw from p.
            return Err(self.draw(&d));
        }
        Err(self.draw(&rest))
    }

    /// Sampling temperature (0: greedy).
    pub fn temperature(&self) -> f32 {
        self.temperature
    }

    /// The step's distribution over the kept candidates (presence penalty, then greedy, or
    /// temperature + top-k + top-p), as (id, probability) summing to 1.
    fn distribution(&self, cands: &[(u32, f32)]) -> Vec<(u32, f64)> {
        let mut c: Vec<(u32, f32)> = cands
            .iter()
            .map(|&(id, l)| {
                let pen = if self.seen.contains(&id) {
                    self.presence_penalty
                } else {
                    0.0
                };
                (id, l - pen)
            })
            .collect();
        // Stable: equal logits keep the device's order (lower id first).
        c.sort_by(|a, b| b.1.total_cmp(&a.1));
        if c.is_empty() {
            return vec![(0, 1.0)];
        }
        if self.temperature <= 0.0 {
            return vec![(c[0].0, 1.0)];
        }
        c.truncate(self.top_k);
        let max = c[0].1;
        let mut probs: Vec<f64> = c
            .iter()
            .map(|&(_, l)| (f64::from((l - max) / self.temperature)).exp())
            .collect();
        let total: f64 = probs.iter().sum();
        let mut cum = 0.0;
        let mut keep = probs.len();
        for (n, p) in probs.iter_mut().enumerate() {
            *p /= total;
            cum += *p;
            if cum >= f64::from(self.top_p) {
                keep = n + 1;
                break;
            }
        }
        let mass: f64 = probs[..keep].iter().sum();
        c[..keep]
            .iter()
            .zip(&probs)
            .map(|(&(id, _), &p)| (id, p / mass))
            .collect()
    }

    /// Draws from a distribution (probabilities summing to at most 1; renormalized).
    fn draw(&mut self, d: &[(u32, f64)]) -> u32 {
        if d.len() == 1 {
            return d[0].0;
        }
        let mass: f64 = d.iter().map(|x| x.1).sum();
        let mut r = f64::from(self.rng.uniform() * 0.5 + 0.5) * mass;
        for &(id, p) in d {
            r -= p;
            if r <= 0.0 {
                return id;
            }
        }
        d[d.len() - 1].0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn cands(v: &[f32]) -> Vec<(u32, f32)> {
        let mut c: Vec<(u32, f32)> = v.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
        c.sort_by(|a, b| b.1.total_cmp(&a.1));
        c
    }

    #[test]
    fn temperature_zero_is_argmax() {
        let mut s = Sampler::from_request(&json!({"temperature": 0}), false);
        assert_eq!(s.sample(&cands(&[0.1, 3.0, 0.2, 9.0])), 3);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s =
            Sampler::from_request(&json!({"temperature": 1.0, "top_k": 1, "seed": 3}), false);
        for _ in 0..10 {
            assert_eq!(s.sample(&cands(&[0.0, 2.0, 1.0])), 1);
        }
    }

    #[test]
    fn defaults_follow_the_qwen_presets() {
        let s = Sampler::from_request(&json!({}), false);
        assert_eq!(
            (s.temperature, s.top_k, s.top_p, s.presence_penalty),
            (0.7, 20, 0.8, 1.5)
        );
        let t = Sampler::from_request(&json!({}), true);
        assert_eq!(
            (t.temperature, t.top_k, t.top_p, t.presence_penalty),
            (1.0, 20, 0.95, 0.0)
        );
    }

    #[test]
    fn speculative_sampling_preserves_the_distribution() {
        // Target p = softmax over three candidates; a fixed draft accepted with p(draft) and the
        // residual otherwise must reproduce p.
        let c = cands(&[2.0, 1.0, 0.5]);
        let mut s = Sampler::from_request(
            &json!({"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0, "seed": 9}),
            false,
        );
        let p = s.distribution(&c);
        let mut counts = [0usize; 3];
        let n = 200_000;
        for _ in 0..n {
            let tok = match s.verify_draft(&c, 1) {
                Ok(()) => 1,
                Err(t) => t,
            };
            counts[tok as usize] += 1;
        }
        for &(id, prob) in &p {
            let got = counts[id as usize] as f64 / n as f64;
            assert!((got - prob).abs() < 0.01, "token {id}: {got} vs {prob}");
        }
    }

    #[test]
    fn proposal_distribution_verification_preserves_the_distribution() {
        let c = cands(&[2.0, 1.0, 0.5, 0.0]);
        let mut s = Sampler::from_request(
            &json!({"temperature": 1.0, "top_p": 1.0, "presence_penalty": 0.0, "seed": 5}),
            false,
        );
        let p = s.distribution(&c);
        // A proposal that over-weights token 2 and ignores token 0.
        let q = [(1u32, 0.3f32), (2, 0.6), (3, 0.1)];
        let mut rng = Rng::new(11);
        let mut counts = [0usize; 4];
        let n = 300_000;
        for _ in 0..n {
            let u = rng.uniform() * 0.5 + 0.5;
            let d = if u < 0.3 {
                1
            } else if u < 0.9 {
                2
            } else {
                3
            };
            let tok = match s.verify_draft_q(&c, d, &q) {
                Ok(()) => d,
                Err(t) => t,
            };
            counts[tok as usize] += 1;
        }
        for &(id, prob) in &p {
            let got = counts[id as usize] as f64 / n as f64;
            assert!((got - prob).abs() < 0.01, "token {id}: {got} vs {prob}");
        }
    }

    #[test]
    fn greedy_accepts_only_the_argmax() {
        let mut s = Sampler::from_request(&json!({"temperature": 0}), false);
        let c = cands(&[0.0, 3.0, 1.0]);
        assert_eq!(s.verify_draft(&c, 1), Ok(()));
        assert_eq!(s.verify_draft(&c, 2), Err(1));
    }

    #[test]
    fn presence_penalty_demotes_generated_tokens() {
        let mut s =
            Sampler::from_request(&json!({"temperature": 0, "presence_penalty": 1.5}), false);
        let c = cands(&[5.0, 4.0]);
        assert_eq!(s.sample(&c), 0);
        s.observe(0);
        assert_eq!(s.sample(&c), 1);
    }
}
