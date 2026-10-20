//! Token sampling: greedy at temperature 0, otherwise temperature + top-k + top-p.

use serde_json::Value;
use velox_core::rng::Rng;

pub struct Sampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: Rng,
}

impl Sampler {
    /// OpenAI defaults: temperature 1, top_p 1; `top_k` is a common extension.
    pub fn from_request(req: &Value) -> Self {
        let f = |k: &str, d: f64| req.get(k).and_then(Value::as_f64).unwrap_or(d);
        let seed = req.get("seed").and_then(Value::as_u64).unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(1, |d| d.as_nanos() as u64)
        });
        Sampler {
            temperature: f("temperature", 1.0) as f32,
            top_k: req.get("top_k").and_then(Value::as_u64).unwrap_or(0) as usize,
            top_p: f("top_p", 1.0) as f32,
            rng: Rng::new(seed),
        }
    }

    pub fn greedy() -> Self {
        Sampler {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            rng: Rng::new(1),
        }
    }

    /// Samples among the first `limit` logits.
    pub fn sample(&mut self, logits: &[f32], limit: usize) -> u32 {
        let logits = &logits[..limit.min(logits.len())];
        let argmax = || {
            let mut best = 0;
            for (i, &v) in logits.iter().enumerate() {
                if v > logits[best] {
                    best = i;
                }
            }
            best as u32
        };
        if self.temperature <= 0.0 {
            return argmax();
        }
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        if self.top_k > 0 {
            idx.truncate(self.top_k);
        }
        let max = logits[idx[0]];
        let mut probs: Vec<f64> = idx
            .iter()
            .map(|&i| (f64::from((logits[i] - max) / self.temperature)).exp())
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
        let mut r = f64::from(self.rng.uniform() * 0.5 + 0.5) * mass;
        for (n, &p) in probs[..keep].iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                return idx[n] as u32;
            }
        }
        idx[keep - 1] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn temperature_zero_is_argmax_within_limit() {
        let mut s = Sampler::from_request(&json!({"temperature": 0}));
        assert_eq!(s.sample(&[0.1, 3.0, 0.2, 9.0], 3), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::from_request(&json!({"temperature": 1.0, "top_k": 1, "seed": 3}));
        for _ in 0..10 {
            assert_eq!(s.sample(&[0.0, 2.0, 1.0], 3), 1);
        }
    }
}
