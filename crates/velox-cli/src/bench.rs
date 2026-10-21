//! Synthetic decoder-stack benchmark. Weights are random f32 at the chosen model's shapes, so it
//! measures kernels and scheduling, not model quality. Decode is reported as achieved weight
//! bandwidth because single-token decode is bandwidth-bound.

use std::time::Instant;

use velox_core::model::{HostLayer, Stack};
use velox_core::rng::Rng;
use velox_core::{Backend, ModelSpec, Result};

pub struct Config {
    pub model: ModelSpec,
    pub layers: usize,
    pub context: usize,
    pub prefill: usize,
    pub iters: usize,
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn run<B: Backend>(b: &B, c: &Config) -> Result<()> {
    let spec = &c.model;
    let host: Vec<HostLayer> = (0..c.layers)
        .map(|i| HostLayer::random(spec, i as u64))
        .collect();
    let weight_bytes: usize = host.iter().map(HostLayer::bytes).sum();
    let max_seq = c.context + c.prefill + 1;
    let mut stack = Stack::new(b, spec, &host, max_seq, c.prefill.max(1))?;
    drop(host);

    let mut rng = Rng::new(1);
    let mut x = b.upload(&rng.vec(c.prefill.max(1) * spec.hidden, 1.0))?;
    // Fill the cache up to `context` in prefill-sized chunks.
    let mut pos = 0;
    while pos < c.context {
        let t = c.prefill.min(c.context - pos);
        stack.forward(b, &mut x, t, pos)?;
        pos += t;
    }
    b.synchronize()?;

    let mut time = |tokens: usize| -> Result<f64> {
        for _ in 0..3 {
            stack.forward(b, &mut x, tokens, c.context)?;
        }
        b.synchronize()?;
        let mut samples = Vec::with_capacity(c.iters);
        for _ in 0..c.iters.max(1) {
            let t0 = Instant::now();
            stack.forward(b, &mut x, tokens, c.context)?;
            b.synchronize()?;
            samples.push(t0.elapsed().as_secs_f64());
        }
        Ok(median(samples))
    };

    let decode = time(1)?;
    let prefill = if c.prefill > 0 {
        Some(time(c.prefill)?)
    } else {
        None
    };

    println!("backend   {}", b.describe());
    println!(
        "model     {} shapes, {} of {} layers, f32 weights {:.2} GiB, context {}",
        spec.name,
        c.layers,
        spec.layers,
        weight_bytes as f64 / (1u64 << 30) as f64,
        c.context
    );
    let gbs = weight_bytes as f64 / decode / 1e9;
    let peak = b.profile().map(|p| {
        format!(
            " ({:.0}% of {} GB/s peak)",
            100.0 * gbs / f64::from(p.mem_bandwidth_gbs),
            p.mem_bandwidth_gbs
        )
    });
    println!(
        "decode    {:.3} ms/token for {} layers = {:.3} ms/layer, weights at {:.0} GB/s{}",
        decode * 1e3,
        c.layers,
        decode * 1e3 / c.layers as f64,
        gbs,
        peak.unwrap_or_default()
    );
    if let Some(p) = prefill {
        println!(
            "prefill   {} tokens in {:.3} ms = {:.0} tokens/s",
            c.prefill,
            p * 1e3,
            c.prefill as f64 / p
        );
    }
    Ok(())
}

/// `velox bench-linear`: one `linear` at a real model shape with random weights in a stored
/// format, timed per token count. Reports achieved weight bandwidth and tensor throughput.
pub struct LinearConfig {
    pub format: velox_core::WeightFormat,
    pub n: usize,
    pub k: usize,
    pub ms: Vec<usize>,
    pub act: velox_core::ActPolicy,
    pub iters: usize,
}

pub fn run_linear<B: Backend>(b: &B, c: &LinearConfig) -> Result<()> {
    let mut rng = velox_core::rng::Rng::new(11);
    let bytes = velox_conformance::random_weight(&mut rng, c.format, c.n, c.k);
    // Rotate through enough copies to exceed the L2 cache: a model never finds its weights there.
    let copies = (384_000_000 / bytes.len().max(1)).clamp(1, 64);
    let wds = (0..copies)
        .map(|_| b.upload_bytes(&bytes))
        .collect::<Result<Vec<_>>>()?;
    let views: Vec<_> = wds
        .iter()
        .map(|wd| velox_core::WeightView::full(wd, c.format, c.n, c.k).with_act(c.act))
        .collect();
    let view = &views[0];
    println!(
        "linear {} [{}, {}]  {:.1} MB  on {}",
        c.format.name(),
        c.n,
        c.k,
        bytes.len() as f64 / 1e6,
        b.describe()
    );
    for &m in &c.ms {
        let x = b.upload(&rng.vec(m * c.k, 1.0))?;
        let mut y = b.alloc(m * c.n)?;
        for v in &views {
            b.linear(&x, v, &mut y, m)?;
        }
        b.synchronize()?;
        // Back-to-back batches over all copies, so host launch and sync latency amortize.
        let rounds = c.iters.div_ceil(copies).max(5);
        let mut samples = Vec::with_capacity(rounds);
        for _ in 0..rounds {
            let t0 = Instant::now();
            for v in &views {
                b.linear(&x, v, &mut y, m)?;
            }
            b.synchronize()?;
            samples.push(t0.elapsed().as_secs_f64() / copies as f64);
        }
        let t = median(samples);
        println!(
            "  m={m:<5} {:>9.1} us  {:>7.0} GB/s weights  {:>7.1} TFLOPS  act {:?}",
            t * 1e6,
            bytes.len() as f64 / t / 1e9,
            2.0 * (m * c.n * c.k) as f64 / t / 1e12,
            b.linear_act(view, m)
        );
    }
    Ok(())
}

/// `velox bench-gdn`: one `gated_delta_net` call at a model head geometry with random inputs,
/// timed per token count (decode/verify sizes and prefill chunks).
pub struct GdnConfig {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    pub tokens: Vec<usize>,
    pub iters: usize,
}

pub fn run_gdn<B: Backend>(b: &B, c: &GdnConfig) -> Result<()> {
    let mut rng = Rng::new(13);
    let a_log = b.upload(&rng.vec(c.v_heads, 1.0))?;
    let dt_bias = b.upload(&rng.vec(c.v_heads, 1.0))?;
    let mut state = b.upload(&rng.vec(c.v_heads * c.v_dim * c.k_dim, 0.1))?;
    println!(
        "gated_delta_net heads {}/{} d {}/{}  on {}",
        c.k_heads,
        c.v_heads,
        c.k_dim,
        c.v_dim,
        b.describe()
    );
    for &tokens in &c.tokens {
        let s = velox_core::GdnShape {
            tokens,
            k_heads: c.k_heads,
            v_heads: c.v_heads,
            k_dim: c.k_dim,
            v_dim: c.v_dim,
            row0: 0,
        };
        let qkv = b.upload(&rng.vec(tokens * s.qkv_width(), 1.0))?;
        let ab = b.upload(&rng.vec(tokens * 2 * c.v_heads, 2.0))?;
        let mut out = b.alloc(tokens * c.v_heads * c.v_dim)?;
        for _ in 0..3 {
            b.gated_delta_net(&qkv, &ab, &a_log, &dt_bias, &mut state, &mut out, s)?;
        }
        b.synchronize()?;
        let mut samples = Vec::with_capacity(c.iters);
        for _ in 0..c.iters {
            let t0 = Instant::now();
            b.gated_delta_net(&qkv, &ab, &a_log, &dt_bias, &mut state, &mut out, s)?;
            b.synchronize()?;
            samples.push(t0.elapsed().as_secs_f64());
        }
        let best = samples.iter().copied().fold(f64::INFINITY, f64::min);
        let t = median(samples);
        println!(
            "  tokens={tokens:<5} {:>9.1} us (min {:>9.1})  {:>8.3} us/token",
            t * 1e6,
            best * 1e6,
            t * 1e6 / tokens as f64
        );
    }
    Ok(())
}
