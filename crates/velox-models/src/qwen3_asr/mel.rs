//! Whisper-style log-mel features (docs/qwen3_asr.md#log-mel-whisperfeatureextractor), computed
//! on the host: an f32 DFT over frames split across threads, then sparse mel filters and the
//! log in f64.

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
const BINS: usize = N_FFT / 2 + 1;

/// Frames per thread below which splitting does not pay for the spawn.
const FRAMES_PER_THREAD: usize = 64;

/// Precomputed window, DFT twiddles and mel filters; build once and reuse.
pub struct MelFrontend {
    window: Vec<f32>,
    /// `[BINS, N_FFT]` each.
    cos: Vec<f32>,
    sin: Vec<f32>,
    /// Per mel band: first FFT bin and the weights from there (the non-zero span of its
    /// triangle).
    filters: Vec<(usize, Vec<f64>)>,
}

fn hz_to_mel(f: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (f_sp, min_log_hz) = (200.0 / 3.0, 1000.0);
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.0;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

impl MelFrontend {
    pub fn new() -> Self {
        let window: Vec<f64> = (0..N_FFT)
            .map(|n| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / N_FFT as f64).cos())
            .collect();
        let mut cos = vec![0.0; BINS * N_FFT];
        let mut sin = vec![0.0; BINS * N_FFT];
        for k in 0..BINS {
            for n in 0..N_FFT {
                // Index the angle modulo N_FFT so large k*n stays exact.
                let a = 2.0 * std::f64::consts::PI * ((k * n) % N_FFT) as f64 / N_FFT as f64;
                cos[k * N_FFT + n] = a.cos();
                sin[k * N_FFT + n] = a.sin();
            }
        }
        let fft_freqs: Vec<f64> = (0..BINS)
            .map(|i| i as f64 * (SAMPLE_RATE as f64 / 2.0) / (BINS - 1) as f64)
            .collect();
        let (m0, m1) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
        let edges: Vec<f64> = (0..N_MELS + 2)
            .map(|i| mel_to_hz(m0 + (m1 - m0) * i as f64 / (N_MELS + 1) as f64))
            .collect();
        let mut filters = Vec::with_capacity(N_MELS);
        for m in 0..N_MELS {
            let (lo, c, hi) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = 2.0 / (hi - lo);
            let w: Vec<f64> = fft_freqs
                .iter()
                .map(|&f| ((f - lo) / (c - lo)).min((hi - f) / (hi - c)).max(0.0) * norm)
                .collect();
            let first = w.iter().position(|&v| v > 0.0).unwrap_or(0);
            let last = w.iter().rposition(|&v| v > 0.0).map_or(first, |i| i + 1);
            filters.push((first, w[first..last.max(first)].to_vec()));
        }
        MelFrontend {
            window: window.iter().map(|&v| v as f32).collect(),
            cos: cos.iter().map(|&v| v as f32).collect(),
            sin: sin.iter().map(|&v| v as f32).collect(),
            filters,
        }
    }

    /// Frames produced for `samples` input samples.
    pub fn frames(samples: usize) -> usize {
        samples / HOP
    }

    /// Power spectra of frames `t0 .. t0 + out.len() / BINS` into `out` (`[frames, BINS]`).
    fn power(&self, x: &[f32], t0: usize, out: &mut [f64]) {
        let n = x.len() as isize;
        let pad = (N_FFT / 2) as isize;
        // numpy / torch "reflect": the edge sample is not repeated.
        let at = |i: isize| -> f32 {
            if n == 1 {
                return x[0];
            }
            let mut j = i - pad;
            while j < 0 || j >= n {
                if j < 0 {
                    j = -j;
                }
                if j >= n {
                    j = 2 * (n - 1) - j;
                }
            }
            x[j as usize]
        };
        let mut frame = [0f32; N_FFT];
        for (f, p) in out.chunks_exact_mut(BINS).enumerate() {
            let start = ((t0 + f) * HOP) as isize;
            for (i, v) in frame.iter_mut().enumerate() {
                *v = at(start + i as isize) * self.window[i];
            }
            for (k, pk) in p.iter_mut().enumerate() {
                let (c, s) = (
                    &self.cos[k * N_FFT..][..N_FFT],
                    &self.sin[k * N_FFT..][..N_FFT],
                );
                // Eight partial sums keep the loop vectorizable and the f32 error small.
                let (mut re, mut im) = ([0f32; 8], [0f32; 8]);
                for ((fr, cc), ss) in frame
                    .chunks_exact(8)
                    .zip(c.chunks_exact(8))
                    .zip(s.chunks_exact(8))
                {
                    for j in 0..8 {
                        re[j] += fr[j] * cc[j];
                        im[j] += fr[j] * ss[j];
                    }
                }
                let re: f64 = re.iter().map(|&v| f64::from(v)).sum();
                let im: f64 = im.iter().map(|&v| f64::from(v)).sum();
                *pk = re * re + im * im;
            }
        }
    }

    /// `[N_MELS, frames]` row-major log-mel of 16 kHz mono PCM.
    pub fn compute(&self, pcm: &[i16]) -> Vec<f32> {
        let frames = Self::frames(pcm.len());
        if frames == 0 {
            return Vec::new();
        }
        let x: Vec<f32> = pcm.iter().map(|&s| f32::from(s) / 32768.0).collect();
        let mut power = vec![0f64; frames * BINS];
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(frames.div_ceil(FRAMES_PER_THREAD))
            .max(1);
        let per = frames.div_ceil(threads);
        std::thread::scope(|sc| {
            for (i, part) in power.chunks_mut(per * BINS).enumerate() {
                let x = &x;
                sc.spawn(move || self.power(x, i * per, part));
            }
        });
        let mut logmel = vec![0f64; N_MELS * frames];
        for t in 0..frames {
            let p = &power[t * BINS..][..BINS];
            for (m, (first, w)) in self.filters.iter().enumerate() {
                let e: f64 = w.iter().zip(&p[*first..]).map(|(a, b)| a * b).sum();
                logmel[m * frames + t] = e.max(1e-10).log10();
            }
        }
        let max = logmel.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        logmel
            .iter()
            .map(|&v| ((v.max(max - 8.0) + 4.0) / 4.0) as f32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slaney_scale_round_trips() {
        for f in [0.0, 300.0, 999.0, 1000.0, 4000.0, 8000.0] {
            assert!((mel_to_hz(hz_to_mel(f)) - f).abs() < 1e-9, "{f}");
        }
    }

    #[test]
    fn one_second_is_one_hundred_frames() {
        let m = MelFrontend::new();
        let pcm: Vec<i16> = (0..SAMPLE_RATE)
            .map(|i| ((i as f64 * 0.05).sin() * 8000.0) as i16)
            .collect();
        let out = m.compute(&pcm);
        assert_eq!(out.len(), N_MELS * 100);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
