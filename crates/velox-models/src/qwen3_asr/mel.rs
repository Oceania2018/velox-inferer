//! Whisper-style log-mel features (docs/qwen3_asr.md#log-mel-whisperfeatureextractor), computed
//! on the host in f64.

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 128;
const BINS: usize = N_FFT / 2 + 1;

/// Precomputed window, DFT twiddles and mel filters; build once and reuse.
pub struct MelFrontend {
    window: Vec<f64>,
    cos: Vec<f64>,
    sin: Vec<f64>,
    /// `[N_MELS, BINS]`
    filters: Vec<f64>,
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
        let mut filters = vec![0.0; N_MELS * BINS];
        for m in 0..N_MELS {
            let (lo, c, hi) = (edges[m], edges[m + 1], edges[m + 2]);
            let norm = 2.0 / (hi - lo);
            for (b, &f) in fft_freqs.iter().enumerate() {
                let down = (f - lo) / (c - lo);
                let up = (hi - f) / (hi - c);
                filters[m * BINS + b] = down.min(up).max(0.0) * norm;
            }
        }
        MelFrontend {
            window,
            cos,
            sin,
            filters,
        }
    }

    /// Frames produced for `samples` input samples.
    pub fn frames(samples: usize) -> usize {
        samples / HOP
    }

    /// `[N_MELS, frames]` row-major log-mel of 16 kHz mono PCM.
    pub fn compute(&self, pcm: &[i16]) -> Vec<f32> {
        let frames = Self::frames(pcm.len());
        if frames == 0 {
            return Vec::new();
        }
        let x: Vec<f64> = pcm.iter().map(|&s| f64::from(s) / 32768.0).collect();
        let pad = N_FFT / 2;
        // Reflect padding (numpy / torch "reflect": the edge sample is not repeated).
        let at = |i: isize| -> f64 {
            let n = x.len() as isize;
            let mut j = i - pad as isize;
            if n == 1 {
                return x[0];
            }
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
        let mut logmel = vec![0f64; N_MELS * frames];
        let mut frame = vec![0f64; N_FFT];
        let mut power = vec![0f64; BINS];
        for t in 0..frames {
            for (n, v) in frame.iter_mut().enumerate() {
                *v = at((t * HOP + n) as isize) * self.window[n];
            }
            for (k, p) in power.iter_mut().enumerate() {
                let (c, s) = (
                    &self.cos[k * N_FFT..(k + 1) * N_FFT],
                    &self.sin[k * N_FFT..(k + 1) * N_FFT],
                );
                let (mut re, mut im) = (0.0, 0.0);
                for n in 0..N_FFT {
                    re += frame[n] * c[n];
                    im -= frame[n] * s[n];
                }
                *p = re * re + im * im;
            }
            for m in 0..N_MELS {
                let f = &self.filters[m * BINS..(m + 1) * BINS];
                let e: f64 = f.iter().zip(&power).map(|(a, b)| a * b).sum();
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
