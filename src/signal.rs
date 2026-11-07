//! Frequency grids, interpolation, minimum-phase reconstruction, smoothing, decibels.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::constants::{F_HI, F_LO, PER_OCTAVE};
use crate::special::{interp, linefit};
use crate::C64;

/// Logarithmically spaced grid from ``f_lo`` to (at least) ``f_hi``.
pub fn log_grid(f_lo: f64, f_hi: f64, per_octave: usize) -> Vec<f64> {
    let n = ((f_hi / f_lo).log2() * per_octave as f64).ceil() as usize + 1;
    (0..n).map(|i| f_lo * 2f64.powf(i as f64 / per_octave as f64)).collect()
}

/// The study's grid: 150 Hz to 20 kHz at 1/96 octave.
pub fn study_grid() -> Vec<f64> {
    log_grid(F_LO, F_HI, PER_OCTAVE)
}

pub fn rfft_freqs(n: usize, fs: f64) -> Vec<f64> {
    (0..=n / 2).map(|i| i as f64 * fs / n as f64).collect()
}

/// Linear interpolation in log frequency, ends held.
pub fn interp_log(f_src: &[f64], values: &[f64], f_dst: &[f64]) -> Vec<f64> {
    let xs: Vec<f64> = f_src.iter().map(|f| f.log2()).collect();
    f_dst.iter().map(|f| interp(f.log2(), &xs, values)).collect()
}

/// Minimum-phase spectrum with the given magnitude, on the bins of an even-length real FFT
/// (homomorphic method: the real cepstrum of ``ln|H|`` folded onto positive quefrencies).
pub fn minimum_phase(magnitude: &[f64]) -> Vec<C64> {
    let m = magnitude.len();
    let n = 2 * (m - 1);
    let max = magnitude.iter().cloned().fold(0.0, f64::max);
    let floor = 1e-12 * max;
    let mut planner = FftPlanner::<f64>::new();
    // Real, even spectrum ln|H| on the full circle.
    let mut buf: Vec<Complex<f64>> = (0..n)
        .map(|k| {
            let i = if k <= n / 2 { k } else { n - k };
            Complex::new(magnitude[i].max(floor).ln(), 0.0)
        })
        .collect();
    planner.plan_fft_inverse(n).process(&mut buf);
    let scale = 1.0 / n as f64;
    let mut fold = vec![Complex::new(0.0, 0.0); n];
    fold[0] = buf[0] * scale;
    for i in 1..n / 2 {
        fold[i] = buf[i] * (2.0 * scale);
    }
    fold[n / 2] = buf[n / 2] * scale;
    planner.plan_fft_forward(n).process(&mut fold);
    fold[..=n / 2].iter().map(|c| C64::new(c.re, c.im).exp()).collect()
}

/// Unwrapped phase, as NumPy's ``np.unwrap``.
pub fn unwrap(phase: &[f64]) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut out = phase.to_vec();
    let mut correction = 0.0;
    for i in 1..phase.len() {
        let dd = phase[i] - phase[i - 1];
        let mut ddmod = (dd + PI).rem_euclid(2.0 * PI) - PI;
        if ddmod == -PI && dd > 0.0 {
            ddmod = PI;
        }
        let ph_correct = if dd.abs() < PI { 0.0 } else { ddmod - dd };
        correction += ph_correct;
        out[i] = phase[i] + correction;
    }
    out
}

/// Minimum-phase response with the magnitude ``magnitude`` given on the log grid ``freqs``.
///
/// The log-magnitude is interpolated on a fine real-FFT grid (192 kHz, 2^20 points), extended
/// beyond the grid with the slopes of its last sixth of an octave for ``reach`` octaves and then
/// held, its minimum-phase version computed there and sampled back at ``freqs``.
pub fn minimum_phase_at(freqs: &[f64], magnitude: &[f64]) -> Vec<C64> {
    minimum_phase_at_with(freqs, magnitude, 192_000.0, 1 << 20, 2.0)
}

pub fn minimum_phase_at_with(freqs: &[f64], magnitude: &[f64], fs: f64, n: usize, reach: f64) -> Vec<C64> {
    let log_mag: Vec<f64> = magnitude.iter().map(|m| m.max(1e-300).ln()).collect();
    let fine = rfft_freqs(n, fs);
    let x: Vec<f64> = freqs.iter().map(|f| (f / freqs[0]).log2()).collect();
    let top = *x.last().unwrap();
    let (xl, yl): (Vec<f64>, Vec<f64>) = x.iter().zip(&log_mag).filter(|(xi, _)| **xi <= 1.0 / 6.0).map(|(a, b)| (*a, *b)).unzip();
    let (xh, yh): (Vec<f64>, Vec<f64>) =
        x.iter().zip(&log_mag).filter(|(xi, _)| **xi >= top - 1.0 / 6.0).map(|(a, b)| (*a, *b)).unzip();
    let below = linefit(&xl, &yl).0;
    let above = linefit(&xh, &yh).0;
    let first = log_mag[0];
    let last = *log_mag.last().unwrap();
    let lm: Vec<f64> = fine
        .iter()
        .map(|&f| {
            let o = (f.max(1e-9) / freqs[0]).log2();
            if o < 0.0 {
                first + below * o.max(-reach)
            } else if o > top {
                last + above * (o - top).min(reach)
            } else {
                interp(o, &x, &log_mag)
            }
        })
        .collect();
    let mx = lm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mag: Vec<f64> = lm.iter().map(|v| (v - mx).exp()).collect();
    let h = minimum_phase(&mag);
    let phase: Vec<f64> = unwrap(&h[1..].iter().map(|c| c.arg()).collect::<Vec<_>>());
    let fine_pos = &fine[1..];
    freqs
        .iter()
        .zip(magnitude)
        .map(|(f, m)| C64::from_polar(*m, interp(*f, fine_pos, &phase)))
        .collect()
}

/// Fractional-octave smoothing of a power spectrum (rectangular window in log frequency):
/// ``out[i]`` is the mean of ``power`` over the points within ``freqs[i] * 2^(+-1/(2 fraction))``.
pub fn smooth_power(freqs: &[f64], power: &[f64], fraction: f64) -> Vec<f64> {
    let half = 2f64.powf(1.0 / (2.0 * fraction));
    let mut csum = vec![0.0; power.len() + 1];
    for (i, p) in power.iter().enumerate() {
        csum[i + 1] = csum[i] + p;
    }
    freqs
        .iter()
        .map(|&f| {
            let lo = freqs.partition_point(|&v| v < f / half);
            let hi = freqs.partition_point(|&v| v <= f * half);
            let (lo, hi) = if hi <= lo {
                let near = freqs.partition_point(|&v| v < f).min(freqs.len() - 1);
                (near, near + 1)
            } else {
                (lo, hi)
            };
            (csum[hi] - csum[lo]) / (hi - lo) as f64
        })
        .collect()
}

pub fn db10(p: f64) -> f64 {
    10.0 * p.max(1e-30).log10()
}

pub fn db20(x: f64) -> f64 {
    20.0 * x.abs().max(1e-15).log10()
}

pub fn db10_vec(p: &[f64]) -> Vec<f64> {
    p.iter().map(|v| db10(*v)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_phase_of_a_first_order_low_pass() {
        // |H| of 1/(1 + j f/fc): the minimum phase is -atan(f/fc).
        let f = log_grid(100.0, 20_000.0, 24);
        let fc = 1000.0;
        let mag: Vec<f64> = f.iter().map(|x| 1.0 / (1.0 + (x / fc).powi(2)).sqrt()).collect();
        let h = minimum_phase_at(&f, &mag);
        // The magnitude is extended two octaves beyond the grid, then held: the phase misses the
        // slope beyond, by about f / 80 kHz radians.
        for (x, c) in f.iter().zip(&h) {
            if *x > 300.0 && *x < 1500.0 {
                assert!((c.arg() + (x / fc).atan()).abs() < 0.03, "{} {}", x, c.arg());
            }
        }
    }

    #[test]
    fn smoothing_keeps_a_flat_spectrum() {
        let f = log_grid(100.0, 1000.0, 48);
        let p = vec![2.0; f.len()];
        assert!(smooth_power(&f, &p, 6.0).iter().all(|v| (v - 2.0).abs() < 1e-12));
    }
}
