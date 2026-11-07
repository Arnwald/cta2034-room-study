//! Crossovers: acoustic targets on the reference axis, and the filters that reach them.
//!
//! Each driver is equalized to an acoustic target on the reference axis: its filter is
//! ``X_i = T_i E_i``, ``T_i`` the target and ``E_i`` the minimum-phase inverse of the driver's
//! 1/6-octave smoothed on-axis magnitude. Targets, for a crossover frequency ``fc``:
//!
//! * LR2, LR4, LR8: Linkwitz-Riley of order 2, 4, 8 (squared Butterworth of order 1, 2, 4), minimum
//!   phase; the LR2's high-pass inverted so that both outputs are in phase at ``fc``;
//! * FIR: the brickwall linear-phase crossover of the four-way article, a windowed sinc of 1501
//!   taps at 93.75 kHz (Kaiser, beta = 8) and its complement (an impulse minus the low-pass); its
//!   common delay is left out.

use crate::signal::{minimum_phase_at, smooth_power};
use crate::special::kaiser;
use crate::speaker::Layout;
use crate::C64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Lr2,
    Lr4,
    Lr8,
    Fir,
}

pub const KINDS: [Kind; 4] = [Kind::Lr2, Kind::Lr4, Kind::Lr8, Kind::Fir];

impl Kind {
    pub fn name(&self) -> &'static str {
        match self {
            Kind::Lr2 => "lr2",
            Kind::Lr4 => "lr4",
            Kind::Lr8 => "lr8",
            Kind::Fir => "fir",
        }
    }

    pub fn from_name(s: &str) -> Option<Kind> {
        KINDS.iter().copied().find(|k| k.name() == s)
    }
}

pub const FIR_TAPS: usize = 1501;
pub const FIR_FS: f64 = 93_750.0;
pub const FIR_BETA: f64 = 8.0;

/// Analog Butterworth low-pass (``hp = false``) or high-pass of any order at frequency ``f``.
pub fn butterworth(f: f64, fc: f64, order: usize, hp: bool) -> C64 {
    let mut s = C64::new(0.0, f / fc);
    if hp {
        s = C64::new(1.0, 0.0) / s;
    }
    let mut out = C64::new(1.0, 0.0);
    for k in 1..=order / 2 {
        let theta = std::f64::consts::PI * (2 * k + order - 1) as f64 / (2 * order) as f64;
        out /= s * s - s * (2.0 * theta.cos()) + 1.0;
    }
    if order % 2 == 1 {
        out /= s + 1.0;
    }
    out
}

pub fn linkwitz_riley(f: f64, fc: f64, order: usize, hp: bool) -> C64 {
    let b = butterworth(f, fc, order / 2, hp);
    b * b
}

/// The brickwall low-pass's taps (centred, summing to 1).
pub fn fir_lowpass_taps(fc: f64) -> Vec<f64> {
    let m = (FIR_TAPS - 1) / 2;
    let win = kaiser(FIR_TAPS, FIR_BETA);
    let mut h: Vec<f64> = (0..FIR_TAPS)
        .map(|i| {
            let n = i as f64 - m as f64;
            let x = 2.0 * fc / FIR_FS * n;
            let sinc = if x == 0.0 { 1.0 } else { (std::f64::consts::PI * x).sin() / (std::f64::consts::PI * x) };
            2.0 * fc / FIR_FS * sinc * win[i]
        })
        .collect();
    let sum: f64 = h.iter().sum();
    h.iter_mut().for_each(|v| *v /= sum);
    h
}

/// Zero-phase amplitude of the brickwall low-pass (its bulk delay removed) at ``f``.
pub fn fir_lowpass_amplitude(taps: &[f64], f: f64) -> f64 {
    let m = (taps.len() - 1) / 2;
    let mut a = taps[m];
    for n in 1..=m {
        a += 2.0 * taps[m + n] * (2.0 * std::f64::consts::PI * f * n as f64 / FIR_FS).cos();
    }
    a
}

/// Low-pass and high-pass target of one crossover at the frequencies ``f``.
pub fn pair(kind: Kind, f: &[f64], fc: f64) -> (Vec<C64>, Vec<C64>) {
    match kind {
        Kind::Fir => {
            let taps = fir_lowpass_taps(fc);
            let lp: Vec<C64> = f.iter().map(|&x| C64::new(fir_lowpass_amplitude(&taps, x), 0.0)).collect();
            let hp = lp.iter().map(|l| C64::new(1.0, 0.0) - l).collect();
            (lp, hp)
        }
        _ => {
            let order = match kind {
                Kind::Lr2 => 2,
                Kind::Lr4 => 4,
                _ => 8,
            };
            let sign = if order % 4 == 2 { -1.0 } else { 1.0 };
            let lp = f.iter().map(|&x| linkwitz_riley(x, fc, order, false)).collect();
            let hp = f.iter().map(|&x| linkwitz_riley(x, fc, order, true) * sign).collect();
            (lp, hp)
        }
    }
}

/// Acoustic targets of the ways of a layout, lowest first. At each crossover the way just below
/// gets the low-pass, every way above the high-pass, and every way further below the
/// crossover's all-pass (its low-pass plus high-pass), so that the ways sum to a pure all-pass (a
/// delay for the brickwall).
pub fn targets(kind: Kind, f: &[f64], crossovers: &[f64]) -> Vec<Vec<C64>> {
    let n = crossovers.len() + 1;
    let mut out = vec![vec![C64::new(1.0, 0.0); f.len()]; n];
    for (i, &fc) in crossovers.iter().enumerate() {
        let (lp, hp) = pair(kind, f, fc);
        for (w, way) in out.iter_mut().enumerate() {
            for j in 0..f.len() {
                way[j] *= match w.cmp(&i) {
                    std::cmp::Ordering::Less => lp[j] + hp[j],
                    std::cmp::Ordering::Equal => lp[j],
                    std::cmp::Ordering::Greater => hp[j],
                };
            }
        }
    }
    out
}

/// Minimum-phase inverse of a driver's 1/6-octave smoothed on-axis magnitude.
pub fn equalizer(freqs: &[f64], on_axis: &[C64]) -> Vec<C64> {
    let power: Vec<f64> = on_axis.iter().map(|c| c.norm_sqr()).collect();
    let smooth = smooth_power(freqs, &power, 6.0);
    let inverse: Vec<f64> = smooth.iter().map(|p| 1.0 / p.sqrt().max(1e-9)).collect();
    minimum_phase_at(freqs, &inverse)
}

/// The filters ``X_i = T_w E_w`` of a layout's drivers for one crossover kind: every driver of a
/// way ``w`` gets the way's target times the inverse of the way's summed on-axis response.
pub fn filters(kind: Kind, freqs: &[f64], layout: &Layout, on_axis: &[Vec<C64>]) -> Vec<Vec<C64>> {
    let t = targets(kind, freqs, &layout.crossovers);
    let ways = layout.crossovers.len() + 1;
    let x: Vec<Vec<C64>> = (0..ways)
        .map(|w| {
            let mut h = vec![C64::new(0.0, 0.0); freqs.len()];
            for (d, r) in layout.drivers.iter().zip(on_axis) {
                if d.way == w {
                    h.iter_mut().zip(r).for_each(|(a, b)| *a += b);
                }
            }
            let eq = equalizer(freqs, &h);
            t[w].iter().zip(&eq).map(|(a, b)| a * b).collect()
        })
        .collect();
    layout.drivers.iter().map(|d| x[d.way].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signal::log_grid;

    #[test]
    fn linkwitz_riley_outputs_sum_to_an_all_pass() {
        let f = log_grid(100.0, 20_000.0, 48);
        for kind in [Kind::Lr2, Kind::Lr4, Kind::Lr8] {
            let (lp, hp) = pair(kind, &f, 2200.0);
            for (a, b) in lp.iter().zip(&hp) {
                assert!(((a + b).norm() - 1.0).abs() < 1e-9);
            }
            let j = f.iter().enumerate().min_by(|a, b| (a.1 - 2200.0).abs().partial_cmp(&(b.1 - 2200.0).abs()).unwrap()).unwrap().0;
            assert!((lp[j] / hp[j]).arg().abs() < 0.05);
            assert!((20.0 * lp[j].norm().log10() + 6.0).abs() < 0.2);
        }
    }

    #[test]
    fn multi_way_targets_sum_to_an_all_pass() {
        let f = log_grid(100.0, 20_000.0, 24);
        for kind in KINDS {
            for crossovers in [&[2200.0][..], &[450.0, 3000.0], &[250.0, 900.0, 3200.0]] {
                let t = targets(kind, &f, crossovers);
                for j in 0..f.len() {
                    let s: C64 = t.iter().map(|w| w[j]).sum();
                    assert!((s.norm() - 1.0).abs() < 1e-6, "{kind:?} {crossovers:?} {}", f[j]);
                }
            }
        }
    }

    #[test]
    fn fir_outputs_sum_to_a_delay_and_are_steep() {
        let f = log_grid(100.0, 20_000.0, 48);
        let (lp, hp) = pair(Kind::Fir, &f, 2200.0);
        for (i, x) in f.iter().enumerate() {
            assert!(((lp[i] + hp[i]) - 1.0).norm() < 1e-12);
            if *x > 2500.0 {
                assert!(lp[i].norm() < 10f64.powf(-70.0 / 20.0));
            }
            if *x < 1900.0 {
                assert!(hp[i].norm() < 10f64.powf(-70.0 / 20.0));
            }
        }
    }

    #[test]
    fn butterworth_is_maximally_flat() {
        let f = log_grid(100.0, 20_000.0, 48);
        for n in 1..=4 {
            for &x in &f {
                let h = butterworth(x, 1000.0, n, false);
                let expected = 1.0 / (1.0 + (x / 1000.0).powi(2 * n as i32));
                assert!((h.norm_sqr() - expected).abs() < 1e-9 * expected.max(1e-12) + 1e-15);
            }
        }
    }
}
