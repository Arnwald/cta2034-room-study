//! Do gentle crossover slopes harm what a listener receives in a room? The computation of
//! `protocol/crossover-slopes.md`.
//!
//! The instrument: excitation patterns (the power spectrum smoothed by the rounded-exponential
//! auditory filters of Glasberg and Moore, one ERB wide, at centres every 0.25 Cam from 200 Hz to
//! 10 kHz), the difference between two of them in dB, and two candidate summaries of it: A, its
//! root sum of squares over ERB number, and B, its largest value. Step 0 calibrates them on
//! resonances whose detection thresholds are published.

use crate::coherent::erb;
use crate::signal::db10;

/// Band and spacing of the excitation patterns.
pub const EX_LO: f64 = 200.0;
pub const EX_HI: f64 = 10_000.0;
pub const EX_STEP: f64 = 0.25;

/// ERB number (Cam) of a frequency (Hz), Glasberg and Moore (1990).
pub fn cam(f: f64) -> f64 {
    21.4 * (4.37 * f / 1000.0 + 1.0).log10()
}

/// Frequency (Hz) of an ERB number (Cam).
pub fn cam_inv(c: f64) -> f64 {
    (10f64.powf(c / 21.4) - 1.0) / 4.37 * 1000.0
}

/// The centres of the excitation patterns (Hz).
pub fn centres() -> Vec<f64> {
    let (a, b) = (cam(EX_LO), cam(EX_HI));
    let n = ((b - a) / EX_STEP).floor() as usize + 1;
    (0..n).map(|i| cam_inv(a + i as f64 * EX_STEP)).collect()
}

/// Rounded-exponential auditory filters over any frequency grid, each sample weighted by the
/// bandwidth it stands for, so that a response sampled on a logarithmic grid and the same response
/// on a linear grid give the same excitation.
pub struct Excitation {
    rows: Vec<(usize, Vec<f64>)>,
}

impl Excitation {
    pub fn new(f: &[f64]) -> Self {
        let n = f.len();
        let width: Vec<f64> = (0..n)
            .map(|k| {
                let lo = if k == 0 { f[0] } else { 0.5 * (f[k - 1] + f[k]) };
                let hi = if k + 1 == n { f[n - 1] } else { 0.5 * (f[k] + f[k + 1]) };
                hi - lo
            })
            .collect();
        let rows = centres()
            .iter()
            .map(|&fc| {
                let p = 4.0 * fc / erb(fc);
                let half = 12.0 * fc / p;
                let lo = f.partition_point(|&x| x < fc - half);
                let hi = f.partition_point(|&x| x <= fc + half).max(lo + 1);
                let mut w: Vec<f64> = (lo..hi)
                    .map(|k| {
                        let g = (f[k] - fc).abs() / fc;
                        (1.0 + p * g) * (-p * g).exp() * width[k]
                    })
                    .collect();
                let s: f64 = w.iter().sum();
                w.iter_mut().for_each(|v| *v /= s);
                (lo, w)
            })
            .collect();
        Excitation { rows }
    }

    /// Excitation pattern (power) of a power spectrum on the grid given to [`Excitation::new`].
    pub fn apply(&self, power: &[f64]) -> Vec<f64> {
        self.rows.iter().map(|(lo, w)| w.iter().zip(&power[*lo..]).map(|(a, b)| a * b).sum()).collect()
    }

    /// Excitation pattern in dB.
    pub fn db(&self, power: &[f64]) -> Vec<f64> {
        self.apply(power).iter().map(|v| db10(*v)).collect()
    }
}

/// The difference of two excitation patterns in dB, channel by channel.
pub fn difference(a_db: &[f64], b_db: &[f64]) -> Vec<f64> {
    a_db.iter().zip(b_db).map(|(a, b)| a - b).collect()
}

/// Summary A: root of the sum of squares over ERB number (dB per square root of a Cam).
pub fn summary_a(delta: &[f64]) -> f64 {
    (delta.iter().map(|d| d * d).sum::<f64>() * EX_STEP).sqrt()
}

/// Summary B: the largest change in any channel (dB).
pub fn summary_b(delta: &[f64]) -> f64 {
    delta.iter().fold(0.0, |m, d| m.max(d.abs()))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Summary {
    A,
    B,
}

impl Summary {
    pub fn of(&self, delta: &[f64]) -> f64 {
        match self {
            Summary::A => summary_a(delta),
            Summary::B => summary_b(delta),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Summary::A => "A, root sum of squares over ERB number",
            Summary::B => "B, largest change in one channel",
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Step 0: calibration on resonances
// ---------------------------------------------------------------------------------------------

/// Power response of an analog peaking filter (RBJ form) of centre ``f0``, quality factor ``q``
/// and gain ``gain_db`` at its centre (negative: a dip).
pub fn peaking_power(f: f64, f0: f64, q: f64, gain_db: f64) -> f64 {
    let a = 10f64.powf(gain_db / 40.0);
    let w = f / f0;
    let re = 1.0 - w * w;
    (re * re + (w * a / q).powi(2)) / (re * re + (w / (a * q)).powi(2))
}

pub const CAL_FREQS: [f64; 3] = [1000.0, 2200.0, 5000.0];
pub const CAL_QS: [f64; 3] = [1.0, 10.0, 50.0];
/// The published anchors (peaks, music): about 1-2 dB for Q = 1, about 10 dB for Q = 50.
pub const ANCHOR_Q1_DB: f64 = 1.5;
pub const ANCHOR_Q50_DB: f64 = 10.0;

pub struct Calibrator {
    f: Vec<f64>,
    ex: Excitation,
    flat: Vec<f64>,
}

impl Calibrator {
    pub fn new() -> Self {
        // 1 Hz from 50 Hz to 14 kHz: the narrowest resonance (Q = 50 at 1 kHz, 20 Hz wide) is
        // sampled twenty times over its bandwidth.
        let f: Vec<f64> = (50..=14_000).map(|v| v as f64).collect();
        let ex = Excitation::new(&f);
        let flat = ex.db(&vec![1.0; f.len()]);
        Calibrator { f, ex, flat }
    }

    /// The difference pattern of a resonance against a flat response.
    pub fn delta(&self, f0: f64, q: f64, gain_db: f64) -> Vec<f64> {
        let p: Vec<f64> = self.f.iter().map(|&x| peaking_power(x, f0, q, gain_db)).collect();
        difference(&self.ex.db(&p), &self.flat)
    }

    pub fn value(&self, s: Summary, f0: f64, q: f64, gain_db: f64) -> f64 {
        s.of(&self.delta(f0, q, gain_db))
    }

    /// The level (dB, signed as ``sign``) at which a resonance reaches ``target``, by bisection
    /// up to ``max_db``; ``None`` if it never does.
    pub fn level_for(&self, s: Summary, f0: f64, q: f64, sign: f64, target: f64, max_db: f64) -> Option<f64> {
        if self.value(s, f0, q, sign * max_db) < target {
            return None;
        }
        let (mut lo, mut hi) = (0.0, max_db);
        for _ in 0..60 {
            let mid = 0.5 * (lo + hi);
            if self.value(s, f0, q, sign * mid) < target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Some(0.5 * (lo + hi))
    }
}

impl Default for Calibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// The calibration of one summary: the level of each resonance that gives the same value as a
/// +1.5 dB, Q = 1 peak at the same frequency, and the unit T.
pub struct Calibration {
    pub summary: Summary,
    /// ``[frequency][q]``: equal-value peak level (dB).
    pub peaks: Vec<Vec<Option<f64>>>,
    /// ``[frequency][q]``: equal-value dip depth (dB, positive).
    pub dips: Vec<Vec<Option<f64>>>,
    /// Mean over the frequencies (dB) of the Q = 50 equal-value peak level.
    pub l50: f64,
    /// The unit: the value of a +1.5 dB, Q = 1 peak at 2.2 kHz.
    pub unit: f64,
}

pub fn calibrate(c: &Calibrator, s: Summary) -> Calibration {
    let mut peaks = Vec::new();
    let mut dips = Vec::new();
    for &f0 in &CAL_FREQS {
        let target = c.value(s, f0, 1.0, ANCHOR_Q1_DB);
        peaks.push(CAL_QS.iter().map(|&q| c.level_for(s, f0, q, 1.0, target, 40.0)).collect());
        dips.push(CAL_QS.iter().map(|&q| c.level_for(s, f0, q, -1.0, target, 60.0)).collect());
    }
    let q50: Vec<f64> = peaks.iter().map(|row: &Vec<Option<f64>>| row[2].unwrap_or(f64::INFINITY)).collect();
    let l50 = q50.iter().sum::<f64>() / q50.len() as f64;
    Calibration { summary: s, peaks, dips, l50, unit: c.value(s, 2200.0, 1.0, ANCHOR_Q1_DB) }
}

/// The protocol's choice: the summary whose ``L50`` is closest to 10 dB in ratio, accepted if
/// within 5-20 dB.
pub fn choose(cals: &[Calibration]) -> (usize, bool) {
    let score = |c: &Calibration| (c.l50 / ANCHOR_Q50_DB).ln().abs();
    let best = (0..cals.len()).min_by(|&a, &b| score(&cals[a]).partial_cmp(&score(&cals[b])).unwrap()).unwrap();
    let ok = cals[best].l50 >= 5.0 && cals[best].l50 <= 20.0;
    (best, ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cam_round_trips() {
        for f in [200.0, 1000.0, 2200.0, 10_000.0] {
            assert!((cam_inv(cam(f)) - f).abs() < 1e-9 * f);
        }
        // 1 kHz is about 15.6 Cam.
        assert!((cam(1000.0) - 15.6).abs() < 0.05);
    }

    #[test]
    fn excitation_keeps_a_flat_spectrum_on_any_grid() {
        let lin: Vec<f64> = (50..=14_000).map(|v| v as f64).collect();
        let log = crate::signal::log_grid(50.0, 14_000.0, 96);
        for f in [lin, log] {
            for v in Excitation::new(&f).apply(&vec![2.0; f.len()]) {
                assert!((v - 2.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn excitation_does_not_depend_on_the_grid() {
        // A broad peak, on a 1 Hz grid and at 1/96 octave: the same pattern within 0.05 dB.
        let lin: Vec<f64> = (50..=14_000).map(|v| v as f64).collect();
        let log = crate::signal::log_grid(50.0, 14_000.0, 96);
        let pat = |f: &[f64]| Excitation::new(f).db(&f.iter().map(|&x| peaking_power(x, 2200.0, 2.0, 6.0)).collect::<Vec<_>>());
        for (a, b) in pat(&lin).iter().zip(pat(&log)) {
            assert!((a - b).abs() < 0.05, "{a} {b}");
        }
    }

    #[test]
    fn peaking_filter_has_its_gain_at_its_centre() {
        for g in [-12.0, 3.0, 10.0] {
            assert!((db10(peaking_power(2200.0, 2200.0, 10.0, g)) - g).abs() < 1e-9);
        }
        assert!(db10(peaking_power(200.0, 2200.0, 10.0, 10.0)).abs() < 0.01);
    }

    #[test]
    fn the_ear_fills_a_narrow_notch() {
        // A total notch 40 Hz wide at 2.2 kHz (a sixth of the 262 Hz ERB) leaves a dip of
        // excitation below 1.5 dB; a 6 dB dip one octave wide keeps most of its depth.
        let f: Vec<f64> = (50..=14_000).map(|v| v as f64).collect();
        let ex = Excitation::new(&f);
        let flat = ex.db(&vec![1.0; f.len()]);
        let notch: Vec<f64> = f.iter().map(|&x| if (x - 2200.0).abs() < 20.0 { 0.0 } else { 1.0 }).collect();
        let broad: Vec<f64> = f.iter().map(|&x| if x > 2200.0 / 2f64.sqrt() && x < 2200.0 * 2f64.sqrt() { 0.25 } else { 1.0 }).collect();
        let narrow = summary_b(&difference(&ex.db(&notch), &flat));
        let wide = summary_b(&difference(&ex.db(&broad), &flat));
        assert!(narrow < 1.5, "{narrow}");
        assert!(wide > 5.0, "{wide}");
    }
}
