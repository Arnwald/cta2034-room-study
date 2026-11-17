//! Do gentle crossover slopes harm what a listener receives in a room? The computation of
//! `protocol/crossover-slopes.md`.
//!
//! The instrument: excitation patterns (the power spectrum smoothed by the rounded-exponential
//! auditory filters of Glasberg and Moore, one ERB wide, at centres every 0.25 Cam from 200 Hz to
//! 10 kHz), the difference between two of them in dB, and two candidate summaries of it: A, its
//! root sum of squares over ERB number, and B, its largest value. Step 0 calibrates them on
//! resonances whose detection thresholds are published.

use std::f64::consts::PI;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::coherent::erb;
use crate::constants::C;
use crate::crossover::{self, KINDS};
use crate::mat::Mat;
use crate::room::{self, Setup};
use crate::signal::{db10, study_grid};
use crate::speaker::{roll, FarField, Layout};
use crate::sphere;
use crate::study::{MAX_ORDER, N_GRID};
use crate::{dot, norm, sub, unit, Vec3, C32, C64};

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

// ---------------------------------------------------------------------------------------------
// The pathways: each variant against its interference-free twin
// ---------------------------------------------------------------------------------------------

/// Linear grid of the early sound added with its phases (Hz).
pub const LIN_LO: f64 = 150.0;
pub const LIN_HI: f64 = 12_000.0;
pub const LIN_DF: f64 = 4.0;
/// Ears of a standing listener (m).
pub const STANDING_EAR: f64 = 1.65;
/// Band over which a path's weight is averaged (Hz).
pub const WEIGHT_BAND: (f64, f64) = (600.0, 8000.0);
pub const PATHWAYS: [&str; 5] = ["P1s", "P1d", "P2", "P3", "P4"];

/// One room's distances, every crossover (in the order of [`KINDS`]), in the raw units of summary
/// A (divide by T).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoomValues {
    /// ``[kind][pathway]``: the variant against its twin.
    pub d: Vec<[f64; 5]>,
    /// ``[kind]``: P4, the twin against the FIR's twin (how the slope blends the directivities).
    pub blend: Vec<f64>,
    /// ``[kind]``: P2 over the floor and ceiling bounces only.
    pub p2_vertical: Vec<f64>,
    /// ``[kind]``: P3 with the two ears of each head averaged in power, mean over the seven heads.
    pub p3_ears: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LayoutValues {
    pub name: String,
    pub label: String,
    pub roll: f64,
    pub crossovers: Vec<f64>,
    pub rooms: Vec<RoomValues>,
}

pub fn linear_grid() -> Vec<f64> {
    let n = ((LIN_HI - LIN_LO) / LIN_DF).round() as usize;
    (0..=n).map(|i| LIN_LO + i as f64 * LIN_DF).collect()
}

/// Complex values on ``src`` interpolated linearly in frequency onto ``dst``.
fn interp_complex(src: &[f64], v: &[C64], dst: &[f64]) -> Vec<C64> {
    dst.iter()
        .map(|&x| {
            let j = src.partition_point(|&s| s <= x).saturating_sub(1).min(src.len() - 2);
            let t = (x - src[j]) / (src[j + 1] - src[j]);
            v[j] * (1.0 - t) + v[j + 1] * t
        })
        .collect()
}

/// ``sum_d X_d H_d(u)``, the twin's with each driver moved to the baffle centre:
/// ``H_d(u) exp(-j k u.c_d)``. ``u``: the direction in the loudspeaker's own frame.
fn combine(x: &[Vec<C64>], h: &[Vec<C32>], u: &Vec3, centres: &[Vec3], k: &[f64], twin: bool) -> Vec<C64> {
    let nf = k.len();
    let mut out = vec![C64::new(0.0, 0.0); nf];
    for (d, hd) in h.iter().enumerate() {
        let us = dot(u, &centres[d]);
        for j in 0..nf {
            let mut v = x[d][j] * C64::new(hd[j].re as f64, hd[j].im as f64);
            if twin {
                v *= C64::from_polar(1.0, -k[j] * us);
            }
            out[j] += v;
        }
    }
    out
}

fn band_mean(f: &[f64], v: &[f64]) -> f64 {
    let (s, n) = f.iter().zip(v).filter(|(x, _)| **x >= WEIGHT_BAND.0 && **x <= WEIGHT_BAND.1).fold((0.0, 0), |(s, n), (_, y)| (s + y, n + 1));
    s / n as f64
}

/// The distances of one layout, rolled by ``roll_deg`` about its axis, in every setup given
/// (placed already: a tower's reference point half its height above the floor).
pub fn run_layout(layout: &Layout, roll_deg: f64, setups: &[Setup]) -> LayoutValues {
    let freqs = study_grid();
    let lin = linear_grid();
    let ex_s = Excitation::new(&freqs);
    let ex_l = Excitation::new(&lin);
    let far_s = FarField::new(layout, &freqs);
    let far_l = FarField::new(layout, &lin);
    let k_s: Vec<f64> = freqs.iter().map(|f| 2.0 * PI * f / C).collect();
    let k_l: Vec<f64> = lin.iter().map(|f| 2.0 * PI * f / C).collect();
    let centres: Vec<Vec3> = layout.drivers.iter().map(|d| d.centre()).collect();
    let nd = centres.len();
    let on = far_s.on_axis();
    let xs: Vec<Vec<Vec<C64>>> = KINDS.iter().map(|&k| crossover::filters(k, &freqs, layout, &on)).collect();
    let xl: Vec<Vec<Vec<C64>>> = xs.iter().map(|x| x.iter().map(|d| interp_complex(&freqs, d, &lin)).collect()).collect();

    // The power on the quadrature, for the diffuse field: [kind][0 variant, 1 twin].
    let grid = sphere::fibonacci(N_GRID);
    let solid_angle = 4.0 * PI / N_GRID as f64;
    let rgrid = roll(&grid, roll_deg);
    let resp_grid = far_s.responses(&rgrid);
    let nf = freqs.len();
    let grid_power: Vec<[Mat<f32>; 2]> = (0..KINDS.len())
        .map(|ki| {
            [false, true].map(|twin| {
                let mut m = Mat::<f32>::zeros(N_GRID, nf);
                m.data.par_chunks_mut(nf).enumerate().for_each(|(g, row)| {
                    let h: Vec<Vec<C32>> = (0..nd).map(|d| resp_grid[d].row(g).to_vec()).collect();
                    for (o, v) in row.iter_mut().zip(combine(&xs[ki], &h, &rgrid[g], &centres, &k_s, twin)) {
                        *o = v.norm_sqr() as f32;
                    }
                });
                m
            })
        })
        .collect();

    let rooms: Vec<RoomValues> = setups
        .par_iter()
        .map(|setup| {
            let im = room::image_sources(setup, MAX_ORDER);
            assert_eq!(im.order[0], 0, "the direct sound first");
            let n_img = im.order.len();
            let w = room::specular_weights(setup, &im, &freqs);
            let wl = room::specular_weights(setup, &im, &lin);
            let dif = room::diffuse_weights(setup, &grid, solid_angle, &freqs, MAX_ORDER);
            // Directions in the loudspeaker's own frame: the images, then the seated and the
            // standing listener's direct sound.
            let r = setup.receiver;
            let stand = unit(&setup.to_speaker(&sub(&[r[0], r[1], STANDING_EAR], &setup.source)));
            let mut dirs: Vec<Vec3> = im.direction.iter().map(unit).collect();
            dirs.push(dirs[0]);
            dirs.push(stand);
            let dirs = roll(&dirs, roll_deg);
            let resp = |far: &FarField, u: &Vec3| -> Vec<Vec<C32>> {
                (0..nd)
                    .map(|d| {
                        let mut row = vec![C32::new(0.0, 0.0); far.freqs.len()];
                        far.response(d, u, &mut row);
                        row
                    })
                    .collect()
            };
            let h_s: Vec<Vec<Vec<C32>>> = dirs.iter().map(|u| resp(&far_s, u)).collect();
            let early: Vec<usize> = (0..n_img).filter(|&i| im.order[i] <= 1).collect();
            let h_l: Vec<Vec<Vec<C32>>> = early.iter().map(|&i| resp(&far_l, &dirs[i])).collect();
            let pts = crate::coherent::points(setup);
            let first: Vec<usize> = (0..n_img).filter(|&i| im.order[i] == 1).collect();

            // Excitation patterns (dB) of every response: [kind][version] -> (P1s, P1d, P2 by
            // first-order path, P3 by point, P4).
            struct Patterns {
                p1s: Vec<f64>,
                p1d: Vec<f64>,
                p2: Vec<Vec<f64>>,
                p3: Vec<Vec<f64>>,
                p3_pow: Vec<Vec<f64>>,
                p4: Vec<f64>,
                band_power: Vec<f64>,
            }
            let patterns = |ki: usize, twin: bool| -> Patterns {
                let spectra: Vec<Vec<f64>> = dirs
                    .iter()
                    .zip(&h_s)
                    .map(|(u, h)| combine(&xs[ki], h, u, &centres, &k_s, twin).iter().map(|c| c.norm_sqr()).collect())
                    .collect();
                let mut e = vec![0.0; nf];
                for i in 0..n_img {
                    for j in 0..nf {
                        e[j] += w.row(i)[j] * spectra[i][j];
                    }
                }
                let gp = &grid_power[ki][twin as usize];
                for g in 0..N_GRID {
                    let (dr, pr) = (dif.row(g), gp.row(g));
                    for j in 0..nf {
                        e[j] += dr[j] * pr[j] as f64;
                    }
                }
                // The early sound with its phases at the fourteen points.
                let a: Vec<Vec<C64>> = early
                    .iter()
                    .zip(&h_l)
                    .map(|(&i, h)| {
                        let c = combine(&xl[ki], h, &dirs[i], &centres, &k_l, twin);
                        c.iter().zip(wl.row(i)).map(|(v, wk)| v * (wk.sqrt() * im.distance[i])).collect()
                    })
                    .collect();
                let p3_pow: Vec<Vec<f64>> = pts
                    .iter()
                    .map(|p| {
                        let mut s = vec![C64::new(0.0, 0.0); lin.len()];
                        for (ai, &i) in a.iter().zip(&early) {
                            let d = norm(&sub(p, &im.position[i]));
                            for (j, sj) in s.iter_mut().enumerate() {
                                *sj += ai[j] * C64::from_polar(1.0 / d, -k_l[j] * d);
                            }
                        }
                        ex_l.apply(&s.iter().map(|c| c.norm_sqr()).collect::<Vec<_>>())
                    })
                    .collect();
                Patterns {
                    p1s: ex_s.db(&spectra[n_img]),
                    p1d: ex_s.db(&spectra[n_img + 1]),
                    p2: first.iter().map(|&i| ex_s.db(&spectra[i])).collect(),
                    p3: p3_pow.iter().map(|p| p.iter().map(|v| db10(*v)).collect()).collect(),
                    p3_pow,
                    p4: ex_s.db(&e),
                    band_power: first.iter().map(|&i| band_mean(&freqs, &spectra[i])).collect(),
                }
            };
            let pats: Vec<[Patterns; 2]> = (0..KINDS.len()).map(|ki| [patterns(ki, false), patterns(ki, true)]).collect();
            let dist = |a: &[f64], b: &[f64]| summary_a(&difference(a, b));

            // The weight of each first-order path: its room weight times the twins' power in its
            // direction (the four crossovers averaged), over 600 Hz-8 kHz.
            let path_w: Vec<f64> = first
                .iter()
                .enumerate()
                .map(|(n, &i)| band_mean(&freqs, w.row(i)) * pats.iter().map(|p| p[1].band_power[n]).sum::<f64>() / KINDS.len() as f64)
                .collect();
            let vertical: Vec<bool> = first.iter().map(|&i| matches!(im.first[i], Some(4) | Some(5))).collect();
            let weighted = |v: &dyn Fn(usize) -> f64, keep: &dyn Fn(usize) -> bool| {
                let (mut s, mut n) = (0.0, 0.0);
                for (k, wk) in path_w.iter().enumerate() {
                    if keep(k) {
                        s += wk * v(k);
                        n += wk;
                    }
                }
                s / n
            };
            let fir = KINDS.len() - 1;
            let mut out = RoomValues::default();
            for [v, t] in pats.iter() {
                let p2 = |k: usize| dist(&v.p2[k], &t.p2[k]);
                let p3 = v.p3.iter().zip(&t.p3).map(|(a, b)| dist(a, b)).sum::<f64>() / pts.len() as f64;
                out.d.push([dist(&v.p1s, &t.p1s), dist(&v.p1d, &t.p1d), weighted(&p2, &|_| true), p3, dist(&v.p4, &t.p4)]);
                out.blend.push(dist(&t.p4, &pats[fir][1].p4));
                out.p2_vertical.push(weighted(&p2, &|k| vertical[k]));
                let ears = |p: &Patterns| -> Vec<Vec<f64>> {
                    p.p3_pow.chunks(2).map(|e| e[0].iter().zip(&e[1]).map(|(a, b)| db10(0.5 * (a + b))).collect()).collect()
                };
                let (ev, et) = (ears(v), ears(t));
                out.p3_ears.push(ev.iter().zip(&et).map(|(a, b)| dist(a, b)).sum::<f64>() / ev.len() as f64);
            }
            out
        })
        .collect();
    LayoutValues { name: layout.name.clone(), label: layout.label.clone(), roll: roll_deg, crossovers: layout.crossovers.clone(), rooms }
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
