//! Curves and metrics from the results.
//!
//! In-room responses are relative to an omnidirectional source at the same place: in **steady
//! state** (every path), and for the **early** sound (the direct sound and the six first-order
//! reflections, what CTA-2034's Listening Window and Early Reflections stand for).
//!
//! Deviation of a curve: the curve in dB on the 1/96-octave grid minus the straight line (dB
//! against log frequency) that best fits it from 600 Hz to 8 kHz; its RMS over that band
//! ([`anomaly`]), its RMS within one octave of the 2.2 kHz crossover ([`local_anomaly`]), the
//! deepest dip there ([`notch_depth`]).

use std::ops::Range;
use std::path::Path;

use crate::cta2034::{self, Curves};
use crate::results::Results;
use crate::signal::db10;
use crate::special::{linefit, mean};

pub const ANOMALY_BAND: (f64, f64) = (600.0, 8000.0);
pub const CROSSOVER_BAND: (f64, f64) = (1100.0, 4400.0);
pub const ERROR_BAND: (f64, f64) = (300.0, 16000.0);
/// The first (training) half of the random rooms fits the updated weights; the second tests them.
pub const SPLIT: usize = 150;
/// Categories of the early sound: the direct sound and the six first reflections.
pub const EARLY: usize = 7;

pub fn band(freqs: &[f64], lo_hi: (f64, f64)) -> Vec<bool> {
    freqs.iter().map(|&f| f >= lo_hi.0 && f <= lo_hi.1).collect()
}

/// Residual (dB) of the straight-line fit over ``lo_hi`` and its RMS.
pub fn line_residual(freqs: &[f64], power: &[f64], lo_hi: (f64, f64)) -> (Vec<f64>, f64) {
    let m = band(freqs, lo_hi);
    let x: Vec<f64> = freqs.iter().zip(&m).filter(|(_, b)| **b).map(|(f, _)| f.log2()).collect();
    let y: Vec<f64> = power.iter().zip(&m).filter(|(_, b)| **b).map(|(p, _)| db10(*p)).collect();
    let (a, b) = linefit(&x, &y);
    let r: Vec<f64> = x.iter().zip(&y).map(|(x, y)| y - (a * x + b)).collect();
    let rms = (r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64).sqrt();
    (r, rms)
}

pub fn anomaly(freqs: &[f64], power: &[f64]) -> f64 {
    line_residual(freqs, power, ANOMALY_BAND).1
}

/// Residual of the line fitted from 600 Hz to 8 kHz, kept within one octave of the crossover.
pub fn local_residual(freqs: &[f64], power: &[f64]) -> Vec<f64> {
    let (r, _) = line_residual(freqs, power, ANOMALY_BAND);
    let f_fit: Vec<f64> = freqs.iter().cloned().filter(|&f| f >= ANOMALY_BAND.0 && f <= ANOMALY_BAND.1).collect();
    r.iter()
        .zip(&f_fit)
        .filter(|(_, f)| **f >= CROSSOVER_BAND.0 && **f <= CROSSOVER_BAND.1)
        .map(|(v, _)| *v)
        .collect()
}

pub fn local_anomaly(freqs: &[f64], power: &[f64]) -> f64 {
    let r = local_residual(freqs, power);
    (r.iter().map(|v| v * v).sum::<f64>() / r.len() as f64).sqrt()
}

/// Depth of the deepest dip below the straight line within one octave of the crossover (dB).
pub fn notch_depth(freqs: &[f64], power: &[f64]) -> f64 {
    -local_residual(freqs, power).iter().cloned().fold(f64::INFINITY, f64::min)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Which {
    Steady,
    Early,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sp {
    TwoOrbits,
    Sphere,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Group {
    Steady,
    Early,
    Reflections,
    Vertical,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Metric {
    Anom,
    Local,
    Notch,
}

pub struct Study {
    pub res: Results,
}

impl Study {
    pub fn load(path: &Path) -> Self {
        Study { res: Results::load(path) }
    }

    pub fn freqs(&self) -> &[f64] {
        &self.res.freqs
    }

    pub fn n_random(&self) -> usize {
        self.res.n_random
    }

    pub fn random(&self) -> Range<usize> {
        0..self.res.n_random
    }

    pub fn train(&self) -> Range<usize> {
        0..SPLIT
    }

    pub fn test(&self) -> Range<usize> {
        SPLIT..self.res.n_random
    }

    pub fn variants(&self) -> Vec<String> {
        self.res.variants.iter().map(|v| v.name.clone()).collect()
    }

    pub fn named_index(&self, key: &str) -> usize {
        self.res.keys.iter().position(|k| k.as_deref() == Some(key)).expect("named room")
    }

    fn n_cat(&self) -> usize {
        self.res.categories.len()
    }

    /// Omnidirectional energy of one room, summed over the first ``upto`` categories.
    pub fn omni_sum(&self, room: usize, upto: usize) -> Vec<f64> {
        let nf = self.res.freqs.len();
        let mut out = vec![0.0; nf];
        for c in 0..upto {
            for (o, v) in out.iter_mut().zip(self.res.omni.row(room * self.n_cat() + c)) {
                *o += *v as f64;
            }
        }
        out
    }

    /// Normalized in-room response (power) of one room.
    pub fn room_response(&self, variant: &str, which: Which, room: usize) -> Vec<f64> {
        let v = self.res.variant(variant);
        let upto = match which {
            Which::Steady => self.n_cat(),
            Which::Early => EARLY,
        };
        let nf = self.res.freqs.len();
        let mut e = vec![0.0; nf];
        for c in 0..upto {
            for (o, x) in e.iter_mut().zip(v.rooms.row(room * self.n_cat() + c)) {
                *o += *x as f64;
            }
        }
        let g = self.omni_sum(room, upto);
        e.iter().zip(&g).map(|(a, b)| a / b).collect()
    }

    pub fn rooms(&self, variant: &str, which: Which, rooms: Range<usize>) -> Vec<Vec<f64>> {
        rooms.map(|r| self.room_response(variant, which, r)).collect()
    }

    pub fn curves(&self, variant: &str) -> Curves {
        let v = self.res.variant(variant);
        cta2034::curves(&v.cta, v.sp_full.clone())
    }

    /// Share of each category in the energy at the listener (omnidirectional source):
    /// ``[room][category][freq]``.
    pub fn shares(&self, rooms: Range<usize>) -> Vec<Vec<Vec<f64>>> {
        rooms
            .map(|r| {
                let total = self.omni_sum(r, self.n_cat());
                (0..self.n_cat())
                    .map(|c| self.res.omni.row(r * self.n_cat() + c).iter().zip(&total).map(|(v, t)| *v as f64 / t).collect())
                    .collect()
            })
            .collect()
    }
}

/// ``w_LW LW + w_ER ER + w_SP SP``, SP from the two orbits or the sphere.
pub fn pir(c: &Curves, w: [f64; 3], sp: Sp) -> Vec<f64> {
    let s = match sp {
        Sp::TwoOrbits => &c.sp,
        Sp::Sphere => &c.sp_sphere,
    };
    (0..c.lw.len()).map(|i| w[0] * c.lw[i] + w[1] * c.er[i] + w[2] * s[i]).collect()
}

/// Weights ``(w_LW, w_ER, w_SP)``, summing to 1, that best predict (least squares on the level
/// ratio) the mean in-room response of ``rooms`` for every variant in ``variants``.
pub fn fit_weights_on(study: &Study, variants: &[String], sp: Sp, rooms: Range<usize>, which: Which) -> [f64; 3] {
    let freqs = study.freqs();
    let m = band(freqs, ERROR_BAND);
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for v in variants {
        let c = study.curves(v);
        let rs = study.rooms(v, which, rooms.clone());
        let s = match sp {
            Sp::TwoOrbits => &c.sp,
            Sp::Sphere => &c.sp_sphere,
        };
        for j in 0..freqs.len() {
            if !m[j] {
                continue;
            }
            let target = rs.iter().map(|r| r[j]).sum::<f64>() / rs.len() as f64;
            let x1 = (c.lw[j] - s[j]) / target;
            let x2 = (c.er[j] - s[j]) / target;
            let y = 1.0 - s[j] / target;
            a11 += x1 * x1;
            a12 += x1 * x2;
            a22 += x2 * x2;
            b1 += x1 * y;
            b2 += x2 * y;
        }
    }
    let det = a11 * a22 - a12 * a12;
    let w1 = (b1 * a22 - b2 * a12) / det;
    let w2 = (a11 * b2 - a12 * b1) / det;
    [w1, w2, 1.0 - w1 - w2]
}

pub fn fit_weights(study: &Study, sp: Sp, which: Which) -> [f64; 3] {
    fit_weights_on(study, &study.variants(), sp, study.train(), which)
}

/// Per-room error ``prediction - room`` (dB) over 300 Hz-16 kHz: its RMS per room, and the mean
/// error against frequency (the bias).
pub fn errors(study: &Study, variant: &str, prediction: &[f64], rooms: Range<usize>, which: Which) -> (Vec<f64>, Vec<f64>) {
    let freqs = study.freqs();
    let m = band(freqs, ERROR_BAND);
    let rs = study.rooms(variant, which, rooms);
    let mut bias = vec![0.0; freqs.len()];
    let mut rms = Vec::with_capacity(rs.len());
    for r in &rs {
        let mut s = 0.0;
        let mut n = 0;
        for j in 0..freqs.len() {
            let e = db10(prediction[j]) - db10(r[j]);
            bias[j] += e / rs.len() as f64;
            if m[j] {
                s += e * e;
                n += 1;
            }
        }
        rms.push((s / n as f64).sqrt());
    }
    (rms, bias)
}

// ---------------------------------------------------------------------------------------------
// Colouration of the paths themselves
// ---------------------------------------------------------------------------------------------

fn metric_key(m: Metric) -> usize {
    match m {
        Metric::Anom => 0,
        Metric::Local => 1,
        Metric::Notch => 2,
    }
}

fn metric_values<'a>(v: &'a crate::results::VariantResults, m: Metric, which: &str) -> &'a [f32] {
    match (metric_key(m), which) {
        (0, "cta") => &v.anom_cta,
        (0, "grid") => &v.anom_grid,
        (0, _) => &v.anom_img,
        (1, "cta") => &v.local_cta,
        (1, "grid") => &v.local_grid,
        (1, _) => &v.local_img,
        (_, "cta") => &v.notch_cta,
        (_, "grid") => &v.notch_grid,
        _ => &v.notch_img,
    }
}

/// Band-averaged power of the four crossovers of the same loudspeaker, averaged: weights that do
/// not depend on the crossover, so that a notch does not lower the weight of the path it colours.
fn common_power(study: &Study, variant: &str, which: &str) -> Vec<f64> {
    let prefix = variant.rsplit_once('-').unwrap().0;
    let vs: Vec<&crate::results::VariantResults> =
        study.res.variants.iter().filter(|v| v.name.rsplit_once('-').unwrap().0 == prefix).collect();
    let pick = |v: &crate::results::VariantResults| -> Vec<f32> {
        match which {
            "cta" => v.pband_cta.clone(),
            "grid" => v.pband_grid.clone(),
            _ => v.pband_img.clone(),
        }
    };
    let n = pick(vs[0]).len();
    let mut out = vec![0.0; n];
    for v in &vs {
        for (o, x) in out.iter_mut().zip(pick(v)) {
            *o += x as f64 / vs.len() as f64;
        }
    }
    out
}

fn own_power(study: &Study, variant: &str, which: &str) -> Vec<f64> {
    let v = study.res.variant(variant);
    let p = match which {
        "cta" => &v.pband_cta,
        "grid" => &v.pband_grid,
        _ => &v.pband_img,
    };
    p.iter().map(|x| *x as f64).collect()
}

/// Colouration of the paths, per room: each path's own deviation, weighted by the energy it
/// brings (the room's weight of the path times the loudspeaker's power in its direction, both
/// averaged over 600 Hz-8 kHz; with ``common`` the power of the four crossovers averaged).
pub fn path_anomaly(study: &Study, variant: &str, group: Group, rooms: Range<usize>, metric: Metric, common: bool) -> Vec<f64> {
    let res = &study.res;
    let v = res.variant(variant);
    let p_img = if common { common_power(study, variant, "img") } else { own_power(study, variant, "img") };
    let p_grid = if common { common_power(study, variant, "grid") } else { own_power(study, variant, "grid") };
    let a_img = metric_values(v, metric, "img");
    let a_grid = metric_values(v, metric, "grid");
    rooms
        .map(|r| {
            let (s0, s1) = (res.image_starts[r], res.image_starts[r + 1]);
            let mut num = 0.0;
            let mut den = 0.0;
            for i in s0..s1 {
                let c = res.image_cat[i];
                let keep = match group {
                    Group::Steady => true,
                    Group::Early => c <= 6,
                    Group::Reflections => (1..=6).contains(&c),
                    Group::Vertical => c == 5 || c == 6,
                };
                if keep {
                    let w = res.image_weights_band[i] as f64 * p_img[i];
                    num += w * a_img[i] as f64;
                    den += w;
                }
            }
            if group == Group::Steady {
                let d = res.diffuse_band.row(r);
                for k in 0..d.len() {
                    let w = d[k] as f64 * p_grid[k];
                    num += w * a_grid[k] as f64;
                    den += w;
                }
            }
            num / den
        })
        .collect()
}

/// The same colouration measure weighted by a CTA-2034-style kernel (``w_LW`` over the listening
/// window, ``w_ER`` over the early reflections' directions, ``w_SP`` over the 70 directions or
/// uniformly over the sphere), each direction also weighted by the loudspeaker's power there.
pub fn cta_path_anomaly(study: &Study, variant: &str, w: [f64; 3], sp: Sp, metric: Metric, common: bool) -> f64 {
    let v = study.res.variant(variant);
    let a = metric_values(v, metric, "cta");
    let p = if common { common_power(study, variant, "cta") } else { own_power(study, variant, "cta") };
    let (lw, er, spk) = (cta2034::kernel("lw"), cta2034::kernel("er"), cta2034::kernel("sp"));
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..a.len() {
        let k = w[0] * lw[i] + w[1] * er[i];
        num += k * p[i] * a[i] as f64;
        den += k * p[i];
    }
    match sp {
        Sp::TwoOrbits => {
            for i in 0..a.len() {
                num += w[2] * spk[i] * p[i] * a[i] as f64;
                den += w[2] * spk[i] * p[i];
            }
        }
        Sp::Sphere => {
            let pg = if common { common_power(study, variant, "grid") } else { own_power(study, variant, "grid") };
            let ag = metric_values(v, metric, "grid");
            let n = pg.len() as f64;
            num += w[2] * pg.iter().zip(ag).map(|(p, a)| p * *a as f64).sum::<f64>() / n;
            den += w[2] * pg.iter().sum::<f64>() / n;
        }
    }
    num / den
}

pub fn median_of(values: &[f64]) -> f64 {
    crate::special::median(values)
}

pub fn mean_of(values: &[f64]) -> f64 {
    mean(values)
}
