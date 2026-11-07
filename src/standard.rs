//! A two-orbit standard faithful to the rooms, tested on the loudspeaker library.
//!
//! Three parts, each computed from the two orbits of CTA-2034:
//!
//! 1. **Sound power.** CTA-2034 weighs each orbit direction by the "spherical square" around it.
//!    The *ring* estimate weighs the four half-orbits (right, up, left, down) at an angle from
//!    the axis by the area of the band of the sphere at that angle. The *ways* estimate rebuilds
//!    the sphere driver by driver from their own orbits, with phase, and adds them with their
//!    positions ([`crate::study::sp_from_ways`]).
//! 2. **In-room curves.** Weights of LW, ER and SP fitted to the rooms, checked on rooms and
//!    loudspeaker families left out of the fit.
//! 3. **Colouration of the reflections.** The roughness of each early-reflection direction of the
//!    orbits, read one by one instead of averaged, weighted by the energy each reflection brings:
//!    its share for an omnidirectional source in the rooms times the loudspeaker's own output in
//!    that direction.

use std::ops::Range;
use std::path::Path;

use crate::analysis::{self, Group, Metric, Study, Which, ANOMALY_BAND, ERROR_BAND};
use crate::cta2034::{self, Curves};
use crate::signal::db10;
use crate::special::median;
use crate::speaker::{library, Layout};

/// The library's results: one study per layout.
pub struct Library {
    pub layouts: Vec<Layout>,
    pub studies: Vec<Study>,
}

impl Library {
    pub fn load(dir: &Path) -> Self {
        let layouts = library();
        let studies = layouts.iter().map(|l| Study::load(&dir.join(format!("{}.bin", l.name)))).collect();
        Library { layouts, studies }
    }

    /// ``(layout index, variant name)`` of every variant.
    pub fn variants(&self) -> Vec<(usize, String)> {
        self.studies.iter().enumerate().flat_map(|(i, s)| s.variants().into_iter().map(move |v| (i, v))).collect()
    }
}

/// Sources of the sound power.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpFrom {
    /// The whole sphere (the reference).
    Sphere,
    /// CTA-2034's weights of the two orbits.
    Cta,
    /// The four half-orbits weighted by the area of their band of the sphere.
    Ring,
    /// The drivers' own orbits, rebuilt over the sphere with phase.
    Ways,
}

/// Weights of the 70 directions for the ring estimate of the sound power.
pub fn ring_weights() -> Vec<f64> {
    let band = |a: f64| ((a - 5.0).max(0.0).to_radians()).cos() - ((a + 5.0).min(180.0).to_radians()).cos();
    let keys = cta2034::keys();
    let mut w: Vec<f64> = keys
        .iter()
        .map(|&(_, a)| {
            let a = a.unsigned_abs() as f64;
            if a == 0.0 || a == 180.0 {
                band(a)
            } else {
                band(a) / 4.0
            }
        })
        .collect();
    let s: f64 = w.iter().sum();
    w.iter_mut().for_each(|v| *v /= s);
    w
}

/// The sound power of a variant from the given source.
pub fn sound_power(study: &Study, variant: &str, from: SpFrom) -> Vec<f64> {
    let v = study.res.variant(variant);
    match from {
        SpFrom::Sphere => v.sp_full.clone(),
        SpFrom::Cta => study.curves(variant).sp,
        SpFrom::Ring => cta2034::weighted(&ring_weights(), &v.cta),
        SpFrom::Ways => v.sp_ways.clone(),
    }
}

/// RMS and largest error (dB) of a sound-power estimate against the sphere, 200 Hz to 10 kHz.
pub fn sp_error(study: &Study, variant: &str, from: SpFrom) -> (f64, f64) {
    let f = study.freqs();
    let (a, b) = (sound_power(study, variant, from), sound_power(study, variant, SpFrom::Sphere));
    let e: Vec<f64> = (0..f.len()).filter(|&j| f[j] >= 200.0 && f[j] <= 10000.0).map(|j| db10(a[j]) - db10(b[j])).collect();
    ((e.iter().map(|x| x * x).sum::<f64>() / e.len() as f64).sqrt(), e.iter().fold(0.0f64, |m, x| m.max(x.abs())))
}

/// ``w_LW LW + w_ER ER + w_SP SP`` with the sound power from ``from``.
pub fn curve(c: &Curves, sp: &[f64], w: [f64; 3]) -> Vec<f64> {
    (0..c.lw.len()).map(|i| w[0] * c.lw[i] + w[1] * c.er[i] + w[2] * sp[i]).collect()
}

/// Weights ``(w_LW, w_ER, w_SP)``, summing to 1, that best predict (least squares on the level
/// ratio) the mean response of ``rooms`` for the variants of the layouts ``layouts``.
pub fn fit(lib: &Library, layouts: &[usize], from: SpFrom, rooms: Range<usize>, which: Which) -> [f64; 3] {
    let (mut a11, mut a12, mut a22, mut b1, mut b2) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &li in layouts {
        let study = &lib.studies[li];
        let freqs = study.freqs();
        let m = analysis::band(freqs, ERROR_BAND);
        for v in study.variants() {
            let c = study.curves(&v);
            let s = sound_power(study, &v, from);
            let rs = study.rooms(&v, which, rooms.clone());
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
    }
    let det = a11 * a22 - a12 * a12;
    let w1 = (b1 * a22 - b2 * a12) / det;
    let w2 = (a11 * b2 - a12 * b1) / det;
    [w1, w2, 1.0 - w1 - w2]
}

/// Median over ``rooms`` of the RMS error (dB, 300 Hz to 16 kHz) of a prediction, and its largest
/// mean error.
pub fn prediction_error(study: &Study, variant: &str, prediction: &[f64], rooms: Range<usize>, which: Which) -> (f64, f64) {
    let (rms, bias) = analysis::errors(study, variant, prediction, rooms, which);
    let f = study.freqs();
    let m = analysis::band(f, ERROR_BAND);
    (median(&rms), bias.iter().zip(&m).filter(|(_, b)| **b).fold(0.0f64, |a, (x, _)| a.max(x.abs())))
}

/// How far the far-field model (every path from the reference point) is from the paths taken from
/// each driver: median over the random rooms of the RMS difference (dB, 300 Hz to 16 kHz).
pub fn far_field_error(study: &Study, variant: &str, which: Which) -> f64 {
    let f = study.freqs();
    let m = analysis::band(f, ERROR_BAND);
    let v = study.res.variant(variant);
    let (kind, upto) = match which {
        Which::Steady => (0, study.res.categories.len()),
        Which::Early => (1, analysis::EARLY),
    };
    let e: Vec<f64> = study
        .random()
        .map(|r| {
            let near = study.room_response(variant, which, r);
            let omni = study.omni_sum(r, upto);
            let far = v.rooms_far.row(r * 2 + kind);
            let s: Vec<f64> = (0..f.len()).filter(|&j| m[j]).map(|j| (db10(near[j]) - db10(far[j] as f64 / omni[j])).powi(2)).collect();
            (s.iter().sum::<f64>() / s.len() as f64).sqrt()
        })
        .collect();
    median(&e)
}

/// The early-reflection bounces of CTA-2034 and the rooms' categories they stand for.
pub const BOUNCES: [(&str, &[u8]); 5] = [("floor", &[5]), ("ceiling", &[6]), ("side", &[1, 2]), ("front", &[3]), ("rear", &[4])];

/// Share of each bounce in the first reflections' energy of an omnidirectional source, from 600 Hz
/// to 8 kHz (median of the random rooms, normalized to sum to 1).
pub fn bounce_shares(study: &Study) -> [f64; 5] {
    let res = &study.res;
    let f = study.freqs();
    let m = analysis::band(f, ANOMALY_BAND);
    let n_cat = res.categories.len();
    let per_room: Vec<[f64; 5]> = study
        .random()
        .map(|r| {
            let cat = |c: usize| -> f64 {
                let row = res.omni.row(r * n_cat + c);
                row.iter().zip(&m).filter(|(_, b)| **b).map(|(v, _)| *v as f64).sum()
            };
            let raw: Vec<f64> = BOUNCES.iter().map(|(_, cs)| cs.iter().map(|&c| cat(c as usize)).sum()).collect();
            let s: f64 = raw.iter().sum();
            [raw[0] / s, raw[1] / s, raw[2] / s, raw[3] / s, raw[4] / s]
        })
        .collect();
    let mut out = [0.0; 5];
    for (k, o) in out.iter_mut().enumerate() {
        *o = median(&per_room.iter().map(|r| r[k]).collect::<Vec<_>>());
    }
    let s: f64 = out.iter().sum();
    out.map(|v| v / s)
}

/// The colouration of the reflections read on the two orbits: the roughness of every direction
/// of CTA-2034's early-reflection bounces, weighted by the bounce's share (``shares``) spread over
/// its directions and by the loudspeaker's output there (the power of the four crossovers of the
/// layout averaged, as for the rooms). ``only``: a subset of the bounces.
pub fn orbit_index(study: &Study, variant: &str, shares: &[f64; 5], only: Option<&[usize]>) -> f64 {
    let v = study.res.variant(variant);
    let prefix = variant.rsplit_once('-').unwrap().0;
    let siblings: Vec<&crate::results::VariantResults> = study.res.variants.iter().filter(|x| x.name.rsplit_once('-').unwrap().0 == prefix).collect();
    let power: Vec<f64> = (0..v.pband_cta.len()).map(|i| siblings.iter().map(|x| x.pband_cta[i] as f64).sum::<f64>() / siblings.len() as f64).collect();
    let (mut num, mut den) = (0.0, 0.0);
    for (b, (name, _)) in BOUNCES.iter().enumerate() {
        if only.is_some_and(|o| !o.contains(&b)) {
            continue;
        }
        let k = cta2034::kernel(name);
        for i in 0..k.len() {
            if k[i] > 0.0 {
                let w = shares[b] * k[i] * power[i];
                num += w * v.rough_cta[i] as f64;
                den += w;
            }
        }
    }
    num / den
}

/// The rooms' colouration of the paths with the roughness: median over the random rooms.
pub fn room_colouration(study: &Study, variant: &str, group: Group) -> f64 {
    median(&analysis::path_anomaly(study, variant, group, study.random(), Metric::Rough, true))
}

/// Roughness of a curve.
pub fn curve_roughness(study: &Study, p: &[f64]) -> f64 {
    analysis::roughness(study.freqs(), p)
}

/// Pearson correlation.
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let (mx, my) = (x.iter().sum::<f64>() / n, y.iter().sum::<f64>() / n);
    let sxy: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum();
    let syy: f64 = y.iter().map(|b| (b - my).powi(2)).sum();
    sxy / (sxx * syy).sqrt()
}
