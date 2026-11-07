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

/// The proposed in-room curves: steady state, and the early sound (direct sound and first
/// reflections), as weights of LW, ER and SP (rebuilt driver by driver).
pub const PROPOSED_STEADY: [f64; 3] = [0.1, 0.0, 0.9];
pub const PROPOSED_EARLY: [f64; 3] = [0.4, 0.4, 0.2];

/// One crossover of one loudspeaker.
#[derive(Clone, Debug)]
pub struct KindSummary {
    pub variant: String,
    /// The rooms' colouration of the paths (roughness, median of the random rooms): every path,
    /// the first reflections, the floor and ceiling bounces.
    pub rooms: [f64; 3],
    /// The colouration read on the orbits: every bounce, the floor and ceiling only.
    pub index: [f64; 2],
    /// Roughness of the averaged curves: CTA PIR, ER, the rooms' mean response.
    pub curves: [f64; 3],
}

/// One loudspeaker of the library.
#[derive(Clone, Debug)]
pub struct FamilySummary {
    pub name: String,
    /// Sound power from two orbits against the sphere (dB), worst of the four crossovers: RMS and
    /// largest error with CTA-2034's weights, the rings, driver by driver.
    pub sp_rms: [f64; 3],
    pub sp_max: [f64; 3],
    /// The far-field model against the paths from each driver (median RMS difference, dB, median of
    /// the four crossovers): steady state, early sound.
    pub far: [f64; 2],
    /// Prediction of the test rooms (median RMS error, dB, median of the four crossovers), steady
    /// state: CTA PIR, the proposed weights with CTA-2034's sound power, with the sound power rebuilt
    /// driver by driver.
    pub steady: [f64; 3],
    /// Early sound: CTA PIR, the proposed early weights.
    pub early: [f64; 2],
    pub kinds: Vec<KindSummary>,
}

/// Everything the report and the articles quote about the library.
#[derive(Clone, Debug)]
pub struct Summary {
    pub families: Vec<FamilySummary>,
    /// Shares of the first reflections' energy: floor, ceiling, side walls, front, rear.
    pub shares: [f64; 5],
    /// Weights fitted on the training rooms of every family (sound power driver by driver).
    pub steady_fit: [f64; 3],
    pub early_fit: [f64; 3],
    /// Range of the steady-state LW and SP weights fitted with each family left out.
    pub lofo_lw: (f64, f64),
    pub lofo_sp: (f64, f64),
    /// Ranges over the 28 variants of the median RMS error of the test rooms (dB), and the largest
    /// mean error: CTA PIR, proposed (driver by driver), proposed with CTA-2034's sound power, the
    /// PIR's weights with the sound power rebuilt; early: CTA PIR, proposed.
    pub errors: Vec<(String, f64, f64, f64)>,
    /// Correlation with the rooms' colouration of the first reflections: the orbit index, the
    /// roughness of the ER curve, of the PIR, of the rooms' mean response; and the orbit index
    /// of the floor and ceiling against the rooms' floor and ceiling bounces.
    pub corr: [f64; 5],
}

fn medianv(v: &[f64]) -> f64 {
    median(v)
}

pub fn summary(lib: &Library) -> Summary {
    let all: Vec<usize> = (0..lib.layouts.len()).collect();
    let shares = bounce_shares(&lib.studies[0]);
    let steady_fit = fit(lib, &all, SpFrom::Ways, 0..150, Which::Steady);
    let early_fit = fit(lib, &all, SpFrom::Ways, 0..150, Which::Early);
    let lofo: Vec<[f64; 3]> = all
        .iter()
        .map(|&li| {
            let others: Vec<usize> = all.iter().cloned().filter(|&x| x != li).collect();
            fit(lib, &others, SpFrom::Ways, 0..150, Which::Steady)
        })
        .collect();
    let range = |k: usize| (lofo.iter().map(|w| w[k]).fold(f64::MAX, f64::min), lofo.iter().map(|w| w[k]).fold(f64::MIN, f64::max));
    let predictions: [(&str, Which, Option<[f64; 3]>, SpFrom); 6] = [
        ("pirSteady", Which::Steady, None, SpFrom::Cta),
        ("propSteady", Which::Steady, Some(PROPOSED_STEADY), SpFrom::Ways),
        ("propSteadyCtaSp", Which::Steady, Some(PROPOSED_STEADY), SpFrom::Cta),
        ("pirWeightsWaysSp", Which::Steady, Some(cta2034::PIR_WEIGHTS), SpFrom::Ways),
        ("pirEarly", Which::Early, None, SpFrom::Cta),
        ("propEarly", Which::Early, Some(PROPOSED_EARLY), SpFrom::Ways),
    ];
    let mut per: Vec<Vec<Vec<f64>>> = vec![vec![Vec::new(); lib.layouts.len()]; predictions.len()];
    let mut errors = Vec::new();
    for (pi, (name, which, w, from)) in predictions.iter().enumerate() {
        let (mut lo, mut hi, mut bias) = (f64::MAX, 0.0f64, 0.0f64);
        for (li, s) in lib.studies.iter().enumerate() {
            for v in s.variants() {
                let c = s.curves(&v);
                let p = match w {
                    None => c.pir.clone(),
                    Some(w) => curve(&c, &sound_power(s, &v, *from), *w),
                };
                let (rms, b) = prediction_error(s, &v, &p, s.test(), *which);
                per[pi][li].push(rms);
                lo = lo.min(rms);
                hi = hi.max(rms);
                bias = bias.max(b);
            }
        }
        errors.push((name.to_string(), lo, hi, bias));
    }
    let mut families = Vec::new();
    let mut rooms_refl = Vec::new();
    let mut cols: [Vec<f64>; 6] = Default::default();
    for (li, s) in lib.studies.iter().enumerate() {
        let vs = s.variants();
        let sp: Vec<[(f64, f64); 3]> = vs.iter().map(|v| [SpFrom::Cta, SpFrom::Ring, SpFrom::Ways].map(|f| sp_error(s, v, f))).collect();
        let worst = |k: usize, max: bool| sp.iter().map(|e| if max { e[k].1 } else { e[k].0 }).fold(0.0f64, f64::max);
        let kinds: Vec<KindSummary> = vs
            .iter()
            .map(|v| {
                let c = s.curves(v);
                let k = KindSummary {
                    variant: v.clone(),
                    rooms: [Group::Steady, Group::Reflections, Group::Vertical].map(|g| room_colouration(s, v, g)),
                    index: [orbit_index(s, v, &shares, None), orbit_index(s, v, &shares, Some(&[0, 1]))],
                    curves: [curve_roughness(s, &c.pir), curve_roughness(s, &c.er), curve_roughness(s, &s.room_response_mean(v))],
                };
                rooms_refl.push(k.rooms[1]);
                cols[0].push(k.index[0]);
                cols[1].push(k.curves[1]);
                cols[2].push(k.curves[0]);
                cols[3].push(k.curves[2]);
                cols[4].push(k.index[1]);
                cols[5].push(k.rooms[2]);
                k
            })
            .collect();
        families.push(FamilySummary {
            name: lib.layouts[li].name.clone(),
            sp_rms: [worst(0, false), worst(1, false), worst(2, false)],
            sp_max: [worst(0, true), worst(1, true), worst(2, true)],
            far: [Which::Steady, Which::Early].map(|w| medianv(&vs.iter().map(|v| far_field_error(s, v, w)).collect::<Vec<_>>())),
            steady: [medianv(&per[0][li]), medianv(&per[2][li]), medianv(&per[1][li])],
            early: [medianv(&per[4][li]), medianv(&per[5][li])],
            kinds,
        });
    }
    let corr = [
        correlation(&cols[0], &rooms_refl),
        correlation(&cols[1], &rooms_refl),
        correlation(&cols[2], &rooms_refl),
        correlation(&cols[3], &rooms_refl),
        correlation(&cols[4], &cols[5]),
    ];
    Summary { families, shares, steady_fit, early_fit, lofo_lw: range(0), lofo_sp: range(2), errors, corr }
}
