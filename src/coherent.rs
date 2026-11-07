//! Interference between the paths: the images added with their phases at the listener's ears.
//!
//! The study adds the paths in power, as if each were heard on its own. Here the direct sound and
//! the specular images up to the third order are added as pressures at fourteen points: the two
//! ears (17.5 cm apart, without a head) of seven head positions, the reference one, 10 cm to either
//! side, 10 cm forward and back, 5 cm up and down. Each image keeps the square root of its energy
//! weight (a real, positive reflection factor: hard surfaces), the loudspeaker's response in the
//! direction the path leaves in (taken for the reference position, a few degrees off for the
//! others) and the delay of its path to the point. The diffuse field is added in energy. Every
//! response is then smoothed as the ear analyses frequencies, by the rounded-exponential auditory
//! filters of Glasberg and Moore, one ERB wide, before the deviation metrics of the study.
//!
//! The drivers' responses and the crossover filters are those of the study, on its 1/96-octave
//! grid, interpolated (complex, linear in frequency) on a linear grid of 4 Hz: fine enough for the
//! combs of the longest paths.

use std::f64::consts::PI;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::iter::once;
use std::path::Path;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::analysis::{anomaly, local_anomaly, EARLY};
use crate::constants::C;
use crate::crossover::{self, KINDS};
use crate::results::Results;
use crate::room::{self, Setup};
use crate::signal::db10;
use crate::special::interp;
use crate::speaker::{FarField, Layout};
use crate::study::{variant_name, MAX_ORDER};
use crate::{norm, sub, Vec3, C64};

/// Linear grid of the coherent sums (Hz).
pub const F_LO: f64 = 300.0;
pub const F_HI: f64 = 12_000.0;
pub const DF: f64 = 4.0;
/// Band of the smoothed responses, on the study's grid (Hz).
pub const OUT: (f64, f64) = (500.0, 9000.0);
/// Half the distance between the ears (m).
pub const EAR: f64 = 0.0875;
/// Head positions relative to the reference one, in the room frame (across toward the right,
/// along toward the loudspeakers, up), metres.
pub const HEADS: [Vec3; 7] = [
    [0.0, 0.0, 0.0],
    [-0.1, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, -0.1, 0.0],
    [0.0, 0.1, 0.0],
    [0.0, 0.0, -0.05],
    [0.0, 0.0, 0.05],
];

/// Name of the omnidirectional source among the variants.
pub const OMNI: &str = "omnidirectional";

/// Equivalent rectangular bandwidth of the auditory filter centred at ``f`` (Glasberg and Moore).
pub fn erb(f: f64) -> f64 {
    24.7 * (4.37 * f / 1000.0 + 1.0)
}

pub fn linear_grid() -> Vec<f64> {
    let n = ((F_HI - F_LO) / DF).round() as usize;
    (0..=n).map(|i| F_LO + i as f64 * DF).collect()
}

/// The fourteen points: the left and right ear of each head position, in that order.
pub fn points(setup: &Setup) -> Vec<Vec3> {
    let r = setup.receiver;
    HEADS.iter().flat_map(|h| [-EAR, EAR].map(|e| [r[0] + h[0] + e, r[1] + h[1], r[2] + h[2]])).collect()
}

/// Rounded-exponential auditory filters centred at ``centres``, over a linear grid: the weight
/// ``(1 + p g) exp(-p g)``, ``g = |f - fc| / fc``, ``p = 4 fc / ERB(fc)``, normalized.
pub struct Smoother {
    rows: Vec<(usize, Vec<f64>)>,
}

impl Smoother {
    pub fn new(f: &[f64], centres: &[f64]) -> Self {
        let df = f[1] - f[0];
        let rows = centres
            .iter()
            .map(|&fc| {
                let p = 4.0 * fc / erb(fc);
                let half = 12.0 * fc / p;
                let lo = ((fc - half - f[0]) / df).floor().max(0.0) as usize;
                let hi = (((fc + half - f[0]) / df).ceil().max(0.0) as usize).min(f.len() - 1);
                let mut w: Vec<f64> = (lo..=hi)
                    .map(|k| {
                        let g = (f[k] - fc).abs() / fc;
                        (1.0 + p * g) * (-p * g).exp()
                    })
                    .collect();
                let s: f64 = w.iter().sum();
                w.iter_mut().for_each(|v| *v /= s);
                (lo, w)
            })
            .collect();
        Smoother { rows }
    }

    pub fn apply(&self, power: &[f64]) -> Vec<f64> {
        self.rows.iter().map(|(lo, w)| w.iter().zip(&power[*lo..]).map(|(a, b)| a * b).sum()).collect()
    }
}

/// Linear interpolation from ``src`` onto ``dst``: the index of the point below and the weight of
/// the next one.
fn table(src: &[f64], dst: &[f64]) -> Vec<(usize, f64)> {
    dst.iter()
        .map(|&x| {
            let j = src.partition_point(|&v| v <= x).saturating_sub(1).min(src.len() - 2);
            (j, (x - src[j]) / (src[j + 1] - src[j]))
        })
        .collect()
}

/// A power spectrum on the study's grid, interpolated in dB against log frequency.
fn power_on(study_f: &[f64], p: &[f64], f: &[f64]) -> Vec<f64> {
    let x: Vec<f64> = study_f.iter().map(|v| v.log2()).collect();
    let y: Vec<f64> = p.iter().map(|v| db10(*v)).collect();
    f.iter().map(|v| 10f64.powf(interp(v.log2(), &x, &y) / 10.0)).collect()
}

fn mean_curves(curves: &[Vec<f64>]) -> Vec<f64> {
    let n = curves.len() as f64;
    (0..curves[0].len()).map(|k| curves.iter().map(|c| c[k]).sum::<f64>() / n).collect()
}

/// The study's deviation metrics of one response (dB): RMS within one octave of the crossover,
/// and over 600 Hz to 8 kHz.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Metrics {
    pub local: f32,
    pub full: f32,
}

impl Metrics {
    fn of(f: &[f64], p: &[f64]) -> Self {
        Metrics { local: local_anomaly(f, p) as f32, full: anomaly(f, p) as f32 }
    }
}

/// One room, one loudspeaker variant, one kind of response (steady state or early sound).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RoomMetrics {
    /// The paths in power at the reference point: the study's model, smoothed the same way.
    pub energy: Metrics,
    /// The paths with their phases at each of the fourteen points.
    pub points: Vec<Metrics>,
    /// The two ears of each head position, averaged in power.
    pub ears: Vec<Metrics>,
    /// The fourteen points averaged in power.
    pub spatial: Metrics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CoherentResults {
    /// The upright variants, layout by layout, and the omnidirectional source ([`OMNI`]).
    pub variants: Vec<String>,
    /// Frequencies of the smoothed responses (the study's grid within ``OUT``).
    pub freqs: Vec<f64>,
    pub n_random: usize,
    /// ``[variant][room][0: steady state, 1: early sound]``.
    pub rooms: Vec<Vec<[RoomMetrics; 2]>>,
    /// The living room's smoothed responses relative to the omnidirectional source:
    /// ``[variant][kind][0: in power at the reference point, 1..=14: the points]``.
    pub living_index: usize,
    pub living: Vec<[Vec<Vec<f32>>; 2]>,
    /// Largest difference (dB) between the energy of the images here and in the study, from
    /// 500 Hz to 9 kHz.
    pub check_db: f64,
}

impl CoherentResults {
    pub fn save(&self, path: &Path) {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        let mut w = BufWriter::new(File::create(path).unwrap());
        bincode::serde::encode_into_std_write(self, &mut w, bincode::config::standard()).unwrap();
    }

    pub fn load(path: &Path) -> Self {
        let mut r = BufReader::new(File::open(path).unwrap_or_else(|e| panic!("{}: {e} (run the coherent study first)", path.display())));
        bincode::serde::decode_from_std_read(&mut r, bincode::config::standard()).unwrap()
    }

    pub fn variant(&self, name: &str) -> usize {
        self.variants.iter().position(|v| v == name).unwrap_or_else(|| panic!("no variant {name}"))
    }
}

/// The coherent responses of the upright variants of ``layouts`` in every setup, from the
/// study's results (its filters, its diffuse field, its omnidirectional source).
pub fn run(setups: &[Setup], res: &Results, layouts: &[Layout], living: usize) -> CoherentResults {
    let study_f = &res.freqs;
    let n_cat = res.categories.len();
    let f = linear_grid();
    let nf = f.len();
    let lo = study_f.partition_point(|&x| x < F_LO).saturating_sub(1);
    let hi = (study_f.partition_point(|&x| x <= F_HI) + 1).min(study_f.len());
    let fs = &study_f[lo..hi];
    let tab = table(fs, &f);
    let out_f: Vec<f64> = study_f.iter().cloned().filter(|&x| x >= OUT.0 && x <= OUT.1).collect();
    let smoother = Smoother::new(&f, &out_f);

    // ``layout: None``: an omnidirectional source, the room's own combs.
    struct Var {
        layout: Option<usize>,
        name: String,
        filters: Vec<Vec<C64>>,
        study: usize,
    }
    let fars: Vec<FarField> = layouts.iter().map(|l| FarField::new(l, fs)).collect();
    let mut vars = Vec::new();
    for (li, layout) in layouts.iter().enumerate() {
        let on = FarField::new(layout, study_f).on_axis();
        for k in KINDS {
            let filters = crossover::filters(k, study_f, layout, &on).into_iter().map(|d| d[lo..hi].to_vec()).collect();
            let name = variant_name(layout, 0.0, k);
            let study = res.variants.iter().position(|v| v.name == name).expect("variant in the study");
            vars.push(Var { layout: Some(li), name, filters, study });
        }
    }
    vars.push(Var { layout: None, name: OMNI.into(), filters: Vec::new(), study: 0 });

    struct RoomOut {
        metrics: Vec<[RoomMetrics; 2]>,
        living: Vec<[Vec<Vec<f32>>; 2]>,
        check: f64,
    }
    let outs: Vec<RoomOut> = setups
        .par_iter()
        .enumerate()
        .map(|(ri, setup)| {
            let im = room::image_sources(setup, MAX_ORDER);
            let n_img = im.order.len();
            let w = room::specular_weights(setup, &im, &f);
            let w_fs = room::specular_weights(setup, &im, fs);
            let dist: Vec<Vec<f64>> = points(setup).iter().map(|p| im.position.iter().map(|u| norm(&sub(p, u))).collect()).collect();
            let omni = |upto: usize| {
                let mut s = vec![0.0; study_f.len()];
                for c in 0..upto {
                    for (o, v) in s.iter_mut().zip(res.omni.row(ri * n_cat + c)) {
                        *o += *v as f64;
                    }
                }
                power_on(study_f, &s, &f)
            };
            let omni = [omni(n_cat), omni(EARLY)];
            let mut out = RoomOut { metrics: Vec::new(), living: Vec::new(), check: 0.0 };
            for li in (0..fars.len()).map(Some).chain(once(None)) {
                let resp = li.map(|l| fars[l].responses(&im.direction)).unwrap_or_default();
                for var in vars.iter().filter(|v| v.layout == li) {
                    // Each image's response on the study's grid (its energy there checks the
                    // computation against the study's), then on the linear grid, with the
                    // image's amplitude at 1 m.
                    let mut e_fs = vec![0.0; fs.len()];
                    let a: Vec<Vec<C64>> = (0..n_img)
                        .map(|i| {
                            let h: Vec<C64> = (0..fs.len())
                                .map(|j| {
                                    if li.is_none() {
                                        return C64::new(1.0, 0.0);
                                    }
                                    let mut s = C64::new(0.0, 0.0);
                                    for (x, r) in var.filters.iter().zip(&resp) {
                                        let v = r.row(i)[j];
                                        s += x[j] * C64::new(v.re as f64, v.im as f64);
                                    }
                                    s
                                })
                                .collect();
                            for (e, (wj, hj)) in e_fs.iter_mut().zip(w_fs.row(i).iter().zip(&h)) {
                                *e += wj * hj.norm_sqr();
                            }
                            let gain = im.distance[i];
                            tab.iter().zip(w.row(i)).map(|(&(j, t), wk)| (h[j] * (1.0 - t) + h[j + 1] * t) * (wk.sqrt() * gain)).collect()
                        })
                        .collect();
                    let study_rows = |c: usize| if li.is_some() { res.variants[var.study].rooms.row(ri * n_cat + c) } else { res.omni.row(ri * n_cat + c) };
                    for (j, e) in e_fs.iter().enumerate() {
                        if fs[j] >= OUT.0 && fs[j] <= OUT.1 {
                            let s: f64 = (0..n_cat - 1).map(|c| study_rows(c)[lo + j] as f64).sum();
                            out.check = out.check.max((db10(*e) - db10(s)).abs());
                        }
                    }
                    let diffuse_study: Vec<f64> = study_rows(n_cat - 1).iter().map(|x| *x as f64).collect();
                    let diffuse = power_on(study_f, &diffuse_study, &f);
                    // The paths in power at the reference point.
                    let mut energy = [diffuse.clone(), vec![0.0; nf]];
                    for (i, ai) in a.iter().enumerate() {
                        let g = 1.0 / (im.distance[i] * im.distance[i]);
                        for (k, v) in ai.iter().enumerate() {
                            let e = v.norm_sqr() * g;
                            energy[0][k] += e;
                            if im.order[i] <= 1 {
                                energy[1][k] += e;
                            }
                        }
                    }
                    // With their phases at every point.
                    let mut coherent: [Vec<Vec<f64>>; 2] = [Vec::new(), Vec::new()];
                    for d in &dist {
                        let mut all = vec![C64::new(0.0, 0.0); nf];
                        let mut early = vec![C64::new(0.0, 0.0); nf];
                        for i in 0..n_img {
                            let k0 = -2.0 * PI * d[i] / C;
                            let step = C64::from_polar(1.0, k0 * DF);
                            let mut z = C64::new(0.0, 0.0);
                            for k in 0..nf {
                                if k % 256 == 0 {
                                    z = C64::from_polar(1.0 / d[i], k0 * f[k]);
                                }
                                let p = a[i][k] * z;
                                all[k] += p;
                                if im.order[i] <= 1 {
                                    early[k] += p;
                                }
                                z *= step;
                            }
                        }
                        coherent[0].push((0..nf).map(|k| (all[k].norm_sqr() + diffuse[k]) / omni[0][k]).collect());
                        coherent[1].push((0..nf).map(|k| early[k].norm_sqr() / omni[1][k]).collect());
                    }
                    let mut m: [RoomMetrics; 2] = Default::default();
                    let mut curves: [Vec<Vec<f32>>; 2] = Default::default();
                    for kind in 0..2 {
                        let e = smoother.apply(&energy[kind].iter().zip(&omni[kind]).map(|(a, b)| a / b).collect::<Vec<_>>());
                        let p: Vec<Vec<f64>> = coherent[kind].iter().map(|c| smoother.apply(c)).collect();
                        m[kind].energy = Metrics::of(&out_f, &e);
                        m[kind].points = p.iter().map(|c| Metrics::of(&out_f, c)).collect();
                        m[kind].ears = p.chunks(2).map(|pair| Metrics::of(&out_f, &mean_curves(pair))).collect();
                        m[kind].spatial = Metrics::of(&out_f, &mean_curves(&p));
                        if ri == living {
                            curves[kind] = once(&e).chain(&p).map(|c| c.iter().map(|v| *v as f32).collect()).collect();
                        }
                    }
                    out.metrics.push(m);
                    if ri == living {
                        out.living.push(curves);
                    }
                }
            }
            out
        })
        .collect();
    let check_db = outs.iter().map(|o| o.check).fold(0.0, f64::max);
    let living_curves = outs[living].living.clone();
    let rooms = (0..vars.len()).map(|vi| outs.iter().map(|o| o.metrics[vi].clone()).collect()).collect();
    CoherentResults {
        variants: vars.into_iter().map(|v| v.name).collect(),
        freqs: out_f,
        n_random: res.n_random,
        rooms,
        living_index: living,
        living: living_curves,
        check_db,
    }
}

/// Medians over the random rooms of the deviation within one octave of the crossover of one
/// variant (``kind``: 0 steady state, 1 early sound): the paths in power, one point, two ears,
/// the fourteen points.
pub fn values(coh: &CoherentResults, variant: &str, kind: usize) -> [f64; 4] {
    let rooms = &coh.rooms[coh.variant(variant)][..coh.n_random];
    let med = |v: Vec<f64>| crate::special::median(&v);
    [
        med(rooms.iter().map(|r| r[kind].energy.local as f64).collect()),
        med(rooms.iter().flat_map(|r| r[kind].points.iter().map(|m| m.local as f64)).collect()),
        med(rooms.iter().flat_map(|r| r[kind].ears.iter().map(|m| m.local as f64)).collect()),
        med(rooms.iter().map(|r| r[kind].spatial.local as f64).collect()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditory_filters_are_one_erb_wide_and_keep_a_flat_spectrum() {
        let f = linear_grid();
        let centres = [500.0, 2200.0, 8000.0];
        let s = Smoother::new(&f, &centres);
        for v in s.apply(&vec![1.0; f.len()]) {
            assert!((v - 1.0).abs() < 1e-12);
        }
        // The rounded exponential's equivalent rectangular bandwidth is the ERB.
        for fc in centres {
            let p = 4.0 * fc / erb(fc);
            let area: f64 = f.iter().map(|x| {
                let g = (x - fc).abs() / fc;
                (1.0 + p * g) * (-p * g).exp() * DF
            }).sum();
            assert!((area / erb(fc) - 1.0).abs() < 0.01, "{fc}: {area}");
        }
    }

    #[test]
    fn a_comb_smoothed_by_the_ear_comes_back_to_its_energy() {
        // A reflection 20 ms late at half the amplitude: within an ERB at 4 kHz (456 Hz) the comb
        // averages to 1 + 0.25; a reflection 0.5 ms late (2 kHz period) keeps its ripple.
        let f = linear_grid();
        let s = Smoother::new(&f, &[4000.0]);
        for (tau, kept) in [(0.020, false), (0.0005, true)] {
            let p: Vec<f64> = f.iter().map(|x| (C64::new(1.0, 0.0) + C64::from_polar(0.5, -2.0 * PI * x * tau)).norm_sqr()).collect();
            let v = s.apply(&p)[0];
            assert_eq!((v / 1.25 - 1.0).abs() > 0.05, kept, "{tau}: {v}");
        }
    }

    #[test]
    fn interpolated_responses_follow_the_far_field() {
        // The drivers' responses on the study's grid, interpolated, against the far field
        // computed on the linear grid itself, toward the floor and ceiling bounces.
        let layout = crate::speaker::two_way();
        let study_f = crate::signal::study_grid();
        let lo = study_f.partition_point(|&x| x < 1000.0) - 1;
        let hi = study_f.partition_point(|&x| x <= 8000.0) + 1;
        let fs = &study_f[lo..hi];
        let f: Vec<f64> = (0..=1750).map(|i| 1000.0 + 4.0 * i as f64).collect();
        let tab = table(fs, &f);
        let dirs = [crate::sphere::from_angles(0.0, -33.0), crate::sphere::from_angles(20.0, 42.0)];
        let coarse = FarField::new(&layout, fs).responses(&dirs);
        let fine = FarField::new(&layout, &f).responses(&dirs);
        for (c, d) in coarse.iter().zip(&fine) {
            for i in 0..dirs.len() {
                for (k, &(j, t)) in tab.iter().enumerate() {
                    let a = c.row(i)[j] * (1.0 - t as f32) + c.row(i)[j + 1] * t as f32;
                    let b = d.row(i)[k];
                    assert!((a - b).norm() < 0.01 * b.norm().max(0.05), "{} Hz", f[k]);
                }
            }
        }
    }
}
