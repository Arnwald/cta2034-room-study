//! The computation: room kernels, loudspeaker responses, curves.
//!
//! For every setup the room is reduced to weights over directions ([`crate::room`]): the images
//! (direct sound and specular reflections up to order 3) and the diffuse field on a quadrature of
//! the sphere. For every loudspeaker variant (layout, orientation, crossover) the far-field power
//! is computed in all these directions, at the 70 directions of CTA-2034 and on the quadrature;
//! the energy at the listener is the weighted sum. In-room responses are given relative to an
//! omnidirectional source in the same place.

use rayon::prelude::*;

use crate::analysis::{anomaly, local_anomaly, notch_depth, ANOMALY_BAND};
use crate::crossover::{self, Kind, KINDS};
use crate::cta2034;
use crate::mat::Mat;
use crate::results::{Results, VariantResults};
use crate::room::{self, Setup};
use crate::speaker::{roll, FarField, Layout};
use crate::sphere;
use crate::{Vec3, C32, C64};

/// Categories of energy at the listener.
pub const CATEGORIES: [&str; 10] = [
    "direct",
    "first_left",
    "first_right",
    "first_listener_wall",
    "first_speaker_wall",
    "first_floor",
    "first_ceiling",
    "second",
    "third",
    "diffuse",
];

pub const N_GRID: usize = 3000;
pub const MAX_ORDER: usize = 3;
pub const ROLLS: [f64; 2] = [0.0, 90.0];
pub const MAP_FREQS: [f64; 5] = [500.0, 1000.0, 2200.0, 4000.0, 8000.0];

/// One setup reduced to weights: images (directions in the speaker frame, weights per frequency,
/// category, delay) and the diffuse field over the quadrature.
pub struct RoomKernel {
    pub directions: Vec<Vec3>,
    pub weights: Mat<f64>,
    pub category: Vec<u8>,
    pub delay: Vec<f64>,
    pub diffuse: Mat<f64>,
}

pub fn room_kernel(setup: &Setup, freqs: &[f64], grid: &[Vec3], solid_angle: f64) -> RoomKernel {
    let images = room::image_sources(setup, MAX_ORDER);
    let weights = room::specular_weights(setup, &images, freqs);
    let category = images
        .order
        .iter()
        .zip(&images.first)
        .map(|(&o, f)| match o {
            0 => 0u8,
            1 => 1 + f.unwrap() as u8,
            2 => 7,
            _ => 8,
        })
        .collect();
    let diffuse = room::diffuse_weights(setup, grid, solid_angle, freqs, MAX_ORDER);
    RoomKernel { directions: images.direction, weights, category, delay: images.delay, diffuse }
}

/// Energy at the listener per category, ``(n_categories, n_freqs)``; ``power_images`` the power
/// in the images' directions, ``power_grid`` on the quadrature (``None``: an omnidirectional
/// source).
pub fn category_sums(kernel: &RoomKernel, power_images: Option<&Mat<f32>>, power_grid: Option<&Mat<f32>>) -> Vec<Vec<f64>> {
    let nf = kernel.weights.cols;
    let mut out = vec![vec![0.0; nf]; CATEGORIES.len()];
    for i in 0..kernel.weights.rows {
        let c = kernel.category[i] as usize;
        let w = kernel.weights.row(i);
        match power_images {
            Some(p) => {
                let p = p.row(i);
                for j in 0..nf {
                    out[c][j] += w[j] * p[j] as f64;
                }
            }
            None => {
                for j in 0..nf {
                    out[c][j] += w[j];
                }
            }
        }
    }
    let last = CATEGORIES.len() - 1;
    for k in 0..kernel.diffuse.rows {
        let d = kernel.diffuse.row(k);
        match power_grid {
            Some(p) => {
                let p = p.row(k);
                for j in 0..nf {
                    out[last][j] += d[j] * p[j] as f64;
                }
            }
            None => {
                for j in 0..nf {
                    out[last][j] += d[j];
                }
            }
        }
    }
    out
}

/// ``|sum_i X_i H_i|^2`` for every direction.
pub fn power(filters: &[Vec<C64>], responses: &[Mat<C32>]) -> Mat<f32> {
    let (rows, cols) = (responses[0].rows, responses[0].cols);
    let fx: Vec<Vec<C32>> = filters.iter().map(|f| f.iter().map(|c| C32::new(c.re as f32, c.im as f32)).collect()).collect();
    let mut out = Mat::<f32>::zeros(rows, cols);
    out.data.par_chunks_mut(cols).enumerate().for_each(|(i, row)| {
        for j in 0..cols {
            let mut h = C32::new(0.0, 0.0);
            for (d, r) in responses.iter().enumerate() {
                h += fx[d][j] * r.row(i)[j];
            }
            row[j] = h.norm_sqr();
        }
    });
    out
}

/// A loudspeaker variant: a layout, rolled by ``roll_deg`` about its axis, with a crossover.
pub fn variant_name(layout: &Layout, roll_deg: f64, kind: Kind) -> String {
    format!("{}-{}-{}", layout.name, if roll_deg != 0.0 { "side" } else { "upright" }, kind.name())
}

fn band_mask(freqs: &[f64], lo: f64, hi: f64) -> Vec<bool> {
    freqs.iter().map(|&f| f >= lo && f <= hi).collect()
}

fn band_mean(values: &[f64], mask: &[bool]) -> f64 {
    let (s, n) = values.iter().zip(mask).filter(|(_, m)| **m).fold((0.0, 0), |(s, n), (v, _)| (s + v, n + 1));
    s / n as f64
}

/// Per-direction metrics of a power matrix: anomaly, local anomaly, notch depth, band power.
fn direction_metrics(freqs: &[f64], p: &Mat<f32>, band: &[bool]) -> [Vec<f32>; 4] {
    let rows: Vec<[f32; 4]> = (0..p.rows)
        .into_par_iter()
        .map(|i| {
            let row: Vec<f64> = p.row(i).iter().map(|v| (*v as f64).max(1e-12)).collect();
            [
                anomaly(freqs, &row) as f32,
                local_anomaly(freqs, &row) as f32,
                notch_depth(freqs, &row) as f32,
                band_mean(&p.row(i).iter().map(|v| *v as f64).collect::<Vec<_>>(), band) as f32,
            ]
        })
        .collect();
    [0, 1, 2, 3].map(|k| rows.iter().map(|r| r[k]).collect())
}

/// Everything the figures and tables need, for the setups and layouts given.
pub fn run(setups: &[Setup], n_random: usize, layouts: &[Layout], freqs: &[f64]) -> Results {
    let grid = sphere::fibonacci(N_GRID);
    let solid_angle = 4.0 * std::f64::consts::PI / N_GRID as f64;
    let nf = freqs.len();
    let band = band_mask(freqs, ANOMALY_BAND.0, ANOMALY_BAND.1);
    let map_idx: Vec<usize> = MAP_FREQS
        .iter()
        .map(|f| (0..nf).min_by(|&a, &b| (freqs[a] - f).abs().partial_cmp(&(freqs[b] - f).abs()).unwrap()).unwrap())
        .collect();
    let cta_dirs = cta2034::directions();

    // Far fields, filters, and the power at the 70 directions and on the quadrature.
    struct Orientation {
        layout: usize,
        roll: f64,
        filters: Vec<(Kind, Vec<Vec<C64>>)>,
    }
    let fars: Vec<FarField> = layouts.iter().map(|l| FarField::new(l, freqs)).collect();
    let mut orientations = Vec::new();
    let mut variants: Vec<VariantResults> = Vec::new();
    let mut grid_powers: Vec<Mat<f32>> = Vec::new();
    for (li, layout) in layouts.iter().enumerate() {
        let on = fars[li].on_axis();
        let filters: Vec<(Kind, Vec<Vec<C64>>)> = KINDS.iter().map(|&k| (k, crossover::filters(k, freqs, layout, &on))).collect();
        for &r in &ROLLS {
            let resp_cta = fars[li].responses(&roll(&cta_dirs, r));
            let resp_grid = fars[li].responses(&roll(&grid, r));
            for (kind, x) in &filters {
                let p_cta = power(x, &resp_cta);
                let p_grid = power(x, &resp_grid);
                let [anom_cta, local_cta, notch_cta, pband_cta] = direction_metrics(freqs, &p_cta, &band);
                let [anom_grid, local_grid, notch_grid, pband_grid] = direction_metrics(freqs, &p_grid, &band);
                variants.push(VariantResults {
                    name: variant_name(layout, r, *kind),
                    sp_full: p_grid.column_mean(),
                    cta: p_cta,
                    anom_cta,
                    anom_grid,
                    local_cta,
                    local_grid,
                    notch_cta,
                    notch_grid,
                    pband_cta,
                    pband_grid,
                    ..Default::default()
                });
                grid_powers.push(p_grid);
            }
            orientations.push(Orientation { layout: li, roll: r, filters: filters.clone() });
        }
    }
    eprintln!("far fields: {} variants", variants.len());

    // Room by room: kernels, energy per category, the images' own metrics.
    struct RoomOut {
        omni: Vec<Vec<f64>>,
        sums: Vec<Vec<Vec<f64>>>, // [variant][category][freq]
        dirs: Vec<Vec3>,
        cat: Vec<u8>,
        delay: Vec<f64>,
        map_w: Vec<Vec<f64>>,
        band_w: Vec<f64>,
        diffuse_band: Vec<f64>,
        img_metrics: Vec<[Vec<f32>; 4]>, // [variant]
    }
    let outs: Vec<RoomOut> = setups
        .par_iter()
        .map(|setup| {
            let k = room_kernel(setup, freqs, &grid, solid_angle);
            let omni = category_sums(&k, None, None);
            let mut sums = Vec::new();
            let mut img_metrics = Vec::new();
            let mut vi = 0;
            for o in &orientations {
                let dirs = roll(&k.directions, o.roll);
                let resp: Vec<Mat<C32>> = (0..layouts[o.layout].drivers.len())
                    .map(|d| {
                        let mut m = Mat::<C32>::zeros(dirs.len(), nf);
                        for (i, u) in dirs.iter().enumerate() {
                            fars[o.layout].response(d, u, m.row_mut(i));
                        }
                        m
                    })
                    .collect();
                for (_, x) in &o.filters {
                    let p_img = power_serial(x, &resp);
                    sums.push(category_sums(&k, Some(&p_img), Some(&grid_powers[vi])));
                    img_metrics.push(direction_metrics_serial(freqs, &p_img, &band));
                    vi += 1;
                }
            }
            let map_w = (0..k.weights.rows).map(|i| map_idx.iter().map(|&j| k.weights.row(i)[j]).collect()).collect();
            let band_w = (0..k.weights.rows).map(|i| band_mean(k.weights.row(i), &band)).collect();
            let diffuse_band = (0..k.diffuse.rows).map(|i| band_mean(k.diffuse.row(i), &band)).collect();
            RoomOut { omni, sums, dirs: k.directions, cat: k.category, delay: k.delay, map_w, band_w, diffuse_band, img_metrics }
        })
        .collect();
    eprintln!("rooms: {}", outs.len());

    // The diffuse field's mean share per direction (random rooms).
    let diffuse_mean = {
        let acc = setups[..n_random]
            .par_iter()
            .map(|setup| {
                let d = room::diffuse_weights(setup, &grid, solid_angle, freqs, MAX_ORDER);
                let k = room_kernel(setup, freqs, &grid, solid_angle);
                let omni = category_sums(&k, None, None);
                let total: Vec<f64> = (0..nf).map(|j| omni.iter().map(|c| c[j]).sum()).collect();
                let mut m = Mat::<f64>::zeros(d.rows, nf);
                for i in 0..d.rows {
                    for ((o, v), t) in m.row_mut(i).iter_mut().zip(d.row(i)).zip(&total) {
                        *o = v / t;
                    }
                }
                m
            })
            .reduce(
                || Mat::<f64>::zeros(N_GRID, nf),
                |mut a, b| {
                    a.data.iter_mut().zip(&b.data).for_each(|(x, y)| *x += y);
                    a
                },
            );
        Mat { rows: N_GRID, cols: nf, data: acc.data.iter().map(|v| (v / n_random as f64) as f32).collect() }
    };

    // Assemble.
    let n_rooms = setups.len();
    let n_cat = CATEGORIES.len();
    let mut omni = Mat::<f32>::zeros(n_rooms * n_cat, nf);
    for (r, o) in outs.iter().enumerate() {
        for c in 0..n_cat {
            for (dst, v) in omni.row_mut(r * n_cat + c).iter_mut().zip(&o.omni[c]) {
                *dst = *v as f32;
            }
        }
    }
    for (vi, v) in variants.iter_mut().enumerate() {
        let mut m = Mat::<f32>::zeros(n_rooms * n_cat, nf);
        for (r, o) in outs.iter().enumerate() {
            for c in 0..n_cat {
                for (dst, x) in m.row_mut(r * n_cat + c).iter_mut().zip(&o.sums[vi][c]) {
                    *dst = *x as f32;
                }
            }
        }
        v.rooms = m;
        for o in &outs {
            v.anom_img.extend(&o.img_metrics[vi][0]);
            v.local_img.extend(&o.img_metrics[vi][1]);
            v.notch_img.extend(&o.img_metrics[vi][2]);
            v.pband_img.extend(&o.img_metrics[vi][3]);
        }
    }
    let mut image_starts = vec![0usize];
    let mut image_dirs = Vec::new();
    let mut image_cat = Vec::new();
    let mut image_delay = Vec::new();
    let mut map_rows = Vec::new();
    let mut image_weights_band = Vec::new();
    for o in &outs {
        image_dirs.extend(o.dirs.iter().map(|d| [d[0] as f32, d[1] as f32, d[2] as f32]));
        image_cat.extend(&o.cat);
        image_delay.extend(o.delay.iter().map(|v| *v as f32));
        map_rows.extend(o.map_w.iter().map(|r| r.iter().map(|v| *v as f32).collect::<Vec<f32>>()));
        image_weights_band.extend(o.band_w.iter().map(|v| *v as f32));
        image_starts.push(image_starts.last().unwrap() + o.dirs.len());
    }
    let diffuse_band = Mat::from_rows(outs.iter().map(|o| o.diffuse_band.iter().map(|v| *v as f32).collect()).collect());
    let octaves = crate::constants::OCTAVES;
    Results {
        freqs: freqs.to_vec(),
        grid,
        categories: CATEGORIES.iter().map(|s| s.to_string()).collect(),
        labels: setups.iter().map(|s| s.label.clone()).collect(),
        keys: setups.iter().map(|s| s.key.clone()).collect(),
        n_random,
        omni,
        image_dirs,
        image_starts,
        image_cat,
        image_delay,
        map_freqs: MAP_FREQS.to_vec(),
        image_weights: Mat::from_rows(map_rows),
        image_weights_band,
        diffuse_band,
        diffuse_mean,
        meta: setups.iter().map(|s| s.meta.clone()).collect(),
        rt60: setups.iter().map(|s| {
            let v = s.room.rt60(&octaves);
            [v[0], v[1], v[2], v[3], v[4], v[5], v[6]]
        }).collect(),
        variants,
    }
}

fn power_serial(filters: &[Vec<C64>], responses: &[Mat<C32>]) -> Mat<f32> {
    let (rows, cols) = (responses[0].rows, responses[0].cols);
    let mut out = Mat::<f32>::zeros(rows, cols);
    for i in 0..rows {
        for (j, o) in out.row_mut(i).iter_mut().enumerate() {
            let mut h = C64::new(0.0, 0.0);
            for (x, r) in filters.iter().zip(responses) {
                let v = r.row(i)[j];
                h += x[j] * C64::new(v.re as f64, v.im as f64);
            }
            *o = h.norm_sqr() as f32;
        }
    }
    out
}

fn direction_metrics_serial(freqs: &[f64], p: &Mat<f32>, band: &[bool]) -> [Vec<f32>; 4] {
    let mut out: [Vec<f32>; 4] = Default::default();
    for i in 0..p.rows {
        let raw: Vec<f64> = p.row(i).iter().map(|v| *v as f64).collect();
        let row: Vec<f64> = raw.iter().map(|v| v.max(1e-12)).collect();
        out[0].push(anomaly(freqs, &row) as f32);
        out[1].push(local_anomaly(freqs, &row) as f32);
        out[2].push(notch_depth(freqs, &row) as f32);
        out[3].push(band_mean(&raw, band) as f32);
    }
    out
}
