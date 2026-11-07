//! The computation: room kernels, loudspeaker responses, curves.
//!
//! For every setup the room is reduced to weights over directions ([`crate::room`]): the images
//! (direct sound and specular reflections up to order 3) and the diffuse field on a quadrature of
//! the sphere. For every loudspeaker variant (layout, orientation, crossover) the far-field power
//! is computed in all these directions, at the 70 directions of CTA-2034 and on the quadrature;
//! the energy at the listener is the weighted sum. In-room responses are given relative to an
//! omnidirectional source in the same place.

use rayon::prelude::*;

use crate::analysis::{anomaly, local_anomaly, notch_depth, roughness, ANOMALY_BAND};
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
    pub images: room::Images,
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
    RoomKernel { directions: images.direction.clone(), delay: images.delay.clone(), images, weights, category, diffuse }
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

/// Per-direction metrics of a power matrix: anomaly, local anomaly, notch depth, band power,
/// roughness.
fn direction_metrics(freqs: &[f64], p: &Mat<f32>, band: &[bool]) -> [Vec<f32>; 5] {
    let rows: Vec<[f32; 5]> = (0..p.rows)
        .into_par_iter()
        .map(|i| {
            let row: Vec<f64> = p.row(i).iter().map(|v| (*v as f64).max(1e-12)).collect();
            [
                anomaly(freqs, &row) as f32,
                local_anomaly(freqs, &row) as f32,
                notch_depth(freqs, &row) as f32,
                band_mean(&p.row(i).iter().map(|v| *v as f64).collect::<Vec<_>>(), band) as f32,
                roughness(freqs, &row) as f32,
            ]
        })
        .collect();
    [0, 1, 2, 3, 4].map(|k| rows.iter().map(|r| r[k]).collect())
}

/// How the variants are computed.
#[derive(Clone, Debug)]
pub struct Options {
    /// Orientations: rolls about the axis (degrees).
    pub rolls: Vec<f64>,
    /// Take every path from each driver's own position (its own direction, length and
    /// reflection), the drivers adding with their phases along it; otherwise every path leaves
    /// the reference point in the far field. Upright only.
    pub per_driver: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { rolls: ROLLS.to_vec(), per_driver: false }
    }
}

/// Everything the figures and tables need, for the setups and layouts given.
pub fn run(setups: &[Setup], n_random: usize, layouts: &[Layout], freqs: &[f64]) -> Results {
    run_with(setups, n_random, layouts, freqs, &Options::default())
}

/// The responses of a layout's drivers along every path of a room, each path taken from the
/// driver's own position: ``H_d(u_d) exp(-j k (u_d . c_d + r_d - r_0)) sqrt(w_d / w_0)``, ``u_d``,
/// ``r_d`` and ``w_d`` the direction, length and energy weight of the path from driver ``d``,
/// ``r_0`` and ``w_0`` those from the reference point (whose weights the room kernel holds).
pub fn driver_paths(setup: &Setup, kernel: &RoomKernel, far: &FarField, freqs: &[f64]) -> Vec<Mat<C32>> {
    let axes = setup.axes();
    let reference = &kernel.images;
    let n_img = reference.order.len();
    let nf = freqs.len();
    far.layout
        .drivers
        .iter()
        .enumerate()
        .map(|(d, drv)| {
            let c = drv.centre();
            let mut st = setup.clone();
            st.source = [0, 1, 2].map(|i| setup.source[i] + c[0] * axes[0][i] + c[1] * axes[1][i] + c[2] * axes[2][i]);
            let im = room::image_sources(&st, MAX_ORDER);
            let w = room::specular_weights(&st, &im, freqs);
            let index: std::collections::HashMap<[i8; 6], usize> = im.ident.iter().enumerate().map(|(i, id)| (*id, i)).collect();
            let mut m = Mat::<C32>::zeros(n_img, nf);
            let mut row = vec![C32::new(0.0, 0.0); nf];
            for i in 0..n_img {
                let j = index[&reference.ident[i]];
                let u = im.direction[j];
                far.response(d, &u, &mut row);
                let shift = crate::dot(&u, &c) + im.distance[j] - reference.distance[i];
                for (q, (out, h)) in m.row_mut(i).iter_mut().zip(&row).enumerate() {
                    let amp = (w.row(j)[q] / kernel.weights.row(i)[q]).sqrt();
                    let v = C64::new(h.re as f64, h.im as f64) * C64::from_polar(amp, -far.k[q] * shift);
                    *out = C32::new(v.re as f32, v.im as f32);
                }
            }
            m
        })
        .collect()
}

/// Sound power over the sphere rebuilt from the two orbits driver by driver: each driver's
/// complex response, relative to its own centre, interpolated over the sphere between the four
/// half-orbits (linearly in the angle from the axis, by ``cos^2`` and ``sin^2`` of the angle
/// around it), then the drivers added with their positions. ``responses``: the drivers at the 70
/// directions.
pub fn sp_from_ways(filters: &[Vec<C64>], responses: &[Mat<C32>], far: &FarField, grid: &[Vec3]) -> Vec<f64> {
    let dirs = cta2034::directions();
    let nf = far.k.len();
    let centres: Vec<Vec3> = far.layout.drivers.iter().map(|d| d.centre()).collect();
    // Each driver relative to its own centre, filtered, at the 70 directions.
    let g: Vec<Vec<Vec<C64>>> = (0..centres.len())
        .map(|d| {
            (0..dirs.len())
                .map(|i| {
                    let us = crate::dot(&dirs[i], &centres[d]);
                    (0..nf)
                        .map(|j| {
                            let h = responses[d].row(i)[j];
                            filters[d][j] * C64::new(h.re as f64, h.im as f64) * C64::from_polar(1.0, -far.k[j] * us)
                        })
                        .collect()
                })
                .collect()
        })
        .collect();
    // Four orbit samples and their weights for every direction of the grid.
    let corners: Vec<[(usize, f64); 4]> = grid
        .iter()
        .map(|u| {
            let alpha = u[2].clamp(-1.0, 1.0).acos().to_degrees();
            let psi = u[1].atan2(u[0]).to_degrees().rem_euclid(360.0);
            let q = ((psi / 90.0).floor() as usize).min(3);
            let t = (psi - 90.0 * q as f64).to_radians();
            let a0 = ((alpha / 10.0).floor() * 10.0).min(170.0);
            let s = (alpha - a0) / 10.0;
            // Half-orbits in order around the axis: right, up, left, down.
            let half = |h: usize, a: f64| -> usize {
                let a = a.round() as i32;
                match h % 4 {
                    0 => cta2034::index_of('h', a),
                    1 => cta2034::index_of('v', a),
                    2 => cta2034::index_of('h', -a),
                    _ => cta2034::index_of('v', -a),
                }
            };
            let (wa, wb) = (t.cos().powi(2), t.sin().powi(2));
            [
                (half(q, a0), wa * (1.0 - s)),
                (half(q, a0 + 10.0), wa * s),
                (half(q + 1, a0), wb * (1.0 - s)),
                (half(q + 1, a0 + 10.0), wb * s),
            ]
        })
        .collect();
    let total: Vec<f64> = grid
        .par_iter()
        .zip(corners.par_iter())
        .map(|(u, c)| {
            let mut p = vec![0.0; nf];
            for (j, pj) in p.iter_mut().enumerate() {
                let mut h = C64::new(0.0, 0.0);
                for (d, gd) in g.iter().enumerate() {
                    let mut v = C64::new(0.0, 0.0);
                    for (i, w) in c {
                        v += gd[*i][j] * *w;
                    }
                    h += v * C64::from_polar(1.0, far.k[j] * crate::dot(u, &centres[d]));
                }
                *pj = h.norm_sqr();
            }
            p
        })
        .reduce(|| vec![0.0; nf], |a, b| a.iter().zip(&b).map(|(x, y)| x + y).collect());
    total.iter().map(|v| v / grid.len() as f64).collect()
}

/// Everything the figures and tables need, for the setups and layouts given, computed as
/// ``opts`` says.
pub fn run_with(setups: &[Setup], n_random: usize, layouts: &[Layout], freqs: &[f64], opts: &Options) -> Results {
    assert!(!opts.per_driver || opts.rolls.iter().all(|r| *r == 0.0), "per-driver paths: upright only");
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
        for &r in &opts.rolls {
            let resp_cta = fars[li].responses(&roll(&cta_dirs, r));
            let resp_grid = fars[li].responses(&roll(&grid, r));
            for (kind, x) in &filters {
                let p_cta = power(x, &resp_cta);
                let p_grid = power(x, &resp_grid);
                let [anom_cta, local_cta, notch_cta, pband_cta, rough_cta] = direction_metrics(freqs, &p_cta, &band);
                let [anom_grid, local_grid, notch_grid, pband_grid, rough_grid] = direction_metrics(freqs, &p_grid, &band);
                variants.push(VariantResults {
                    name: variant_name(layout, r, *kind),
                    sp_full: p_grid.column_mean(),
                    sp_ways: if r == 0.0 { sp_from_ways(x, &resp_cta, &fars[li], &grid) } else { Vec::new() },
                    cta: p_cta,
                    anom_cta,
                    anom_grid,
                    local_cta,
                    local_grid,
                    notch_cta,
                    notch_grid,
                    pband_cta,
                    pband_grid,
                    rough_cta,
                    rough_grid,
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
        img_metrics: Vec<[Vec<f32>; 5]>, // [variant]
        far: Vec<[Vec<f64>; 2]>,         // [variant]: steady state, early sound (per-driver runs)
    }
    let outs: Vec<RoomOut> = setups
        .par_iter()
        .map(|setup| {
            let k = room_kernel(setup, freqs, &grid, solid_angle);
            let omni = category_sums(&k, None, None);
            let mut sums = Vec::new();
            let mut img_metrics = Vec::new();
            let mut far = Vec::new();
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
                let near = opts.per_driver.then(|| driver_paths(setup, &k, &fars[o.layout], freqs));
                for (_, x) in &o.filters {
                    let p_img = power_serial(x, near.as_ref().unwrap_or(&resp));
                    sums.push(category_sums(&k, Some(&p_img), Some(&grid_powers[vi])));
                    img_metrics.push(direction_metrics_serial(freqs, &p_img, &band));
                    if opts.per_driver {
                        let c = category_sums(&k, Some(&power_serial(x, &resp)), Some(&grid_powers[vi]));
                        let total = |upto: usize| (0..nf).map(|j| c[..upto].iter().map(|v| v[j]).sum()).collect::<Vec<f64>>();
                        far.push([total(CATEGORIES.len()), total(crate::analysis::EARLY)]);
                    }
                    vi += 1;
                }
            }
            let map_w = (0..k.weights.rows).map(|i| map_idx.iter().map(|&j| k.weights.row(i)[j]).collect()).collect();
            let band_w = (0..k.weights.rows).map(|i| band_mean(k.weights.row(i), &band)).collect();
            let diffuse_band = (0..k.diffuse.rows).map(|i| band_mean(k.diffuse.row(i), &band)).collect();
            RoomOut { omni, sums, dirs: k.directions, cat: k.category, delay: k.delay, map_w, band_w, diffuse_band, img_metrics, far }
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
            v.rough_img.extend(&o.img_metrics[vi][4]);
        }
        if opts.per_driver {
            let mut m = Mat::<f32>::zeros(n_rooms * 2, nf);
            for (r, o) in outs.iter().enumerate() {
                for kind in 0..2 {
                    for (dst, x) in m.row_mut(r * 2 + kind).iter_mut().zip(&o.far[vi][kind]) {
                        *dst = *x as f32;
                    }
                }
            }
            v.rooms_far = m;
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

fn direction_metrics_serial(freqs: &[f64], p: &Mat<f32>, band: &[bool]) -> [Vec<f32>; 5] {
    let mut out: [Vec<f32>; 5] = Default::default();
    for i in 0..p.rows {
        let raw: Vec<f64> = p.row(i).iter().map(|v| *v as f64).collect();
        let row: Vec<f64> = raw.iter().map(|v| v.max(1e-12)).collect();
        out[0].push(anomaly(freqs, &row) as f32);
        out[1].push(local_anomaly(freqs, &row) as f32);
        out[2].push(notch_depth(freqs, &row) as f32);
        out[3].push(band_mean(&raw, band) as f32);
        out[4].push(roughness(freqs, &row) as f32);
    }
    out
}
