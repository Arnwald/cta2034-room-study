//! Shoebox rooms: image sources for the early sound, a diffuse field for the rest.
//!
//! Room frame (metres): ``x`` across the room, toward the listener's right; ``y`` along it, from the
//! wall behind the listener (``y = 0``) to the wall behind the loudspeakers (``y = Ly``); ``z`` up.
//!
//! The sound reaching the listener is split three ways, in energy (paths added in power):
//!
//! 1. **Specular images** up to order ``N`` (3). The image ``u = 2 n L + (1 - 2 q) s`` has bounced
//!    ``|n - q|`` times on the wall at 0 and ``|n|`` times on the wall at ``L``. The loudspeaker
//!    emits that path in the direction ``P (r - u) / |r - u|``, ``P`` flipping the axes with an
//!    odd number of bounces. Weight: ``prod (1 - alpha)(1 - s) exp(-m |r - u|) / |r - u|^2``.
//! 2. **Diffuse field.** At each bounce of a ray launched in a direction ``Omega``, the scattered
//!    share ``(1 - alpha) s`` joins a diffuse field; after the ``N``-th bounce, all of what is
//!    left does: ``g(Omega)``. In steady state that field gives everywhere
//!    ``p^2 = 4 / A * integral g |H|^2 dOmega``, ``A = -S ln(1 - mean alpha) + 4 m V``.
//! 3. The **direct sound**, the image of order 0.
//!
//! Absorbing patches replace a surface's coefficients where a ray or a first-order reflection lands
//! on them; higher-order images and the diffuse field use each surface's area-weighted mean.
//! Furniture is an equivalent absorption area ``A_f`` spread over the volume, an attenuation
//! ``m_f = A_f / 4V`` per metre applied to images of order 2 and more, to the rays and to the
//! diffuse field.

use crate::constants::{AIR_HUMIDITY, AIR_PRESSURE, AIR_TEMPERATURE, C, OCTAVES};
use crate::mat::Mat;
use crate::signal::interp_log;
use crate::{dot, norm, sub, Vec3};

pub const SURFACES: [&str; 6] = ["left", "right", "listener_wall", "speaker_wall", "floor", "ceiling"];

/// (axis, side) of each surface: side 0 is the plane at 0, side 1 the plane at L. The index of a
/// surface is ``2 axis + side``.
pub fn plane(surface: usize) -> (usize, usize) {
    (surface / 2, surface % 2)
}

pub fn surface_index(name: &str) -> usize {
    SURFACES.iter().position(|s| *s == name).expect("unknown surface")
}

/// Octave-band coefficients interpolated in log frequency, ends held.
pub fn bands(values: &[f64; 7], freqs: &[f64]) -> Vec<f64> {
    interp_log(&OCTAVES, values, freqs)
}

/// Energy attenuation of air per metre, ``m`` (1/m), ISO 9613-1.
pub fn air_attenuation(freqs: &[f64]) -> Vec<f64> {
    let t = AIR_TEMPERATURE + 273.15;
    let (t0, t01) = (293.15, 273.16);
    let pa = AIR_PRESSURE / 101.325;
    let c_sat = -6.8346 * (t01 / t).powf(1.261) + 4.6151;
    let h = AIR_HUMIDITY * 10f64.powf(c_sat) / pa;
    let fr_o = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    let fr_n = pa * (t / t0).powf(-0.5) * (9.0 + 280.0 * h * (-4.170 * ((t / t0).powf(-1.0 / 3.0) - 1.0)).exp());
    freqs
        .iter()
        .map(|&f| {
            let alpha_db = 8.686
                * f
                * f
                * (1.84e-11 / pa * (t / t0).sqrt()
                    + (t / t0).powf(-2.5)
                        * (0.01275 * (-2239.1 / t).exp() / (fr_o + f * f / fr_o)
                            + 0.1068 * (-3352.0 / t).exp() / (fr_n + f * f / fr_n)));
            alpha_db / (10.0 * std::f64::consts::E.log10())
        })
        .collect()
}

/// A rectangle of another material on one surface: ``rect = (u0, u1, v0, v1)`` in the surface's
/// two in-plane coordinates, in increasing axis order (floor: x, y; side walls: y, z; end walls:
/// x, z).
#[derive(Clone, Debug)]
pub struct Patch {
    pub surface: usize,
    pub rect: [f64; 4],
    pub absorption: [f64; 7],
    pub scattering: [f64; 7],
    pub label: String,
}

impl Patch {
    pub fn area(&self) -> f64 {
        (self.rect[1] - self.rect[0]) * (self.rect[3] - self.rect[2])
    }
}

#[derive(Clone, Debug)]
pub struct Room {
    pub dims: Vec3,
    pub absorption: [[f64; 7]; 6],
    pub scattering: [[f64; 7]; 6],
    pub patches: Vec<Patch>,
    /// Equivalent absorption area of the furniture (m^2), octave bands.
    pub fittings: [f64; 7],
    pub label: String,
}

impl Room {
    pub fn volume(&self) -> f64 {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    pub fn area(&self, surface: usize) -> f64 {
        let (axis, _) = plane(surface);
        let o: Vec<f64> = (0..3).filter(|&i| i != axis).map(|i| self.dims[i]).collect();
        o[0] * o[1]
    }

    pub fn total_area(&self) -> f64 {
        (0..6).map(|s| self.area(s)).sum()
    }

    /// Area-weighted absorption and scattering of a surface, patches included.
    pub fn mean_coefficients(&self, surface: usize, freqs: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let a = self.area(surface);
        let base_a = bands(&self.absorption[surface], freqs);
        let base_s = bands(&self.scattering[surface], freqs);
        let mut alpha: Vec<f64> = base_a.iter().map(|v| v * a).collect();
        let mut scat: Vec<f64> = base_s.iter().map(|v| v * a).collect();
        for p in self.patches.iter().filter(|p| p.surface == surface) {
            let pa = bands(&p.absorption, freqs);
            let ps = bands(&p.scattering, freqs);
            for i in 0..freqs.len() {
                alpha[i] += (pa[i] - base_a[i]) * p.area();
                scat[i] += (ps[i] - base_s[i]) * p.area();
            }
        }
        (alpha.iter().map(|v| v / a).collect(), scat.iter().map(|v| v / a).collect())
    }

    pub fn furniture_attenuation(&self, freqs: &[f64]) -> Vec<f64> {
        bands(&self.fittings, freqs).iter().map(|v| v / (4.0 * self.volume())).collect()
    }

    pub fn mean_absorption(&self, freqs: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; freqs.len()];
        for s in 0..6 {
            let (a, _) = self.mean_coefficients(s, freqs);
            let area = self.area(s);
            for (o, v) in out.iter_mut().zip(a) {
                *o += v * area;
            }
        }
        let total = self.total_area();
        out.iter().map(|v| v / total).collect()
    }

    /// Eyring equivalent absorption area with air and furniture, ``A`` (m^2).
    pub fn absorption_area(&self, freqs: &[f64]) -> Vec<f64> {
        let air = air_attenuation(freqs);
        let fit = self.furniture_attenuation(freqs);
        let mean = self.mean_absorption(freqs);
        (0..freqs.len())
            .map(|i| -self.total_area() * (1.0 - mean[i]).ln() + 4.0 * (air[i] + fit[i]) * self.volume())
            .collect()
    }

    pub fn rt60(&self, freqs: &[f64]) -> Vec<f64> {
        self.absorption_area(freqs).iter().map(|a| 24.0 * 10f64.ln() / C * self.volume() / a).collect()
    }

    /// Absorption and scattering of every material at every frequency: the six surfaces, then the
    /// patches (index 6 + i).
    pub fn material_tables(&self, freqs: &[f64]) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut a: Vec<Vec<f64>> = (0..6).map(|s| bands(&self.absorption[s], freqs)).collect();
        let mut s: Vec<Vec<f64>> = (0..6).map(|i| bands(&self.scattering[i], freqs)).collect();
        for p in &self.patches {
            a.push(bands(&p.absorption, freqs));
            s.push(bands(&p.scattering, freqs));
        }
        (a, s)
    }

    /// Material index of a hit point on a surface.
    pub fn material_at(&self, surface: usize, point: &Vec3) -> usize {
        let mut out = surface;
        for (i, p) in self.patches.iter().enumerate() {
            if p.surface != surface {
                continue;
            }
            let (axis, _) = plane(surface);
            let uv: Vec<usize> = (0..3).filter(|&j| j != axis).collect();
            let [u0, u1, v0, v1] = p.rect;
            if point[uv[0]] >= u0 && point[uv[0]] <= u1 && point[uv[1]] >= v0 && point[uv[1]] <= v1 {
                out = 6 + i;
            }
        }
        out
    }
}

/// A loudspeaker (its reference point ``source``, its axis turned by ``yaw`` toward +x) and a
/// listener's head (``receiver``) in a room.
#[derive(Clone, Debug)]
pub struct Setup {
    pub room: Room,
    pub source: Vec3,
    pub receiver: Vec3,
    pub yaw: f64,
    pub label: String,
    pub key: Option<String>,
    pub meta: Meta,
}

/// What the setup was drawn from.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct Meta {
    pub dims: Vec<f64>,
    pub wall: String,
    pub ceiling: String,
    pub floor: String,
    pub rug: bool,
    pub panels: bool,
    pub cloud: bool,
    pub furniture: f64,
    pub spacing: f64,
    pub wall_distance: f64,
    pub listen_distance: f64,
    pub speaker_height: f64,
    pub ear_height: f64,
    pub toe: String,
}

impl Setup {
    /// Rows: the speaker frame's x (right as seen from the listener), y (up), z (forward), in room
    /// coordinates.
    pub fn axes(&self) -> [Vec3; 3] {
        let (s, c) = (self.yaw.sin(), self.yaw.cos());
        [[c, s, 0.0], [0.0, 0.0, 1.0], [s, -c, 0.0]]
    }

    pub fn to_speaker(&self, d: &Vec3) -> Vec3 {
        let a = self.axes();
        [dot(d, &a[0]), dot(d, &a[1]), dot(d, &a[2])]
    }

    pub fn to_room(&self, d: &Vec3) -> Vec3 {
        let a = self.axes();
        [
            d[0] * a[0][0] + d[1] * a[1][0] + d[2] * a[2][0],
            d[0] * a[0][1] + d[1] * a[1][1] + d[2] * a[2][1],
            d[0] * a[0][2] + d[1] * a[1][2] + d[2] * a[2][2],
        ]
    }
}

/// Yaw that points the loudspeaker's axis at the receiver (in the horizontal plane).
pub fn aim(source: &Vec3, receiver: &Vec3) -> f64 {
    let d = sub(receiver, source);
    d[0].atan2(-d[1])
}

/// The images of a setup, sorted by distance.
#[derive(Clone, Debug, Default)]
pub struct Images {
    pub order: Vec<usize>,
    pub counts: Vec<[usize; 6]>,
    pub distance: Vec<f64>,
    /// Emission direction in the speaker frame.
    pub direction: Vec<Vec3>,
    /// Delay after the direct sound (s).
    pub delay: Vec<f64>,
    /// First-order images: the surface, the material and the reflection point.
    pub first: Vec<Option<usize>>,
    pub material: Vec<Option<usize>>,
    pub point: Vec<Option<Vec3>>,
}

pub fn image_sources(setup: &Setup, max_order: usize) -> Images {
    let l = setup.room.dims;
    let s = setup.source;
    let r = setup.receiver;
    let m = max_order as i64;
    let mut rows = Vec::new(); // (n[3], q[3], counts[6], order)
    for nx in -m..=m {
        for qx in 0..2i64 {
            let cx = [(nx - qx).unsigned_abs() as usize, nx.unsigned_abs() as usize];
            if cx[0] + cx[1] > max_order {
                continue;
            }
            for ny in -m..=m {
                for qy in 0..2i64 {
                    let cy = [(ny - qy).unsigned_abs() as usize, ny.unsigned_abs() as usize];
                    if cx[0] + cx[1] + cy[0] + cy[1] > max_order {
                        continue;
                    }
                    for nz in -m..=m {
                        for qz in 0..2i64 {
                            let cz = [(nz - qz).unsigned_abs() as usize, nz.unsigned_abs() as usize];
                            let order = cx[0] + cx[1] + cy[0] + cy[1] + cz[0] + cz[1];
                            if order > max_order {
                                continue;
                            }
                            rows.push(([nx, ny, nz], [qx, qy, qz], [cx[0], cx[1], cy[0], cy[1], cz[0], cz[1]], order));
                        }
                    }
                }
            }
        }
    }
    let direct = norm(&sub(&r, &s));
    let mut items: Vec<(f64, usize)> = Vec::new();
    let mut out = Images::default();
    let mut tmp = Vec::new();
    for (n, q, counts, order) in rows {
        let u: Vec3 = [0, 1, 2].map(|i| 2.0 * n[i] as f64 * l[i] + (1 - 2 * q[i]) as f64 * s[i]);
        let v = sub(&r, &u);
        let dist = norm(&v);
        let d_room: Vec3 = [0, 1, 2].map(|i| (1 - 2 * (q[i] % 2)) as f64 * v[i] / dist);
        let (mut first, mut material, mut point) = (None, None, None);
        if order == 1 {
            let k = counts.iter().position(|&c| c > 0).unwrap();
            let (axis, side) = plane(k);
            let wall = side as f64 * l[axis];
            let t = (wall - u[axis]) / (r[axis] - u[axis]);
            let p: Vec3 = [0, 1, 2].map(|i| u[i] + t * (r[i] - u[i]));
            first = Some(k);
            material = Some(setup.room.material_at(k, &p));
            point = Some(p);
        }
        items.push((dist, tmp.len()));
        tmp.push((order, counts, dist, setup.to_speaker(&d_room), (dist - direct) / C, first, material, point));
    }
    items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (_, i) in items {
        let (order, counts, dist, dir, delay, first, material, point) = tmp[i];
        out.order.push(order);
        out.counts.push(counts);
        out.distance.push(dist);
        out.direction.push(dir);
        out.delay.push(delay);
        out.first.push(first);
        out.material.push(material);
        out.point.push(point);
    }
    out
}

/// Energy weight of every image at every frequency, ``(n_images, n_freqs)``.
pub fn specular_weights(setup: &Setup, images: &Images, freqs: &[f64]) -> Mat<f64> {
    let room = &setup.room;
    let nf = freqs.len();
    let (alpha_tab, scat_tab) = room.material_tables(freqs);
    let keep: Vec<Vec<f64>> = (0..6)
        .map(|s| {
            let (a, sc) = room.mean_coefficients(s, freqs);
            a.iter().zip(&sc).map(|(a, s)| (1.0 - a) * (1.0 - s)).collect()
        })
        .collect();
    let m_air = air_attenuation(freqs);
    let m_fit = room.furniture_attenuation(freqs);
    let mut w = Mat::<f64>::zeros(images.order.len(), nf);
    for i in 0..images.order.len() {
        let d = images.distance[i];
        let o = images.order[i];
        let row = w.row_mut(i);
        for j in 0..nf {
            let mut v = (-d * m_air[j]).exp() / (d * d);
            if o >= 2 {
                v *= (-d * m_fit[j]).exp();
            }
            if o == 1 {
                let mat = images.material[i].unwrap();
                v *= (1.0 - alpha_tab[mat][j]) * (1.0 - scat_tab[mat][j]);
            } else if o >= 2 {
                for (k, &c) in images.counts[i].iter().enumerate() {
                    if c > 0 {
                        v *= keep[k][j].powi(c as i32);
                    }
                }
            }
            row[j] = v;
        }
    }
    w
}

/// Rays from the source in the given directions (speaker frame): for each bounce, the material hit
/// and the path length from the source, ``[bounce][ray]``.
pub fn trace(setup: &Setup, directions: &[Vec3], bounces: usize) -> (Vec<Vec<usize>>, Vec<Vec<f64>>) {
    let l = setup.room.dims;
    let mut materials = vec![vec![0usize; directions.len()]; bounces];
    let mut lengths = vec![vec![0.0; directions.len()]; bounces];
    for (ri, u) in directions.iter().enumerate() {
        let mut d = setup.to_room(u);
        let mut p = setup.source;
        let mut length = 0.0;
        for b in 0..bounces {
            let mut tmin = f64::INFINITY;
            let mut axis = 0;
            for a in 0..3 {
                if d[a] != 0.0 {
                    let target = if d[a] > 0.0 { l[a] } else { 0.0 };
                    let t = ((target - p[a]) / d[a]).max(0.0);
                    if t < tmin {
                        tmin = t;
                        axis = a;
                    }
                }
            }
            p = [p[0] + tmin * d[0], p[1] + tmin * d[1], p[2] + tmin * d[2]];
            length += tmin;
            let side = if d[axis] > 0.0 { 1 } else { 0 };
            materials[b][ri] = setup.room.material_at(2 * axis + side, &p);
            lengths[b][ri] = length;
            d[axis] = -d[axis];
        }
    }
    (materials, lengths)
}

/// ``g(Omega, f)``: the share of the energy emitted in each direction that joins the diffuse
/// field, ``(n_directions, n_freqs)``.
pub fn diffuse_share(setup: &Setup, directions: &[Vec3], freqs: &[f64], bounces: usize) -> Mat<f64> {
    let room = &setup.room;
    let nf = freqs.len();
    let (alpha_tab, scat_tab) = room.material_tables(freqs);
    let air = air_attenuation(freqs);
    let fit = room.furniture_attenuation(freqs);
    let m: Vec<f64> = air.iter().zip(&fit).map(|(a, b)| a + b).collect();
    let (materials, lengths) = trace(setup, directions, bounces);
    let mut g = Mat::<f64>::zeros(directions.len(), nf);
    for ri in 0..directions.len() {
        let row = g.row_mut(ri);
        let mut carried = vec![1.0; nf];
        for b in 0..bounces {
            let mat = materials[b][ri];
            let seg = lengths[b][ri] - if b > 0 { lengths[b - 1][ri] } else { 0.0 };
            for j in 0..nf {
                let arriving = carried[j] * (-seg * m[j]).exp();
                let a = alpha_tab[mat][j];
                let sc = scat_tab[mat][j];
                if b < bounces - 1 {
                    row[j] += arriving * (1.0 - a) * sc;
                    carried[j] = arriving * (1.0 - a) * (1.0 - sc);
                } else {
                    row[j] += arriving * (1.0 - a);
                }
            }
        }
    }
    g
}

/// Weights of the diffuse field over a quadrature of the sphere: ``4 / A * dOmega * g``.
pub fn diffuse_weights(setup: &Setup, directions: &[Vec3], solid_angle: f64, freqs: &[f64], bounces: usize) -> Mat<f64> {
    let mut g = diffuse_share(setup, directions, freqs, bounces);
    let a = setup.room.absorption_area(freqs);
    for ri in 0..g.rows {
        for (v, aj) in g.row_mut(ri).iter_mut().zip(&a) {
            *v *= 4.0 / aj * solid_angle;
        }
    }
    g
}
