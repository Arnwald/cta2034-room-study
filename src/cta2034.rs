//! The spatial averages of CTA-2034-A, and the weights they give to each direction.
//!
//! Two orbits of 10-degree steps around the reference axis: horizontal, angle ``h`` positive
//! toward +x, ``u = (sin h, 0, cos h)``; vertical, angle ``v`` positive upward,
//! ``u = (0, sin v, cos v)``. Every curve is a weighted mean of the power responses at the 70
//! directions, with weights summing to 1 ([`kernel`]):
//!
//! * Listening Window: 0, +-10 degrees vertical, +-10, +-20, +-30 degrees horizontal;
//! * Early Reflections: the mean of five bounces, each the mean of its directions: floor (-20,
//!   -30, -40 vertical), ceiling (40, 50, 60 vertical), front wall (0, +-10, +-20, +-30
//!   horizontal), side walls (+-40 ... +-80 horizontal), rear wall (+-90, 180 horizontal);
//! * Sound Power: all 70 directions, weighted by the solid angle each stands for (the bands
//!   between "spherical squares", Appendix C of the standard);
//! * Predicted In-Room Response: ``0.12 LW + 0.44 ER + 0.44 SP``.

use std::sync::OnceLock;

use crate::mat::Mat;
use crate::Vec3;

pub const LISTENING_WINDOW_H: [i32; 7] = [0, 10, -10, 20, -20, 30, -30];
pub const LISTENING_WINDOW_V: [i32; 2] = [10, -10];
pub const FRONT_WALL: [i32; 7] = [0, 10, -10, 20, -20, 30, -30];
pub const SIDE_WALL: [i32; 10] = [40, -40, 50, -50, 60, -60, 70, -70, 80, -80];
pub const REAR_WALL: [i32; 3] = [90, -90, 180];
pub const FLOOR: [i32; 3] = [-20, -30, -40];
pub const CEILING: [i32; 3] = [40, 50, 60];
pub const BOUNCES: [&str; 5] = ["floor", "ceiling", "front", "side", "rear"];
pub const PIR_WEIGHTS: [f64; 3] = [0.12, 0.44, 0.44];

/// A direction of the orbits: ``('h', angle)`` or ``('v', angle)``.
pub type Key = (char, i32);

fn square_solid_angle(alpha_deg: f64) -> f64 {
    let a = alpha_deg.to_radians();
    let gamma = (a.cos() * a.cos()).acos();
    let corner_a = (a.sin() / a.tan()).atan();
    let corner = (-(corner_a.cos().powi(2)) + corner_a.sin().powi(2) * gamma.cos()).acos();
    4.0 * corner - 2.0 * std::f64::consts::PI
}

/// Weight of every angle 0 ... 180 of the 10-degree orbit (index ``angle / 10``).
pub fn sound_power_weights() -> [f64; 19] {
    let edges: Vec<f64> = (0..9).map(|i| i as f64 * 10.0 + 5.0).chain(std::iter::once(90.0)).collect();
    let areas: Vec<f64> = edges.iter().map(|&e| square_solid_angle(e)).collect();
    let mut w = vec![areas[0]];
    for i in 1..10 {
        w.push(areas[i] - areas[i - 1]);
    }
    w[9] *= 2.0;
    let mut out = [0.0; 19];
    for (i, weight) in w.iter().enumerate() {
        out[i] = *weight;
        out[18 - i] = *weight;
    }
    out
}

/// The 70 directions: the whole horizontal orbit, and the vertical one without 0 and 180 degrees.
pub fn keys() -> &'static [Key] {
    static K: OnceLock<Vec<Key>> = OnceLock::new();
    K.get_or_init(|| {
        let mut k: Vec<Key> = (-17..=18).map(|i| ('h', i * 10)).collect();
        k.extend((-17..=18).map(|i| i * 10).filter(|&a| a != 0 && a != 180).map(|a| ('v', a)));
        k
    })
}

pub fn direction(plane: char, angle: f64) -> Vec3 {
    let a = angle.to_radians();
    if plane == 'h' {
        [a.sin(), 0.0, a.cos()]
    } else {
        [0.0, a.sin(), a.cos()]
    }
}

pub fn directions() -> Vec<Vec3> {
    keys().iter().map(|&(p, a)| direction(p, a as f64)).collect()
}

fn normalize_key(plane: char, angle: i32) -> Key {
    let mut a = angle.rem_euclid(360);
    if a > 180 {
        a -= 360;
    }
    if a == 0 || a == 180 {
        ('h', a)
    } else {
        (plane, a)
    }
}

pub fn index_of(plane: char, angle: i32) -> usize {
    let k = normalize_key(plane, angle);
    keys().iter().position(|&x| x == k).expect("not an orbit direction")
}

/// Weights of the 70 directions of a curve: ``lw``, ``er``, ``sp``, ``pir``, or one bounce of the
/// early reflections (``floor``, ``ceiling``, ``front``, ``side``, ``rear``).
pub fn kernel(curve: &str) -> Vec<f64> {
    let n = keys().len();
    let mut w = vec![0.0; n];
    let add = |w: &mut Vec<f64>, plane: char, angles: &[i32]| {
        for &a in angles {
            w[index_of(plane, a)] += 1.0 / angles.len() as f64;
        }
    };
    match curve {
        "lw" => {
            let total = (LISTENING_WINDOW_H.len() + LISTENING_WINDOW_V.len()) as f64;
            for &a in &LISTENING_WINDOW_H {
                w[index_of('h', a)] += 1.0 / total;
            }
            for &a in &LISTENING_WINDOW_V {
                w[index_of('v', a)] += 1.0 / total;
            }
        }
        "floor" => add(&mut w, 'v', &FLOOR),
        "ceiling" => add(&mut w, 'v', &CEILING),
        "front" => add(&mut w, 'h', &FRONT_WALL),
        "side" => add(&mut w, 'h', &SIDE_WALL),
        "rear" => add(&mut w, 'h', &REAR_WALL),
        "er" => {
            for b in BOUNCES {
                for (o, v) in w.iter_mut().zip(kernel(b)) {
                    *o += v / BOUNCES.len() as f64;
                }
            }
        }
        "sp" => {
            let spw = sound_power_weights();
            for (i, &(_, a)) in keys().iter().enumerate() {
                w[i] = spw[(a.unsigned_abs() / 10) as usize];
            }
            let s: f64 = w.iter().sum();
            w.iter_mut().for_each(|v| *v /= s);
        }
        "pir" => {
            for (c, weight) in ["lw", "er", "sp"].iter().zip(PIR_WEIGHTS) {
                for (o, v) in w.iter_mut().zip(kernel(c)) {
                    *o += weight * v;
                }
            }
        }
        _ => panic!("unknown curve {curve}"),
    }
    w
}

/// The curves of the standard from the power responses at the 70 directions.
#[derive(Clone, Debug)]
pub struct Curves {
    pub on_axis: Vec<f64>,
    pub lw: Vec<f64>,
    pub er: Vec<f64>,
    pub sp: Vec<f64>,
    pub pir: Vec<f64>,
    pub floor: Vec<f64>,
    pub ceiling: Vec<f64>,
    pub front: Vec<f64>,
    pub side: Vec<f64>,
    pub rear: Vec<f64>,
    /// Sound power over the whole sphere, when known.
    pub sp_sphere: Vec<f64>,
}

pub fn weighted(kernel: &[f64], power: &Mat<f32>) -> Vec<f64> {
    let mut out = vec![0.0; power.cols];
    for (w, row) in kernel.iter().zip(power.rows_iter()) {
        if *w != 0.0 {
            for (o, p) in out.iter_mut().zip(row) {
                *o += w * *p as f64;
            }
        }
    }
    out
}

pub fn curves(power: &Mat<f32>, sp_sphere: Vec<f64>) -> Curves {
    let on = index_of('h', 0);
    Curves {
        on_axis: power.row(on).iter().map(|v| *v as f64).collect(),
        lw: weighted(&kernel("lw"), power),
        er: weighted(&kernel("er"), power),
        sp: weighted(&kernel("sp"), power),
        pir: weighted(&kernel("pir"), power),
        floor: weighted(&kernel("floor"), power),
        ceiling: weighted(&kernel("ceiling"), power),
        front: weighted(&kernel("front"), power),
        side: weighted(&kernel("side"), power),
        rear: weighted(&kernel("rear"), power),
        sp_sphere,
    }
}

impl Curves {
    pub fn get(&self, name: &str) -> &[f64] {
        match name {
            "on_axis" => &self.on_axis,
            "lw" => &self.lw,
            "er" => &self.er,
            "sp" => &self.sp,
            "pir" => &self.pir,
            "floor" => &self.floor,
            "ceiling" => &self.ceiling,
            "front" => &self.front,
            "side" => &self.side,
            "rear" => &self.rear,
            "sp_sphere" => &self.sp_sphere,
            _ => panic!("unknown curve {name}"),
        }
    }
}
