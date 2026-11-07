//! CTA-2034 in simulated rooms: what the listener receives, direction by direction.
//!
//! The spatial averages of ANSI/CTA-2034-A are weighted means of the power a loudspeaker radiates
//! in 70 directions. The energy reaching a listener in a room is also a weighted mean of the power
//! radiated in every direction. This crate simulates loudspeakers whose far field is known over the
//! whole sphere ([`speaker`]), shoebox rooms reduced to weights over directions ([`room`]), and
//! compares the two ([`study`], [`analysis`]).

pub mod analysis;
pub mod articles;
pub mod coherent;
pub mod constants;
pub mod crossover;
pub mod cta2034;
pub mod directivity;
pub mod ensemble;
pub mod figures;
pub mod library_figures;
pub mod mat;
pub mod npy;
pub mod plot;
pub mod results;
pub mod rng;
pub mod room;
pub mod signal;
pub mod special;
pub mod speaker;
pub mod standard;
pub mod sphere;
pub mod study;
pub mod tables;

pub type C64 = num_complex::Complex<f64>;
pub type C32 = num_complex::Complex<f32>;

/// A direction or a point in 3D.
pub type Vec3 = [f64; 3];

pub fn dot(a: &Vec3, b: &Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

pub fn norm(a: &Vec3) -> f64 {
    dot(a, a).sqrt()
}

pub fn sub(a: &Vec3, b: &Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

pub fn scale(a: &Vec3, s: f64) -> Vec3 {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub fn unit(a: &Vec3) -> Vec3 {
    scale(a, 1.0 / norm(a))
}
