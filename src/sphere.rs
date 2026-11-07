//! Directions on the sphere: conventions, quadrature, angles.
//!
//! Speaker frame: ``x`` to the right as seen from the listener, ``y`` up, ``z`` forward. Azimuth
//! ``az = atan2(x, z)`` (0 on axis, +90 to the right), elevation ``el = asin(y)``.

use crate::Vec3;

/// ``n`` nearly equally spaced unit vectors (Fibonacci lattice), each standing for ``4 pi / n``
/// steradians; the lattice's pole is the ``y`` axis.
pub fn fibonacci(n: usize) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            let i = i as f64 + 0.5;
            let y = 1.0 - 2.0 * i / n as f64;
            let r = (1.0 - y * y).sqrt();
            let phi = std::f64::consts::PI * (1.0 + 5f64.sqrt()) * i;
            [r * phi.sin(), y, r * phi.cos()]
        })
        .collect()
}

pub fn from_angles(az_deg: f64, el_deg: f64) -> Vec3 {
    let (az, el) = (az_deg.to_radians(), el_deg.to_radians());
    [el.cos() * az.sin(), el.sin(), el.cos() * az.cos()]
}

/// Azimuth and elevation (degrees) of a unit vector.
pub fn to_angles(u: &Vec3) -> (f64, f64) {
    (u[0].atan2(u[2]).to_degrees(), u[1].clamp(-1.0, 1.0).asin().to_degrees())
}

/// Angle from the reference axis (degrees).
pub fn off_axis_angle(u: &Vec3) -> f64 {
    u[2].clamp(-1.0, 1.0).acos().to_degrees()
}
