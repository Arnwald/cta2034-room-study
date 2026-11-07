//! Far-field directivity of the radiators.
//!
//! * Cone: rigid flat circular piston of radius ``a`` in an infinite baffle,
//!   ``D(theta) = 2 J1(k a sin theta) / (k a sin theta)``.
//! * Dome: rigid convex spherical cap (base radius ``a``, height ``h``) moving axially. The far
//!   field is the Rayleigh integral over the projected disc, each ring of radius ``r`` advanced by
//!   the cap height ``z(r)`` above the rim plane:
//!   ``A(theta, k) = (2 / a^2) int_0^a J0(k r sin theta) exp(j k z(r) cos theta) r dr``,
//!   ``z(r) = sqrt(R^2 - r^2) - (R - h)``, ``R = (a^2 + h^2) / (2 h)``, and
//!   ``D(theta) = A(theta, k) / A(0, k)``.
//! * Waveguide: a phenomenological constant-directivity source, the piston's directivity with an
//!   effective radius that stops growing in wavelengths above the frequency where
//!   ``k a = ka_max``: ``D(theta) = 2 J1(x) / x``, ``x = min(k a, ka_max) sin theta``.
//!
//! All are normalized to 1 on their own axis.

use crate::special::{j0, leggauss, piston};
use crate::C64;

#[derive(Clone, Debug)]
pub struct Radiator {
    pub radius: f64,
    pub height: f64,
    pub waveguide_ka: Option<f64>,
    r: Vec<f64>,
    w: Vec<f64>,
    z: Vec<f64>,
}

impl Radiator {
    /// A piston (``height = 0``) or a dome, with 96 Gauss-Legendre nodes over its radius.
    pub fn new(radius: f64, height: f64) -> Self {
        let (x, wx) = leggauss(96);
        let r: Vec<f64> = x.iter().map(|x| 0.5 * radius * (x + 1.0)).collect();
        let w: Vec<f64> = wx.iter().map(|w| 0.5 * radius * w).collect();
        let mut out = Radiator { radius, height, waveguide_ka: None, r, w, z: Vec::new() };
        out.z = out.r.iter().map(|&r| out.surface(r)).collect();
        out
    }

    /// The constant-directivity source of radius ``radius`` up to ``k a = ka_max``.
    pub fn waveguide(radius: f64, ka_max: f64) -> Self {
        let mut out = Self::new(radius, 0.0);
        out.waveguide_ka = Some(ka_max);
        out
    }

    /// Axial position ``z(r)`` of the radiating surface above the rim plane (m).
    pub fn surface(&self, r: f64) -> f64 {
        if self.height > 0.0 {
            let sphere = (self.radius * self.radius + self.height * self.height) / (2.0 * self.height);
            (sphere * sphere - r * r).sqrt() - (sphere - self.height)
        } else {
            0.0
        }
    }

    pub fn is_piston(&self) -> bool {
        self.height == 0.0
    }

    /// Unnormalized far-field factor ``A(theta, k)``.
    pub fn aperture(&self, k: f64, theta: f64) -> C64 {
        if let Some(ka) = self.waveguide_ka {
            return C64::new(piston((k * self.radius).min(ka) * theta.sin()), 0.0);
        }
        if self.is_piston() {
            return C64::new(piston(k * self.radius * theta.sin()), 0.0);
        }
        let (st, ct) = (theta.sin(), theta.cos());
        let mut acc = C64::new(0.0, 0.0);
        for i in 0..self.r.len() {
            acc += C64::from_polar(j0(k * self.r[i] * st) * self.r[i] * self.w[i], k * self.z[i] * ct);
        }
        acc * (2.0 / (self.radius * self.radius))
    }

    /// ``A(0, k)``: 1 for the piston, the phase-spread loss of the dome otherwise.
    pub fn on_axis(&self, k: f64) -> C64 {
        self.aperture(k, 0.0)
    }

    /// ``D(theta) = A(theta, k) / A(0, k)``.
    pub fn directivity(&self, k: f64, theta: f64) -> C64 {
        if self.is_piston() {
            self.aperture(k, theta)
        } else {
            self.aperture(k, theta) / self.on_axis(k)
        }
    }
}
