//! The CTA-2034 curves: kernels, orbits, and what two orbits miss of the sound power.

use inroom::constants::C;
use inroom::cta2034;
use inroom::mat::Mat;
use inroom::special::piston;
use inroom::{norm, sphere, Vec3};

#[test]
fn kernels_sum_to_one() {
    for c in ["lw", "er", "sp", "pir", "floor", "ceiling", "front", "side", "rear"] {
        let s: f64 = cta2034::kernel(c).iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "{c}: {s}");
    }
}

#[test]
fn a_source_radiating_the_same_everywhere_has_equal_curves() {
    let p = Mat::from_rows(vec![vec![3.0f32; 5]; 70]);
    let c = cta2034::curves(&p, vec![3.0; 5]);
    for k in ["lw", "er", "sp", "pir"] {
        assert!(c.get(k).iter().all(|v| (v - 3.0).abs() < 1e-9), "{k}");
    }
    // Sound power directivity index: zero.
    assert!(c.lw.iter().zip(&c.sp).all(|(l, s)| (10.0 * (l / s).log10()).abs() < 1e-9));
}

#[test]
fn listening_window_and_bounces() {
    let keys = cta2034::keys();
    let picked = |curve: &str| {
        let mut k: Vec<(char, i32)> = cta2034::kernel(curve).iter().zip(keys).filter(|(w, _)| **w > 0.0).map(|(_, k)| *k).collect();
        k.sort();
        k
    };
    let mut lw: Vec<(char, i32)> = [0, 10, -10, 20, -20, 30, -30].iter().map(|a| ('h', *a)).collect();
    lw.extend([('v', 10), ('v', -10)]);
    lw.sort();
    assert_eq!(picked("lw"), lw);
    assert_eq!(picked("floor"), vec![('v', -40), ('v', -30), ('v', -20)]);
    // 0 and 180 degrees are shared by both orbits: 70 directions.
    assert_eq!(keys.len(), 70);
    assert!(cta2034::directions().iter().all(|u| (norm(u) - 1.0).abs() < 1e-12));
}

/// Power over the sphere of the Fibonacci lattice and from the two orbits, in dB.
fn two_orbit_bias(power: impl Fn(&Vec3) -> f64, n: usize) -> (f64, f64) {
    let exact = sphere::fibonacci(n).iter().map(&power).sum::<f64>() / n as f64;
    let two: f64 = cta2034::kernel("sp").iter().zip(cta2034::directions()).map(|(w, u)| w * power(&u)).sum();
    (exact, 10.0 * (two / exact).log10())
}

/// For a source symmetric about its axis, the two orbits are exact when it radiates the same
/// everywhere and overestimate a directional one a little: the response on an orbit stands for a
/// whole spherical square, whose corners lie farther off axis.
#[test]
fn sound_power_of_an_axisymmetric_source_from_two_orbits() {
    let bias = |ka: f64| {
        two_orbit_bias(
            |u| {
                let s = (1.0 - u[2] * u[2]).max(0.0).sqrt();
                (piston(ka * s) * 0.5 * (1.0 + u[2])).powi(2)
            },
            20000,
        )
        .1
    };
    let (low, high) = (bias(0.3), bias(3.0));
    assert!(low.abs() < 0.1, "{low}");
    assert!(0.2 < high && high < 0.6, "{high}");
}

/// Two in-phase point sources 140 mm apart vertically: the sphere gives the textbook power, the
/// two orbits overestimate it near the first null.
#[test]
fn two_point_sources_two_orbits_miss_the_interference() {
    let (d, f) = (0.14, 2200.0);
    let k = 2.0 * std::f64::consts::PI * f / C;
    let p = |u: &Vec3| {
        let ph = k * d * u[1];
        ((1.0 + ph.cos()).powi(2) + ph.sin().powi(2)) / 4.0
    };
    let analytic = 0.5 * (1.0 + (k * d).sin() / (k * d));
    let (exact, bias) = two_orbit_bias(p, 40000);
    assert!((exact / analytic - 1.0).abs() < 1e-3);
    let two: f64 = cta2034::kernel("sp").iter().zip(cta2034::directions()).map(|(w, u)| w * p(&u)).sum();
    assert!(10.0 * (two / analytic).log10() > 1.0);
    assert!(bias > 1.0);
}
