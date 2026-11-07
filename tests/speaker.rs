//! The loudspeakers' far field: baffle, symmetry, roll, equalized sum, waveguide.

use inroom::crossover::{self, KINDS};
use inroom::signal::log_grid;
use inroom::speaker::{library, roll, two_way, two_way_waveguide, FarField};
use inroom::{norm, sphere, C64};

fn freqs() -> Vec<f64> {
    log_grid(150.0, 20000.0, 24)
}

#[test]
fn baffle_step_and_symmetry() {
    let f = freqs();
    let far = FarField::new(&two_way(), &f);
    let u = [sphere::from_angles(0.0, 0.0), sphere::from_angles(40.0, 10.0), sphere::from_angles(-40.0, 10.0)];
    for r in far.responses(&u) {
        // Full space below the baffle step, half space well above it.
        assert!((r.row(0)[0].norm() as f64 - 0.5).abs() < 0.15);
        let hi: Vec<f64> = f.iter().zip(r.row(0)).filter(|(f, _)| **f > 10000.0).map(|(_, h)| h.norm() as f64).collect();
        assert!((20.0 * (hi.iter().sum::<f64>() / hi.len() as f64).log10()).abs() < 1.0);
        // Drivers on the vertical middle line of a symmetric baffle: left and right are equal.
        for (a, b) in r.row(1).iter().zip(r.row(2)) {
            assert!((a.norm() - b.norm()).abs() <= 1e-4 * b.norm() + 1e-6);
        }
    }
}

#[test]
fn roll_maps_directions() {
    // Looking straight up at the rolled loudspeaker is looking at the unrolled one's left side.
    let up = roll(&[[0.0, 1.0, 0.0]], 90.0)[0];
    assert!((up[0] + 1.0).abs() < 1e-12 && up[1].abs() < 1e-12 && up[2].abs() < 1e-12);
    assert!(roll(&sphere::fibonacci(50), 90.0).iter().all(|v| (norm(v) - 1.0).abs() < 1e-12));
}

#[test]
fn equalized_sum_is_flat_on_axis() {
    let f = freqs();
    for layout in library() {
        let far = FarField::new(&layout, &f);
        let on = far.on_axis();
        for k in KINDS {
            let x = crossover::filters(k, &f, &layout, &on);
            for j in (0..f.len()).filter(|&j| f[j] > 300.0 && f[j] < 16000.0) {
                let s: C64 = (0..on.len()).map(|d| x[d][j] * on[d][j]).sum();
                let level = 20.0 * s.norm().log10();
                assert!(level.abs() < 0.5, "{} {}: {level} dB at {} Hz", layout.name, k.name(), f[j]);
            }
        }
    }
}

#[test]
fn waveguide_holds_its_beamwidth() {
    let f = freqs();
    let layout = two_way_waveguide();
    let far = FarField::new(&layout, &f);
    let t = layout.names().iter().position(|n| n == "tweeter").unwrap();
    let h = &far.responses(&[sphere::from_angles(0.0, 0.0), sphere::from_angles(60.0, 0.0)])[t];
    let rel: Vec<f64> = (0..f.len()).filter(|&j| f[j] > 4000.0 && f[j] < 16000.0).map(|j| 20.0 * (h.row(1)[j].norm() as f64 / h.row(0)[j].norm() as f64).log10()).collect();
    let spread = rel.iter().cloned().fold(f64::MIN, f64::max) - rel.iter().cloned().fold(f64::MAX, f64::min);
    assert!(spread < 3.0, "{spread}");
}
