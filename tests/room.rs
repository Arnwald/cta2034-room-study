//! Rooms as weights over the directions the energy leaves the loudspeaker in: images, rays, the
//! diffuse field, and the ensemble.

use std::f64::consts::PI;
use std::path::Path;

use inroom::ensemble;
use inroom::room::{self, Meta, Room, Setup, SURFACES};
use inroom::{dot, norm, sphere, sub};

fn uniform(alpha: f64, s: f64) -> Room {
    Room { dims: [4.3, 6.0, 2.6], absorption: [[alpha; 7]; 6], scattering: [[s; 7]; 6], patches: vec![], fittings: [0.0; 7], label: String::new() }
}

fn setup(room: Room) -> Setup {
    Setup { room, source: [1.0, 5.2, 1.0], receiver: [2.15, 2.2, 1.15], yaw: 0.3, label: String::new(), key: None, meta: Meta::default() }
}

fn named(key: &str) -> Setup {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rooms.json");
    ensemble::load(&path).into_iter().find(|s| s.key.as_deref() == Some(key)).expect("named room")
}

/// Energy of the images up to order 3 and of the diffuse field at 1 kHz.
fn hybrid(st: &Setup) -> f64 {
    let f = [1000.0];
    let grid = sphere::fibonacci(3000);
    let im = room::image_sources(st, 3);
    let specular: f64 = room::specular_weights(st, &im, &f).data.iter().sum();
    let diffuse: f64 = room::diffuse_weights(st, &grid, 4.0 * PI / 3000.0, &f, 3).data.iter().sum();
    specular + diffuse
}

#[test]
fn image_count_and_first_order_geometry() {
    let st = setup(uniform(0.2, 0.0));
    let im = room::image_sources(&st, 3);
    let mut counts = [0; 4];
    for o in &im.order {
        counts[*o] += 1;
    }
    assert_eq!(counts, [1, 6, 18, 38]);
    let (s, r) = (st.source, st.receiver);
    for i in (0..im.order.len()).filter(|&i| im.order[i] == 1) {
        let (axis, side) = room::plane(im.first[i].unwrap());
        let p = im.point[i].unwrap();
        // The reflection point lies on the surface, and both legs add up to the image distance.
        assert!((p[axis] - side as f64 * st.room.dims[axis]).abs() < 1e-9);
        assert!((norm(&sub(&p, &s)) + norm(&sub(&r, &p)) - im.distance[i]).abs() < 1e-9);
        // The emission direction points from the source to the reflection point.
        let d = st.to_room(&im.direction[i]);
        let e = sub(&p, &s);
        assert!((0..3).all(|k| (d[k] - e[k] / norm(&e)).abs() < 1e-9));
    }
    assert!((im.distance[0] - norm(&sub(&r, &s))).abs() < 1e-12);
}

#[test]
fn speaker_frame_is_right_handed_and_aims() {
    let st = named("living");
    let a = st.axes();
    for i in 0..3 {
        for j in 0..3 {
            assert!((dot(&a[i], &a[j]) - if i == j { 1.0 } else { 0.0 }).abs() < 1e-12);
        }
    }
    let cross = [a[0][1] * a[1][2] - a[0][2] * a[1][1], a[0][2] * a[1][0] - a[0][0] * a[1][2], a[0][0] * a[1][1] - a[0][1] * a[1][0]];
    assert!((0..3).all(|k| (cross[k] - a[2][k]).abs() < 1e-12));
    let mut d = sub(&st.receiver, &st.source);
    d[2] = 0.0;
    assert!((0..3).all(|k| (a[2][k] - d[k] / norm(&d)).abs() < 1e-12));
}

#[test]
fn air_attenuation_iso_9613() {
    // About 4.7 dB/km at 1 kHz and 105 dB/km at 8 kHz (20 degC, 50 %).
    let m: Vec<f64> = room::air_attenuation(&[1000.0, 8000.0]).iter().map(|v| v * 10.0 * std::f64::consts::E.log10() * 1000.0).collect();
    assert!(4.0 < m[0] && m[0] < 5.5, "{}", m[0]);
    assert!(90.0 < m[1] && m[1] < 120.0, "{}", m[1]);
}

#[test]
fn hybrid_matches_hopkins_stryker_when_every_reflection_scatters() {
    let st = setup(uniform(0.2, 1.0));
    let f = [1000.0];
    let a = st.room.absorption_area(&f)[0];
    let r = norm(&sub(&st.receiver, &st.source));
    let expected = 1.0 / (r * r) + 16.0 * PI * 0.8 / a * (-room::air_attenuation(&f)[0] * 2.5).exp();
    let ratio = hybrid(&st) / expected;
    assert!((ratio - 1.0).abs() < 0.03, "{ratio}");
}

/// Purely specular room: the hybrid's energy within 10 % of 25 000 images.
#[test]
fn hybrid_against_a_full_image_method() {
    let st = setup(uniform(0.2, 0.0));
    let full = room::image_sources(&st, 26);
    let m = room::air_attenuation(&[1000.0])[0];
    let images: f64 = full.order.iter().zip(&full.distance).map(|(o, d)| 0.8f64.powi(*o as i32) * (-d * m).exp() / (d * d)).sum();
    let ratio = hybrid(&st) / images;
    assert!((ratio - 1.0).abs() < 0.1, "{ratio}");
}

#[test]
fn rays_and_patches() {
    let st = named("treated");
    // A ray toward the left wall's first reflection point lands on the panel.
    let im = room::image_sources(&st, 1);
    let i = (0..im.order.len()).find(|&j| im.order[j] == 1 && SURFACES[im.first[j].unwrap()] == "left").unwrap();
    let material = im.material[i].unwrap();
    assert!(material >= 6);
    let (materials, lengths) = room::trace(&st, &[im.direction[i]], 1);
    assert_eq!(materials[0][0], material);
    let l = norm(&sub(&im.point[i].unwrap(), &st.source));
    assert!((lengths[0][0] / l - 1.0).abs() < 1e-9);
}

#[test]
fn ensemble_is_reproducible_and_sane() {
    let a = ensemble::sample(20, 1, 0.25);
    let b = ensemble::sample(20, 1, 0.25);
    assert_eq!(a.iter().map(|s| &s.label).collect::<Vec<_>>(), b.iter().map(|s| &s.label).collect::<Vec<_>>());
    for s in &a {
        let [lx, ly, _] = s.room.dims;
        assert!(0.0 < s.source[0] && s.source[0] < lx);
        assert!(0.0 < s.receiver[1] && s.receiver[1] < s.source[1] && s.source[1] < ly);
        let rt = s.room.rt60(&[1000.0])[0];
        assert!(0.2 < rt && rt < 1.2, "{rt}");
    }
}

#[test]
fn the_rooms_file_holds_the_study_ensemble() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("data/rooms.json");
    let setups = ensemble::load(&path);
    let keys: Vec<&str> = setups.iter().filter_map(|s| s.key.as_deref()).collect();
    assert_eq!(setups.len(), 303);
    assert_eq!(keys, ["living", "treated", "small"]);
    for s in &setups {
        let rt = s.room.rt60(&[1000.0])[0];
        assert!(0.15 < rt && rt < 1.5, "{}: {rt}", s.label);
    }
}
