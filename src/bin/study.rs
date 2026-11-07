//! Every simulated quantity of the study, saved to results/study.bin.
//!
//! ``cargo run --release --bin study [-- --npy]``: ``--npy`` also writes the main arrays as NumPy
//! files in results/npy/.

use std::path::PathBuf;
use std::time::Instant;

use inroom::{ensemble, npy, signal, speaker, study};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let t0 = Instant::now();
    let setups = ensemble::load(&root.join("data/rooms.json"));
    let n_random = setups.iter().filter(|s| s.key.is_none()).count();
    let layouts = vec![speaker::two_way(), speaker::two_way_waveguide()];
    let freqs = signal::study_grid();
    eprintln!("{} setups ({} random), {} frequencies", setups.len(), n_random, freqs.len());
    let res = study::run(&setups, n_random, &layouts, &freqs);
    let out = root.join("results/study.bin");
    res.save(&out);
    eprintln!("saved {} in {:.1} s", out.display(), t0.elapsed().as_secs_f64());

    if std::env::args().any(|a| a == "--npy") {
        let dir = root.join("results/npy");
        std::fs::create_dir_all(&dir).unwrap();
        let nf = res.freqs.len();
        let n_rooms = setups.len();
        let n_cat = res.categories.len();
        npy::write(&dir.join("freqs.npy"), &[nf], &res.freqs).unwrap();
        npy::write(&dir.join("omni.npy"), &[n_rooms, n_cat, nf], &res.omni.data).unwrap();
        npy::write(&dir.join("diffuse_mean.npy"), &[res.diffuse_mean.rows, nf], &res.diffuse_mean.data).unwrap();
        npy::write(&dir.join("image_weights_band.npy"), &[res.image_weights_band.len()], &res.image_weights_band).unwrap();
        let dirs: Vec<f32> = res.image_dirs.iter().flat_map(|d| d.iter().copied()).collect();
        npy::write(&dir.join("image_dirs.npy"), &[res.image_dirs.len(), 3], &dirs).unwrap();
        for v in &res.variants {
            npy::write(&dir.join(format!("{}.cta.npy", v.name)), &[70, nf], &v.cta.data).unwrap();
            npy::write(&dir.join(format!("{}.sp_full.npy", v.name)), &[nf], &v.sp_full).unwrap();
            npy::write(&dir.join(format!("{}.rooms.npy", v.name)), &[n_rooms, n_cat, nf], &v.rooms.data).unwrap();
            for (name, data) in [("anom_img", &v.anom_img), ("local_img", &v.local_img), ("notch_img", &v.notch_img),
                                 ("pband_img", &v.pband_img), ("anom_grid", &v.anom_grid), ("local_cta", &v.local_cta)] {
                npy::write(&dir.join(format!("{}.{}.npy", v.name, name)), &[data.len()], data).unwrap();
            }
        }
        eprintln!("npy files in {}", dir.display());
    }
}
