//! The paths added with their phases at the listener's ears, from results/study.bin, saved to
//! results/coherent.bin. ``cargo run --release --bin coherent``.

use std::path::PathBuf;
use std::time::Instant;

use inroom::coherent::{self, CoherentResults};
use inroom::results::Results;
use inroom::special::median;
use inroom::{ensemble, speaker};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let t0 = Instant::now();
    let res = Results::load(&root.join("results/study.bin"));
    let setups = ensemble::load(&root.join("data/rooms.json"));
    let living = res.keys.iter().position(|k| k.as_deref() == Some("living")).expect("living room");
    let layouts = vec![speaker::two_way(), speaker::two_way_waveguide()];
    let out = coherent::run(&setups, &res, &layouts, living);
    let path = root.join("results/coherent.bin");
    out.save(&path);
    eprintln!("saved {} in {:.1} s; images against the study: {:.2e} dB at most", path.display(), t0.elapsed().as_secs_f64(), out.check_db);
    summary(&out);
}

/// Medians over the random rooms (and over the points or head positions) of the deviation within
/// one octave of the crossover.
fn summary(out: &CoherentResults) {
    let n = out.n_random;
    for (kind, label) in ["steady state", "early sound"].iter().enumerate() {
        eprintln!("\n{label}: deviation within one octave of the crossover, dB RMS (median of {n} rooms)");
        eprintln!("{:<26} {:>7} {:>7} {:>7} {:>7}", "variant", "energy", "point", "ears", "14 pts");
        for (vi, name) in out.variants.iter().enumerate() {
            let rooms = &out.rooms[vi][..n];
            let energy: Vec<f64> = rooms.iter().map(|r| r[kind].energy.local as f64).collect();
            let points: Vec<f64> = rooms.iter().flat_map(|r| r[kind].points.iter().map(|m| m.local as f64)).collect();
            let ears: Vec<f64> = rooms.iter().flat_map(|r| r[kind].ears.iter().map(|m| m.local as f64)).collect();
            let spatial: Vec<f64> = rooms.iter().map(|r| r[kind].spatial.local as f64).collect();
            eprintln!("{:<26} {:>7.2} {:>7.2} {:>7.2} {:>7.2}", name, median(&energy), median(&points), median(&ears), median(&spatial));
        }
    }
}
