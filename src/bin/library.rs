//! The loudspeaker library in the rooms: every layout of the library, upright, with the four
//! crossovers, every path taken from each driver. One results file per layout in
//! results/library/. ``cargo run --release --bin library [-- layout ...]``.

use std::path::PathBuf;
use std::time::Instant;

use inroom::room::Setup;
use inroom::study::{self, Options};
use inroom::{ensemble, signal, speaker};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let setups = ensemble::load(&root.join("data/rooms.json"));
    let n_random = setups.iter().filter(|s| s.key.is_none()).count();
    let freqs = signal::study_grid();
    let only: Vec<String> = std::env::args().skip(1).collect();
    for layout in speaker::library() {
        if !only.is_empty() && !only.contains(&layout.name) {
            continue;
        }
        let t0 = Instant::now();
        // A tower stands on the floor; a bookshelf loudspeaker stays on its stand.
        let placed: Vec<Setup> = setups
            .iter()
            .map(|s| {
                let mut s = s.clone();
                if layout.floor {
                    s.source[2] = layout.cabinet.height / 2.0;
                }
                s
            })
            .collect();
        let opts = Options { rolls: vec![0.0], per_driver: true };
        let res = study::run_with(&placed, n_random, std::slice::from_ref(&layout), &freqs, &opts);
        let path = root.join(format!("results/library/{}.bin", layout.name));
        res.save(&path);
        eprintln!("{}: {:.1} s", layout.name, t0.elapsed().as_secs_f64());
    }
}
