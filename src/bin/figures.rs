//! Figures (paper/figures) and tables (paper/tables) from results/study.bin, in English and in
//! French (paper/figures/fr, paper/tables/fr). ``cargo run --release --bin figures [-- en|fr]``.

use std::path::PathBuf;
use std::time::Instant;

use inroom::analysis::Study;
use inroom::coherent::CoherentResults;
use inroom::standard::{self, Library};
use inroom::plot::Lang;
use inroom::{ensemble, figures, tables};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let t0 = Instant::now();
    let study = Study::load(&root.join("results/study.bin"));
    let setups = ensemble::load(&root.join("data/rooms.json"));
    let args: Vec<String> = std::env::args().skip(1).collect();
    let langs: Vec<Lang> = if args.iter().any(|a| a == "fr") && !args.iter().any(|a| a == "en") {
        vec![Lang::Fr]
    } else if args.iter().any(|a| a == "en") && !args.iter().any(|a| a == "fr") {
        vec![Lang::En]
    } else {
        vec![Lang::En, Lang::Fr]
    };
    let spectra = figures::path_spectra(&setups);
    let coh_path = root.join("results/coherent.bin");
    let coh = coh_path.exists().then(|| CoherentResults::load(&coh_path));
    let lib_dir = root.join("results/library");
    let lib = lib_dir.join("four_way_tower.bin").exists().then(|| standard::summary(&Library::load(&lib_dir)));
    for lang in langs {
        let (fig, tab) = match lang {
            Lang::En => (root.join("paper/figures"), root.join("paper/tables")),
            Lang::Fr => (root.join("paper/figures/fr"), root.join("paper/tables/fr")),
        };
        figures::all(&study, &setups, lang, &fig, &spectra, coh.as_ref());
        if let Some(lib) = &lib {
            inroom::library_figures::all(lib, lang, &fig);
        }
        let numbers = tables::all(&study, lang, &tab, &spectra, coh.as_ref(), lib.as_ref());
        eprintln!("{:?}: {} figures, {} numbers ({:.1} s)", lang, 14 + if coh.is_some() { 2 } else { 0 } + if lib.is_some() { 3 } else { 0 }, numbers.len(), t0.elapsed().as_secs_f64());
    }
}
