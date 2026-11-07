//! The figures of the two blog articles, SVG for the web, in articles/figures/.
//! ``cargo run --release --bin articles`` (after the study and the library).

use std::path::PathBuf;

use inroom::analysis::Study;
use inroom::ensemble;
use inroom::standard::{self, Library};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let study = Study::load(&root.join("results/study.bin"));
    let summary = standard::summary(&Library::load(&root.join("results/library")));
    let setups = ensemble::load(&root.join("data/rooms.json"));
    inroom::articles::all(&study, &setups, &summary, &root.join("articles/figures"));
    eprintln!("articles/figures written");
}
