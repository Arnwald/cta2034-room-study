//! The figures of the two blog articles, SVG for the web, in articles/figures/.
//! ``cargo run --release --bin articles`` (after the study and the library).

use std::path::PathBuf;

use inroom::analysis::Study;
use inroom::standard::{self, Library};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let study = Study::load(&root.join("results/study.bin"));
    let summary = standard::summary(&Library::load(&root.join("results/library")));
    inroom::articles::all(&study, &summary, &root.join("articles/figures"));
    eprintln!("articles/figures written");
}
