//! The two-orbit standard on the loudspeaker library: prints what the report and the articles
//! quote. ``cargo run --release --bin standard`` (after ``--bin library``).

use std::path::PathBuf;

use inroom::standard::{self, Library};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lib = Library::load(&root.join("results/library"));
    let s = standard::summary(&lib);
    println!("{:18} {:>16} {:>16} {:>16} {:>11} {:>17} {:>11}", "", "SP CTA rms/max", "SP rings", "SP ways", "far st/ea", "steady pir/c/w", "early p/w");
    for f in &s.families {
        println!(
            "{:18} {:>7.2}/{:<8.2} {:>7.2}/{:<8.2} {:>7.2}/{:<8.2} {:>5.2}/{:<5.2} {:>5.2}/{:.2}/{:<5.2} {:>5.2}/{:<5.2}",
            f.name, f.sp_rms[0], f.sp_max[0], f.sp_rms[1], f.sp_max[1], f.sp_rms[2], f.sp_max[2], f.far[0], f.far[1], f.steady[0], f.steady[1], f.steady[2], f.early[0], f.early[1]
        );
    }
    println!("\nfitted: steady {:.2?}, early {:.2?}; family left out: LW {:.2?}, SP {:.2?}", s.steady_fit, s.early_fit, s.lofo_lw, s.lofo_sp);
    for (name, lo, hi, bias) in &s.errors {
        println!("{name:18} median RMS {lo:.2}-{hi:.2} dB, largest mean error {bias:.2}");
    }
    println!("\nshares (floor, ceiling, side, front, rear) {:.2?}", s.shares);
    println!("correlation with the rooms' first reflections: orbit index {:.3}, ER {:.3}, PIR {:.3}, room curve {:.3}; floor+ceiling {:.3}", s.corr[0], s.corr[1], s.corr[2], s.corr[3], s.corr[4]);
    for f in &s.families {
        for k in &f.kinds {
            println!("{:30} rooms {:.2?} index {:.2?} curves {:.2?}", k.variant, k.rooms, k.index, k.curves);
        }
    }
}
