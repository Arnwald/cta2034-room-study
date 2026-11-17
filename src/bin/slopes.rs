//! Do gentle crossover slopes harm what a listener receives in a room? (`protocol/crossover-slopes.md`)
//!
//! ``cargo run --release --bin slopes -- calibrate``: step 0, written to
//! `protocol/crossover-slopes-calibration.md`.

use std::fmt::Write as _;
use std::path::PathBuf;

use inroom::slopes::{self, Calibration, Calibrator, Summary, ANCHOR_Q1_DB, CAL_FREQS, CAL_QS};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    match std::env::args().nth(1).as_deref() {
        Some("calibrate") => calibrate(&root),
        _ => eprintln!("usage: slopes calibrate"),
    }
}

fn level(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "never".into())
}

fn calibrate(root: &std::path::Path) {
    let c = Calibrator::new();
    let cals: Vec<Calibration> = [Summary::A, Summary::B].iter().map(|&s| slopes::calibrate(&c, s)).collect();
    let (best, ok) = slopes::choose(&cals);
    let mut md = String::new();
    writeln!(md, "# Step 0: calibration of the instrument\n").unwrap();
    writeln!(md, "Output of `cargo run --release --bin slopes -- calibrate`, run before any loudspeaker was computed").unwrap();
    writeln!(md, "(protocol: `crossover-slopes.md`).\n").unwrap();
    writeln!(
        md,
        "Each table gives, for resonances (RBJ peaking filters) at {} Hz, the level that reaches the same value \
         of the summary as a +{ANCHOR_Q1_DB} dB, Q = 1 peak at the same frequency. Published anchors (peaks, music, \
         Toole and Olive 1988): about 1-2 dB for Q = 1, about 10 dB for Q = 50. Dips: depth (dB) for the same value; \
         \"never\" means not reached at 60 dB.\n",
        CAL_FREQS.map(|f| format!("{f:.0}")).join(", ")
    )
    .unwrap();
    for cal in &cals {
        writeln!(md, "## Summary {}\n", cal.summary.name()).unwrap();
        writeln!(md, "| | {} |", CAL_QS.map(|q| format!("peak, Q = {q:.0}")).join(" | ") + " | " + &CAL_QS.map(|q| format!("dip, Q = {q:.0}")).join(" | ")).unwrap();
        writeln!(md, "|---|{}", "---|".repeat(2 * CAL_QS.len())).unwrap();
        for (i, f0) in CAL_FREQS.iter().enumerate() {
            let p: Vec<String> = cal.peaks[i].iter().map(|v| level(*v)).collect();
            let d: Vec<String> = cal.dips[i].iter().map(|v| level(*v)).collect();
            writeln!(md, "| {f0:.0} Hz | {} | {} |", p.join(" | "), d.join(" | ")).unwrap();
        }
        writeln!(md, "\nL50 (Q = 50 peak, mean of the three frequencies): **{:.1} dB**. Value of a +{ANCHOR_Q1_DB} dB, Q = 1 peak at 2.2 kHz: {:.3}.\n", cal.l50, cal.unit).unwrap();
    }
    let chosen = &cals[best];
    writeln!(md, "## Choice\n").unwrap();
    if ok {
        writeln!(
            md,
            "Summary {} is kept: its L50, {:.1} dB, is the closest to 10 dB and within 5-20 dB. The unit T is {:.3} \
             (its value for a +{ANCHOR_Q1_DB} dB, Q = 1 peak at 2.2 kHz).",
            chosen.summary.name(),
            chosen.l50,
            chosen.unit
        )
        .unwrap();
    } else {
        writeln!(md, "Neither summary has an L50 within 5-20 dB (closest: {}, {:.1} dB). The protocol stops here.", chosen.summary.name(), chosen.l50).unwrap();
    }
    print!("{md}");
    let path = root.join("protocol/crossover-slopes-calibration.md");
    std::fs::write(&path, md).unwrap();
    eprintln!("written to {}", path.display());
}
