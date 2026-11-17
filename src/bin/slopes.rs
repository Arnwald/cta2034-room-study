//! Do gentle crossover slopes harm what a listener receives in a room? (`protocol/crossover-slopes.md`)
//!
//! ``cargo run --release --bin slopes -- calibrate``: step 0, written to
//! `protocol/crossover-slopes-calibration.md`.
//! ``cargo run --release --bin slopes -- run``: every loudspeaker and the positive control in the
//! 300 random rooms (results/slopes/), then the report.
//! ``cargo run --release --bin slopes -- report``: the report alone, from results/slopes/, written to
//! `protocol/crossover-slopes-results.md`.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use inroom::crossover::KINDS;
use inroom::room::Setup;
use inroom::slopes::{self, Calibration, Calibrator, LayoutValues, Summary, ANCHOR_Q1_DB, CAL_FREQS, CAL_QS, PATHWAYS};
use inroom::special::{median, percentile};
use inroom::{ensemble, speaker};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    match std::env::args().nth(1).as_deref() {
        Some("calibrate") => calibrate(&root),
        Some("run") => {
            run(&root);
            report(&root);
        }
        Some("report") => report(&root),
        _ => eprintln!("usage: slopes calibrate | run | report"),
    }
}

fn level(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}")).unwrap_or_else(|| "never".into())
}

fn calibrate(root: &Path) {
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

/// The random rooms, a tower's reference point half its height above the floor.
fn placed(setups: &[Setup], layout: &speaker::Layout) -> Vec<Setup> {
    setups
        .iter()
        .filter(|s| s.key.is_none())
        .map(|s| {
            let mut s = s.clone();
            if layout.floor {
                s.source[2] = layout.cabinet.height / 2.0;
            }
            s
        })
        .collect()
}

fn file_name(v: &LayoutValues) -> String {
    if v.roll != 0.0 {
        format!("{}-side.json", v.name)
    } else {
        format!("{}.json", v.name)
    }
}

fn run(root: &Path) {
    let setups = ensemble::load(&root.join("data/rooms.json"));
    let dir = root.join("results/slopes");
    std::fs::create_dir_all(&dir).unwrap();
    let mut jobs: Vec<(speaker::Layout, f64)> = speaker::library().into_iter().map(|l| (l, 0.0)).collect();
    // The positive control: the two-way laid on its side.
    jobs.push((speaker::two_way(), 90.0));
    for (layout, roll) in jobs {
        let t0 = Instant::now();
        let v = slopes::run_layout(&layout, roll, &placed(&setups, &layout));
        std::fs::write(dir.join(file_name(&v)), serde_json::to_string(&v).unwrap()).unwrap();
        eprintln!("{} (roll {roll}): {} rooms, {:.1} s", layout.name, v.rooms.len(), t0.elapsed().as_secs_f64());
    }
}

fn load(root: &Path, name: &str) -> LayoutValues {
    let path = root.join("results/slopes").join(name);
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e} (run first)", path.display()))).unwrap()
}

fn cell(v: &[f64]) -> String {
    format!("{:.2} ({:.2}–{:.2})", median(v), percentile(v, 10.0), percentile(v, 90.0))
}

fn report(root: &Path) {
    let t = slopes::calibrate(&Calibrator::new(), Summary::A).unit;
    let library: Vec<LayoutValues> = speaker::library().iter().map(|l| load(root, &format!("{}.json", l.name))).collect();
    let side = load(root, "two_way-side.json");
    let names: Vec<&str> = KINDS.iter().map(|k| k.name()).collect();
    let k_index = |name: &str| names.iter().position(|n| *n == name).unwrap();
    // Values in T: [room] for a layout, a kind, a pathway.
    let path_values = |v: &LayoutValues, ki: usize, p: usize| -> Vec<f64> { v.rooms.iter().map(|r| r.d[ki][p] / t).collect() };
    let mut md = String::new();
    writeln!(md, "# Results: do gentle crossover slopes harm what a listener receives in a room?\n").unwrap();
    writeln!(md, "Output of `cargo run --release --bin slopes -- run`, following `crossover-slopes.md` with its amendment 1,").unwrap();
    writeln!(md, "committed before this run. Distances to each variant's interference-free twin, in T (summary A, T = {t:.3}, from").unwrap();
    writeln!(md, "`crossover-slopes-calibration.md`). Median over the {} random rooms, 10–90 % in brackets.\n", library[0].rooms.len()).unwrap();

    // Checks.
    let coax = library.iter().find(|v| v.name == "coaxial").unwrap();
    let coax_max = coax.rooms.iter().flat_map(|r| r.d.iter().flat_map(|d| d.iter())).fold(0.0f64, |m, v| m.max(*v / t));
    // Amendment 1: the 75th percentile, the study's quarter of the listeners.
    let control = percentile(&path_values(&side, k_index("lr4"), 0), 75.0);
    writeln!(md, "## Checks\n").unwrap();
    writeln!(md, "- The coaxial against its twin: {coax_max:.2e} T at most (must be zero to rounding).").unwrap();
    writeln!(
        md,
        "- Positive control, the two-way laid on its side, P1s with LR4: 75th percentile {control:.2} T (must exceed 1 T, amendment 1): {}.",
        if control > 1.0 { "passed" } else { "FAILED, the protocol stops" }
    )
    .unwrap();
    writeln!(md, "  Every crossover, P1s: {}.", names.iter().enumerate().map(|(ki, n)| format!("{n} {}", cell(&path_values(&side, ki, 0)))).collect::<Vec<_>>().join(", ")).unwrap();
    // Amendment 1: P1, P2 and P4 are powers, which the twin's centre cannot change.
    let run1 = root.join("results/slopes-run1");
    if run1.is_dir() {
        let mut worst = 0.0f64;
        for v in &library {
            let path = run1.join(format!("{}.json", v.name));
            let old: LayoutValues = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            for (a, b) in v.rooms.iter().zip(&old.rooms) {
                for ki in 0..KINDS.len() {
                    for p in [0, 1, 2, 4] {
                        worst = worst.max((a.d[ki][p] - b.d[ki][p]).abs() / t);
                    }
                }
            }
        }
        writeln!(md, "- P1, P2 and P4 against the first run (they must not change, amendment 1): {worst:.2e} T at most.").unwrap();
    }
    writeln!(md).unwrap();

    // Tables per pathway.
    let labels = ["P1s, direct sound, seated", "P1d, direct sound, standing", "P2, each first reflection's spectrum", "P3, early sound at the ears", "P4, steady state"];
    for (p, label) in labels.iter().enumerate() {
        writeln!(md, "## {label}\n").unwrap();
        writeln!(md, "| loudspeaker (crossovers, Hz) | {} |", names.join(" | ")).unwrap();
        writeln!(md, "|---|{}", "---|".repeat(names.len())).unwrap();
        for v in &library {
            let xo: Vec<String> = v.crossovers.iter().map(|f| format!("{f:.0}")).collect();
            let cells: Vec<String> = (0..KINDS.len()).map(|ki| cell(&path_values(v, ki, p))).collect();
            writeln!(md, "| {} ({}) | {} |", v.name, xo.join(", "), cells.join(" | ")).unwrap();
        }
        writeln!(md).unwrap();
    }

    // The decision.
    let tested: Vec<&LayoutValues> = library.iter().filter(|v| v.name != "coaxial").collect();
    let pairs = [("lr2", "fir"), ("lr4", "fir"), ("lr8", "fir"), ("lr2", "lr4")];
    writeln!(md, "## Decision\n").unwrap();
    writeln!(md, "Median over the rooms of the paired difference δ (T); a pathway supports view 1 when δ(LR2 − FIR) ≥ +0.5 T").unwrap();
    writeln!(md, "for at least 5 of the 6 loudspeakers, contradicts it when δ ≤ −0.5 T for at least 5 of 6.\n").unwrap();
    for (a, b) in pairs {
        let (ka, kb) = (k_index(a), k_index(b));
        writeln!(md, "### {} − {}\n", a.to_uppercase(), b.to_uppercase()).unwrap();
        writeln!(md, "| pathway | {} | ≥ +0.5 | ≤ −0.5 | outcome |", tested.iter().map(|v| v.name.as_str()).collect::<Vec<_>>().join(" | ")).unwrap();
        writeln!(md, "|---|{}---|---|---|", "---|".repeat(tested.len())).unwrap();
        for (p, name) in PATHWAYS.iter().enumerate() {
            let deltas: Vec<f64> = tested
                .iter()
                .map(|v| median(&v.rooms.iter().map(|r| (r.d[ka][p] - r.d[kb][p]) / t).collect::<Vec<_>>()))
                .collect();
            let up = deltas.iter().filter(|d| **d >= 0.5).count();
            let down = deltas.iter().filter(|d| **d <= -0.5).count();
            let audible = library.iter().any(|v| (0..KINDS.len()).any(|ki| median(&path_values(v, ki, p)) >= 0.5));
            let outcome = if !audible {
                "below audibility"
            } else if up >= 5 {
                "supported"
            } else if down >= 5 {
                "contradicted"
            } else {
                "no general answer"
            };
            let cells: Vec<String> = deltas.iter().map(|d| format!("{d:+.2}")).collect();
            writeln!(md, "| {name} | {} | {up} | {down} | {outcome} |", cells.join(" | ")).unwrap();
        }
        writeln!(md).unwrap();
    }

    // Secondary, descriptive.
    writeln!(md, "## Secondary, descriptive\n").unwrap();
    let secondary: [(&str, Box<dyn Fn(&inroom::slopes::RoomValues, usize) -> f64>); 3] = [
        ("P4, each crossover's twin against the FIR's twin (the blending of the directivities, no lobing)", Box::new(|r, ki| r.blend[ki])),
        ("P2 over the floor and ceiling bounces only", Box::new(|r, ki| r.p2_vertical[ki])),
        ("P3 with the two ears of each head averaged in power", Box::new(|r, ki| r.p3_ears[ki])),
    ];
    for (label, f) in secondary.iter() {
        writeln!(md, "### {label}\n").unwrap();
        writeln!(md, "| loudspeaker | {} |", names.join(" | ")).unwrap();
        writeln!(md, "|---|{}", "---|".repeat(names.len())).unwrap();
        for v in &library {
            let cells: Vec<String> = (0..KINDS.len()).map(|ki| cell(&v.rooms.iter().map(|r| f(r, ki) / t).collect::<Vec<_>>())).collect();
            writeln!(md, "| {} | {} |", v.name, cells.join(" | ")).unwrap();
        }
        writeln!(md).unwrap();
    }
    print!("{md}");
    let path = root.join("protocol/crossover-slopes-results.md");
    std::fs::write(&path, md).unwrap();
    eprintln!("written to {}", path.display());
}
