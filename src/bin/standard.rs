//! The two-orbit standard on the loudspeaker library: prints what the report and the articles
//! quote. ``cargo run --release --bin standard`` (after ``--bin library``).

use std::path::PathBuf;

use inroom::analysis::{Group, Which};
use inroom::special::median;
use inroom::standard::{self, Library, SpFrom};

fn main() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let lib = Library::load(&root.join("results/library"));
    let all: Vec<usize> = (0..lib.layouts.len()).collect();

    println!("== Sound power from two orbits (dB RMS / max, 200 Hz-10 kHz, worst of the 4 crossovers)");
    for (li, s) in lib.studies.iter().enumerate() {
        let row: Vec<String> = [SpFrom::Cta, SpFrom::Ring, SpFrom::Ways]
            .iter()
            .map(|&from| {
                let e: Vec<(f64, f64)> = s.variants().iter().map(|v| standard::sp_error(s, v, from)).collect();
                format!("{:.2}/{:.2}", e.iter().fold(0.0f64, |a, x| a.max(x.0)), e.iter().fold(0.0f64, |a, x| a.max(x.1)))
            })
            .collect();
        println!("{:18} CTA {:>10}  ring {:>10}  ways {:>10}", lib.layouts[li].name, row[0], row[1], row[2]);
    }

    println!("\n== Paths from each driver against the far field (median RMS difference, dB)");
    for (li, s) in lib.studies.iter().enumerate() {
        let vs = s.variants();
        let st: Vec<f64> = vs.iter().map(|v| standard::far_field_error(s, v, Which::Steady)).collect();
        let ea: Vec<f64> = vs.iter().map(|v| standard::far_field_error(s, v, Which::Early)).collect();
        println!("{:18} steady {:.2}  early {:.2}", lib.layouts[li].name, median(&st), median(&ea));
    }

    for which in [Which::Steady, Which::Early] {
        println!("\n== In-room curve, {:?}: weights fitted on the training rooms of every layout", which);
        for from in [SpFrom::Sphere, SpFrom::Ways, SpFrom::Ring, SpFrom::Cta] {
            let w = standard::fit(&lib, &all, from, 0..150, which);
            println!("SP {:?}: {:.2} LW + {:.2} ER + {:.2} SP", from, w[0], w[1], w[2]);
        }
        let w_all = standard::fit(&lib, &all, SpFrom::Ways, 0..150, which);
        println!("test rooms, median RMS error / largest mean error (dB): CTA PIR | fitted (ways) | fitted without this family (ways)");
        for (li, s) in lib.studies.iter().enumerate() {
            let others: Vec<usize> = all.iter().cloned().filter(|&x| x != li).collect();
            let w_out = standard::fit(&lib, &others, SpFrom::Ways, 0..150, which);
            let mut cols = [Vec::new(), Vec::new(), Vec::new()];
            for v in s.variants() {
                let c = s.curves(&v);
                let sp = standard::sound_power(s, &v, SpFrom::Ways);
                let preds = [c.pir.clone(), standard::curve(&c, &sp, w_all), standard::curve(&c, &sp, w_out)];
                for (k, p) in preds.iter().enumerate() {
                    cols[k].push(standard::prediction_error(s, &v, p, s.test(), which));
                }
            }
            let fmt = |e: &Vec<(f64, f64)>| format!("{:.2}-{:.2} / {:.2}", e.iter().map(|x| x.0).fold(f64::MAX, f64::min), e.iter().map(|x| x.0).fold(0.0, f64::max), e.iter().map(|x| x.1).fold(0.0, f64::max));
            println!("{:18} {:>18} | {:>18} | {:>18}   (left out: {:.2}/{:.2}/{:.2})", lib.layouts[li].name, fmt(&cols[0]), fmt(&cols[1]), fmt(&cols[2]), w_out[0], w_out[1], w_out[2]);
        }
    }

    println!("\n== Colouration of the reflections (roughness, dB RMS)");
    let shares = standard::bounce_shares(&lib.studies[0]);
    println!("bounce shares (floor, ceiling, side, front, rear): {:.2?}", shares);
    let mut rows = Vec::new();
    for (li, s) in lib.studies.iter().enumerate() {
        for v in s.variants() {
            let c = s.curves(&v);
            let r = [
                standard::room_colouration(s, &v, Group::Steady),
                standard::room_colouration(s, &v, Group::Reflections),
                standard::room_colouration(s, &v, Group::Vertical),
                standard::orbit_index(s, &v, &shares, None),
                standard::orbit_index(s, &v, &shares, Some(&[0, 1])),
                standard::curve_roughness(s, &c.pir),
                standard::curve_roughness(s, &c.er),
                standard::curve_roughness(s, &s.room_response_mean(&v)),
            ];
            println!("{:30} rooms all {:.2} refl {:.2} floor+ceil {:.2} | orbit index {:.2} floor+ceil {:.2} | PIR {:.2} ER {:.2} room curve {:.2}", v, r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]);
            rows.push((li, r));
        }
    }
    let col = |k: usize| rows.iter().map(|(_, r)| r[k]).collect::<Vec<f64>>();
    println!("correlation, first reflections: orbit index {:.3}, ER curve {:.3}, PIR {:.3}, room curve {:.3}", standard::correlation(&col(3), &col(1)), standard::correlation(&col(6), &col(1)), standard::correlation(&col(5), &col(1)), standard::correlation(&col(7), &col(1)));
    println!("correlation, floor and ceiling: orbit index (floor+ceiling) {:.3}", standard::correlation(&col(4), &col(2)));
}
