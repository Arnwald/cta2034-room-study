//! LaTeX table rows and the numbers quoted in the report (``\n...`` macros), in one language.

use std::collections::BTreeMap;
use std::path::Path;

use crate::analysis::{self, Group, Metric, Sp, Study, Which};
use crate::coherent::{self as coh_mod, CoherentResults, OMNI};
use crate::standard::Summary;
use crate::crossover::{Kind, KINDS};
use crate::figures::{colouration_values, kind_label, name, notch_width, surf_label};
use crate::plot::Lang;
use crate::signal::db10;
use crate::special::{interp, median, percentile};
use crate::sphere;

fn write(dir: &Path, name: &str, text: &str) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(format!("{name}.tex")), text).unwrap();
}

fn kind_word(k: Kind) -> &'static str {
    match k {
        Kind::Lr2 => "LrTwo",
        Kind::Lr4 => "LrFour",
        Kind::Lr8 => "LrEight",
        Kind::Fir => "Fir",
    }
}

fn title_case(s: &str) -> String {
    let s = s.replace('_', "");
    let mut c = s.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

fn min_max(v: &[f64]) -> (f64, f64) {
    (v.iter().cloned().fold(f64::INFINITY, f64::min), v.iter().cloned().fold(f64::NEG_INFINITY, f64::max))
}

fn closest(f: &[f64], x: f64) -> usize {
    (0..f.len()).min_by(|&a, &b| (f[a] - x).abs().partial_cmp(&(f[b] - x).abs()).unwrap()).unwrap()
}

pub use crate::figures::Spectra;

/// Writes every table and ``numbers.tex``; returns the numbers.
pub fn all(study: &Study, lang: Lang, dir: &Path, spectra: &Spectra, coh: Option<&CoherentResults>, lib: Option<&Summary>) -> BTreeMap<String, String> {
    let res = &study.res;
    let f = study.freqs();
    let n = res.n_random;
    let num = |x: f64, p: usize| lang.tex(x, p);
    let mut numbers: BTreeMap<String, String> = BTreeMap::new();

    // The ensemble.
    let meta = &res.meta[..n];
    let rt = &res.rt60[..n];
    let rows: Vec<(String, Vec<f64>)> = vec![
        (lang.t("width (m)", "largeur (m)"), meta.iter().map(|m| m.dims[0]).collect()),
        (lang.t("length (m)", "longueur (m)"), meta.iter().map(|m| m.dims[1]).collect()),
        (lang.t("height (m)", "hauteur (m)"), meta.iter().map(|m| m.dims[2]).collect()),
        ("volume (m$^3$)".into(), meta.iter().map(|m| m.dims[0] * m.dims[1] * m.dims[2]).collect()),
        (lang.t("listening distance (m)", "distance d’écoute (m)"), meta.iter().map(|m| m.listen_distance).collect()),
        (lang.t("loudspeaker spacing (m)", "écartement des enceintes (m)"), meta.iter().map(|m| m.spacing).collect()),
        (lang.t("baffle to wall behind (m)", "distance du baffle au mur (m)"), meta.iter().map(|m| m.wall_distance).collect()),
        (lang.t("$T_{60}$ at 500 Hz (s)", "$T_{60}$ à 500 Hz (s)"), rt.iter().map(|r| r[2]).collect()),
        (lang.t("$T_{60}$ at 1 kHz (s)", "$T_{60}$ à 1 kHz (s)"), rt.iter().map(|r| r[3]).collect()),
        (lang.t("$T_{60}$ at 4 kHz (s)", "$T_{60}$ à 4 kHz (s)"), rt.iter().map(|r| r[5]).collect()),
    ];
    let body: Vec<String> = rows
        .iter()
        .map(|(l, v)| format!("{l} & {} & {} & {} \\\\", num(percentile(v, 10.0), 2), num(median(v), 2), num(percentile(v, 90.0), 2)))
        .collect();
    write(dir, "ensemble", &(body.join("\n") + "\n"));
    let rt1k: Vec<f64> = rt.iter().map(|r| r[3]).collect();
    numbers.insert("rtMedian".into(), num(median(&rt1k), 2));
    numbers.insert("rtLo".into(), num(percentile(&rt1k, 10.0), 2));
    numbers.insert("rtHi".into(), num(percentile(&rt1k, 90.0), 2));
    numbers.insert("treated".into(), format!("{}", meta.iter().filter(|m| m.panels).count()));

    // Energy shares.
    let sh = study.shares(study.random());
    let first = lang.t("first order", "premier ordre");
    let groups: Vec<(String, Vec<usize>)> = vec![
        (lang.t("direct sound", "son direct"), vec![0]),
        (format!("{first}: {}", surf_label(lang, "floor")), vec![5]),
        (format!("{first}: {}", surf_label(lang, "ceiling")), vec![6]),
        (format!("{first}: {}", surf_label(lang, "left")), vec![1]),
        (format!("{first}: {}", surf_label(lang, "right")), vec![2]),
        (format!("{first}: {}", surf_label(lang, "listener_wall")), vec![3]),
        (format!("{first}: {}", surf_label(lang, "speaker_wall")), vec![4]),
        (lang.t("second and third order", "deuxième et troisième ordres"), vec![7, 8]),
        (lang.t("diffuse field", "champ diffus"), vec![9]),
    ];
    let (j1k, j4k) = (closest(f, 1000.0), closest(f, 4000.0));
    let mut out = Vec::new();
    for (l, idx) in &groups {
        let cells: Vec<String> = [j1k, j4k]
            .iter()
            .map(|&j| {
                let v: Vec<f64> = sh.iter().map(|r| idx.iter().map(|&k| r[k][j]).sum::<f64>() * 100.0).collect();
                format!("{} & {}--{}", num(median(&v), 1), num(percentile(&v, 10.0), 1), num(percentile(&v, 90.0), 1))
            })
            .collect();
        out.push(format!("{l} & {} \\\\", cells.join(" & ")));
    }
    write(dir, "shares", &(out.join("\n") + "\n"));
    let first_e: Vec<f64> = sh.iter().map(|r| (1..7).map(|k| r[k][j1k]).sum()).collect();
    let vert: Vec<f64> = sh.iter().zip(&first_e).map(|(r, fe)| (r[5][j1k] + r[6][j1k]) / fe).collect();
    let side: Vec<f64> = sh.iter().zip(&first_e).map(|(r, fe)| (r[1][j1k] + r[2][j1k]) / fe).collect();
    let vt: Vec<f64> = vert.iter().zip(meta).filter(|(_, m)| m.panels).map(|(v, _)| *v).collect();
    let vu: Vec<f64> = vert.iter().zip(meta).filter(|(_, m)| !m.panels).map(|(v, _)| *v).collect();
    let late: Vec<f64> = sh.iter().map(|r| (7..10).map(|k| r[k][j1k]).sum()).collect();
    numbers.insert("directShare".into(), num(median(&sh.iter().map(|r| r[0][j1k]).collect::<Vec<_>>()) * 100.0, 1));
    numbers.insert("firstShare".into(), num(median(&first_e) * 100.0, 0));
    numbers.insert("lateShare".into(), num(median(&late) * 100.0, 0));
    numbers.insert("vertShare".into(), num(median(&vert) * 100.0, 0));
    numbers.insert("vertLo".into(), num(percentile(&vert, 10.0) * 100.0, 0));
    numbers.insert("vertHi".into(), num(percentile(&vert, 90.0) * 100.0, 0));
    numbers.insert("sideShare".into(), num(median(&side) * 100.0, 0));
    numbers.insert("vertTreated".into(), num(median(&vt) * 100.0, 0));
    numbers.insert("vertUntreated".into(), num(median(&vu) * 100.0, 0));
    let early: Vec<Vec<f64>> = sh
        .iter()
        .map(|r| {
            let t: f64 = (0..7).map(|k| r[k][j1k]).sum();
            (0..7).map(|k| r[k][j1k] / t).collect()
        })
        .collect();
    numbers.insert("earlyDirect".into(), num(median(&early.iter().map(|e| e[0]).collect::<Vec<_>>()) * 100.0, 0));
    numbers.insert("earlyVert".into(), num(median(&early.iter().map(|e| e[5] + e[6]).collect::<Vec<_>>()) * 100.0, 0));

    // Reflection directions.
    let end = res.image_starts[n];
    let to = lang.t("to", "à");
    let cta_angles = |s: &str| -> &'static str {
        match s {
            "floor" => "$-20, -30, -40$ (V)",
            "ceiling" => "$40, 50, 60$ (V)",
            "left" | "right" => "$\\pm 40 \\ldots \\pm 80$ (H)",
            "listener_wall" => "$0, \\pm 10, \\pm 20, \\pm 30$ (H)",
            _ => "$\\pm 90, 180$ (H)",
        }
    };
    let mut rows = Vec::new();
    for (c, surf) in [(5u8, "floor"), (6u8, "ceiling"), (1u8, "left"), (2u8, "right"), (3u8, "listener_wall"), (4u8, "speaker_wall")] {
        let mut ang = Vec::new();
        let mut del = Vec::new();
        for i in 0..end {
            if res.image_cat[i] == c {
                let d = res.image_dirs[i];
                let (az, el) = sphere::to_angles(&[d[0] as f64, d[1] as f64, d[2] as f64]);
                ang.push(if surf == "floor" || surf == "ceiling" { el } else { az });
                del.push(res.image_delay[i] as f64 * 1e3);
            }
        }
        let kind = if surf == "floor" || surf == "ceiling" { lang.t("elevation", "élévation") } else { lang.t("azimuth", "azimut") };
        rows.push(format!(
            "{} & {kind} & ${:.0}$ & ${:.0}$ {to} ${:.0}$ & {} & {} \\\\",
            surf_label(lang, surf),
            median(&ang),
            percentile(&ang, 10.0),
            percentile(&ang, 90.0),
            cta_angles(surf),
            num(median(&del), 1)
        ));
        numbers.insert(format!("delay{}", title_case(surf)), num(median(&del), 1));
        numbers.insert(format!("angle{}", title_case(surf)), format!("{:.0}", median(&ang)));
    }
    write(dir, "directions", &(rows.join("\n") + "\n"));

    // Weights and prediction errors.
    let w = analysis::fit_weights(study, Sp::Sphere, Which::Steady);
    let w2 = analysis::fit_weights(study, Sp::TwoOrbits, Which::Steady);
    let we = analysis::fit_weights(study, Sp::Sphere, Which::Early);
    for (k, v) in [
        ("wLw", w[0]),
        ("wEr", w[1]),
        ("wSp", w[2]),
        ("wLwTwo", w2[0]),
        ("wErTwo", w2[1]),
        ("wSpTwo", w2[2]),
        ("wLwEarly", we[0]),
        ("wErEarly", we[1]),
        ("wSpEarly", we[2]),
    ] {
        numbers.insert(k.into(), num(v, 2));
    }
    let preds: Vec<(String, [f64; 3], Sp)> = vec![
        (lang.t("CTA-2034 PIR, $0.12/0.44/0.44$, two orbits", "PIR CTA-2034, $0,12/0,44/0,44$, deux orbites"), [0.12, 0.44, 0.44], Sp::TwoOrbits),
        (lang.t("same weights, sound power of the sphere", "mêmes poids, puissance sur la sphère"), [0.12, 0.44, 0.44], Sp::Sphere),
        (format!("{}, ${}/{}/{}$", lang.t("fitted, two orbits", "ajustés, deux orbites"), num(w2[0], 2), num(w2[1], 2), num(w2[2], 2)), w2, Sp::TwoOrbits),
        (format!("{}, ${}/{}/{}$", lang.t("fitted, sphere", "ajustés, sphère"), num(w[0], 2), num(w[1], 2), num(w[2], 2)), w, Sp::Sphere),
    ];
    let m = analysis::band(f, analysis::ERROR_BAND);
    let mut rows = Vec::new();
    for (l, weights, sp) in &preds {
        let mut rms_all = Vec::new();
        let mut bias_all = Vec::new();
        for v in study.variants() {
            let (rms, bias) = analysis::errors(study, &v, &analysis::pir(&study.curves(&v), *weights, *sp), study.test(), Which::Steady);
            rms_all.push(median(&rms));
            bias_all.push(bias.iter().zip(&m).filter(|(_, b)| **b).map(|(x, _)| x.abs()).fold(0.0, f64::max));
        }
        let (rn, rx) = min_max(&rms_all);
        let (bn, bx) = min_max(&bias_all);
        rows.push(format!("{l} & {}--{} & {}--{} \\\\", num(rn, 2), num(rx, 2), num(bn, 2), num(bx, 2)));
    }
    let floor: Vec<f64> = study
        .variants()
        .iter()
        .map(|v| {
            let rs = study.rooms(v, Which::Steady, study.train());
            let mean_train: Vec<f64> = (0..f.len()).map(|j| rs.iter().map(|r| r[j]).sum::<f64>() / rs.len() as f64).collect();
            median(&analysis::errors(study, v, &mean_train, study.test(), Which::Steady).0)
        })
        .collect();
    let (fn_, fx) = min_max(&floor);
    rows.push(format!(
        "{} & {}--{} & -- \\\\",
        lang.t("mean of the 150 training rooms (room-to-room floor)", "moyenne des 150 pièces d’apprentissage (plancher entre pièces)"),
        num(fn_, 2),
        num(fx, 2)
    ));
    write(dir, "errors", &(rows.join("\n") + "\n"));

    // Two orbits against the sphere at 2 kHz.
    let jx = closest(f, 2000.0);
    for k in KINDS {
        let c = study.curves(&name("two_way", "upright", k));
        numbers.insert(format!("spBias{}", kind_word(k)), num(db10(c.sp[jx]) - db10(c.sp_sphere[jx]), 1));
    }

    // Averaged curves within an octave of the crossover.
    let mut rows = Vec::new();
    let mut curve_vals: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for layout in ["two_way", "two_way_wg"] {
        for orient in ["upright", "side"] {
            for k in KINDS {
                let v = name(layout, orient, k);
                let c = study.curves(&v);
                let pf = analysis::pir(&c, w, Sp::Sphere);
                let steady = study.rooms(&v, Which::Steady, study.random());
                let early_r = study.rooms(&v, Which::Early, study.random());
                let vals = vec![
                    analysis::local_anomaly(f, &c.pir),
                    analysis::local_anomaly(f, &pf),
                    median(&steady.iter().map(|r| analysis::local_anomaly(f, r)).collect::<Vec<_>>()),
                    median(&early_r.iter().map(|r| analysis::local_anomaly(f, r)).collect::<Vec<_>>()),
                    analysis::notch_depth(f, &c.pir),
                    median(&steady.iter().map(|r| analysis::notch_depth(f, r)).collect::<Vec<_>>()),
                ];
                let short = if layout == "two_way" { lang.t("dome", "dôme") } else { lang.t("waveguide", "guide d’onde") };
                let o = if orient == "upright" { lang.t("upright", "debout") } else { lang.t("on its side", "couchée") };
                rows.push(format!("{short}, {o}, {} & {} \\\\", kind_label(lang, k), vals.iter().map(|x| num(*x, 2)).collect::<Vec<_>>().join(" & ")));
                curve_vals.insert(v, vals);
            }
            rows.push("\\addlinespace".into());
        }
    }
    rows.pop();
    write(dir, "anomalies", &(rows.join("\n") + "\n"));
    for (layout, tag) in [("two_way", "Dome"), ("two_way_wg", "Wg")] {
        for k in KINDS {
            let vals = &curve_vals[&name(layout, "upright", k)];
            for (i, key) in ["Pir", "Fit", "Room", "Early", "PirDip", "RoomDip"].iter().enumerate() {
                numbers.insert(format!("curve{tag}{}{key}", kind_word(k)), num(vals[i], 2));
            }
        }
    }

    // Colouration of the paths.
    let mut rows = Vec::new();
    let mut col_vals: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for layout in ["two_way", "two_way_wg"] {
        for k in KINDS {
            let v = name(layout, "upright", k);
            let mut vals = colouration_values(study, &v, w).to_vec();
            let dip = median(&analysis::path_anomaly(study, &v, Group::Vertical, study.random(), Metric::Notch, true));
            let short = if layout == "two_way" { lang.t("dome", "dôme") } else { lang.t("waveguide", "guide d’onde") };
            rows.push(format!(
                "{short}, {} & {} & {} \\\\",
                kind_label(lang, k),
                vals.iter().map(|x| num(*x, 2)).collect::<Vec<_>>().join(" & "),
                num(dip, 1)
            ));
            vals.push(dip);
            col_vals.insert(v, vals);
        }
        rows.push("\\addlinespace".into());
    }
    rows.pop();
    write(dir, "colouration", &(rows.join("\n") + "\n"));
    for (layout, tag) in [("two_way", "Dome"), ("two_way_wg", "Wg")] {
        let base = col_vals[&name(layout, "upright", Kind::Lr4)].clone();
        for k in KINDS {
            let vals = &col_vals[&name(layout, "upright", k)];
            for (i, key) in ["Pir", "Fit", "Steady", "Refl", "Vert", "VertDip"].iter().enumerate() {
                numbers.insert(format!("col{tag}{}{key}", kind_word(k)), num(vals[i], if *key == "VertDip" { 1 } else { 2 }));
                if k == Kind::Fir {
                    numbers.insert(format!("col{tag}Red{key}"), num((1.0 - vals[i] / base[i]) * 100.0, 0));
                }
                if k == Kind::Lr2 && *key != "VertDip" {
                    numbers.insert(format!("col{tag}LrTwoMore{key}"), num((vals[i] / base[i] - 1.0) * 100.0, 0));
                }
            }
        }
    }

    // The PIR above the rooms.
    let v = name("two_way", "upright", Kind::Lr4);
    let (_, bias) = analysis::errors(study, &v, &study.curves(&v).pir, study.random(), Which::Steady);
    numbers.insert("pirBiasMax".into(), num(bias.iter().zip(&m).filter(|(_, b)| **b).map(|(x, _)| *x).fold(f64::NEG_INFINITY, f64::max), 1));
    numbers.insert("pirBiasLow".into(), num(interp(300.0, f, &bias), 1));

    // The living room's floor and ceiling bounces.
    let (fs, per_layout) = spectra;
    for (layout, per_kind) in per_layout {
        let tag = if layout == "two_way" { "Dome" } else { "Wg" };
        for (k, paths) in per_kind {
            if *k == Kind::Lr8 {
                continue;
            }
            for path in ["floor", "ceiling"] {
                let p = &paths.iter().find(|x| x.0 == path).unwrap().1;
                let (width, _, depth) = notch_width(fs, p, 6.0);
                numbers.insert(format!("notch{tag}{}{}Width", kind_word(*k), title_case(path)), num(width, 0));
                numbers.insert(format!("notch{tag}{}{}Depth", kind_word(*k), title_case(path)), num(depth, 0));
            }
        }
    }

    // The notches the Early Reflections curve averages: each floor and ceiling direction's depth
    // below the axis and frequency, the curve's own dip below its straight line.
    for k in KINDS {
        let (dirs, dip) = crate::figures::er_notches(k);
        let (dn, dx) = min_max(&dirs.iter().map(|d| d.0).collect::<Vec<_>>());
        let (fn_, fx) = min_max(&dirs.iter().map(|d| d.1 / 1000.0).collect::<Vec<_>>());
        numbers.insert(format!("erDir{}DepthMin", kind_word(k)), num(dn, 0));
        numbers.insert(format!("erDir{}DepthMax", kind_word(k)), num(dx, 0));
        numbers.insert(format!("erDir{}FreqMin", kind_word(k)), num(fn_, 1));
        numbers.insert(format!("erDir{}FreqMax", kind_word(k)), num(fx, 1));
        numbers.insert(format!("erCurve{}Dip", kind_word(k)), num(dip, 1));
    }

    // The living room's averaged response: its deepest dip below the straight line.
    let living = study.named_index("living");
    for (layout, tag) in [("two_way", "Dome"), ("two_way_wg", "Wg")] {
        for k in KINDS {
            let r = study.room_response(&name(layout, "upright", k), Which::Steady, living);
            numbers.insert(format!("living{tag}{}Dip", kind_word(k)), num(analysis::notch_depth(f, &r), 1));
        }
    }

    if let Some(coh) = coh {
        coherent(coh, lang, dir, &mut numbers);
    }
    if let Some(lib) = lib {
        library(lib, lang, dir, &mut numbers);
    }

    let text: Vec<String> = numbers.iter().map(|(k, v)| format!("\\newcommand{{\\n{k}}}{{{v}}}")).collect();
    write(dir, "numbers", &(text.join("\n") + "\n"));
    numbers
}

/// The table of the paths added with their phases, and its numbers (``coh...`` macros).
fn coherent(coh: &CoherentResults, lang: Lang, dir: &Path, numbers: &mut BTreeMap<String, String>) {
    let num = |x: f64| lang.tex(x, 2);
    let keys = ["Energy", "Point", "Ears", "Spatial"];
    let mut body = Vec::new();
    let mut rows: Vec<(String, String, String)> = Vec::new();
    for layout in ["two_way", "two_way_wg"] {
        for k in KINDS {
            let tag = if layout == "two_way" { "Dome" } else { "Wg" };
            let label = format!("{}, {}", if layout == "two_way" { lang.t("dome", "dôme") } else { lang.t("waveguide", "guide d’onde") }, kind_label(lang, k));
            rows.push((name(layout, "upright", k), label, format!("{tag}{}", kind_word(k))));
        }
    }
    rows.push((OMNI.into(), lang.t("omnidirectional source", "source omnidirectionnelle"), "Omni".into()));
    for (i, (variant, label, tag)) in rows.iter().enumerate() {
        let steady = coh_mod::values(coh, variant, 0);
        let early = coh_mod::values(coh, variant, 1);
        for (which, vals) in [("Steady", steady), ("Early", early)] {
            for (key, v) in keys.iter().zip(vals) {
                numbers.insert(format!("coh{tag}{which}{key}"), num(v));
            }
        }
        if i == 4 || i == 8 {
            body.push("\\addlinespace".to_string());
        }
        let cells: Vec<String> = steady.iter().chain(&early).map(|v| num(*v)).collect();
        body.push(format!("{label} & {} \\\\", cells.join(" & ")));
    }
    write(dir, "coherent", &(body.join("\n") + "\n"));
    numbers.insert("cohCheck".into(), format!("10^{{{}}}", coh.check_db.log10().ceil()));
    // What the phases add in steady state at one point, from the gentlest to the largest.
    let gains: Vec<f64> = rows[..8].iter().map(|(v, _, _)| {
        let s = coh_mod::values(coh, v, 0);
        s[1] - s[0]
    }).collect();
    numbers.insert("cohSteadyGainMin".into(), num(gains.iter().cloned().fold(f64::INFINITY, f64::min)));
    numbers.insert("cohSteadyGainMax".into(), num(gains.iter().cloned().fold(f64::NEG_INFINITY, f64::max)));
    numbers.insert("erbCrossover".into(), format!("{:.0}", coh_mod::erb(2200.0)));
}

/// Tag of a loudspeaker in the macros.
fn family_tag(name: &str) -> &'static str {
    match name {
        "two_way" => "Dome",
        "two_way_wg" => "Wg",
        "coaxial" => "Coax",
        "mtm" => "Mtm",
        "three_way_monitor" => "Monitor",
        "three_way_tower" => "TowerThree",
        _ => "TowerFour",
    }
}

/// The library's table and numbers (``lib...`` macros).
fn library(s: &Summary, lang: Lang, dir: &Path, numbers: &mut BTreeMap<String, String>) {
    let num = |x: f64, p: usize| lang.tex(x, p);
    let layouts = crate::speaker::library();
    let mut body = Vec::new();
    for (f, l) in s.families.iter().zip(&layouts) {
        let name = crate::library_figures::family_label(lang, &f.name).replace('\n', " ");
        let c = &l.cabinet;
        let dims = format!("{:.0}\\,$\\times$\\,{:.0}\\,$\\times$\\,{:.0}", c.width * 1000.0, c.height * 1000.0, c.depth * 1000.0);
        let xo: Vec<String> = l.crossovers.iter().map(|x| lang.tex(*x, 0)).collect();
        let placed = if l.floor { lang.t("floor", "sol") } else { lang.t("stand", "pied") };
        body.push(format!(
            "{name} & {dims} & {} & {} & {placed} & {} & {} & {} \\\\",
            l.drivers.len(),
            xo.join(", "),
            num(f.far[0], 2),
            num(f.far[1], 2),
            num(f.sp_rms[0], 2),
        ));
        let tag = family_tag(&f.name);
        numbers.insert(format!("lib{tag}SpCta"), num(f.sp_rms[0], 2));
        numbers.insert(format!("lib{tag}SpCtaMax"), num(f.sp_max[0], 1));
        numbers.insert(format!("lib{tag}SpRing"), num(f.sp_rms[1], 2));
        numbers.insert(format!("lib{tag}SpWays"), num(f.sp_rms[2], 2));
        numbers.insert(format!("lib{tag}FarSteady"), num(f.far[0], 2));
        numbers.insert(format!("lib{tag}FarEarly"), num(f.far[1], 2));
        for k in &f.kinds {
            let kw = kind_word(Kind::from_name(k.variant.rsplit_once('-').unwrap().1).unwrap());
            for (key, v) in [("All", k.rooms[0]), ("Refl", k.rooms[1]), ("Vert", k.rooms[2]), ("Index", k.index[0]), ("Er", k.curves[1]), ("Pir", k.curves[0]), ("Room", k.curves[2])] {
                numbers.insert(format!("lib{tag}{kw}{key}"), num(v, 2));
            }
        }
    }
    write(dir, "library", &(body.join("\n") + "\n"));
    let range = |v: Vec<f64>| (v.iter().cloned().fold(f64::MAX, f64::min), v.iter().cloned().fold(f64::MIN, f64::max));
    let (a, b) = range(s.families.iter().map(|f| f.sp_rms[0]).collect());
    numbers.insert("libSpCtaMin".into(), num(a, 2));
    numbers.insert("libSpCtaMax".into(), num(b, 2));
    numbers.insert("libSpCtaPeak".into(), num(s.families.iter().map(|f| f.sp_max[0]).fold(0.0, f64::max), 1));
    let (a, b) = range(s.families.iter().map(|f| f.sp_rms[2]).collect());
    numbers.insert("libSpWaysMin".into(), num(a, 2));
    numbers.insert("libSpWaysMax".into(), num(b, 2));
    numbers.insert("libSpWaysPeak".into(), num(s.families.iter().map(|f| f.sp_max[2]).fold(0.0, f64::max), 2));
    for (k, w) in ["Lw", "Er", "Sp"].iter().enumerate() {
        numbers.insert(format!("libSteady{w}"), num(s.steady_fit[k], 2));
        numbers.insert(format!("libEarly{w}"), num(s.early_fit[k], 2));
    }
    // The ER weight with its sign, for ``a LW <sign> b ER + c SP`` in math mode.
    let signed = |x: f64| format!("{} {}", if x < 0.0 { "-" } else { "+" }, num(x.abs(), 2));
    numbers.insert("libSteadyErSigned".into(), signed(s.steady_fit[1]));
    numbers.insert("libEarlyErSigned".into(), signed(s.early_fit[1]));
    numbers.insert("libLofoLwMin".into(), num(s.lofo_lw.0, 2));
    numbers.insert("libLofoLwMax".into(), num(s.lofo_lw.1, 2));
    numbers.insert("libLofoSpMin".into(), num(s.lofo_sp.0, 2));
    numbers.insert("libLofoSpMax".into(), num(s.lofo_sp.1, 2));
    for (name, lo, hi, bias) in &s.errors {
        let key = title_case(name);
        numbers.insert(format!("lib{key}Min"), num(*lo, 2));
        numbers.insert(format!("lib{key}Max"), num(*hi, 2));
        numbers.insert(format!("lib{key}Bias"), num(*bias, 1));
    }
    for (k, name) in ["Floor", "Ceiling", "Side", "Front", "Rear"].iter().enumerate() {
        numbers.insert(format!("libShare{name}"), num(s.shares[k] * 100.0, 0));
    }
    for (k, name) in ["Index", "Er", "Pir", "Room", "IndexVert"].iter().enumerate() {
        numbers.insert(format!("libCorr{name}"), num(s.corr[k], 2));
    }
}
