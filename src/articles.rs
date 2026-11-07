//! The figures of the two blog articles (English, for the web: SVG with the text as paths, a
//! narrower page so that the text reads larger).

use std::iter::once;
use std::path::Path;

use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, VPos};

use crate::analysis::{Study, Which};
use crate::crossover::{self, Kind, KINDS};
use crate::figures::{kind_color, kind_label};
use crate::library_figures::{bar_axes, bar_panel, family_label};
use crate::plot::{band, label, layout, legend, legend_in, line, outline, pt, render, save_web, steps_of, sw, tint, Axes, Dash, Ground, Lang, Place, Swatch, INK, LW, MUTED, SLOTS};
use crate::signal::{db10, log_grid, smooth_power};
use crate::special::percentile;
use crate::speaker::{two_way, FarField};
use crate::standard::{self, Summary};
use crate::study::power;

/// Width of a figure on the blog (inches).
const W: f64 = 6.0;
const EN: Lang = Lang::En;

fn db(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| db10(*x)).collect()
}

/// Where the energy at the listener comes from in the rooms, and what the PIR assumes.
pub fn energy_shares(study: &Study, dir: &Path) {
    let f = study.freqs();
    let j = (0..f.len()).min_by(|&a, &b| (f[a] - 1000.0).abs().partial_cmp(&(f[b] - 1000.0).abs()).unwrap()).unwrap();
    let sh = study.shares(study.random());
    let med = |cats: &[usize]| percentile(&sh.iter().map(|r| cats.iter().map(|&c| r[c][j]).sum::<f64>()).collect::<Vec<_>>(), 50.0);
    let rooms = [med(&[0]), med(&[1, 2, 3, 4, 5, 6]), med(&[7, 8, 9])];
    let s: f64 = rooms.iter().sum();
    let rooms = rooms.map(|v| v / s * 100.0);
    let pir = [12.0, 44.0, 44.0];
    let colours = [tint(SLOTS[0], 0.8), tint(SLOTS[1], 0.8), tint(SLOTS[2], 0.8)];
    // An invisible title keeps room for the legend above the plot.
    let mut ax = Axes::lin(EN, "\u{a0}", (0.0, 100.0), (-0.55, 1.75), "share of the energy (%)", "");
    ax.x_ticks = steps_of(0.0, 100.0, 20.0);
    ax.y_ticks = vec![];
    ax.show_ylabels = false;
    ax.y_grid = false;
    let svg = render(W, 2.3, |root| {
        let mut ch = ax.build(root);
        let bars = [(1.0, "In 300 rooms (median at 1 kHz)", rooms), (0.0, "In CTA-2034's Predicted In-Room Response", pir)];
        for (y, name, vals) in bars {
            let mut x0 = 0.0;
            for (k, v) in vals.iter().enumerate() {
                ch.draw_series(once(Rectangle::new([(x0, y - 0.27), (x0 + v, y + 0.27)], colours[k].filled()))).unwrap();
                label(root, &format!("{:.0} %", v), ch.backend_coord(&(x0 + v / 2.0, y)), 7.5, &INK, HPos::Center, VPos::Center, false);
                x0 += v;
            }
            label(root, name, ch.backend_coord(&(0.0, y + 0.32)), 8.0, &INK, HPos::Left, VPos::Bottom, false);
        }
        outline(&ch);
        let entries = vec![
            (Swatch::Patch(colours[0]), "direct sound / Listening Window".to_string()),
            (Swatch::Patch(colours[1]), "first reflections / Early Reflections".to_string()),
            (Swatch::Patch(colours[2]), "everything later / Sound Power".to_string()),
        ];
        let (xs, ys) = ch.plotting_area().get_pixel_range();
        let _ = ys;
        crate::plot::legend_at(root, (xs.start, 0), &entries, 3, 6.5, Ground::None);
    });
    save_web(&svg, dir, "energy-shares");
}

/// The floor and ceiling directions of CTA-2034's Early Reflections, each with its notch, and the
/// Early Reflections curve that averages them with the walls' directions (two-way, plain dome).
pub fn averaging(dir: &Path) {
    let f = log_grid(300.0, 20000.0, 96);
    let layout_ = two_way();
    let far = FarField::new(&layout_, &f);
    let on = far.on_axis();
    let dirs = crate::cta2034::directions();
    let resp = far.responses(&dirs);
    let vertical: Vec<usize> = [-20, -30, -40, 40, 50, 60].iter().map(|a| crate::cta2034::index_of('v', *a)).collect();
    let axis = crate::cta2034::index_of('h', 0);
    let er = crate::cta2034::kernel("er");
    let kinds = [Kind::Lr2, Kind::Lr4, Kind::Fir];
    let mut axes: Vec<Axes> = kinds
        .iter()
        .enumerate()
        .map(|(i, k)| {
            let mut a = Axes::freq(EN, &kind_label(EN, *k), (700.0, 7000.0), (-30.0, 4.0), if i == 0 { "level re on axis (dB)" } else { "" });
            a.x_ticks = vec![1000.0, 2000.0, 5000.0];
            a.show_ylabels = i == 0;
            a
        })
        .collect();
    let svg = render(W, 2.7, |root| {
        let p = layout(root, 1, 3, &mut axes);
        for (i, k) in kinds.iter().enumerate() {
            let ax = &axes[i];
            let mut ch = ax.build(&p[i]);
            let x = crossover::filters(*k, &f, &layout_, &on);
            let pw = power(&x, &resp);
            let rel = |d: usize| -> Vec<f64> { (0..f.len()).map(|j| pw.row(d)[j] as f64 / pw.row(axis)[j] as f64).collect() };
            let col = kind_color(*k);
            for &d in &vertical {
                line(&mut ch, ax, &f, &db(&rel(d)), col.mix(0.6), 0.7, Dash::Solid, None);
            }
            let curve: Vec<f64> = (0..f.len()).map(|j| er.iter().enumerate().map(|(d, w)| w * pw.row(d)[j] as f64).sum::<f64>() / pw.row(axis)[j] as f64).collect();
            line(&mut ch, ax, &f, &db(&curve), INK, 1.6, Dash::Solid, None);
            if i == 0 {
                let entries = vec![
                    (Swatch::Line(tint(col, 0.6), 0.7, Dash::Solid), "its six floor and ceiling directions".to_string()),
                    (Swatch::Line(INK, 1.6, Dash::Solid), "the Early Reflections curve".to_string()),
                ];
                legend_in(root, &ch, Place::LowerLeft, &entries, 1, 6.5, Ground::Plain);
            }
        }
    });
    save_web(&svg, dir, "averaging");
}

/// The two-orbit sound power of the two-way against the sphere's.
pub fn sound_power(study: &Study, dir: &Path) {
    let f = study.freqs();
    let mut ax = Axes::freq(EN, "", (300.0, 20000.0), (-0.1, 2.0), "two orbits − sphere (dB)");
    ax.y_ticks = steps_of(0.0, 2.0, 0.5);
    let svg = render(W, 2.5, |root| {
        let mut ch = ax.build(root);
        for k in KINDS {
            let c = study.curves(&format!("two_way-upright-{}", k.name()));
            let d: Vec<f64> = c.sp.iter().zip(&c.sp_sphere).map(|(a, b)| db10(*a) - db10(*b)).collect();
            line(&mut ch, &ax, f, &d, kind_color(k), LW, Dash::Solid, Some(&kind_label(EN, k)));
        }
        crate::plot::hline(&mut ch, &ax, 0.0, MUTED, 0.6);
        legend(&mut ch, SeriesLabelPosition::UpperRight, 7.0);
    });
    save_web(&svg, dir, "sound-power");
}

/// The rooms' response against the CTA-2034 PIR and the proposed curve, two-way with LR4.
pub fn pir_vs_rooms(study: &Study, dir: &Path) {
    let f = study.freqs();
    let v = "two_way-upright-lr4";
    let c = study.curves(v);
    let rooms: Vec<Vec<f64>> = study.rooms(v, Which::Steady, study.random()).iter().map(|r| db(r)).collect();
    let pct = |q: f64| (0..f.len()).map(|j| percentile(&rooms.iter().map(|r| r[j]).collect::<Vec<_>>(), q)).collect::<Vec<f64>>();
    let sp = standard::sound_power(study, v, standard::SpFrom::Ways);
    let proposed = standard::curve(&c, &sp, standard::PROPOSED_STEADY);
    let sm = |p: &[f64]| db(&smooth_power(f, p, 12.0));
    let ax = Axes::freq(EN, "", (150.0, 20000.0), (-12.0, 1.0), "level re omnidirectional source (dB)");
    let svg = render(W, 2.9, |root| {
        let mut ch = ax.build(root);
        band(&mut ch, &ax, f, &pct(10.0), &pct(90.0), SLOTS[2], 0.25, Some("300 rooms, 10 to 90 %"));
        line(&mut ch, &ax, f, &pct(50.0), SLOTS[2], LW, Dash::Solid, Some("300 rooms, median"));
        line(&mut ch, &ax, f, &sm(&c.pir), INK, LW, Dash::Dashed, Some("CTA-2034 PIR"));
        line(&mut ch, &ax, f, &sm(&proposed), INK, LW, Dash::Dotted, Some("0.1 LW + 0.9 SP"));
        legend(&mut ch, SeriesLabelPosition::LowerLeft, 7.0);
    });
    save_web(&svg, dir, "pir-vs-rooms");
}

/// The sound power from two orbits, every loudspeaker of the library.
pub fn sp_library(s: &Summary, dir: &Path) {
    let names: Vec<String> = s.families.iter().map(|f| family_label(EN, &f.name)).collect();
    let values: Vec<Vec<f64>> = s.families.iter().map(|f| vec![f.sp_rms[0], f.sp_rms[2]]).collect();
    let colours = [SLOTS[1], SLOTS[0]];
    let entries = vec![(Swatch::Patch(colours[0]), "CTA-2034's weights".to_string()), (Swatch::Patch(colours[1]), "driver by driver, with phase".to_string())];
    let ax = bar_axes(EN, "", (0.0, 1.8), "sound power from two orbits,\nerror against the sphere (dB RMS)", names.len(), 2, 6.5);
    let svg = render(W, 2.6, |root| bar_panel(root, root, &ax, &values, &colours, &names, &entries, 6.5));
    save_web(&svg, dir, "sp-library");
}

/// How well the PIR and the proposed curves predict the rooms, every loudspeaker.
pub fn prediction_library(s: &Summary, dir: &Path) {
    let names: Vec<String> = s.families.iter().map(|f| family_label(EN, &f.name)).collect();
    let steady: Vec<Vec<f64>> = s.families.iter().map(|f| vec![f.steady[0], f.steady[2]]).collect();
    let early: Vec<Vec<f64>> = s.families.iter().map(|f| f.early.to_vec()).collect();
    let colours = [SLOTS[1], SLOTS[0]];
    let e1 = vec![(Swatch::Patch(colours[0]), "CTA-2034 PIR".to_string()), (Swatch::Patch(colours[1]), "0.1 LW + 0.9 SP".to_string())];
    let e2 = vec![(Swatch::Patch(colours[0]), "CTA-2034 PIR".to_string()), (Swatch::Patch(colours[1]), "0.4 LW + 0.4 ER + 0.2 SP".to_string())];
    let ylabel = "error on 150 rooms,\nmedian RMS (dB)";
    let mut axes = vec![
        bar_axes(EN, "steady state", (0.0, 3.0), ylabel, names.len(), 2, 6.5),
        bar_axes(EN, "direct sound and first reflections", (0.0, 3.0), ylabel, names.len(), 2, 6.5),
    ];
    let svg = render(W, 4.6, |root| {
        let p = layout(root, 2, 1, &mut axes);
        bar_panel(root, &p[0], &axes[0], &steady, &colours, &names, &e1, 6.5);
        bar_panel(root, &p[1], &axes[1], &early, &colours, &names, &e2, 6.5);
    });
    save_web(&svg, dir, "prediction-library");
}

/// The colouration of the reflections: read on the orbits, and the roughness of the ER curve.
pub fn index_library(s: &Summary, dir: &Path) {
    let xl = "rooms: colouration of the reflections (dB)";
    let mut axes = vec![
        Axes::lin(EN, &format!("read direction by direction (r = {:.2})", s.corr[0]), (0.0, 2.2), (0.0, 2.2), xl, "estimate from the orbits (dB)"),
        Axes::lin(EN, &format!("the ER curve (r = {:.2})", s.corr[1]), (0.0, 2.2), (0.0, 2.2), xl, ""),
    ];
    for a in axes.iter_mut() {
        a.x_ticks = steps_of(0.0, 2.0, 0.5);
        a.y_ticks = steps_of(0.0, 2.0, 0.5);
        a.title_size = 8.0;
    }
    axes[1].show_ylabels = false;
    let svg = render(W, 3.3, |root| {
        let p = layout(root, 1, 2, &mut axes);
        for (i, ax) in axes.iter().enumerate() {
            let mut ch = ax.build(&p[i]);
            ch.draw_series(once(PathElement::new(vec![(0.0, 0.0), (2.2, 2.2)], MUTED.stroke_width(sw(0.6))))).unwrap();
            let mut entries = Vec::new();
            for (fi, fam) in s.families.iter().enumerate() {
                let col = SLOTS[fi % SLOTS.len()];
                ch.draw_series(fam.kinds.iter().map(|k| Circle::new((k.rooms[1], if i == 0 { k.index[0] } else { k.curves[1] }), pt(2.6), col.filled()))).unwrap();
                entries.push((Swatch::Dot(col, 2.6), family_label(EN, &fam.name).replace('\n', " ")));
            }
            if i == 1 {
                legend_in(root, &ch, Place::UpperLeft, &entries, 1, 6.5, Ground::Plain);
            }
        }
    });
    save_web(&svg, dir, "index-library");
}

/// Every figure of both articles: ``dir/standard-problems`` and ``dir/standard-proposal``.
pub fn all(study: &Study, s: &Summary, dir: &Path) {
    let a = dir.join("standard-problems");
    energy_shares(study, &a);
    averaging(&a);
    sound_power(study, &a);
    pir_vs_rooms(study, &a);
    let b = dir.join("standard-proposal");
    sp_library(s, &b);
    prediction_library(s, &b);
    index_library(s, &b);
}
