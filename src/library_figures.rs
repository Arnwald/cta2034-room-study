//! The figures of the loudspeaker library (the two-orbit standard), in English or French.

use std::iter::once;
use std::path::Path;

use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, VPos};

use crate::figures::W2;
use crate::plot::{label, layout, legend_in, outline, pt, render, save, steps_of, sw, tint, Axes, Dash, Ground, Lang, Place, Swatch, INK, INK_2, MUTED, PT, SLOTS};
use crate::standard::Summary;

/// A loudspeaker's name, on one or two lines.
pub fn family_label(lang: Lang, name: &str) -> String {
    match name {
        "two_way" => lang.t("two-way", "deux voies"),
        "two_way_wg" => lang.t("two-way,\nwaveguide", "deux voies,\nguide d’onde"),
        "coaxial" => lang.t("coaxial", "coaxiale"),
        "mtm" => "MTM".into(),
        "three_way_monitor" => lang.t("3-way\nmonitor", "moniteur\n3 voies"),
        "three_way_tower" => lang.t("3-way\ntower", "colonne\n3 voies"),
        _ => lang.t("4-way\ntower", "colonne\n4 voies"),
    }
}

/// Grouped bars, one group per loudspeaker, with the names under the axis.
#[allow(clippy::too_many_arguments)]
fn bar_panel<DB: DrawingBackend>(
    root: &DrawingArea<DB, plotters::coord::Shift>,
    area: &DrawingArea<DB, plotters::coord::Shift>,
    ax: &Axes,
    values: &[Vec<f64>],
    colours: &[RGBColor],
    names: &[String],
    entries: &[(Swatch, String)],
    small: f64,
) where
    DB::ErrorType: 'static,
{
    let mut ch = ax.build(area);
    let n = colours.len();
    let width = 0.8 / n as f64;
    for (g, vals) in values.iter().enumerate() {
        for (k, v) in vals.iter().enumerate() {
            let x0 = g as f64 - 0.4 + k as f64 * width;
            ch.draw_series(once(Rectangle::new([(x0, 0.0), (x0 + width, *v)], colours[k].filled()))).unwrap();
        }
    }
    outline(&ch);
    let (_, ys) = ch.plotting_area().get_pixel_range();
    for (g, text) in names.iter().enumerate() {
        let (bx, _) = ch.backend_coord(&(g as f64, 0.0));
        for (li, l) in text.lines().enumerate() {
            let y = ys.end + pt(6.0) + (li as f64 * small * PT * 1.2).round() as i32;
            label(root, l, (bx, y), small, &INK_2, HPos::Center, VPos::Top, false);
        }
    }
    if !entries.is_empty() {
        legend_in(root, &ch, Place::UpperLeft, entries, 1, 6.5, Ground::Plain);
    }
}

fn bar_axes(lang: Lang, title: &str, y: (f64, f64), ylabel: &str, n: usize, lines: usize, small: f64) -> Axes {
    let mut a = Axes::lin(lang, title, (-0.6, n as f64 - 0.4), y, "", ylabel);
    a.x_ticks = steps_of(0.0, n as f64 - 1.0, 1.0);
    a.show_xlabels = false;
    a.x_grid = false;
    a.x_label_area = Some(6.0 + lines as f64 * small * 1.2 + 2.0);
    a
}

/// The sound power from two orbits against the sphere, for every loudspeaker.
pub fn library_sp(s: &Summary, lang: Lang, dir: &Path) {
    let names: Vec<String> = s.families.iter().map(|f| family_label(lang, &f.name)).collect();
    let values: Vec<Vec<f64>> = s.families.iter().map(|f| f.sp_rms.to_vec()).collect();
    let colours = [SLOTS[1], tint(SLOTS[3], 0.85), SLOTS[0]];
    let entries = vec![
        (Swatch::Patch(colours[0]), lang.t("CTA-2034's weights", "poids de la CTA-2034")),
        (Swatch::Patch(colours[1]), lang.t("rings", "anneaux")),
        (Swatch::Patch(colours[2]), lang.t("driver by driver, with phase", "haut-parleur par haut-parleur, avec la phase")),
    ];
    let ax = bar_axes(lang, "", (0.0, 1.8), &lang.t("sound power from two orbits:\nerror against the sphere (dB RMS)", "puissance sur deux orbites~:\nécart à la sphère (dB RMS)").replace("~", "\u{a0}"), names.len(), 2, 6.5);
    let svg = render(W2, 2.6, |root| bar_panel(root, root, &ax, &values, &colours, &names, &entries, 6.5));
    save(&svg, dir, "library_sp");
}

/// How well each curve predicts the test rooms, for every loudspeaker.
pub fn library_prediction(s: &Summary, lang: Lang, dir: &Path) {
    let names: Vec<String> = s.families.iter().map(|f| family_label(lang, &f.name)).collect();
    let steady: Vec<Vec<f64>> = s.families.iter().map(|f| f.steady.to_vec()).collect();
    let early: Vec<Vec<f64>> = s.families.iter().map(|f| f.early.to_vec()).collect();
    let c3 = [SLOTS[1], tint(SLOTS[0], 0.55), SLOTS[0]];
    let c2 = [SLOTS[1], SLOTS[0]];
    let num = |x: f64| lang.num(x, 1);
    let prop = |w: [f64; 3]| {
        if w[1] == 0.0 {
            format!("{} LW + {} SP", num(w[0]), num(w[2]))
        } else {
            format!("{} LW + {} ER + {} SP", num(w[0]), num(w[1]), num(w[2]))
        }
    };
    let e3 = vec![
        (Swatch::Patch(c3[0]), lang.t("CTA-2034 PIR", "PIR de la CTA-2034")),
        (Swatch::Patch(c3[1]), format!("{}, {}", prop(crate::standard::PROPOSED_STEADY), lang.t("CTA-2034's SP", "SP de la CTA-2034"))),
        (Swatch::Patch(c3[2]), format!("{}, {}", prop(crate::standard::PROPOSED_STEADY), lang.t("SP driver by driver", "SP haut-parleur par haut-parleur"))),
    ];
    let e2 = vec![
        (Swatch::Patch(c2[0]), lang.t("CTA-2034 PIR", "PIR de la CTA-2034")),
        (Swatch::Patch(c2[1]), format!("{}, {}", prop(crate::standard::PROPOSED_EARLY), lang.t("SP driver by driver", "SP haut-parleur par haut-parleur"))),
    ];
    let ylabel = lang.t("error on the test rooms,\nmedian RMS (dB)", "écart aux pièces de test,\nRMS médian (dB)");
    let mut axes = vec![
        bar_axes(lang, &lang.t("steady state", "régime établi"), (0.0, 3.2), &ylabel, names.len(), 2, 6.5),
        bar_axes(lang, &lang.t("early sound: direct sound and first reflections", "son précoce~: son direct et premières réflexions").replace("~", "\u{a0}"), (0.0, 3.2), &ylabel, names.len(), 2, 6.5),
    ];
    let svg = render(W2, 4.8, |root| {
        let p = layout(root, 2, 1, &mut axes);
        bar_panel(root, &p[0], &axes[0], &steady, &c3, &names, &e3, 6.5);
        bar_panel(root, &p[1], &axes[1], &early, &c2, &names, &e2, 6.5);
    });
    save(&svg, dir, "library_prediction");
}

/// The colouration of the reflections read on the orbits, and the roughness of the ER curve,
/// against the rooms.
pub fn library_index(s: &Summary, lang: Lang, dir: &Path) {
    let xl = lang.t("rooms: colouration of the first reflections (dB RMS)", "pièces~: coloration des premières réflexions (dB RMS)").replace("~", "\u{a0}");
    let r = |x: f64| lang.num(x, 2);
    let mut axes = vec![
        Axes::lin(lang, &format!("{} (r = {})", lang.t("read on the orbits, direction by direction", "lue sur les orbites, direction par direction"), r(s.corr[0])), (0.0, 2.2), (0.0, 2.2), &xl, &lang.t("estimate (dB RMS)", "estimation (dB RMS)")),
        Axes::lin(lang, &format!("{} (r = {})", lang.t("roughness of the ER curve", "rugosité de la courbe ER"), r(s.corr[1])), (0.0, 2.2), (0.0, 2.2), &xl, ""),
    ];
    for a in axes.iter_mut() {
        a.x_ticks = steps_of(0.0, 2.0, 0.5);
        a.y_ticks = steps_of(0.0, 2.0, 0.5);
        a.title_size = 8.0;
    }
    axes[1].show_ylabels = false;
    let svg = render(W2, 3.4, |root| {
        let p = layout(root, 1, 2, &mut axes);
        for (i, ax) in axes.iter().enumerate() {
            let mut ch = ax.build(&p[i]);
            ch.draw_series(once(PathElement::new(vec![(0.0, 0.0), (2.2, 2.2)], MUTED.stroke_width(sw(0.6))))).unwrap();
            let mut entries = Vec::new();
            for (fi, f) in s.families.iter().enumerate() {
                let col = SLOTS[fi % SLOTS.len()];
                ch.draw_series(f.kinds.iter().map(|k| Circle::new((k.rooms[1], if i == 0 { k.index[0] } else { k.curves[1] }), pt(2.6), col.filled()))).unwrap();
                entries.push((Swatch::Dot(col, 2.6), family_label(lang, &f.name).replace('\n', " ")));
            }
            if i == 1 {
                legend_in(root, &ch, Place::UpperLeft, &entries, 1, 6.5, Ground::Plain);
            }
        }
        let _ = Dash::Solid;
    });
    save(&svg, dir, "library_index");
}

pub fn all(s: &Summary, lang: Lang, dir: &Path) {
    library_sp(s, lang, dir);
    library_prediction(s, lang, dir);
    library_index(s, lang, dir);
    let _ = INK;
}
