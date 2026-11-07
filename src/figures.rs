//! The report's figures, in English or French, from the results.

use std::iter::once;
use std::path::Path;

use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, VPos};

use crate::analysis::{self, Group, Metric, Sp, Study, Which};
use crate::coherent::{self, CoherentResults, OMNI};
use crate::crossover::{self, Kind, KINDS};
use crate::cta2034;
use crate::plot::{
    self, band, colour_bar, colour_bar_width, hline, label, layout, legend, legend_at, legend_in, legend_size, line, outline, pt, render, save, sw, tint, vline, Axes, Dash, Frame,
    Ground, Lang, Place, Swatch, AXIS, DESC, INK, INK_2, LW, MUTED, NEUTRAL, PPI, PT, SLOTS,
};
use crate::room::{self, Setup, SURFACES};
use crate::signal::{db10, log_grid, smooth_power};
use crate::special::percentile;
use crate::speaker::{two_way, two_way_waveguide, FarField, Layout};
use crate::sphere;
use crate::study::power;
use crate::Vec3;

/// Width of a figure across the page (inches).
pub const W2: f64 = 7.0;

pub fn kind_color(k: Kind) -> RGBColor {
    match k {
        Kind::Lr2 => SLOTS[3],
        Kind::Lr4 => SLOTS[0],
        Kind::Lr8 => SLOTS[6],
        Kind::Fir => SLOTS[1],
    }
}

pub fn kind_label(lang: Lang, k: Kind) -> String {
    match k {
        Kind::Lr2 => "LR2".into(),
        Kind::Lr4 => "LR4".into(),
        Kind::Lr8 => "LR8".into(),
        Kind::Fir => lang.t("brickwall FIR", "FIR brickwall"),
    }
}

pub fn surf_label(lang: Lang, s: &str) -> String {
    match s {
        "left" => lang.t("near side wall", "mur latéral proche"),
        "right" => lang.t("far side wall", "mur latéral éloigné"),
        "listener_wall" => lang.t("wall behind listener", "mur derrière l’auditeur"),
        "speaker_wall" => lang.t("wall behind loudspeaker", "mur derrière l’enceinte"),
        "floor" => lang.t("floor", "sol"),
        _ => lang.t("ceiling", "plafond"),
    }
}

pub fn surf_color(s: &str) -> RGBColor {
    match s {
        "floor" => SLOTS[1],
        "ceiling" => SLOTS[3],
        "left" => SLOTS[0],
        "right" => SLOTS[6],
        "listener_wall" => SLOTS[2],
        _ => SLOTS[4],
    }
}

pub fn layout_label(lang: Lang, layout: &str) -> String {
    if layout == "two_way" {
        lang.t("dome", "dôme")
    } else {
        lang.t("dome in a waveguide", "dôme en guide d’onde")
    }
}

pub fn name(layout: &str, orient: &str, kind: Kind) -> String {
    format!("{layout}-{orient}-{}", kind.name())
}

fn db(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| db10(*x)).collect()
}

fn smooth12(f: &[f64], p: &[f64]) -> Vec<f64> {
    db(&smooth_power(f, p, 12.0))
}

/// Percentile across rooms at every frequency (dB curves in, dB out).
fn pct(rows: &[Vec<f64>], q: f64) -> Vec<f64> {
    (0..rows[0].len()).map(|j| percentile(&rows.iter().map(|r| r[j]).collect::<Vec<_>>(), q)).collect()
}

fn db_rows(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    rows.iter().map(|r| db(r)).collect()
}

fn closest(v: &[f64], x: f64) -> usize {
    (0..v.len()).min_by(|&a, &b| (v[a] - x).abs().partial_cmp(&(v[b] - x).abs()).unwrap()).unwrap()
}

/// Ticks from ``lo`` to ``hi`` every ``step``.
fn steps(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let n = ((hi - lo) / step + 1e-9).floor() as usize;
    (0..=n).map(|i| lo + i as f64 * step).collect()
}

pub struct Ctx<'a> {
    pub study: &'a Study,
    pub lang: Lang,
    pub dir: &'a Path,
    pub setups: &'a [Setup],
}

impl Ctx<'_> {
    fn t(&self, en: &str, fr: &str) -> String {
        self.lang.t(en, fr)
    }
}

// ---------------------------------------------------------------------------------------------

/// Plan and elevation of the living room, with the first-order reflection paths.
pub fn setup(c: &Ctx) {
    let lang = c.lang;
    let st = c.setups.iter().find(|s| s.key.as_deref() == Some("living")).unwrap();
    let [lx, ly, lz] = st.room.dims;
    let (s, r) = (st.source, st.receiver);
    let im = room::image_sources(st, 1);
    let mut a = Axes::lin(lang, &c.t("plan", "vue en plan"), (-0.15, lx + 0.15), (-0.15, ly + 0.15), "x (m)", "y (m)");
    a.x_ticks = steps(0.0, lx, 1.0);
    a.y_ticks = steps(0.0, ly, 1.0);
    let mut b = Axes::lin(
        lang,
        &c.t("elevation", "coupe"),
        (-0.15, ly + 0.15),
        (-0.15, lz + 0.15),
        &c.t("distance from the wall behind the loudspeakers (m)", "distance au mur derrière les enceintes (m)"),
        "z (m)",
    );
    b.x_ticks = steps(0.0, ly, 1.0);
    b.y_ticks = steps(0.0, lz, 0.5);
    for ax in [&mut a, &mut b] {
        ax.x_grid = false;
        ax.y_grid = false;
    }
    let first: Vec<(usize, &str)> = (0..im.order.len()).filter(|&i| im.order[i] == 1).map(|i| (i, SURFACES[im.first[i].unwrap()])).collect();
    let walls: Vec<(usize, &str)> = first.iter().copied().filter(|(_, s)| *s != "floor" && *s != "ceiling").collect();
    let entries: Vec<(Swatch, String)> = walls.iter().map(|(_, s)| (Swatch::Line(surf_color(s), 0.9, Dash::Solid), surf_label(lang, s))).collect();
    let size = 6.5;
    let (leg_w, _) = legend_size(&entries, 1, size);
    // One scale for both panels (pixels per metre), set by the plan's height: equal aspect.
    let h = (2.9 * PPI).round() as i32;
    let (na, mut nb) = (a.need(), b.need());
    let k = (h - na.top - na.bottom) as f64 / (ly + 0.3);
    let wa = na.left + na.right + (k * (lx + 0.3)).round() as i32;
    let wl = leg_w + pt(12.0);
    // The elevation's axis label is wider than its plot: room for it on the right.
    let over = ((plot::text_width(&b.xlabel, DESC) - k * (ly + 0.3)) / 2.0).ceil() as i32 + pt(2.0);
    nb.right = nb.right.max(over);
    b.frame = Some(nb);
    let wb = nb.left + nb.right + (k * (ly + 0.3)).round() as i32;
    let hb = nb.top + nb.bottom + (k * (lz + 0.3)).round() as i32;
    let rr = (0.12 * k).round() as i32;
    let svg = render((wa + wl + wb) as f64 / PPI, h as f64 / PPI, |root| {
        let parts = root.split_by_breakpoints([wa, wa + wl], Vec::<i32>::new());
        // Plan.
        let mut ch = a.build(&parts[0]);
        for p in st.room.patches.iter().filter(|p| SURFACES[p.surface] == "floor") {
            let [u0, u1, v0, v1] = p.rect;
            ch.draw_series(once(Rectangle::new([(u0, v0), (u1, v1)], NEUTRAL.filled()))).unwrap();
            ch.draw_series(once(Rectangle::new([(u0, v0), (u1, v1)], AXIS.stroke_width(sw(0.5))))).unwrap();
            label(root, &c.t("rug", "tapis"), ch.backend_coord(&(u1 - 0.05, v0 + 0.08)), 6.5, &MUTED, HPos::Right, VPos::Bottom, false);
        }
        ch.draw_series(once(Rectangle::new([(0.0, 0.0), (lx, ly)], INK_2.stroke_width(sw(1.0))))).unwrap();
        for x in [s[0], lx - s[0]] {
            ch.draw_series(once(Rectangle::new([(x - 0.1, s[1]), (x + 0.1, s[1] + 0.28)], INK_2.filled()))).unwrap();
        }
        for (i, surf) in &walls {
            let p = im.point[*i].unwrap();
            ch.draw_series(LineSeries::new(vec![(s[0], s[1]), (p[0], p[1]), (r[0], r[1])], surf_color(surf).stroke_width(sw(0.9)))).unwrap();
        }
        ch.draw_series(LineSeries::new(vec![(s[0], s[1]), (r[0], r[1])], INK.stroke_width(sw(1.1)))).unwrap();
        // The loudspeaker's axis.
        let axis = st.axes()[2];
        let tip = (s[0] + 0.6 * axis[0], s[1] + 0.6 * axis[1]);
        ch.draw_series(LineSeries::new(vec![(s[0], s[1]), tip], MUTED.stroke_width(sw(0.6)))).unwrap();
        let (b0, b1) = (ch.backend_coord(&(s[0], s[1])), ch.backend_coord(&tip));
        let (dx, dy) = ((b1.0 - b0.0) as f64, (b1.1 - b0.1) as f64);
        let n = dx.hypot(dy);
        let (ux, uy) = (dx / n, dy / n);
        let (hl, hw) = (5.0 * PT, 1.8 * PT);
        let base = (b1.0 as f64 - ux * hl, b1.1 as f64 - uy * hl);
        let head = vec![b1, ((base.0 - uy * hw).round() as i32, (base.1 + ux * hw).round() as i32), ((base.0 + uy * hw).round() as i32, (base.1 - ux * hw).round() as i32)];
        root.draw(&Polygon::new(head, MUTED.filled())).unwrap();
        ch.draw_series(once(Circle::new((r[0], r[1]), rr, INK.filled()))).unwrap();
        legend_at(&parts[1], (pt(6.0), na.top), &entries, 1, size, Ground::None);
        // Elevation, centred vertically.
        let top = (h - hb) / 2;
        let area = parts[2].margin(top, h - hb - top, 0, 0);
        let mut ch2 = b.build(&area);
        ch2.draw_series(once(Rectangle::new([(0.0, 0.0), (ly, lz)], INK_2.stroke_width(sw(1.0))))).unwrap();
        ch2.draw_series(once(Rectangle::new([(ly - s[1] - 0.28, s[2] - 0.29), (ly - s[1], s[2] + 0.07)], INK_2.filled()))).unwrap();
        for (i, surf) in first.iter().filter(|(_, s)| *s == "floor" || *s == "ceiling") {
            let p = im.point[*i].unwrap();
            ch2.draw_series(LineSeries::new(vec![(ly - s[1], s[2]), (ly - p[1], p[2]), (ly - r[1], r[2])], surf_color(surf).stroke_width(sw(0.9)))).unwrap();
        }
        ch2.draw_series(LineSeries::new(vec![(ly - s[1], s[2]), (ly - r[1], r[2])], INK.stroke_width(sw(1.1)))).unwrap();
        ch2.draw_series(once(Circle::new((ly - r[1], r[2]), rr, INK.filled()))).unwrap();
        let mid = 0.5 * (ly - s[1] + ly - r[1]);
        for (y, surf) in [(0.12, "floor"), (lz - 0.25, "ceiling")] {
            label(root, &surf_label(lang, surf), ch2.backend_coord(&(mid, y)), 7.0, &surf_color(surf), HPos::Center, VPos::Bottom, false);
        }
    });
    save(&svg, c.dir, "setup");
}

/// Where the energy leaves the loudspeaker: CTA-2034 PIR weights against the rooms.
pub fn kernel_maps(c: &Ctx) {
    let lang = c.lang;
    let res = &c.study.res;
    let n = res.n_random;
    let f = &res.freqs;
    let j = closest(&res.map_freqs, 2200.0);
    let fj = closest(f, 2200.0);
    let mk = |title: String| {
        let mut ax = Axes::lin(
            lang,
            &title,
            (-180.0, 180.0),
            (-90.0, 90.0),
            &c.t("azimuth (deg, + toward the listener's right)", "azimut (°, + vers la droite de l’auditeur)"),
            &c.t("elevation (deg)", "élévation (°)"),
        );
        ax.x_ticks = steps(-180.0, 180.0, 30.0);
        ax.y_ticks = steps(-90.0, 90.0, 30.0);
        ax
    };
    let mut a = mk(c.t(
        "CTA-2034 Predicted In-Room Response: weight of each of the 70 directions (area)",
        &format!("PIR de la CTA-2034{} poids de chacune des 70 directions (surface)", lang.colon()),
    ));
    let mut b = mk(c.t(
        "300 rooms at 2.2 kHz: direct sound and first reflections (area: share of the energy)",
        &format!("300 pièces à 2,2\u{a0}kHz{} son direct et premières réflexions (surface{} part de l’énergie)", lang.colon(), lang.colon()),
    ));
    let (na, nb) = (a.need(), b.need());
    let fr = Frame { left: na.left.max(nb.left), right: na.right.max(nb.right), top: na.top.max(nb.top), bottom: na.bottom.max(nb.bottom) };
    a.frame = Some(fr);
    b.frame = Some(fr);
    let bar_text = c.t("diffuse field: share per direction\n(relative to the mean)", &format!("champ diffus{} part par direction\n(relative à la moyenne)", lang.colon()));
    let ticks = steps(0.4, 1.4, 0.2);
    // Equal aspect: each plotting area twice as wide as high, the colour bar beside the second.
    let mut ph = (3.2 * PPI) as i32 - fr.top - fr.bottom;
    let bar_w = |ph: i32| (0.6 * ph as f64 / 20.0).round() as i32;
    let zone = |ph: i32| pt(4.0) + colour_bar_width(bar_w(ph), &ticks, &bar_text, 7.0, lang);
    while fr.left + fr.right + 2 * ph + zone(ph) > (W2 * PPI) as i32 {
        ph -= 2;
    }
    let panel_w = fr.left + fr.right + 2 * ph;
    let panel_h = fr.top + fr.bottom + ph;
    let svg = render((panel_w + zone(ph)) as f64 / PPI, (2 * panel_h) as f64 / PPI, |root| {
        let parts = root.split_by_breakpoints([panel_w], [panel_h]);
        // The PIR's weights.
        let mut ch = a.build(&parts[0]);
        let dirs = cta2034::directions();
        let curves = [("sp", 0.44, SLOTS[2]), ("er", 0.44, SLOTS[1]), ("lw", 0.12, SLOTS[0])];
        let mut entries = Vec::new();
        for (curve, w, col) in curves {
            let k = cta2034::kernel(curve);
            let pts: Vec<(f64, f64, f64)> = dirs
                .iter()
                .zip(&k)
                .filter(|(_, v)| **v > 0.0)
                .map(|(u, v)| {
                    let (az, el) = sphere::to_angles(u);
                    (az, el, w * v)
                })
                .collect();
            ch.draw_series(pts.iter().map(|(x, y, v)| Circle::new((*x, *y), pt(0.5 * (v * 2500.0).sqrt()), col.mix(0.55).filled()))).unwrap();
            entries.push((Swatch::Dot(tint(col, 0.7), 3.0), format!("{} ({}{})", curve.to_uppercase(), (w * 100.0).round(), lang.percent())));
        }
        ch.draw_series(dirs.iter().map(|u| {
            let (az, el) = sphere::to_angles(u);
            Circle::new((az, el), pt(0.7), INK.filled())
        }))
        .unwrap();
        legend_in(root, &ch, Place::LowerLeft, &entries, 3, 7.0, Ground::Framed);

        // The rooms.
        let mut ch2 = b.build(&parts[2]);
        let dens: Vec<f64> = (0..res.grid.len()).map(|k| res.diffuse_mean.row(k)[fj] as f64).collect();
        let mean = dens.iter().sum::<f64>() / dens.len() as f64;
        ch2.draw_series(res.grid.iter().zip(&dens).map(|(u, d)| {
            let (az, el) = sphere::to_angles(u);
            Circle::new((az, el), pt(1.5), plot::greys((d / mean - 0.3) / 1.1).filled())
        }))
        .unwrap();
        let starts = &res.image_starts;
        let mut entries = Vec::new();
        for cat in 0..7u8 {
            let (col, alpha) = if cat == 0 { (INK, 0.35) } else { (surf_color(SURFACES[cat as usize - 1]), 0.4) };
            let mut pts = Vec::new();
            for room in 0..n {
                let total: f64 = (0..res.categories.len()).map(|cc| res.omni.row(room * res.categories.len() + cc)[fj] as f64).sum();
                for i in starts[room]..starts[room + 1] {
                    if res.image_cat[i] == cat {
                        let d = res.image_dirs[i];
                        let (az, el) = sphere::to_angles(&[d[0] as f64, d[1] as f64, d[2] as f64]);
                        pts.push((az, el, res.image_weights.row(i)[j] as f64 / total));
                    }
                }
            }
            ch2.draw_series(pts.iter().map(|(x, y, s)| Circle::new((*x, *y), pt(0.5 * (s * 600.0).sqrt()), col.mix(alpha).filled()))).unwrap();
            let label = if cat == 0 { c.t("direct sound", "son direct") } else { surf_label(lang, SURFACES[cat as usize - 1]) };
            entries.push((Swatch::Dot(tint(col, if cat == 0 { 0.6 } else { 0.7 }), 2.5), label));
        }
        legend_in(root, &ch2, Place::LowerLeft, &entries, 4, 6.5, Ground::Framed);
        let (_, ys) = ch2.plotting_area().get_pixel_range();
        let bar_h = (0.6 * ph as f64).round() as i32;
        let y0 = (ys.start + ys.end) / 2 - bar_h / 2;
        colour_bar(root, (panel_w + pt(4.0), y0), (bar_w(ph), bar_h), plot::greys, (0.3, 1.4), &ticks, &bar_text, 7.0, lang);
    });
    save(&svg, c.dir, "kernel_maps");
}

/// Shares of the energy at the listener (omnidirectional source) against frequency.
pub fn shares(c: &Ctx) {
    let lang = c.lang;
    let f = c.study.freqs();
    let sh = c.study.shares(c.study.random());
    let groups: [(String, Vec<usize>, RGBColor); 5] = [
        (c.t("direct sound", "son direct"), vec![0], INK),
        (c.t("first order, floor + ceiling", "1er ordre, sol + plafond"), vec![5, 6], SLOTS[1]),
        (c.t("first order, walls", "1er ordre, murs"), vec![1, 2, 3, 4], SLOTS[0]),
        (c.t("second and third order", "2e et 3e ordres"), vec![7, 8], SLOTS[6]),
        (c.t("diffuse field", "champ diffus"), vec![9], SLOTS[2]),
    ];
    // The legend lists the layers as they are stacked.
    let entries: Vec<(Swatch, String)> = groups.iter().rev().map(|(l, _, col)| (Swatch::Patch(tint(*col, 0.55)), l.clone())).collect();
    let size = 6.5;
    let (leg_w, _) = legend_size(&entries, 1, size);
    let svg = render(5.2, 2.6, |root| {
        let (a, _) = root.split_horizontally((5.2 * PPI) as i32 - leg_w - pt(8.0));
        let ax = Axes::freq(
            lang,
            &c.t("energy at the listener, median of 300 rooms", "énergie reçue par l’auditeur, médiane de 300 pièces"),
            (150.0, 20000.0),
            (0.0, 100.0),
            &c.t("share of the energy (%)", "part de l’énergie (%)"),
        );
        let mut ch = ax.build(&a);
        let mut bottom = vec![0.0; f.len()];
        for (_, idx, col) in groups.iter() {
            let val: Vec<f64> = (0..f.len())
                .map(|j| percentile(&sh.iter().map(|r| idx.iter().map(|&k| r[k][j]).sum::<f64>()).collect::<Vec<_>>(), 50.0))
                .collect();
            let top: Vec<f64> = bottom.iter().zip(&val).map(|(b, v)| b + v).collect();
            band(&mut ch, &ax, f, &bottom.iter().map(|v| v * 100.0).collect::<Vec<_>>(), &top.iter().map(|v| v * 100.0).collect::<Vec<_>>(), *col, 0.55, None);
            bottom = top;
        }
        outline(&ch);
        legend_in(root, &ch, Place::Right, &entries, 1, size, Ground::None);
    });
    save(&svg, c.dir, "shares");
}

/// Directions in which the first reflections leave the loudspeaker, against CTA-2034's angles.
pub fn reflection_angles(c: &Ctx) {
    let lang = c.lang;
    let res = &c.study.res;
    let n = res.n_random;
    let end = res.image_starts[n];
    let ang: Vec<(u8, f64, f64)> = (0..end)
        .map(|i| {
            let d = res.image_dirs[i];
            let (az, el) = sphere::to_angles(&[d[0] as f64, d[1] as f64, d[2] as f64]);
            (res.image_cat[i], az, el)
        })
        .collect();
    let hist = |cat: u8, use_el: bool, lo: f64, w: f64, nb: usize| -> Vec<usize> {
        let mut h = vec![0; nb];
        for (c0, az, el) in &ang {
            if *c0 == cat {
                let v = if use_el { *el } else { *az };
                let b = ((v - lo) / w).floor();
                if b >= 0.0 && (b as usize) < nb {
                    h[b as usize] += 1;
                }
            }
        }
        h
    };
    let mut a = Axes::lin(
        lang,
        &c.t("floor and ceiling (dashed: CTA-2034 angles)", &format!("sol et plafond (tirets{} angles de la CTA-2034)", lang.colon())),
        (-70.0, 70.0),
        (0.0, 59.0),
        &c.t("elevation (deg)", "élévation (°)"),
        &c.t("rooms", "pièces"),
    );
    a.x_ticks = steps(-60.0, 60.0, 20.0);
    let mut b = Axes::lin(
        lang,
        &c.t("walls (dashed: side and front walls, dotted: rear wall)", &format!("murs (tirets{} latéraux et avant, pointillés{} arrière)", lang.colon(), lang.colon())),
        (-180.0, 180.0),
        (0.0, 82.0),
        &c.t("azimuth (deg)", "azimut (°)"),
        "",
    );
    b.x_ticks = steps(-180.0, 180.0, 45.0);
    b.y_ticks = steps(0.0, 80.0, 20.0);
    let svg = render(W2, 2.4, |root| {
        let mut axes = vec![a, b];
        let p = layout(root, 1, 2, &mut axes);
        let mut ch = axes[0].build(&p[0]);
        for (cat, surf) in [(5u8, "floor"), (6u8, "ceiling")] {
            let h = hist(cat, true, -90.0, 2.5, 72);
            let col = surf_color(surf);
            ch.draw_series(h.iter().enumerate().filter(|(_, v)| **v > 0).map(|(i, v)| {
                let x0 = -90.0 + i as f64 * 2.5;
                Rectangle::new([(x0, 0.0), (x0 + 2.5, *v as f64)], col.mix(0.7).filled())
            }))
            .unwrap()
            .label(surf_label(lang, surf))
            .legend(move |(x, y)| Rectangle::new([(x, y - pt(2.8)), (x + pt(14.0), y + pt(2.8))], col.mix(0.7).filled()));
        }
        for (angles, surf) in [(&cta2034::FLOOR[..], "floor"), (&cta2034::CEILING[..], "ceiling")] {
            for &an in angles {
                vline(&mut ch, &axes[0], an as f64, surf_color(surf), 0.8, Dash::Dashed);
            }
        }
        legend(&mut ch, SeriesLabelPosition::LowerMiddle, 6.5);
        let mut ch2 = axes[1].build(&p[1]);
        let mut entries = Vec::new();
        for (cat, surf) in [(1u8, "left"), (2u8, "right"), (3u8, "listener_wall"), (4u8, "speaker_wall")] {
            let h = hist(cat, false, -180.0, 5.0, 72);
            let col = surf_color(surf);
            ch2.draw_series(h.iter().enumerate().filter(|(_, v)| **v > 0).map(|(i, v)| {
                let x0 = -180.0 + i as f64 * 5.0;
                Rectangle::new([(x0, 0.0), (x0 + 5.0, *v as f64)], col.mix(0.7).filled())
            }))
            .unwrap();
            entries.push((Swatch::Patch(tint(col, 0.7)), surf_label(lang, surf)));
        }
        for &an in &cta2034::SIDE_WALL {
            vline(&mut ch2, &axes[1], an as f64, SLOTS[0], 0.6, Dash::Dashed);
        }
        for an in cta2034::REAR_WALL.iter().chain(once(&-180)) {
            vline(&mut ch2, &axes[1], *an as f64, surf_color("speaker_wall"), 0.8, Dash::Dotted);
        }
        for &an in &cta2034::FRONT_WALL {
            vline(&mut ch2, &axes[1], an as f64, surf_color("listener_wall"), 0.6, Dash::Dashed);
        }
        legend_in(root, &ch2, Place::UpperCenter, &entries, 2, 6.0, Ground::Plain);
    });
    save(&svg, c.dir, "reflection_angles");
}

fn weights_label(lang: Lang, w: [f64; 3]) -> String {
    format!("{} LW + {} ER + {} SP ({})", lang.num(w[0], 2), lang.num(w[1], 2), lang.num(w[2], 2), lang.t("sphere", "sphère"))
}

/// The CTA-2034 curves of the upright two-way (LR4), and its in-room responses.
pub fn spin(c: &Ctx, layout_name: &str, stem: &str) {
    let lang = c.lang;
    let f = c.study.freqs();
    let v = name(layout_name, "upright", Kind::Lr4);
    let cu = c.study.curves(&v);
    let w = analysis::fit_weights(c.study, Sp::Sphere, Which::Steady);
    let rooms = db_rows(&c.study.rooms(&v, Which::Steady, c.study.random()));
    let early = db_rows(&c.study.rooms(&v, Which::Early, c.study.random()));
    let a = Axes::freq(lang, &c.t("CTA-2034 curves (far field)", "courbes CTA-2034 (champ lointain)"), (150.0, 20000.0), (-14.0, 3.0), &c.t("level (dB)", "niveau (dB)"));
    let mut b = Axes::freq(lang, &c.t("in room, relative to an omnidirectional source", "dans la pièce, relatif à une source omnidirectionnelle"), (150.0, 20000.0), (-14.0, 3.0), "");
    b.show_ylabels = false;
    let svg = render(W2, 2.9, |root| {
        let mut axes = vec![a, b];
        let p = layout(root, 1, 2, &mut axes);
        let mut ch = axes[0].build(&p[0]);
        for (key, label, col, dash) in [
            ("on_axis", c.t("on axis", "dans l’axe"), INK, Dash::Solid),
            ("lw", c.t("listening window", "fenêtre d’écoute"), SLOTS[0], Dash::Solid),
            ("er", c.t("early reflections", "premières réflexions"), SLOTS[1], Dash::Solid),
            ("sp", c.t("sound power, two orbits", "puissance, deux orbites"), SLOTS[2], Dash::Dashed),
            ("sp_sphere", c.t("sound power, sphere", "puissance, sphère"), SLOTS[2], Dash::Solid),
        ] {
            line(&mut ch, &axes[0], f, &smooth12(f, cu.get(key)), col, LW, dash, Some(&label));
        }
        legend(&mut ch, SeriesLabelPosition::LowerLeft, 6.3);
        let mut ch2 = axes[1].build(&p[1]);
        let ax2 = &axes[1];
        band(&mut ch2, ax2, f, &pct(&rooms, 10.0), &pct(&rooms, 90.0), SLOTS[2], 0.2, Some(&c.t("rooms, 10-90 %", &format!("pièces, 10–90{}", lang.percent()))));
        line(&mut ch2, ax2, f, &pct(&rooms, 50.0), SLOTS[2], LW, Dash::Solid, Some(&c.t("rooms, median", "pièces, médiane")));
        line(&mut ch2, ax2, f, &pct(&early, 50.0), SLOTS[1], LW, Dash::Solid, Some(&c.t("rooms, direct + first reflections", "pièces, direct + premières réflexions")));
        line(&mut ch2, ax2, f, &smooth12(f, &cu.pir), INK, LW, Dash::Dashed, Some(&c.t("CTA-2034 PIR", "PIR CTA-2034")));
        line(&mut ch2, ax2, f, &smooth12(f, &analysis::pir(&cu, w, Sp::Sphere)), INK, LW, Dash::Dotted, Some(&weights_label(lang, w)));
        legend(&mut ch2, SeriesLabelPosition::LowerLeft, 6.3);
    });
    save(&svg, c.dir, stem);
}

/// Sound power from two orbits against the sphere, for each crossover.
pub fn sp_orbits(c: &Ctx) {
    let lang = c.lang;
    let f = c.study.freqs();
    let mut axes: Vec<Axes> = ["two_way", "two_way_wg"]
        .iter()
        .enumerate()
        .map(|(i, layout)| {
            let mut a = Axes::freq(
                lang,
                &format!("{}, {}", c.t("two-way", "deux voies"), layout_label(lang, layout)),
                (300.0, 20000.0),
                (-0.05, 2.05),
                &if i == 0 { c.t("SP two orbits - SP sphere (dB)", "SP deux orbites − SP sphère (dB)") } else { String::new() },
            );
            a.show_ylabels = i == 0;
            a
        })
        .collect();
    let svg = render(W2, 2.5, |root| {
        let p = layout(root, 1, 2, &mut axes);
        for (i, layout) in ["two_way", "two_way_wg"].iter().enumerate() {
            let mut ch = axes[i].build(&p[i]);
            for k in KINDS {
                let cu = c.study.curves(&name(layout, "upright", k));
                let d: Vec<f64> = cu.sp.iter().zip(&cu.sp_sphere).map(|(a, b)| db10(*a) - db10(*b)).collect();
                let lab = kind_label(lang, k);
                line(&mut ch, &axes[i], f, &d, kind_color(k), LW, Dash::Solid, if i == 0 { Some(lab.as_str()) } else { None });
            }
            hline(&mut ch, &axes[i], 0.0, AXIS, 0.6);
            if i == 0 {
                legend(&mut ch, SeriesLabelPosition::UpperRight, 7.0);
            }
        }
    });
    save(&svg, c.dir, "sp_orbits");
}

/// Mean error of the predictions against the rooms (test half), every variant.
pub fn errors(c: &Ctx) {
    let lang = c.lang;
    let f = c.study.freqs();
    let w = analysis::fit_weights(c.study, Sp::Sphere, Which::Steady);
    let preds: Vec<(String, [f64; 3], Sp)> = vec![
        (c.t("CTA-2034 PIR", "PIR CTA-2034"), [0.12, 0.44, 0.44], Sp::TwoOrbits),
        (c.t("same weights, sound power of the sphere", "mêmes poids, puissance sur la sphère"), [0.12, 0.44, 0.44], Sp::Sphere),
        (weights_label(lang, w), w, Sp::Sphere),
    ];
    let mut axes: Vec<Axes> = preds
        .iter()
        .enumerate()
        .map(|(i, (title, _, _))| {
            let mut a = Axes::freq(lang, title, (300.0, 16000.0), (-1.8, 3.2), &if i == 0 { c.t("prediction - room (dB)", "prédiction − pièce (dB)") } else { String::new() });
            a.x_ticks = vec![1000.0, 10000.0];
            a.title_size = 7.0;
            a.show_ylabels = i == 0;
            a
        })
        .collect();
    let svg = render(W2, 2.5, |root| {
        let p = layout(root, 1, 3, &mut axes);
        for (i, (_, weights, sp)) in preds.iter().enumerate() {
            let ax = &axes[i];
            let mut ch = ax.build(&p[i]);
            let mut lo = vec![0.0; f.len()];
            let mut hi = vec![0.0; f.len()];
            let vs = c.study.variants();
            let mut lines = Vec::new();
            for v in &vs {
                let cu = c.study.curves(v);
                let pr = analysis::pir(&cu, *weights, *sp);
                let (_, bias) = analysis::errors(c.study, v, &pr, c.study.test(), Which::Steady);
                let rs = db_rows(&c.study.rooms(v, Which::Steady, c.study.test()));
                for j in 0..f.len() {
                    let e: Vec<f64> = rs.iter().map(|r| db10(pr[j]) - r[j]).collect();
                    let m = e.iter().sum::<f64>() / e.len() as f64;
                    lo[j] += (percentile(&e, 10.0) - m) / vs.len() as f64;
                    hi[j] += (percentile(&e, 90.0) - m) / vs.len() as f64;
                }
                lines.push((v.starts_with("two_way-"), bias));
            }
            band(&mut ch, ax, f, &lo, &hi, NEUTRAL, 1.0, None);
            for (dome, bias) in &lines {
                line(&mut ch, ax, f, bias, (if *dome { SLOTS[0] } else { SLOTS[1] }).mix(0.7), 0.6, Dash::Solid, None);
            }
            hline(&mut ch, ax, 0.0, AXIS, 0.6);
            if i == 0 {
                let entries = vec![
                    (Swatch::Patch(NEUTRAL), c.t("room to room, 10-90 %", &format!("d’une pièce à l’autre, 10–90{}", lang.percent()))),
                    (Swatch::Line(SLOTS[0], LW, Dash::Solid), c.t("dome", "dôme")),
                    (Swatch::Line(SLOTS[1], LW, Dash::Solid), c.t("waveguide", "guide d’onde")),
                ];
                legend_in(root, &ch, Place::LowerRight, &entries, 1, 6.5, Ground::None);
            }
        }
    });
    save(&svg, c.dir, "errors");
}

/// The same loudspeaker upright and on its side (LR4).
pub fn rotation(c: &Ctx) {
    let lang = c.lang;
    let f = c.study.freqs();
    let mut axes: Vec<Axes> = [
        "CTA-2034".to_string(),
        c.t("rooms: direct + first reflections", &format!("pièces{} direct + premières réflexions", lang.colon())),
        c.t("rooms: steady state", &format!("pièces{} régime établi", lang.colon())),
    ]
    .iter()
    .enumerate()
    .map(|(i, t)| {
        let mut a = Axes::freq(lang, t, (300.0, 20000.0), (-14.0, 2.0), &if i == 0 { c.t("level (dB)", "niveau (dB)") } else { String::new() });
        a.x_ticks = vec![1000.0, 10000.0];
        a.show_ylabels = i == 0;
        a
    })
    .collect();
    let svg = render(W2, 2.5, |root| {
        let p = layout(root, 1, 3, &mut axes);
        let mut charts: Vec<_> = axes.iter().zip(&p).map(|(a, area)| a.build(area)).collect();
        for (orient, dash) in [("upright", Dash::Solid), ("side", Dash::Dashed)] {
            let v = name("two_way", orient, Kind::Lr4);
            let cu = c.study.curves(&v);
            let lab = if orient == "upright" { c.t("upright", "debout") } else { c.t("on its side", "couchée") };
            line(&mut charts[0], &axes[0], f, &smooth12(f, &cu.lw), SLOTS[0], LW, dash, Some(&format!("LW, {lab}")));
            line(&mut charts[0], &axes[0], f, &smooth12(f, &cu.er), SLOTS[1], LW, dash, Some(&format!("ER, {lab}")));
            line(&mut charts[0], &axes[0], f, &smooth12(f, &cu.pir), INK, LW, dash, Some(&format!("PIR, {lab}")));
            for (k, which) in [(1, Which::Early), (2, Which::Steady)] {
                let r = db_rows(&c.study.rooms(&v, which, c.study.random()));
                line(&mut charts[k], &axes[k], f, &pct(&r, 50.0), INK, LW, dash, Some(&format!("{}, {lab}", c.t("median", "médiane"))));
                line(&mut charts[k], &axes[k], f, &pct(&r, 10.0), SLOTS[1], LW, dash, Some(&format!("10{}, {lab}", lang.percent())));
            }
        }
        for ch in charts.iter_mut() {
            legend(ch, SeriesLabelPosition::LowerLeft, 5.6);
        }
    });
    save(&svg, c.dir, "rotation");
}

/// In-room responses of the four crossovers, dome and waveguide.
pub fn crossovers(c: &Ctx) {
    let lang = c.lang;
    let f = c.study.freqs();
    let layouts = ["two_way", "two_way_wg"];
    let mut axes = Vec::new();
    for row in 0..2 {
        for (col, layout) in layouts.iter().enumerate() {
            let title = if *layout == "two_way" { c.t("dome", "dôme") } else { c.t("waveguide", "guide d’onde") };
            let ylab = if col == 0 { c.t("level, median of 300 rooms (dB)", "niveau, médiane de 300 pièces (dB)") } else { String::new() };
            let mut a = if row == 0 {
                Axes::freq(
                    lang,
                    &c.t(&format!("{title}: steady state (dotted: CTA-2034 PIR)"), &format!("{title}{} régime établi (pointillés{} PIR CTA-2034)", lang.colon(), lang.colon())),
                    (500.0, 10000.0),
                    (-11.0, -1.0),
                    &ylab,
                )
            } else {
                Axes::freq(lang, &c.t(&format!("{title}: direct + first reflections"), &format!("{title}{} direct + premières réflexions", lang.colon())), (500.0, 10000.0), (-5.0, 1.0), &ylab)
            };
            a.title_size = 7.5;
            a.show_xlabels = row == 1;
            axes.push(a);
        }
    }
    let svg = render(W2, 4.6, |root| {
        let p = layout(root, 2, 2, &mut axes);
        for (col, layout) in layouts.iter().enumerate() {
            let mut c0 = axes[col].build(&p[col]);
            let mut c1 = axes[2 + col].build(&p[2 + col]);
            for k in KINDS {
                let v = name(layout, "upright", k);
                let cu = c.study.curves(&v);
                let r = pct(&db_rows(&c.study.rooms(&v, Which::Steady, c.study.random())), 50.0);
                let e = pct(&db_rows(&c.study.rooms(&v, Which::Early, c.study.random())), 50.0);
                line(&mut c0, &axes[col], f, &r, kind_color(k), LW, Dash::Solid, Some(&kind_label(lang, k)));
                line(&mut c0, &axes[col], f, &db(&cu.pir), kind_color(k), 0.9, Dash::Dotted, None);
                line(&mut c1, &axes[2 + col], f, &e, kind_color(k), LW, Dash::Solid, None);
            }
            if col == 0 {
                legend(&mut c0, SeriesLabelPosition::LowerLeft, 6.5);
            }
        }
    });
    save(&svg, c.dir, "crossovers");
}

/// The living room's direct sound, floor and ceiling bounce directions.
pub fn living_paths(setups: &[Setup]) -> Vec<(String, Vec3)> {
    let st = setups.iter().find(|s| s.key.as_deref() == Some("living")).unwrap();
    let im = room::image_sources(st, 1);
    let mut out = vec![("direct".to_string(), im.direction[0])];
    for i in 0..im.order.len() {
        if im.order[i] == 1 {
            let s = SURFACES[im.first[i].unwrap()];
            if s == "floor" || s == "ceiling" {
                out.push((s.to_string(), im.direction[i]));
            }
        }
    }
    out
}

/// Power in the living room's path directions, for each layout and crossover, on a 1/96-octave
/// grid from 300 Hz: ``(freqs, [(layout, [(kind, [(path, power)])])])``.
pub type Spectra = (Vec<f64>, Vec<(String, Vec<(Kind, Vec<(String, Vec<f64>)>)>)>);

pub fn path_spectra(setups: &[Setup]) -> Spectra {
    let f = log_grid(300.0, 20000.0, 96);
    let paths = living_paths(setups);
    let dirs: Vec<Vec3> = paths.iter().map(|p| p.1).collect();
    let mut out = Vec::new();
    for layout in [two_way(), two_way_waveguide()] {
        let far = FarField::new(&layout, &f);
        let on = far.on_axis();
        let resp = far.responses(&dirs);
        let mut per_kind = Vec::new();
        for k in KINDS {
            let p = power(&crossover::filters(k, &f, &layout, &on), &resp);
            per_kind.push((k, paths.iter().enumerate().map(|(i, (n, _))| (n.clone(), p.row(i).iter().map(|v| *v as f64).collect())).collect()));
        }
        out.push((layout.name.clone(), per_kind));
    }
    (f, out)
}

/// Width (Hz) of the band around the deepest dip within an octave of the crossover where the
/// response stays ``depth`` dB below the straight line; the dip's frequency and depth.
pub fn notch_width(f: &[f64], p: &[f64], depth: f64) -> (f64, f64, f64) {
    let (r, _) = analysis::line_residual(f, p, analysis::ANOMALY_BAND);
    let fb: Vec<f64> = f.iter().cloned().filter(|&x| x >= analysis::ANOMALY_BAND.0 && x <= analysis::ANOMALY_BAND.1).collect();
    let sel: Vec<usize> = (0..fb.len()).filter(|&i| fb[i] >= analysis::CROSSOVER_BAND.0 && fb[i] <= analysis::CROSSOVER_BAND.1).collect();
    let i = *sel.iter().min_by(|&&a, &&b| r[a].partial_cmp(&r[b]).unwrap()).unwrap();
    let (mut lo, mut hi) = (i, i);
    while lo > 0 && r[lo - 1] < -depth {
        lo -= 1;
    }
    while hi + 1 < r.len() && r[hi + 1] < -depth {
        hi += 1;
    }
    (if r[i] < -depth { fb[hi] - fb[lo] } else { 0.0 }, fb[i], -r[i])
}

/// What each reflection carries, against the room's average: the living room.
pub fn paths(c: &Ctx, spectra: &Spectra) {
    let lang = c.lang;
    let room_index = c.study.named_index("living");
    let fs = c.study.freqs();
    let (f, per_layout) = spectra;
    let living = living_paths(c.setups);
    let el = |p: &str| -> String {
        let d = living.iter().find(|x| x.0 == p).unwrap().1;
        lang.num(sphere::to_angles(&d).1, 0)
    };
    let mut axes = Vec::new();
    for (row, (layout, _)) in per_layout.iter().enumerate() {
        let mut a0 = Axes::freq(
            lang,
            &c.t(
                &format!("{}: floor ({}°) and ceiling ({}°) bounces", layout_label(lang, layout), el("floor"), el("ceiling")),
                &format!("{}{} rebonds sol ({}°) et plafond ({}°)", layout_label(lang, layout), lang.colon(), el("floor"), el("ceiling")),
            ),
            (500.0, 10000.0),
            (-30.0, 3.0),
            &c.t("level re on axis (dB)", "niveau relatif à l’axe (dB)"),
        );
        a0.title_size = 8.0;
        let mut a1 = Axes::freq(
            lang,
            &c.t(&format!("{}: the same room, steady state", layout_label(lang, layout)), &format!("{}{} la même pièce, régime établi", layout_label(lang, layout), lang.colon())),
            (500.0, 10000.0),
            (-12.0, 0.0),
            &c.t("level re omnidirectional (dB)", "niveau relatif à l’omni (dB)"),
        );
        a0.show_xlabels = row == 1;
        a1.show_xlabels = row == 1;
        axes.push(a0);
        axes.push(a1);
    }
    let svg = render(W2, 4.6, |root| {
        let p = layout(root, 2, 2, &mut axes);
        for (row, (layout, per_kind)) in per_layout.iter().enumerate() {
            let (a0, a1) = (&axes[2 * row], &axes[2 * row + 1]);
            let mut c0 = a0.build(&p[2 * row]);
            let mut c1 = a1.build(&p[2 * row + 1]);
            let mut entries = Vec::new();
            for (k, paths) in per_kind.iter().filter(|(k, _)| *k != Kind::Lr8) {
                for (path, dash) in [("floor", Dash::Solid), ("ceiling", Dash::Dashed)] {
                    let pw = &paths.iter().find(|x| x.0 == path).unwrap().1;
                    line(&mut c0, a0, f, &db(pw), kind_color(*k), 0.9, dash, None);
                    entries.push((Swatch::Line(kind_color(*k), 0.9, dash), format!("{}, {}", kind_label(lang, *k), surf_label(lang, path))));
                }
                let r = c.study.room_response(&name(layout, "upright", *k), Which::Steady, room_index);
                line(&mut c1, a1, fs, &db(&r), kind_color(*k), LW, Dash::Solid, if row == 0 { Some(kind_label(lang, *k)) } else { None }.as_deref());
            }
            if row == 0 {
                legend_in(root, &c0, Place::LowerLeft, &entries, 2, 5.8, Ground::Plain);
                legend(&mut c1, SeriesLabelPosition::LowerLeft, 6.5);
            }
        }
    });
    save(&svg, c.dir, "paths");
}

/// Each direction's own deviation within an octave of the crossover, along the vertical orbit,
/// with the directions of the rooms' floor and ceiling bounces.
pub fn vertical_colouration(c: &Ctx) {
    let lang = c.lang;
    let res = &c.study.res;
    let n = res.n_random;
    let f = log_grid(300.0, 20000.0, 96);
    let angles: Vec<f64> = (0..=180).map(|i| -90.0 + i as f64).collect();
    let dirs: Vec<Vec3> = angles.iter().map(|a| [0.0, a.to_radians().sin(), a.to_radians().cos()]).collect();
    let els: Vec<(u8, f64)> = (0..res.image_starts[n])
        .map(|i| {
            let d = res.image_dirs[i];
            (res.image_cat[i], sphere::to_angles(&[d[0] as f64, d[1] as f64, d[2] as f64]).1)
        })
        .collect();
    let range = |cat: u8| -> (f64, f64) {
        let v: Vec<f64> = els.iter().filter(|(c0, _)| *c0 == cat).map(|(_, e)| *e).collect();
        (percentile(&v, 10.0), percentile(&v, 90.0))
    };
    let layouts = [two_way(), two_way_waveguide()];
    let mut axes: Vec<Axes> = layouts
        .iter()
        .enumerate()
        .map(|(i, layout)| {
            let mut a = Axes::lin(
                lang,
                &format!("{}, {}", c.t("two-way", "deux voies"), layout_label(lang, &layout.name)),
                (-90.0, 90.0),
                (-0.4, 8.6),
                &c.t("vertical angle (deg)", "angle vertical (°)"),
                &if i == 0 { c.t("deviation, 1.1-4.4 kHz (dB RMS)", "écart, 1,1–4,4\u{a0}kHz (dB RMS)") } else { String::new() },
            );
            a.x_ticks = steps(-90.0, 90.0, 30.0);
            a.y_ticks = steps(0.0, 8.0, 2.0);
            a.show_ylabels = i == 0;
            a
        })
        .collect();
    let svg = render(W2, 2.6, |root| {
        let p = layout(root, 1, 2, &mut axes);
        for (i, layout) in layouts.iter().enumerate() {
            let ax = &axes[i];
            let mut ch = ax.build(&p[i]);
            for (cat, surf) in [(5u8, "floor"), (6u8, "ceiling")] {
                let (lo, hi) = range(cat);
                ch.draw_series(once(Rectangle::new([(lo, ax.y.0), (hi, ax.y.1)], tint(surf_color(surf), 0.15).filled()))).unwrap();
            }
            let far = FarField::new(layout, &f);
            let on = far.on_axis();
            let resp = far.responses(&dirs);
            for k in KINDS {
                let pw = power(&crossover::filters(k, &f, layout, &on), &resp);
                let a: Vec<f64> = (0..dirs.len()).map(|d| analysis::local_anomaly(&f, &pw.row(d).iter().map(|v| (*v as f64).max(1e-12)).collect::<Vec<_>>())).collect();
                line(&mut ch, ax, &angles, &a, kind_color(k), LW, Dash::Solid, if i == 0 { Some(kind_label(lang, k)) } else { None }.as_deref());
            }
            outline(&ch);
            if i == 0 {
                legend(&mut ch, SeriesLabelPosition::UpperMiddle, 6.5);
            }
        }
    });
    save(&svg, c.dir, "vertical_colouration");
}

/// The colouration of the paths for one variant: PIR weights, updated weights, rooms (steady,
/// first reflections, floor and ceiling).
pub fn colouration_values(study: &Study, v: &str, w: [f64; 3]) -> [f64; 5] {
    let med = |g: Group| analysis::median_of(&analysis::path_anomaly(study, v, g, study.random(), Metric::Local, true));
    [
        analysis::cta_path_anomaly(study, v, [0.12, 0.44, 0.44], Sp::TwoOrbits, Metric::Local, true),
        analysis::cta_path_anomaly(study, v, w, Sp::Sphere, Metric::Local, true),
        med(Group::Steady),
        med(Group::Reflections),
        med(Group::Vertical),
    ]
}

/// Colouration of the paths: each direction's own deviation within an octave of the crossover,
/// weighted by the energy it brings, for every crossover.
pub fn colouration(c: &Ctx) {
    let lang = c.lang;
    let w = analysis::fit_weights(c.study, Sp::Sphere, Which::Steady);
    let labels: Vec<String> = vec![
        c.t("CTA-2034\nPIR weights", "poids de la\nPIR CTA-2034"),
        c.t("updated\nweights", "poids\nmis à jour"),
        c.t("rooms,\nsteady state", "pièces,\nrégime\nétabli"),
        c.t("rooms, first\nreflections", "pièces,\npremières\nréflexions"),
        c.t("rooms, floor\n+ ceiling", "pièces,\nsol +\nplafond"),
    ];
    let small = 6.0;
    let lines = labels.iter().map(|l| l.lines().count()).max().unwrap();
    let mut axes: Vec<Axes> = ["two_way", "two_way_wg"]
        .iter()
        .enumerate()
        .map(|(i, layout)| {
            let mut a = Axes::lin(
                lang,
                &format!("{}, {}", c.t("two-way", "deux voies"), layout_label(lang, layout)),
                (-0.6, 4.6),
                (0.0, 5.8),
                "",
                &if i == 0 { c.t("colouration of the paths,\n1.1-4.4 kHz (dB RMS)", "coloration des chemins,\n1,1–4,4\u{a0}kHz (dB RMS)") } else { String::new() },
            );
            a.x_ticks = steps(0.0, 4.0, 1.0);
            a.show_xlabels = false;
            a.x_grid = false;
            a.show_ylabels = i == 0;
            a.x_label_area = Some(5.0 + 1.0 + lines as f64 * small * 1.2 + 2.0);
            a
        })
        .collect();
    let svg = render(W2, 2.7, |root| {
        let p = layout(root, 1, 2, &mut axes);
        for (i, layout) in ["two_way", "two_way_wg"].iter().enumerate() {
            let ax = &axes[i];
            let mut ch = ax.build(&p[i]);
            let width = 0.2;
            let mut entries = Vec::new();
            for (ki, k) in KINDS.iter().enumerate() {
                let vals = colouration_values(c.study, &name(layout, "upright", *k), w);
                let col = kind_color(*k);
                ch.draw_series(vals.iter().enumerate().map(|(m, v)| {
                    let x0 = m as f64 + (ki as f64 - 2.0) * width;
                    Rectangle::new([(x0, 0.0), (x0 + width, *v)], col.filled())
                }))
                .unwrap();
                entries.push((Swatch::Patch(col), kind_label(lang, *k)));
            }
            outline(&ch);
            let (_, ys) = ch.plotting_area().get_pixel_range();
            for (m, text) in labels.iter().enumerate() {
                let (bx, _) = ch.backend_coord(&(m as f64, 0.0));
                for (li, l) in text.lines().enumerate() {
                    let y = ys.end + pt(5.0) + pt(1.0) + (li as f64 * small * PT * 1.2).round() as i32;
                    label(root, l, (bx, y), small, &INK_2, HPos::Center, VPos::Top, false);
                }
            }
            if i == 0 {
                legend_in(root, &ch, Place::UpperLeft, &entries, 2, 6.5, Ground::Plain);
            }
        }
    });
    save(&svg, c.dir, "colouration");
}

/// Vertical-plane directivity of the four crossovers, dome and waveguide (relative to the on-axis
/// response).
pub fn polar_maps(c: &Ctx) {
    let lang = c.lang;
    let f = log_grid(300.0, 20000.0, 48);
    let angles: Vec<f64> = (0..=180).map(|i| -90.0 + i as f64).collect();
    let dirs: Vec<Vec3> = angles.iter().map(|a| [0.0, a.to_radians().sin(), a.to_radians().cos()]).collect();
    let layouts: [Layout; 2] = [two_way(), two_way_waveguide()];
    let maps: Vec<Vec<Vec<Vec<f64>>>> = layouts
        .iter()
        .map(|layout| {
            let far = FarField::new(layout, &f);
            let resp = far.responses(&dirs);
            let on: Vec<Vec<crate::C64>> = resp.iter().map(|m| m.row(90).iter().map(|c| crate::C64::new(c.re as f64, c.im as f64)).collect()).collect();
            KINDS
                .iter()
                .map(|&k| {
                    let p = power(&crossover::filters(k, &f, layout, &on), &resp);
                    (0..dirs.len()).map(|d| (0..f.len()).map(|j| db10(p.row(d)[j] as f64) - db10(p.row(90)[j] as f64)).collect()).collect()
                })
                .collect()
        })
        .collect();
    let bar_text = c.t("level re on axis (dB)", "niveau relatif à l’axe (dB)");
    let ticks = steps(-15.0, 15.0, 5.0);
    let bar_w = pt(9.0);
    let zone = pt(6.0) + colour_bar_width(bar_w, &ticks, &bar_text, DESC, lang);
    let mut axes = Vec::new();
    for row in 0..2 {
        for (col, k) in KINDS.iter().enumerate() {
            let ylabel = if col == 0 {
                format!("{}\n{}", if row == 0 { c.t("dome", "dôme") } else { c.t("waveguide", "guide d’onde") }, c.t("vertical angle (deg)", "angle vertical (°)"))
            } else {
                String::new()
            };
            let mut a = Axes::freq(lang, &if row == 0 { kind_label(lang, *k) } else { String::new() }, (300.0, 20000.0), (-90.0, 90.0), &ylabel);
            a.x_ticks = vec![1000.0, 10000.0];
            a.y_ticks = steps(-90.0, 90.0, 45.0);
            a.show_xlabels = row == 1;
            a.show_ylabels = col == 0;
            a.x_grid = false;
            a.y_grid = false;
            axes.push(a);
        }
    }
    let svg = render(W2, 3.8, |root| {
        let grid_w = (W2 * PPI).round() as i32 - zone;
        let (grid, _) = root.split_horizontally(grid_w);
        let p = layout(&grid, 2, 4, &mut axes);
        let lf: Vec<f64> = f.iter().map(|x| x.log10()).collect();
        let (mut top, mut bottom) = (i32::MAX, 0);
        for row in 0..2 {
            for col in 0..4 {
                let mut ch = axes[row * 4 + col].build(&p[row * 4 + col]);
                let m = &maps[row][col];
                let pixels: Vec<RGBColor> = (0..angles.len()).rev().flat_map(|ai| m[ai].iter().map(|v| plot::diverging((v + 18.0) / 36.0)).collect::<Vec<_>>()).collect();
                let marker = plot::raster(f.len(), angles.len(), &pixels);
                ch.draw_series(once(Rectangle::new([(lf[0], -90.0), (lf[f.len() - 1], 90.0)], marker.filled()))).unwrap();
                outline(&ch);
                let (_, ys) = ch.plotting_area().get_pixel_range();
                top = top.min(ys.start);
                bottom = bottom.max(ys.end);
            }
        }
        let bar_h = (0.8 * (bottom - top) as f64).round() as i32;
        let y0 = (top + bottom) / 2 - bar_h / 2;
        colour_bar(root, (grid_w + pt(6.0), y0), (bar_w, bar_h), plot::diverging, (-18.0, 18.0), &ticks, &bar_text, DESC, lang);
    });
    save(&svg, c.dir, "polar_maps");
}

/// The living room's early sound with the paths added in power and with their phases at the
/// fourteen points (LR4 and brickwall, dome and waveguide).
pub fn interference(c: &Ctx, coh: &CoherentResults) {
    let lang = c.lang;
    let f = &coh.freqs;
    let layouts = ["two_way", "two_way_wg"];
    let kinds = [Kind::Lr4, Kind::Fir];
    let mut axes = Vec::new();
    for (row, layout) in layouts.iter().enumerate() {
        for (col, k) in kinds.iter().enumerate() {
            let mut a = Axes::freq(
                lang,
                &format!("{}, {}", layout_label(lang, layout), kind_label(lang, *k)),
                (500.0, 9000.0),
                (-10.0, 4.0),
                &if col == 0 { c.t("early sound re omnidirectional (dB)", "son précoce relatif à l’omni (dB)") } else { String::new() },
            );
            a.x_ticks = vec![500.0, 1000.0, 2000.0, 5000.0];
            a.title_size = 8.0;
            a.show_xlabels = row == 1;
            a.show_ylabels = col == 0;
            axes.push(a);
        }
    }
    let svg = render(W2, 4.6, |root| {
        let p = layout(root, 2, 2, &mut axes);
        for (row, layout) in layouts.iter().enumerate() {
            for (col, k) in kinds.iter().enumerate() {
                let i = 2 * row + col;
                let ax = &axes[i];
                let mut ch = ax.build(&p[i]);
                let curves = &coh.living[coh.variant(&name(layout, "upright", *k))][1];
                let to_db = |v: &[f32]| v.iter().map(|x| db10(*x as f64)).collect::<Vec<f64>>();
                let colour = kind_color(*k);
                for point in &curves[1..] {
                    line(&mut ch, ax, f, &to_db(point), colour.mix(0.35), 0.5, Dash::Solid, None);
                }
                let n = (curves.len() - 1) as f32;
                let mean: Vec<f32> = (0..f.len()).map(|j| curves[1..].iter().map(|c| c[j]).sum::<f32>() / n).collect();
                let first = i == 0;
                line(&mut ch, ax, f, &to_db(&mean), colour, LW, Dash::Solid, first.then(|| c.t("fourteen points, mean", "quatorze points, moyenne")).as_deref());
                line(&mut ch, ax, f, &to_db(&curves[0]), INK, LW, Dash::Dashed, first.then(|| c.t("paths in power", "chemins en énergie")).as_deref());
                if first {
                    ch.draw_series(LineSeries::new(Vec::<(f64, f64)>::new(), colour.mix(0.5).stroke_width(sw(0.5))))
                        .unwrap()
                        .label(c.t("each point, with phases", "chaque point, avec les phases"))
                        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + pt(14.0), y)], colour.mix(0.5).stroke_width(sw(0.8))));
                    legend(&mut ch, SeriesLabelPosition::LowerLeft, 6.5);
                }
            }
        }
    });
    save(&svg, c.dir, "interference");
}

/// Deviation of the early sound within one octave of the crossover, with and without the phases.
pub fn interference_summary(c: &Ctx, coh: &CoherentResults) {
    let lang = c.lang;
    let labels: Vec<String> = vec![
        c.t("paths in\npower", "chemins\nen énergie"),
        c.t("one\npoint", "un\npoint"),
        c.t("two\nears", "deux\noreilles"),
        c.t("fourteen\npoints", "quatorze\npoints"),
    ];
    let small = 6.0;
    let lines = labels.iter().map(|l| l.lines().count()).max().unwrap();
    let omni = coherent::values(coh, OMNI, 1);
    let mut axes: Vec<Axes> = ["two_way", "two_way_wg"]
        .iter()
        .enumerate()
        .map(|(i, layout)| {
            let mut a = Axes::lin(
                lang,
                &format!("{}, {}", c.t("two-way", "deux voies"), layout_label(lang, layout)),
                (-0.6, 3.6),
                (0.0, 1.85),
                "",
                &if i == 0 { c.t("early sound, deviation\n1.1-4.4 kHz (dB RMS)", "son précoce, écart\n1,1–4,4\u{a0}kHz (dB RMS)") } else { String::new() },
            );
            a.x_ticks = steps(0.0, 3.0, 1.0);
            a.y_ticks = steps(0.0, 1.8, 0.2);
            a.show_xlabels = false;
            a.x_grid = false;
            a.show_ylabels = i == 0;
            a.x_label_area = Some(5.0 + 1.0 + lines as f64 * small * 1.2 + 2.0);
            a
        })
        .collect();
    let svg = render(W2, 2.8, |root| {
        let p = layout(root, 1, 2, &mut axes);
        for (i, layout) in ["two_way", "two_way_wg"].iter().enumerate() {
            let ax = &axes[i];
            let mut ch = ax.build(&p[i]);
            let width = 0.2;
            let mut entries = Vec::new();
            for (ki, k) in KINDS.iter().enumerate() {
                let vals = coherent::values(coh, &name(layout, "upright", *k), 1);
                let col = kind_color(*k);
                ch.draw_series(vals.iter().enumerate().map(|(m, v)| {
                    let x0 = m as f64 + (ki as f64 - 2.0) * width;
                    Rectangle::new([(x0, 0.0), (x0 + width, *v)], col.filled())
                }))
                .unwrap();
                entries.push((Swatch::Patch(col), kind_label(lang, *k)));
            }
            for (m, v) in omni.iter().enumerate().skip(1) {
                ch.draw_series(once(PathElement::new(vec![(m as f64 - 0.45, *v), (m as f64 + 0.45, *v)], INK.stroke_width(sw(1.2))))).unwrap();
            }
            entries.push((Swatch::Line(INK, 1.2, Dash::Solid), c.t("omnidirectional source", "source omnidirectionnelle")));
            outline(&ch);
            let (_, ys) = ch.plotting_area().get_pixel_range();
            for (m, text) in labels.iter().enumerate() {
                let (bx, _) = ch.backend_coord(&(m as f64, 0.0));
                for (li, l) in text.lines().enumerate() {
                    let y = ys.end + pt(5.0) + pt(1.0) + (li as f64 * small * PT * 1.2).round() as i32;
                    label(root, l, (bx, y), small, &INK_2, HPos::Center, VPos::Top, false);
                }
            }
            if i == 0 {
                legend_in(root, &ch, Place::UpperLeft, &entries, 2, 6.5, Ground::Plain);
            }
        }
    });
    save(&svg, c.dir, "interference_summary");
}

/// Every figure of the report in one language.
pub fn all(study: &Study, setups: &[Setup], lang: Lang, dir: &Path, spectra: &Spectra, coh: Option<&CoherentResults>) {
    let c = Ctx { study, lang, dir, setups };
    setup(&c);
    kernel_maps(&c);
    shares(&c);
    reflection_angles(&c);
    spin(&c, "two_way", "spin_dome");
    spin(&c, "two_way_wg", "spin_waveguide");
    sp_orbits(&c);
    errors(&c);
    rotation(&c);
    crossovers(&c);
    paths(&c, spectra);
    vertical_colouration(&c);
    colouration(&c);
    polar_maps(&c);
    if let Some(coh) = coh {
        interference(&c, coh);
        interference_summary(&c, coh);
    }
}
