//! Figure helpers: SVG drawn with plotters, converted to PDF (svg2pdf) for the report and to PNG
//! (resvg) for the README, with the DejaVu Sans font of `data/fonts`. The style is the matplotlib
//! one of the turntable study: text and strokes in points, the room around each plotting area
//! fitted to its labels, panels sharing the labels of a common axis.

use std::cell::RefCell;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use plotters::coord::ranged1d::{DefaultFormatting, KeyPointHint};
use plotters::coord::Shift;
use plotters::element::DashedPathElement;
use plotters::prelude::*;
use plotters::style::text_anchor::{HPos, Pos, VPos};

pub const FONT: &str = "DejaVu Sans";
static FONT_DATA: &[u8] = include_bytes!("../data/fonts/DejaVuSans.ttf");

/// Pixels per inch of the SVG canvas, fine enough for plotters' integer coordinates.
pub const PPI: f64 = 300.0;
/// Canvas pixels per point.
pub const PT: f64 = PPI / 72.0;
/// Text sizes (points): tick labels, axis labels, titles.
pub const LABEL: f64 = 7.5;
pub const DESC: f64 = 8.0;
pub const TITLE: f64 = 8.5;
/// Width of the curves (points).
pub const LW: f64 = 1.2;

/// A length in points, in canvas pixels.
pub fn pt(v: f64) -> i32 {
    (v * PT).round() as i32
}

/// A stroke width in points, in canvas pixels.
pub fn sw(w: f64) -> u32 {
    (w * PT).round().max(1.0) as u32
}

/// A colour blended with white (an opaque tint).
pub fn tint(c: RGBColor, alpha: f64) -> RGBColor {
    let m = |x: u8| (255.0 - (255.0 - x as f64) * alpha).round() as u8;
    RGBColor(m(c.0), m(c.1), m(c.2))
}

pub const INK: RGBColor = RGBColor(0x0b, 0x0b, 0x0b);
pub const INK_2: RGBColor = RGBColor(0x52, 0x51, 0x4e);
pub const MUTED: RGBColor = RGBColor(0x89, 0x87, 0x81);
pub const GRID: RGBColor = RGBColor(0xe1, 0xe0, 0xd9);
pub const AXIS: RGBColor = RGBColor(0xc3, 0xc2, 0xb7);
pub const NEUTRAL: RGBColor = RGBColor(0xf0, 0xef, 0xec);
/// Categorical colours, in a fixed order.
pub const SLOTS: [RGBColor; 8] = [
    RGBColor(0x2a, 0x78, 0xd6),
    RGBColor(0xeb, 0x68, 0x34),
    RGBColor(0x1b, 0xaf, 0x7a),
    RGBColor(0xed, 0xa1, 0x00),
    RGBColor(0xe8, 0x7b, 0xa4),
    RGBColor(0x00, 0x83, 0x00),
    RGBColor(0x62, 0x50, 0xd6),
    RGBColor(0xe3, 0x49, 0x48),
];

/// The language of a figure or a table.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Lang {
    En,
    Fr,
}

impl Lang {
    pub fn t(&self, en: &str, fr: &str) -> String {
        match self {
            Lang::En => en.to_string(),
            Lang::Fr => fr.to_string(),
        }
    }

    /// A number with ``prec`` decimals for a figure: decimal comma in French, true minus sign.
    pub fn num(&self, x: f64, prec: usize) -> String {
        let s = format!("{:.*}", prec, x);
        let s = if *self == Lang::Fr { s.replace('.', ",") } else { s };
        s.replace('-', "\u{2212}")
    }

    /// A number for LaTeX: decimal comma in French.
    pub fn tex(&self, x: f64, prec: usize) -> String {
        let s = format!("{:.*}", prec, x);
        if *self == Lang::Fr {
            s.replace('.', ",")
        } else {
            s
        }
    }

    /// A colon, with the no-break space French typography puts before it.
    pub fn colon(&self) -> &'static str {
        match self {
            Lang::En => ":",
            Lang::Fr => "\u{a0}:",
        }
    }

    pub fn percent(&self) -> &'static str {
        match self {
            Lang::En => " %",
            Lang::Fr => "\u{202f}%",
        }
    }
}

fn lerp_colors(stops: &[(u8, u8, u8)], t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0) * (stops.len() - 1) as f64;
    let i = (t.floor() as usize).min(stops.len() - 2);
    let u = t - i as f64;
    let (a, b) = (stops[i], stops[i + 1]);
    let mix = |x: u8, y: u8| (x as f64 + (y as f64 - x as f64) * u).round() as u8;
    RGBColor(mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

/// Diverging map: blue (low) to grey to red (high), ``t`` in [0, 1].
pub fn diverging(t: f64) -> RGBColor {
    lerp_colors(
        &[
            (0x0d, 0x36, 0x6b),
            (0x25, 0x6a, 0xbf),
            (0x6d, 0xa7, 0xec),
            (0xb7, 0xd3, 0xf6),
            (0xf0, 0xef, 0xec),
            (0xf6, 0xc1, 0xbf),
            (0xec, 0x85, 0x84),
            (0xd6, 0x3b, 0x3a),
            (0x7a, 0x1c, 0x1c),
        ],
        t,
    )
}

/// Range (dB) of the directivity maps: the level off the axis, relative to the axis, reaches a few
/// dB above it at most.
pub const MAP_RANGE: (f64, f64) = (-18.0, 3.0);

/// Colour of a level relative to the axis on the directivity maps: the diverging map centred on
/// 0 dB, clipped to [`MAP_RANGE`].
pub fn level_colour(v: f64) -> RGBColor {
    diverging((v.clamp(MAP_RANGE.0, MAP_RANGE.1) + 18.0) / 36.0)
}

/// The same colours along a colour bar, ``t`` in [0, 1] over [`MAP_RANGE`].
pub fn level_bar(t: f64) -> RGBColor {
    level_colour(MAP_RANGE.0 + t * (MAP_RANGE.1 - MAP_RANGE.0))
}

/// Grey map, white to black.
pub fn greys(t: f64) -> RGBColor {
    lerp_colors(&[(0xff, 0xff, 0xff), (0xbd, 0xbd, 0xbd), (0x73, 0x73, 0x73), (0x00, 0x00, 0x00)], t)
}

// ---------------------------------------------------------------------------------------------
// Text
// ---------------------------------------------------------------------------------------------

/// A text style of ``size`` points (plotters' SVG backend writes sizes divided by 1.24).
pub fn font(size: f64) -> TextStyle<'static> {
    font_c(size, &INK)
}

pub fn font_c(size: f64, color: &RGBColor) -> TextStyle<'static> {
    (FONT, size * PT * 1.24).into_font().color(color)
}

fn face() -> &'static ttf_parser::Face<'static> {
    static FACE: OnceLock<ttf_parser::Face<'static>> = OnceLock::new();
    FACE.get_or_init(|| ttf_parser::Face::parse(FONT_DATA, 0).expect("DejaVu Sans"))
}

/// Width of a text of ``size`` points, in canvas pixels.
pub fn text_width(s: &str, size: f64) -> f64 {
    let f = face();
    let em = f.units_per_em() as f64;
    let units: f64 = s
        .chars()
        .map(|c| f.glyph_index(c).and_then(|g| f.glyph_hor_advance(g)).map_or(0.5 * em, |a| a as f64))
        .sum();
    units / em * size * PT
}

/// Writes ``s`` at ``(x, y)``, pixels of ``area``, anchored by ``h`` and ``v``; ``up``: turned to
/// read upward (the anchor's top is then on the left).
#[allow(clippy::too_many_arguments)]
pub fn label<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>, s: &str, (x, y): (i32, i32), size: f64, color: &RGBColor, h: HPos, v: VPos, up: bool) {
    let mut style = font_c(size, color).pos(Pos::new(h, v));
    if up {
        style = style.transform(FontTransform::Rotate270);
    }
    area.draw_text(s, &style, (x, y)).unwrap();
}

/// Formats axis values with the decimals the step needs (the same for every tick, as matplotlib),
/// a decimal comma in French and a true minus sign.
pub fn axis_labels(lang: Lang, values: &[f64]) -> Vec<String> {
    let dec = (0..3).find(|&d| values.iter().all(|v| (v * 10f64.powi(d) - (v * 10f64.powi(d)).round()).abs() < 1e-6)).unwrap_or(3) as usize;
    values.iter().map(|v| lang.num(if v.abs() < 1e-12 { 0.0 } else { *v }, dec)).collect()
}

/// One axis value, with as few decimals as it needs.
pub fn axis_num(lang: Lang, v: f64) -> String {
    axis_labels(lang, &[v]).remove(0)
}

// ---------------------------------------------------------------------------------------------
// Canvas
// ---------------------------------------------------------------------------------------------

thread_local! {
    /// Raster images waiting for their place: the colour of the rectangle marking it, the PNG.
    static IMAGES: RefCell<Vec<(RGBColor, Vec<u8>)>> = const { RefCell::new(Vec::new()) };
}

/// Registers an image of ``w`` x ``h`` pixels (rows from the top) and returns the colour of the
/// rectangle to draw where it goes: [`render`] puts the image in its place.
pub fn raster(w: usize, h: usize, pixels: &[RGBColor]) -> RGBColor {
    assert_eq!(pixels.len(), w * h);
    let mut pm = resvg::tiny_skia::Pixmap::new(w as u32, h as u32).expect("image size");
    for (p, c) in pm.pixels_mut().iter_mut().zip(pixels) {
        *p = resvg::tiny_skia::PremultipliedColorU8::from_rgba(c.0, c.1, c.2, 255).unwrap();
    }
    let png = pm.encode_png().expect("PNG");
    IMAGES.with(|v| {
        let mut v = v.borrow_mut();
        let marker = RGBColor(0x01, 0x02, 0x03 + v.len() as u8);
        v.push((marker, png));
        marker
    })
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(data.len().div_ceil(3) * 4);
    for c in data.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            s.push(if i <= c.len() { T[(n >> (18 - 6 * i)) as usize & 63] as char } else { '=' });
        }
    }
    s
}

/// Puts the registered images in place of their marker rectangles.
fn place_images(svg: &mut String) {
    for (marker, png) in IMAGES.with(|v| std::mem::take(&mut *v.borrow_mut())) {
        let key = format!("fill=\"#{:02X}{:02X}{:02X}\"", marker.0, marker.1, marker.2);
        let at = svg.find(&key).expect("image marker");
        let start = svg[..at].rfind("<rect").expect("marker rectangle");
        let end = at + svg[at..].find("/>").expect("marker end") + 2;
        let tag = svg[start..end].to_string();
        let attr = |name: &str| {
            let k = format!(" {name}=\"");
            let i = tag.find(&k).expect("marker attribute") + k.len();
            tag[i..i + tag[i..].find('"').unwrap()].to_string()
        };
        let image = format!(
            "<image x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\" preserveAspectRatio=\"none\" href=\"data:image/png;base64,{}\"/>",
            attr("x"),
            attr("y"),
            attr("width"),
            attr("height"),
            base64(&png)
        );
        svg.replace_range(start..end, &image);
    }
}

/// Draws a figure of ``w_in`` x ``h_in`` inches and returns its SVG.
pub fn render(w_in: f64, h_in: f64, draw: impl FnOnce(&DrawingArea<SVGBackend, Shift>)) -> String {
    IMAGES.with(|v| v.borrow_mut().clear());
    let mut svg = String::new();
    {
        let size = ((w_in * PPI).round() as u32, (h_in * PPI).round() as u32);
        let root = SVGBackend::with_string(&mut svg, size).into_drawing_area();
        root.fill(&WHITE).unwrap();
        draw(&root);
        root.present().unwrap();
    }
    place_images(&mut svg);
    svg
}

fn usvg_options() -> usvg::Options<'static> {
    let mut opt = usvg::Options { font_family: FONT.to_string(), ..Default::default() };
    let mut db = usvg::fontdb::Database::new();
    db.load_font_data(FONT_DATA.to_vec());
    db.set_sans_serif_family(FONT);
    opt.fontdb = Arc::new(db);
    opt
}

/// Writes ``stem.pdf`` and ``stem.png`` (200 pixels per inch) in ``dir``.
pub fn save(svg: &str, dir: &Path, stem: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let opt = usvg_options();
    let tree = usvg::Tree::from_str(svg, &opt).expect("figure SVG");
    let pdf = svg2pdf::to_pdf(&tree, svg2pdf::ConversionOptions::default(), svg2pdf::PageOptions::default()).expect("PDF");
    std::fs::write(dir.join(format!("{stem}.pdf")), pdf).unwrap();
    let size = tree.size();
    let scale = (200.0 / PPI) as f32;
    let mut pixmap = resvg::tiny_skia::Pixmap::new((size.width() * scale).round() as u32, (size.height() * scale).round() as u32).unwrap();
    pixmap.fill(resvg::tiny_skia::Color::WHITE);
    resvg::render(&tree, resvg::tiny_skia::Transform::from_scale(scale, scale), &mut pixmap.as_mut());
    pixmap.save_png(dir.join(format!("{stem}.png"))).unwrap();
}

/// Writes ``stem.svg`` in ``dir`` for the web: the text turned into paths, so that it looks the
/// same in every browser.
pub fn save_web(svg: &str, dir: &Path, stem: &str) {
    std::fs::create_dir_all(dir).unwrap();
    let tree = usvg::Tree::from_str(svg, &usvg_options()).expect("figure SVG");
    let opt = usvg::WriteOptions { coordinates_precision: 2, transforms_precision: 4, ..Default::default() };
    std::fs::write(dir.join(format!("{stem}.svg")), tree.to_string(&opt)).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------------------------

/// A linear axis with given bold (labelled) and light key points.
#[derive(Clone, Debug)]
pub struct KeyAxis {
    pub lo: f64,
    pub hi: f64,
    pub bold: Vec<f64>,
    pub light: Vec<f64>,
}

impl Ranged for KeyAxis {
    type FormatOption = DefaultFormatting;
    type ValueType = f64;

    fn map(&self, v: &f64, limit: (i32, i32)) -> i32 {
        let t = (v - self.lo) / (self.hi - self.lo);
        (limit.0 as f64 + t * (limit.1 - limit.0) as f64).round() as i32
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<f64> {
        if hint.max_num_points() == 0 {
            Vec::new()
        } else if hint.weight().allow_light_points() {
            self.light.clone()
        } else {
            self.bold.clone()
        }
    }

    fn range(&self) -> Range<f64> {
        self.lo..self.hi
    }
}

pub type Coord = Cartesian2d<KeyAxis, KeyAxis>;

fn nice_step(span: f64, n: usize) -> f64 {
    let raw = span / n as f64;
    let p = 10f64.powf(raw.log10().floor());
    for m in [1.0, 2.0, 2.5, 5.0, 10.0] {
        if m * p >= raw * (1.0 - 1e-9) {
            return m * p;
        }
    }
    10.0 * p
}

fn auto_ticks(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    let step = nice_step(hi - lo, n);
    let first = (lo / step - 1e-9).ceil() * step;
    (0..).map(|i| first + i as f64 * step).take_while(|v| *v <= hi + 1e-9 * step).collect()
}

pub fn freq_label(f: f64) -> String {
    if f >= 1000.0 {
        let k = f / 1000.0;
        if (k - k.round()).abs() < 1e-9 {
            format!("{}k", k.round())
        } else {
            format!("{k}k")
        }
    } else {
        format!("{}", f.round())
    }
}

/// Frequency ticks within a range: 20, 50, 100, 200, 500, 1k ... (``sparse``: decades).
pub fn freq_ticks(lo: f64, hi: f64, sparse: bool) -> Vec<f64> {
    let all: &[f64] = if sparse {
        &[20.0, 100.0, 1000.0, 10000.0]
    } else {
        &[20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
    };
    all.iter().copied().filter(|&f| f >= lo * 0.999 && f <= hi * 1.001).collect()
}

/// Room around a plotting area, in canvas pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Frame {
    pub left: i32,
    pub right: i32,
    pub top: i32,
    pub bottom: i32,
}

/// Settings of a chart.
#[derive(Clone, Debug)]
pub struct Axes {
    pub title: String,
    pub x: (f64, f64),
    pub y: (f64, f64),
    pub xlabel: String,
    /// Lines separated by ``\n``.
    pub ylabel: String,
    /// Major ticks (data units; for a frequency axis, in Hz).
    pub x_ticks: Vec<f64>,
    pub y_ticks: Vec<f64>,
    /// Frequency axis: x is log10(f).
    pub log_x: bool,
    pub title_size: f64,
    pub lang: Lang,
    /// Tick labels; off on the panels that share an axis with a neighbour.
    pub show_xlabels: bool,
    pub show_ylabels: bool,
    pub x_grid: bool,
    pub y_grid: bool,
    /// Room under the axis (points) for tick labels the figure writes itself.
    pub x_label_area: Option<f64>,
    /// Room around the plotting area, set by a layout.
    pub frame: Option<Frame>,
}

/// Distance between an axis and its tick labels: twice the tick length, as plotters puts them.
fn label_dist() -> i32 {
    2 * pt(2.5)
}

impl Axes {
    pub fn freq(lang: Lang, title: &str, f: (f64, f64), y: (f64, f64), ylabel: &str) -> Self {
        Axes {
            x_ticks: freq_ticks(f.0, f.1, false),
            log_x: true,
            ..Axes::lin(lang, title, f, y, "", ylabel)
        }
    }

    pub fn lin(lang: Lang, title: &str, x: (f64, f64), y: (f64, f64), xlabel: &str, ylabel: &str) -> Self {
        Axes {
            title: title.into(),
            x,
            y,
            xlabel: xlabel.into(),
            ylabel: ylabel.into(),
            x_ticks: Vec::new(),
            y_ticks: Vec::new(),
            log_x: false,
            title_size: TITLE,
            lang,
            show_xlabels: true,
            show_ylabels: true,
            x_grid: true,
            y_grid: true,
            x_label_area: None,
            frame: None,
        }
    }

    /// Converts a data x to the chart's x.
    pub fn cx(&self, x: f64) -> f64 {
        if self.log_x {
            x.log10()
        } else {
            x
        }
    }

    fn x_keys(&self) -> Vec<f64> {
        if self.x_ticks.is_empty() {
            auto_ticks(self.cx(self.x.0), self.cx(self.x.1), 9)
        } else {
            self.x_ticks.iter().map(|t| self.cx(*t)).collect()
        }
    }

    fn y_keys(&self) -> Vec<f64> {
        if self.y_ticks.is_empty() {
            auto_ticks(self.y.0, self.y.1, 9)
        } else {
            self.y_ticks.clone()
        }
    }

    fn x_texts(&self) -> Vec<String> {
        let keys = self.x_keys();
        if self.log_x {
            keys.iter().map(|v| freq_label(10f64.powf(*v))).collect()
        } else {
            axis_labels(self.lang, &keys)
        }
    }

    /// The room the title, the labels and the tick labels need around the plotting area.
    pub fn need(&self) -> Frame {
        let tick = pt(2.5) as f64;
        let dist = label_dist() as f64;
        let lines = if self.ylabel.is_empty() { 0 } else { self.ylabel.lines().count() };
        let y_w = if self.show_ylabels { axis_labels(self.lang, &self.y_keys()).iter().map(|s| text_width(s, LABEL)).fold(0.0, f64::max) } else { 0.0 };
        let mut left = if self.show_ylabels { dist + y_w } else { tick };
        if lines > 0 {
            left += PT * 4.0 + lines as f64 * DESC * PT * 1.2;
        }
        left += PT * 2.0;
        let mut right = PT * 4.0;
        if self.show_xlabels {
            let (lo, hi) = (self.cx(self.x.0), self.cx(self.x.1));
            for (k, s) in self.x_keys().iter().zip(self.x_texts()) {
                let half = text_width(&s, LABEL) / 2.0 + PT * 1.5;
                if (k - hi).abs() < 1e-6 * (hi - lo) {
                    right = right.max(half);
                }
                if (k - lo).abs() < 1e-6 * (hi - lo) {
                    left = left.max(half);
                }
            }
        }
        let bottom = match self.x_label_area {
            Some(a) => a * PT,
            None => {
                let ticks = if self.show_xlabels { dist + LABEL * PT * 1.05 } else { tick };
                ticks + if self.xlabel.is_empty() { 0.0 } else { PT * 3.0 + DESC * PT * 1.1 } + PT * 2.0
            }
        };
        let top = if self.title.is_empty() { PT * 5.0 } else { PT * 7.0 + self.title_size * PT * 1.05 };
        Frame { left: left.ceil() as i32, right: right.ceil() as i32, top: top.ceil() as i32, bottom: bottom.ceil() as i32 }
    }

    /// Draws the frame, grid, ticks, labels and title, and returns the chart.
    pub fn build<'a, DB: DrawingBackend>(&self, area: &'a DrawingArea<DB, Shift>) -> ChartContext<'a, DB, Coord> {
        let fr = self.frame.unwrap_or_else(|| self.need());
        let (lo, hi) = (self.cx(self.x.0), self.cx(self.x.1));
        let mut light = Vec::new();
        if self.log_x {
            let mut decade = 10f64.powf(lo.floor());
            while decade < self.x.1 * 10.0 {
                for m in 1..10 {
                    let v = (decade * m as f64).log10();
                    if v > lo && v < hi {
                        light.push(v);
                    }
                }
                decade *= 10.0;
            }
        }
        let xk = self.x_keys();
        let yk = self.y_keys();
        let x_texts = self.x_texts();
        let y_texts = axis_labels(self.lang, &yk);
        let xr = KeyAxis { lo, hi, bold: xk.clone(), light };
        let yr = KeyAxis { lo: self.y.0, hi: self.y.1, bold: yk.clone(), light: Vec::new() };
        let mut chart = ChartBuilder::on(area)
            .margin_top(fr.top)
            .margin_right(fr.right)
            .margin_left(0)
            .margin_bottom(0)
            .x_label_area_size(fr.bottom)
            .y_label_area_size(fr.left)
            .build_cartesian_2d(xr, yr)
            .unwrap();
        let (sx, sy) = (self.show_xlabels, self.show_ylabels);
        let x_fmt = move |v: &f64| {
            let i = xk.iter().position(|k| (k - v).abs() < 1e-9);
            if sx {
                i.map(|i| x_texts[i].clone()).unwrap_or_default()
            } else {
                String::new()
            }
        };
        let y_fmt = move |v: &f64| {
            let i = yk.iter().position(|k| (k - v).abs() < 1e-9);
            if sy {
                i.map(|i| y_texts[i].clone()).unwrap_or_default()
            } else {
                String::new()
            }
        };
        let mut mesh = chart.configure_mesh();
        mesh.light_line_style(tint(GRID, 0.6).stroke_width(sw(0.5)))
            .bold_line_style(GRID.stroke_width(sw(0.5)))
            .axis_style(AXIS.stroke_width(sw(0.6)))
            .set_all_tick_mark_size(pt(2.5))
            .x_label_formatter(&x_fmt)
            .y_label_formatter(&y_fmt)
            .label_style(font_c(LABEL, &INK_2));
        if !self.x_grid {
            mesh.disable_x_mesh();
        }
        if !self.y_grid {
            mesh.disable_y_mesh();
        }
        mesh.draw().unwrap();
        outline(&chart);
        // Title and axis labels, placed against the plotting area as matplotlib does.
        let (xs, ys) = chart.plotting_area().get_pixel_range();
        let (bx, by) = area.get_base_pixel();
        let (l, r, t, b) = (xs.start - bx, xs.end - bx, ys.start - by, ys.end - by);
        if !self.title.is_empty() {
            // Centred on the plotting area, but kept within the panel when it is wider.
            let half = (text_width(&self.title, self.title_size) / 2.0).ceil() as i32 + pt(1.0);
            let aw = area.dim_in_pixel().0 as i32;
            let cx = if 2 * half < aw { ((l + r) / 2).clamp(half, aw - half) } else { (l + r) / 2 };
            label(area, &self.title, (cx, t - pt(5.0)), self.title_size, &INK, HPos::Center, VPos::Bottom, false);
        }
        if !self.xlabel.is_empty() {
            let ticks = if self.show_xlabels { label_dist() + (LABEL * PT * 1.05) as i32 } else { pt(2.5) };
            label(area, &self.xlabel, ((l + r) / 2, b + ticks + pt(3.0)), DESC, &INK_2, HPos::Center, VPos::Top, false);
        }
        if !self.ylabel.is_empty() {
            let y_w = if self.show_ylabels { axis_labels(self.lang, &self.y_keys()).iter().map(|s| text_width(s, LABEL)).fold(0.0, f64::max) } else { 0.0 };
            let edge = l - if self.show_ylabels { label_dist() + y_w.ceil() as i32 } else { pt(2.5) } - pt(4.0);
            let lines: Vec<&str> = self.ylabel.lines().collect();
            for (i, s) in lines.iter().enumerate() {
                let x = edge - ((lines.len() - i) as f64 * DESC * PT * 1.2).round() as i32;
                label(area, s, (x, (t + b) / 2), DESC, &INK_2, HPos::Center, VPos::Top, true);
            }
        }
        chart
    }
}

/// Splits ``area`` into ``rows`` x ``cols`` panels for ``axes`` (row by row) so that every
/// plotting area has the same size: each column gets the room its panels need on the left and on
/// the right, each row the room its panels need above and below (matplotlib's tight layout).
pub fn layout<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>, rows: usize, cols: usize, axes: &mut [Axes]) -> Vec<DrawingArea<DB, Shift>> {
    assert_eq!(axes.len(), rows * cols);
    let need: Vec<Frame> = axes.iter().map(|a| a.need()).collect();
    let col = |f: fn(&Frame) -> i32| -> Vec<i32> { (0..cols).map(|c| (0..rows).map(|r| f(&need[r * cols + c])).max().unwrap()).collect() };
    let row = |f: fn(&Frame) -> i32| -> Vec<i32> { (0..rows).map(|r| (0..cols).map(|c| f(&need[r * cols + c])).max().unwrap()).collect() };
    let (left, right) = (col(|f| f.left), col(|f| f.right));
    let (top, bottom) = (row(|f| f.top), row(|f| f.bottom));
    for r in 0..rows {
        for c in 0..cols {
            axes[r * cols + c].frame = Some(Frame { left: left[c], right: right[c], top: top[r], bottom: bottom[r] });
        }
    }
    let (w, h) = area.dim_in_pixel();
    let pw = (w as i32 - left.iter().sum::<i32>() - right.iter().sum::<i32>()) as f64 / cols as f64;
    let ph = (h as i32 - top.iter().sum::<i32>() - bottom.iter().sum::<i32>()) as f64 / rows as f64;
    let mut xs = Vec::new();
    let mut acc = 0.0;
    for c in 0..cols - 1 {
        acc += (left[c] + right[c]) as f64 + pw;
        xs.push(acc.round() as i32);
    }
    let mut ys = Vec::new();
    acc = 0.0;
    for r in 0..rows - 1 {
        acc += (top[r] + bottom[r]) as f64 + ph;
        ys.push(acc.round() as i32);
    }
    area.split_by_breakpoints(xs, ys)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Dash {
    Solid,
    Dashed,
    Dotted,
}

impl Dash {
    /// Dash and gap (canvas pixels) for a line ``width`` points wide, as matplotlib's.
    fn pattern(self, width: f64) -> (i32, i32) {
        match self {
            Dash::Solid => (i32::MAX / 4, 1),
            Dash::Dashed => (pt(3.7 * width).max(2), pt(1.6 * width).max(2)),
            Dash::Dotted => (pt(width).max(1), pt(1.65 * width).max(2)),
        }
    }
}

/// Length of a line in a legend.
fn handle() -> i32 {
    pt(14.0)
}

/// A line; ``x`` in data units (Hz on a frequency chart), ``width`` in points; ``label`` adds it
/// to the legend.
#[allow(clippy::too_many_arguments)]
pub fn line<DB: DrawingBackend>(
    chart: &mut ChartContext<'_, DB, Coord>,
    ax: &Axes,
    x: &[f64],
    y: &[f64],
    color: impl Into<ShapeStyle>,
    width: f64,
    dash: Dash,
    label: Option<&str>,
) {
    let pts: Vec<(f64, f64)> = x
        .iter()
        .zip(y)
        .filter(|(x, _)| **x >= ax.x.0 * (1.0 - 1e-9) && **x <= ax.x.1 * (1.0 + 1e-9) || !ax.log_x && **x >= ax.x.0 && **x <= ax.x.1)
        .map(|(x, v)| (ax.cx(*x), *v))
        .collect();
    let mut style: ShapeStyle = color.into();
    style.stroke_width = sw(width);
    let (on, off) = dash.pattern(width);
    if let Some(l) = label {
        let len = handle();
        chart
            .draw_series(LineSeries::new(Vec::<(f64, f64)>::new(), style))
            .unwrap()
            .label(l)
            .legend(move |(x, y)| DashedPathElement::new(vec![(x, y), (x + len, y)], on, off, style));
    }
    // plotters pulls the points outside the plotting area onto its edge: clip instead.
    for part in clip_y(&pts, ax.y.0, ax.y.1) {
        match dash {
            Dash::Solid => chart.draw_series(LineSeries::new(part, style)).unwrap(),
            _ => chart.draw_series(DashedLineSeries::new(part, on, off, style)).unwrap(),
        };
    }
}

/// Splits a polyline into the pieces that stay within ``lo <= y <= hi``.
pub fn clip_y(pts: &[(f64, f64)], lo: f64, hi: f64) -> Vec<Vec<(f64, f64)>> {
    let mut out = Vec::new();
    let mut cur: Vec<(f64, f64)> = Vec::new();
    for w in pts.windows(2) {
        let ((x0, y0), (x1, y1)) = (w[0], w[1]);
        let dy = y1 - y0;
        let (t0, t1) = if dy.abs() < 1e-300 {
            if y0 < lo || y0 > hi {
                (1.0, 0.0)
            } else {
                (0.0, 1.0)
            }
        } else {
            let (ta, tb) = ((lo - y0) / dy, (hi - y0) / dy);
            (ta.min(tb).max(0.0), ta.max(tb).min(1.0))
        };
        if t0 > t1 {
            if cur.len() > 1 {
                out.push(std::mem::take(&mut cur));
            }
            cur.clear();
            continue;
        }
        let at = |t: f64| (x0 + t * (x1 - x0), y0 + t * dy);
        if t0 > 0.0 || cur.is_empty() {
            if cur.len() > 1 {
                out.push(std::mem::take(&mut cur));
            }
            cur = vec![at(t0)];
        }
        cur.push(at(t1));
        if t1 < 1.0 {
            out.push(std::mem::take(&mut cur));
        }
    }
    if cur.len() > 1 {
        out.push(cur);
    }
    out
}

/// Draws the frame of a chart's plotting area again, over what covers it.
pub fn outline<DB: DrawingBackend>(chart: &ChartContext<'_, DB, Coord>) {
    let pa = chart.plotting_area().strip_coord_spec();
    let (w, h) = pa.dim_in_pixel();
    pa.draw(&Rectangle::new([(0, 0), (w as i32 - 1, h as i32 - 1)], AXIS.stroke_width(sw(0.6)))).unwrap();
}

/// A horizontal line across the chart.
pub fn hline<DB: DrawingBackend>(chart: &mut ChartContext<'_, DB, Coord>, ax: &Axes, y: f64, color: RGBColor, width: f64) {
    let (a, b) = (ax.cx(ax.x.0), ax.cx(ax.x.1));
    chart.draw_series(std::iter::once(PathElement::new(vec![(a, y), (b, y)], color.stroke_width(sw(width))))).unwrap();
}

/// A vertical line across the chart, ``x`` in data units.
pub fn vline<DB: DrawingBackend>(chart: &mut ChartContext<'_, DB, Coord>, ax: &Axes, x: f64, color: RGBColor, width: f64, dash: Dash) {
    let pts = vec![(ax.cx(x), ax.y.0), (ax.cx(x), ax.y.1)];
    let style = color.stroke_width(sw(width));
    let (on, off) = dash.pattern(width);
    match dash {
        Dash::Solid => chart.draw_series(LineSeries::new(pts, style)).unwrap(),
        _ => chart.draw_series(DashedLineSeries::new(pts, on, off, style)).unwrap(),
    };
}

/// A band between two curves.
#[allow(clippy::too_many_arguments)]
pub fn band<DB: DrawingBackend>(
    chart: &mut ChartContext<'_, DB, Coord>,
    ax: &Axes,
    x: &[f64],
    lo: &[f64],
    hi: &[f64],
    color: RGBColor,
    alpha: f64,
    label: Option<&str>,
) {
    let keep: Vec<usize> = (0..x.len()).filter(|&i| x[i] >= ax.x.0 && x[i] <= ax.x.1).collect();
    let mut pts: Vec<(f64, f64)> = keep.iter().map(|&i| (ax.cx(x[i]), lo[i])).collect();
    pts.extend(keep.iter().rev().map(|&i| (ax.cx(x[i]), hi[i])));
    let fill = tint(color, alpha);
    let anno = chart.draw_series(std::iter::once(Polygon::new(pts, fill.filled()))).unwrap();
    if let Some(l) = label {
        let len = handle();
        anno.label(l).legend(move |(x, y)| Rectangle::new([(x, y - pt(2.8)), (x + len, y + pt(2.8))], fill.filled()));
    }
}

/// The legend of a chart's labelled series, without a frame (matplotlib's style here).
pub fn legend<'a, DB: DrawingBackend + 'a>(chart: &mut ChartContext<'a, DB, Coord>, pos: SeriesLabelPosition, size: f64)
where
    DB::ErrorType: 'static,
{
    chart
        .configure_series_labels()
        .position(pos)
        .background_style(TRANSPARENT)
        .border_style(TRANSPARENT)
        .label_font(font_c(size, &INK))
        .margin(pt(4.0))
        .legend_area_size(handle() + pt(5.0))
        .draw()
        .unwrap();
}

/// What a legend entry shows.
#[derive(Clone, Copy, Debug)]
pub enum Swatch {
    /// A line of a width (points) and a dash.
    Line(RGBColor, f64, Dash),
    /// A filled patch.
    Patch(RGBColor),
    /// A disc of a radius (points).
    Dot(RGBColor, f64),
}

/// The ground of a legend.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Ground {
    None,
    /// Translucent white.
    Plain,
    /// Translucent white with a border.
    Framed,
}

/// Where a legend goes, against a chart's plotting area.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Place {
    UpperLeft,
    UpperCenter,
    UpperRight,
    LowerLeft,
    LowerCenter,
    LowerRight,
    /// Outside, on the right, centred vertically.
    Right,
}

/// Size (canvas pixels) of a legend of ``entries`` in ``ncol`` columns, text of ``size`` points.
pub fn legend_size(entries: &[(Swatch, String)], ncol: usize, size: f64) -> (i32, i32) {
    let em = size * PT;
    let rows = entries.len().div_ceil(ncol);
    let widths: Vec<f64> = (0..ncol).map(|c| entries.iter().skip(c * rows).take(rows).map(|(_, t)| text_width(t, size)).fold(0.0, f64::max)).collect();
    let w = 0.8 * em + widths.iter().map(|tw| handle() as f64 + 0.8 * em + tw).sum::<f64>() + 2.0 * em * (ncol - 1) as f64;
    let h = 0.8 * em + rows as f64 * em + (rows.max(1) - 1) as f64 * 0.5 * em;
    (w.ceil() as i32, h.ceil() as i32)
}

/// Draws a legend of ``entries`` in ``ncol`` columns (filled column by column, as matplotlib),
/// its top left corner at ``(x, y)``, pixels of ``area``.
pub fn legend_at<DB: DrawingBackend>(area: &DrawingArea<DB, Shift>, (x, y): (i32, i32), entries: &[(Swatch, String)], ncol: usize, size: f64, ground: Ground) {
    let em = size * PT;
    let rows = entries.len().div_ceil(ncol);
    let (w, h) = legend_size(entries, ncol, size);
    match ground {
        Ground::None => {}
        Ground::Plain => area.draw(&Rectangle::new([(x, y), (x + w, y + h)], WHITE.mix(0.75).filled())).unwrap(),
        Ground::Framed => {
            area.draw(&Rectangle::new([(x, y), (x + w, y + h)], WHITE.mix(0.9).filled())).unwrap();
            area.draw(&Rectangle::new([(x, y), (x + w, y + h)], AXIS.stroke_width(sw(0.5)))).unwrap();
        }
    }
    let widths: Vec<f64> = (0..ncol).map(|c| entries.iter().skip(c * rows).take(rows).map(|(_, t)| text_width(t, size)).fold(0.0, f64::max)).collect();
    let mut x0 = x as f64 + 0.4 * em;
    for (c, width) in widths.iter().enumerate() {
        for r in 0..rows {
            let Some((swatch, text)) = entries.get(c * rows + r) else { break };
            let yc = (y as f64 + 0.4 * em + r as f64 * 1.5 * em + 0.5 * em).round() as i32;
            let xs = x0.round() as i32;
            match *swatch {
                Swatch::Line(col, width, dash) => {
                    let (on, off) = dash.pattern(width);
                    area.draw(&DashedPathElement::new(vec![(xs, yc), (xs + handle(), yc)], on, off, col.stroke_width(sw(width)))).unwrap();
                }
                Swatch::Patch(col) => {
                    area.draw(&Rectangle::new([(xs, yc - (0.35 * em) as i32), (xs + handle(), yc + (0.35 * em) as i32)], col.filled())).unwrap();
                }
                Swatch::Dot(col, radius) => {
                    area.draw(&Circle::new((xs + handle() / 2, yc), pt(radius), col.filled())).unwrap();
                }
            }
            label(area, text, (xs + handle() + (0.8 * em) as i32, yc), size, &INK, HPos::Left, VPos::Center, false);
        }
        x0 += handle() as f64 + 0.8 * em + width + 2.0 * em;
    }
}

/// Draws a legend on or beside a chart's plotting area; ``area`` must hold it (plotters pulls
/// what falls outside an area onto its edge).
#[allow(clippy::too_many_arguments)]
pub fn legend_in<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    chart: &ChartContext<'_, DB, Coord>,
    place: Place,
    entries: &[(Swatch, String)],
    ncol: usize,
    size: f64,
    ground: Ground,
) {
    let (xs, ys) = chart.plotting_area().get_pixel_range();
    let (bx, by) = area.get_base_pixel();
    let (l, t) = (xs.start - bx, ys.start - by);
    let (aw, ah) = (xs.end - xs.start, ys.end - ys.start);
    let (w, h) = legend_size(entries, ncol, size);
    let pad = (0.5 * size * PT) as i32;
    let at = match place {
        Place::UpperLeft => (pad, pad),
        Place::UpperCenter => ((aw - w) / 2, pad),
        Place::UpperRight => (aw - w - pad, pad),
        Place::LowerLeft => (pad, ah - h - pad),
        Place::LowerCenter => ((aw - w) / 2, ah - h - pad),
        Place::LowerRight => (aw - w - pad, ah - h - pad),
        Place::Right => (aw + pad, (ah - h) / 2),
    };
    legend_at(area, (l + at.0, t + at.1), entries, ncol, size, ground);
}

/// A vertical colour bar: its top left corner at ``(x, y)`` and its size ``(w, h)`` in pixels of
/// ``area``, ``cmap`` over ``range``, with ticks and a label (lines separated by ``\n``, ``size``
/// points) on its right.
#[allow(clippy::too_many_arguments)]
pub fn colour_bar<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    (x, y): (i32, i32),
    (w, h): (i32, i32),
    cmap: impl Fn(f64) -> RGBColor,
    range: (f64, f64),
    ticks: &[f64],
    text: &str,
    size: f64,
    lang: Lang,
) {
    let n = 256;
    let pixels: Vec<RGBColor> = (0..n).map(|i| cmap(1.0 - (i as f64 + 0.5) / n as f64)).collect();
    let marker = raster(1, n, &pixels);
    area.draw(&Rectangle::new([(x, y), (x + w, y + h)], marker.filled())).unwrap();
    area.draw(&Rectangle::new([(x, y), (x + w, y + h)], AXIS.stroke_width(sw(0.6)))).unwrap();
    let texts = axis_labels(lang, ticks);
    let mut tw: f64 = 0.0;
    for (v, s) in ticks.iter().zip(&texts) {
        let yy = y + h - ((v - range.0) / (range.1 - range.0) * h as f64).round() as i32;
        area.draw(&PathElement::new(vec![(x + w, yy), (x + w + pt(2.5), yy)], AXIS.stroke_width(sw(0.6)))).unwrap();
        tw = tw.max(text_width(s, LABEL));
        label(area, s, (x + w + label_dist(), yy), LABEL, &INK_2, HPos::Left, VPos::Center, false);
    }
    for (i, s) in text.lines().enumerate() {
        let xx = x + w + label_dist() + tw.ceil() as i32 + pt(4.0) + (i as f64 * size * PT * 1.2).round() as i32;
        label(area, s, (xx, y + h / 2), size, &INK_2, HPos::Center, VPos::Top, true);
    }
}

/// Width (canvas pixels) a colour bar of width ``w`` takes with its ticks and its label.
pub fn colour_bar_width(w: i32, ticks: &[f64], text: &str, size: f64, lang: Lang) -> i32 {
    let tw = axis_labels(lang, ticks).iter().map(|s| text_width(s, LABEL)).fold(0.0, f64::max);
    w + label_dist() + tw.ceil() as i32 + pt(4.0) + (text.lines().count() as f64 * size * PT * 1.2).ceil() as i32 + pt(2.0)
}

/// Values from ``lo`` to ``hi`` every ``step``.
pub fn steps_of(lo: f64, hi: f64, step: f64) -> Vec<f64> {
    let n = ((hi - lo) / step + 1e-9).floor() as usize;
    (0..=n).map(|i| lo + i as f64 * step).collect()
}

/// Text at a data position of a chart.
pub fn text<DB: DrawingBackend>(chart: &mut ChartContext<'_, DB, Coord>, ax: &Axes, x: f64, y: f64, s: &str, style: TextStyle) {
    chart.draw_series(std::iter::once(Text::new(s.to_string(), (ax.cx(x), y), style))).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_reference() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn axis_labels_share_their_decimals() {
        assert_eq!(axis_labels(Lang::En, &[0.0, 0.25, 0.5]), ["0.00", "0.25", "0.50"]);
        assert_eq!(axis_labels(Lang::Fr, &[-1.0, 0.5]), ["\u{2212}1,0", "0,5"]);
        assert_eq!(axis_labels(Lang::En, &[-14.0, 2.0]), ["\u{2212}14", "2"]);
    }

    #[test]
    fn clipping_keeps_the_pieces_inside() {
        let pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 0.0), (3.0, -2.0), (4.0, 0.0)];
        let parts = clip_y(&pts, -1.0, 1.0);
        assert_eq!(parts.len(), 3);
        assert!(parts.iter().flatten().all(|p| p.1 >= -1.0 - 1e-12 && p.1 <= 1.0 + 1e-12));
        assert_eq!(parts[0], vec![(0.0, 0.0), (0.5, 1.0)]);
        assert_eq!(parts[1], vec![(1.5, 1.0), (2.0, 0.0), (2.5, -1.0)]);
    }

    #[test]
    fn text_width_is_proportional() {
        let w = text_width("10k", 10.0);
        assert!((text_width("10k", 20.0) - 2.0 * w).abs() < 1e-9);
        assert!(w > 0.0 && text_width("10k10k", 10.0) > 1.9 * w);
    }
}
