//! The simulated loudspeakers and their far field over the whole sphere.
//!
//! Speaker frame (metres), as in the turntable study: origin at the centre of the front baffle,
//! ``x`` horizontal and positive to the right as seen from the listening position, ``y`` up, ``z``
//! normal to the baffle and positive toward the listener. The cabinet occupies
//! ``-depth <= z <= 0``; the drivers sit on the baffle plane ``z = 0``.
//!
//! Each driver is a rigid radiator ([`crate::directivity`]) with a flat on-axis half-space
//! response, and the cabinet diffracts it through the Distributed Edge Dipole (DED) model of the
//! turntable study (Urban et al., JAES 52(10), 2004), evaluated in the far field in any direction
//! ``u``, relative to the origin and normalized to 1 m:
//!
//! ``H(u, f) = D(theta, k) (1 + cos theta) / 2 exp(+j k u.s) - u_z sum_e Q_e(f) exp(+j k u.E_e)``,
//!
//! ``s`` the driver centre, ``E_e`` the edge elements, ``cos theta = u_z``. Every driver is then
//! equalized to an acoustic target on the reference axis ([`crate::crossover`]).

use rayon::prelude::*;

use crate::constants::C;
use crate::directivity::Radiator;
use crate::mat::Mat;
use crate::special::{leggauss, CubicSpline};
use crate::{dot, Vec3, C32, C64};

#[derive(Clone, Debug)]
pub struct Cabinet {
    pub width: f64,
    pub height: f64,
    pub depth: f64,
}

/// A driver on the baffle: a piston (``height = 0``) or a dome of base radius ``radius``. With
/// ``waveguide = Some((radius, ka_max))`` the dome sits in a waveguide: it radiates as the
/// constant-directivity source, and the waveguide shields the baffle edges (their strengths are
/// the dome's times the waveguide's directivity at 90 degrees).
#[derive(Clone, Debug)]
pub struct DriverSpec {
    pub name: String,
    pub label: String,
    /// The way the driver belongs to (0: the lowest); drivers of one way share its filter.
    pub way: usize,
    pub x: f64,
    pub y: f64,
    pub radius: f64,
    pub height: f64,
    pub waveguide: Option<(f64, f64)>,
}

impl DriverSpec {
    pub fn new(name: &str, label: &str, way: usize, x: f64, y: f64, radius: f64, height: f64) -> Self {
        DriverSpec { name: name.into(), label: label.into(), way, x, y, radius, height, waveguide: None }
    }

    /// The radiator whose surface drives the edges (the dome itself).
    pub fn radiator(&self) -> Radiator {
        Radiator::new(self.radius, self.height)
    }

    /// The radiator whose directivity the far field follows.
    pub fn far_radiator(&self) -> Radiator {
        match self.waveguide {
            Some((r, ka)) => Radiator::waveguide(r, ka),
            None => self.radiator(),
        }
    }

    pub fn centre(&self) -> Vec3 {
        [self.x, self.y, 0.0]
    }
}

#[derive(Clone, Debug)]
pub struct Layout {
    pub name: String,
    pub label: String,
    pub cabinet: Cabinet,
    /// Drivers, lowest way first.
    pub drivers: Vec<DriverSpec>,
    /// Crossover frequency between consecutive ways (Hz), lowest first.
    pub crossovers: Vec<f64>,
    /// The cabinet stands on the floor (its baffle's centre half its height up); otherwise it sits
    /// on a stand, its baffle's centre at the setup's loudspeaker position.
    pub floor: bool,
}

impl Layout {
    pub fn names(&self) -> Vec<String> {
        self.drivers.iter().map(|d| d.name.clone()).collect()
    }
}

/// The main loudspeaker: an upright two-way bookshelf, 200 x 360 x 280 mm, a 6.5-inch cone (flat
/// piston of 129 cm^2) below a 1.1-inch dome (28 mm, 7.5 mm high), 140 mm apart, crossed over at
/// 2.2 kHz.
pub fn two_way() -> Layout {
    Layout {
        name: "two_way".into(),
        label: "two-way, 6.5 in cone and 1 in dome, 140 mm apart".into(),
        cabinet: Cabinet { width: 0.200, height: 0.360, depth: 0.280 },
        drivers: vec![
            DriverSpec::new("woofer", "6.5 in cone", 0, 0.0, -0.030, 0.064, 0.0),
            DriverSpec::new("tweeter", "1.1 in dome", 1, 0.0, 0.110, 0.014, 0.0075),
        ],
        crossovers: vec![2200.0],
        floor: false,
    }
}

/// The same two-way with its dome in a waveguide whose directivity matches the cone's at the
/// crossover (``k a = 2.58`` at 2.2 kHz for both) and stays constant above.
pub fn two_way_waveguide() -> Layout {
    let mut l = two_way();
    l.name = "two_way_wg".into();
    l.label = "two-way, 6.5 in cone and 1 in dome in a waveguide matched at the crossover".into();
    let ka = 2.0 * std::f64::consts::PI * 2200.0 / C * 0.064;
    let mut tw = DriverSpec::new("tweeter", "1.1 in dome in a waveguide", 1, 0.0, 0.110, 0.014, 0.0075);
    tw.waveguide = Some((0.064, ka));
    l.drivers[1] = tw;
    l
}

/// A dome in a waveguide whose directivity matches a cone of radius ``radius`` at ``fc``.
fn matched(mut dome: DriverSpec, radius: f64, fc: f64) -> DriverSpec {
    dome.waveguide = Some((radius, 2.0 * std::f64::consts::PI * fc / C * radius));
    dome
}

/// A coaxial two-way, 300 x 420 x 300 mm: a 1-inch dome at the centre of an 8-inch cone, which
/// guides it (directivity matched at the 1.8 kHz crossover), no offset between the two.
pub fn coaxial() -> Layout {
    Layout {
        name: "coaxial".into(),
        label: "coaxial two-way, 8 in cone guiding a 1 in dome".into(),
        cabinet: Cabinet { width: 0.300, height: 0.420, depth: 0.300 },
        drivers: vec![
            DriverSpec::new("woofer", "8 in cone", 0, 0.0, 0.020, 0.085, 0.0),
            matched(DriverSpec::new("tweeter", "1 in dome in the cone", 1, 0.0, 0.020, 0.0125, 0.006), 0.085, 1800.0),
        ],
        crossovers: vec![1800.0],
        floor: false,
    }
}

/// A two-way MTM, 200 x 480 x 280 mm: a 1.1-inch dome between two 5.25-inch cones, 115 mm above
/// and below it, crossed over at 2.5 kHz.
pub fn mtm() -> Layout {
    Layout {
        name: "mtm".into(),
        label: "two-way MTM, two 5.25 in cones around a 1 in dome".into(),
        cabinet: Cabinet { width: 0.200, height: 0.480, depth: 0.280 },
        drivers: vec![
            DriverSpec::new("woofer_low", "5.25 in cone, below", 0, 0.0, -0.115, 0.055, 0.0),
            DriverSpec::new("woofer_high", "5.25 in cone, above", 0, 0.0, 0.115, 0.055, 0.0),
            DriverSpec::new("tweeter", "1.1 in dome", 1, 0.0, 0.0, 0.014, 0.0075),
        ],
        crossovers: vec![2500.0],
        floor: false,
    }
}

/// The horizontal three-way monitor of the turntable study, 440 x 290 x 290 mm: an 8-inch woofer
/// on the left, a 3-inch dome midrange and a 1.1-inch dome stacked on the right, crossed over at
/// 450 Hz and 3 kHz.
pub fn three_way_monitor() -> Layout {
    Layout {
        name: "three_way_monitor".into(),
        label: "horizontal three-way monitor, woofer beside the midrange and tweeter".into(),
        cabinet: Cabinet { width: 0.440, height: 0.290, depth: 0.290 },
        drivers: vec![
            DriverSpec::new("woofer", "8 in cone", 0, -0.085, 0.0, 0.0838, 0.0),
            DriverSpec::new("mid", "3 in dome", 1, 0.120, -0.0485, 0.060, 0.0175),
            DriverSpec::new("tweeter", "1.1 in dome", 2, 0.120, 0.0815, 0.014, 0.0075),
        ],
        crossovers: vec![450.0, 3000.0],
        floor: false,
    }
}

/// A three-way tower on the floor, 220 x 1000 x 300 mm: two 6.5-inch woofers, a 4-inch midrange
/// and a 1.1-inch dome 880 mm above the floor, crossed over at 350 Hz and 2.8 kHz.
pub fn three_way_tower() -> Layout {
    Layout {
        name: "three_way_tower".into(),
        label: "three-way tower, two woofers, midrange, tweeter".into(),
        cabinet: Cabinet { width: 0.220, height: 1.000, depth: 0.300 },
        drivers: vec![
            DriverSpec::new("woofer_low", "6.5 in cone, lower", 0, 0.0, -0.300, 0.064, 0.0),
            DriverSpec::new("woofer_high", "6.5 in cone, upper", 0, 0.0, -0.120, 0.064, 0.0),
            DriverSpec::new("mid", "4 in cone", 1, 0.0, 0.240, 0.045, 0.0),
            DriverSpec::new("tweeter", "1.1 in dome", 2, 0.0, 0.380, 0.014, 0.0075),
        ],
        crossovers: vec![350.0, 2800.0],
        floor: true,
    }
}

/// A four-way tower on the floor, 260 x 1100 x 350 mm: an 8-inch woofer, a 6.5-inch low midrange,
/// a 4-inch midrange and a 1.1-inch dome 960 mm above the floor, crossed over at 250 Hz, 900 Hz and
/// 3.2 kHz.
pub fn four_way_tower() -> Layout {
    Layout {
        name: "four_way_tower".into(),
        label: "four-way tower".into(),
        cabinet: Cabinet { width: 0.260, height: 1.100, depth: 0.350 },
        drivers: vec![
            DriverSpec::new("woofer", "8 in cone", 0, 0.0, -0.360, 0.0838, 0.0),
            DriverSpec::new("low_mid", "6.5 in cone", 1, 0.0, -0.080, 0.064, 0.0),
            DriverSpec::new("mid", "4 in cone", 2, 0.0, 0.260, 0.045, 0.0),
            DriverSpec::new("tweeter", "1.1 in dome", 3, 0.0, 0.410, 0.014, 0.0075),
        ],
        crossovers: vec![250.0, 900.0, 3200.0],
        floor: true,
    }
}

/// The loudspeakers of the library: the two two-ways of the study and five other families.
pub fn library() -> Vec<Layout> {
    vec![two_way(), two_way_waveguide(), coaxial(), mtm(), three_way_monitor(), three_way_tower(), four_way_tower()]
}

/// A straight run of edge elements along one side of the front baffle.
#[derive(Clone, Debug)]
pub struct EdgeLine {
    /// Midpoint of the first element.
    pub first: Vec3,
    /// From one element's midpoint to the next.
    pub step: Vec3,
    pub n: usize,
}

/// The front baffle's perimeter as four lines of elements (``step`` metres at most), and every
/// element's midpoint and length.
pub fn front_edges(cabinet: &Cabinet, step: f64) -> (Vec<EdgeLine>, Vec<Vec3>, Vec<f64>) {
    let (w, h) = (cabinet.width / 2.0, cabinet.height / 2.0);
    let corners = [(-w, -h), (w, -h), (w, h), (-w, h)];
    let mut lines = Vec::new();
    let mut points = Vec::new();
    let mut lengths = Vec::new();
    for i in 0..4 {
        let (x0, y0) = corners[i];
        let (x1, y1) = corners[(i + 1) % 4];
        let length = (x1 - x0).hypot(y1 - y0);
        let n = (length / step).ceil() as usize;
        let d = [(x1 - x0) / n as f64, (y1 - y0) / n as f64, 0.0];
        for m in 0..n {
            let t = (m as f64 + 0.5) / n as f64;
            points.push([x0 + t * (x1 - x0), y0 + t * (y1 - y0), 0.0]);
            lengths.push(length / n as f64);
        }
        lines.push(EdgeLine { first: [x0 + 0.5 * d[0], y0 + 0.5 * d[1], 0.0], step: d, n });
    }
    (lines, points, lengths)
}

/// Edge strengths ``Q_e(f)`` of one driver, stored frequency by frequency: ``(n_freqs, n_edges)``.
///
/// ``Q_e = C dl_e G(rho_e, f) / A0(f)``: ``G`` the driving pressure at the edge point (a quadrature
/// of the radiating surface, each point weighted by its obliquity), ``A0`` the radiator's on-axis
/// factor, ``C`` the normalization that makes the direct and edge terms sum to 1/2 on axis at low
/// frequency (full-space radiation below the baffle step).
pub fn edge_strengths(driver: &DriverSpec, cabinet: &Cabinet, freqs: &[f64], step: f64) -> Mat<C64> {
    let coarse_df = 50.0;
    let (_, points, lengths) = front_edges(cabinet, step);
    let rho_e: Vec<f64> = points.iter().map(|p| (p[0] - driver.x).hypot(p[1] - driver.y)).collect();
    let radiator = driver.radiator();
    let a = radiator.radius;
    let f_max = *freqs.last().unwrap();
    let spread = 2.0 * std::f64::consts::PI * f_max / C * 2.0 * a;
    let n_r = (0.7 * spread + 16.0).ceil() as usize;
    let n_phi = (0.5 * spread + 16.0).ceil() as usize;
    let (x, wx) = leggauss(n_r);
    let mut nodes = Vec::with_capacity(n_r * n_phi); // (r, phi, z, weight)
    for i in 0..n_r {
        let r = 0.5 * a * (x[i] + 1.0);
        let wr = 0.5 * a * wx[i];
        let z = radiator.surface(r);
        for j in 0..n_phi {
            let phi = (j as f64 + 0.5) * std::f64::consts::PI / n_phi as f64;
            let w = wr * r * (std::f64::consts::PI / n_phi as f64) * 2.0 / (std::f64::consts::PI * a * a);
            nodes.push((r, phi, z, w));
        }
    }
    let rho_min = rho_e.iter().cloned().fold(f64::INFINITY, f64::min);
    let rho_max = rho_e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let (start, stop) = (rho_min - 0.002, rho_max + 0.003);
    let n_rho = ((stop - start) / 0.001).ceil() as usize;
    let rho_grid: Vec<f64> = (0..n_rho).map(|i| start + i as f64 * 0.001).collect();
    let n_coarse = ((f_max + coarse_df) / coarse_df).ceil() as usize;
    let f_coarse: Vec<f64> = (0..n_coarse).map(|i| i as f64 * coarse_df).collect();
    let k_coarse: Vec<f64> = f_coarse.iter().map(|f| 2.0 * std::f64::consts::PI * f / C).collect();

    // G(rho, f) exp(+j k rho) on the rho grid and the coarse frequencies.
    let g_tab: Vec<Vec<C64>> = rho_grid
        .par_iter()
        .map(|&rho| {
            let mut row = vec![C64::new(0.0, 0.0); n_coarse];
            for &(r, phi, z, w) in &nodes {
                let r1 = ((rho - r * phi.cos()).powi(2) + (r * phi.sin()).powi(2) + z * z).sqrt();
                let amp = w * 0.5 * (1.0 - z / r1) / r1;
                let d = r1 - rho;
                for (g, k) in row.iter_mut().zip(&k_coarse) {
                    *g += C64::from_polar(amp, -k * d);
                }
            }
            row
        })
        .collect();

    // To the edge points (linear in rho), then to the fine frequency grid (cubic).
    let n_e = points.len();
    let mut g_edges: Vec<Vec<C64>> = Vec::with_capacity(n_e);
    for &rho in &rho_e {
        let idx = (rho_grid.partition_point(|&v| v < rho) as isize - 1).clamp(0, n_rho as isize - 2) as usize;
        let t = (rho - rho_grid[idx]) / (rho_grid[idx + 1] - rho_grid[idx]);
        g_edges.push((0..n_coarse).map(|j| g_tab[idx][j] * (1.0 - t) + g_tab[idx + 1][j] * t).collect());
    }
    let normalization = 0.5 / (0..n_e).map(|e| lengths[e] * g_edges[e][0].re).sum::<f64>();
    let ks: Vec<f64> = freqs.iter().map(|f| 2.0 * std::f64::consts::PI * f / C).collect();
    let on_axis: Vec<C64> = ks.iter().map(|&k| radiator.on_axis(k)).collect();
    let shield: Vec<f64> = match driver.waveguide {
        Some(_) => {
            let far = driver.far_radiator();
            ks.iter().map(|&k| far.directivity(k, std::f64::consts::FRAC_PI_2).re).collect()
        }
        None => vec![1.0; freqs.len()],
    };
    let cols: Vec<Vec<C64>> = (0..n_e)
        .into_par_iter()
        .map(|e| {
            let re: Vec<f64> = g_edges[e].iter().map(|c| c.re).collect();
            let im: Vec<f64> = g_edges[e].iter().map(|c| c.im).collect();
            let (sr, si) = (CubicSpline::new(&f_coarse, &re), CubicSpline::new(&f_coarse, &im));
            freqs
                .iter()
                .enumerate()
                .map(|(i, &f)| {
                    let g = C64::new(sr.eval(f), si.eval(f));
                    g * normalization * lengths[e] * C64::from_polar(1.0, -rho_e[e] * ks[i]) / on_axis[i] * shield[i]
                })
                .collect()
        })
        .collect();
    let mut q = Mat::<C64>::zeros(freqs.len(), n_e);
    for (e, col) in cols.iter().enumerate() {
        for (i, v) in col.iter().enumerate() {
            q.data[i * n_e + e] = *v;
        }
    }
    q
}

/// Far-field responses of the drivers of a layout, in any direction of the speaker frame.
pub struct FarField {
    pub layout: Layout,
    pub freqs: Vec<f64>,
    pub k: Vec<f64>,
    lines: Vec<EdgeLine>,
    strengths: Vec<Mat<C64>>,
    radiators: Vec<Radiator>,
}

impl FarField {
    pub fn new(layout: &Layout, freqs: &[f64]) -> Self {
        let step = 0.002;
        let (lines, _, _) = front_edges(&layout.cabinet, step);
        let strengths = layout.drivers.iter().map(|d| edge_strengths(d, &layout.cabinet, freqs, step)).collect();
        let radiators = layout.drivers.iter().map(|d| d.far_radiator()).collect();
        FarField {
            layout: layout.clone(),
            freqs: freqs.to_vec(),
            k: freqs.iter().map(|f| 2.0 * std::f64::consts::PI * f / C).collect(),
            lines,
            strengths,
            radiators,
        }
    }

    /// The response of driver ``d`` in direction ``u`` (unit vector), at every frequency.
    pub fn response(&self, d: usize, u: &Vec3, out: &mut [C32]) {
        let driver = &self.layout.drivers[d];
        let radiator = &self.radiators[d];
        let q = &self.strengths[d];
        let cos_t = u[2].clamp(-1.0, 1.0);
        let theta = cos_t.acos();
        let obliquity = 0.5 * (1.0 + cos_t);
        let us = dot(u, &driver.centre());
        let starts: Vec<f64> = self.lines.iter().map(|l| dot(u, &l.first)).collect();
        let steps: Vec<f64> = self.lines.iter().map(|l| dot(u, &l.step)).collect();
        for (i, &k) in self.k.iter().enumerate() {
            let direct = radiator.directivity(k, theta) * obliquity * C64::from_polar(1.0, k * us);
            let qi = q.row(i);
            let mut acc = C64::new(0.0, 0.0);
            let mut e = 0;
            for (l, line) in self.lines.iter().enumerate() {
                let mut p = C64::from_polar(1.0, k * starts[l]);
                let rot = C64::from_polar(1.0, k * steps[l]);
                for _ in 0..line.n {
                    acc += qi[e] * p;
                    p *= rot;
                    e += 1;
                }
            }
            let h = direct - acc * u[2];
            out[i] = C32::new(h.re as f32, h.im as f32);
        }
    }

    /// ``{driver: (n_directions, n_freqs)}`` for unit vectors ``directions``.
    pub fn responses(&self, directions: &[Vec3]) -> Vec<Mat<C32>> {
        let nf = self.freqs.len();
        (0..self.layout.drivers.len())
            .map(|d| {
                let mut m = Mat::<C32>::zeros(directions.len(), nf);
                m.data.par_chunks_mut(nf).zip(directions.par_iter()).for_each(|(row, u)| {
                    let n = crate::norm(u);
                    let u = [u[0] / n, u[1] / n, u[2] / n];
                    self.response(d, &u, row)
                });
                m
            })
            .collect()
    }

    /// On-axis responses (complex, f64), for the equalizers.
    pub fn on_axis(&self) -> Vec<Vec<C64>> {
        self.responses(&[[0.0, 0.0, 1.0]])
            .into_iter()
            .map(|m| m.row(0).iter().map(|c| C64::new(c.re as f64, c.im as f64)).collect())
            .collect()
    }
}

/// Directions of a loudspeaker rolled by ``angle_deg`` about its axis (positive: its top toward
/// +x), expressed in the frame of the unrolled one: ``H_rolled(u) = H(R^T u)``.
pub fn roll(directions: &[Vec3], angle_deg: f64) -> Vec<Vec3> {
    let a = angle_deg.to_radians();
    let (s, c) = (a.sin(), a.cos());
    directions.iter().map(|u| [u[0] * c - u[1] * s, u[0] * s + u[1] * c, u[2]]).collect()
}
