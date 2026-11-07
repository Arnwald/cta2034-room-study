//! Domestic listening rooms: materials, the study's ensemble (a data file), new ensembles and a
//! few named rooms.
//!
//! Each setup holds one loudspeaker, the left one of a stereo pair, and one listener on the room's
//! middle line, facing the loudspeakers. The distributions (uniform unless stated):
//!
//! * room: width 3.4-5.2 m, length 4.5-7.5 m, height 2.4-2.8 m;
//! * loudspeakers on the short wall, their baffles 0.4-1.2 m from it, 1.8-2.8 m apart (at least
//!   0.45 m from the side walls), the reference axis 0.85-1.15 m above the floor;
//! * listener 2.0-3.8 m from the baffles' line (at least 0.5 m in front of the wall behind), ears
//!   1.0-1.25 m above the floor;
//! * loudspeaker aimed at the listener (half the setups), parallel to the side walls (a quarter),
//!   or halfway (a quarter);
//! * walls of painted plaster on masonry or of plasterboard on studs, one of them with a window
//!   (2-6 m^2) and curtains (0-4 m^2); ceiling of plaster or plasterboard;
//! * floor: parquet or tiles (60 %; half of them with a rug in front of the loudspeakers), or
//!   wall-to-wall carpet (40 %);
//! * furniture: 6-16 m^2 of equivalent absorption at 1 kHz (half of it at 125 Hz);
//! * a quarter of the rooms is treated: panels at the side walls' first reflection points and a rug.
//!
//! The study's 300 rooms were drawn once and are kept in `data/rooms.json`; [`sample`] draws new
//! ensembles from the same distributions.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::rng::Rng;
use crate::room::{aim, surface_index, Meta, Patch, Room, Setup, SURFACES};
use crate::Vec3;

pub fn material(name: &str) -> [f64; 7] {
    match name {
        "plaster" => [0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05],
        "gypsum" => [0.28, 0.12, 0.07, 0.05, 0.06, 0.08, 0.08],
        "glass" => [0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.04],
        "curtain" => [0.07, 0.31, 0.49, 0.75, 0.70, 0.60, 0.60],
        "parquet" => [0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07],
        "tiles" => [0.01, 0.01, 0.015, 0.02, 0.02, 0.02, 0.02],
        "carpet" => [0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73],
        "rug" => [0.05, 0.10, 0.25, 0.45, 0.55, 0.60, 0.60],
        "panel" => [0.15, 0.55, 0.95, 1.00, 0.97, 0.95, 0.95],
        _ => panic!("unknown material {name}"),
    }
}

pub fn scattering(name: &str) -> [f64; 7] {
    match name {
        "wall" => [0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
        "floor" => [0.10, 0.15, 0.25, 0.35, 0.40, 0.45, 0.50],
        "ceiling" => [0.05, 0.05, 0.05, 0.08, 0.10, 0.12, 0.15],
        "flat" => [0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.15],
        _ => panic!("unknown scattering {name}"),
    }
}

/// Frequency shape of the furniture's absorption (times its value at 1 kHz).
pub const FURNITURE_SHAPE: [f64; 7] = [0.50, 0.70, 0.90, 1.0, 1.0, 1.0, 1.0];

// ---------------------------------------------------------------------------------------------
// The data file
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
struct PatchJson {
    surface: String,
    rect: [f64; 4],
    absorption: [f64; 7],
    scattering: [f64; 7],
    label: String,
}

#[derive(Serialize, Deserialize)]
struct SetupJson {
    key: Option<String>,
    label: String,
    dims: [f64; 3],
    absorption: HashMap<String, [f64; 7]>,
    scattering: HashMap<String, [f64; 7]>,
    patches: Vec<PatchJson>,
    fittings: [f64; 7],
    source: [f64; 3],
    receiver: [f64; 3],
    yaw: f64,
    meta: Meta,
}

#[derive(Serialize, Deserialize)]
struct EnsembleJson {
    description: String,
    setups: Vec<SetupJson>,
}

fn from_json(s: SetupJson) -> Setup {
    let mut absorption = [[0.0; 7]; 6];
    let mut scat = [[0.0; 7]; 6];
    for (i, name) in SURFACES.iter().enumerate() {
        absorption[i] = s.absorption[*name];
        scat[i] = s.scattering[*name];
    }
    let patches = s
        .patches
        .into_iter()
        .map(|p| Patch { surface: surface_index(&p.surface), rect: p.rect, absorption: p.absorption, scattering: p.scattering, label: p.label })
        .collect();
    Setup {
        room: Room { dims: s.dims, absorption, scattering: scat, patches, fittings: s.fittings, label: s.label.clone() },
        source: s.source,
        receiver: s.receiver,
        yaw: s.yaw,
        label: s.label,
        key: s.key,
        meta: s.meta,
    }
}

fn to_json(st: &Setup) -> SetupJson {
    SetupJson {
        key: st.key.clone(),
        label: st.label.clone(),
        dims: st.room.dims,
        absorption: SURFACES.iter().enumerate().map(|(i, n)| (n.to_string(), st.room.absorption[i])).collect(),
        scattering: SURFACES.iter().enumerate().map(|(i, n)| (n.to_string(), st.room.scattering[i])).collect(),
        patches: st
            .room
            .patches
            .iter()
            .map(|p| PatchJson { surface: SURFACES[p.surface].into(), rect: p.rect, absorption: p.absorption, scattering: p.scattering, label: p.label.clone() })
            .collect(),
        fittings: st.room.fittings,
        source: st.source,
        receiver: st.receiver,
        yaw: st.yaw,
        meta: st.meta.clone(),
    }
}

/// The setups of a data file: the random ensemble first, then the named rooms (``key`` set).
pub fn load(path: &Path) -> Vec<Setup> {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let data: EnsembleJson = serde_json::from_str(&text).expect("rooms file");
    data.setups.into_iter().map(from_json).collect()
}

pub fn save(path: &Path, setups: &[Setup], description: &str) {
    let data = EnsembleJson { description: description.into(), setups: setups.iter().map(to_json).collect() };
    std::fs::write(path, serde_json::to_string(&data).unwrap()).unwrap();
}

// ---------------------------------------------------------------------------------------------
// Building and drawing setups
// ---------------------------------------------------------------------------------------------

/// The parameters of one setup.
#[derive(Clone, Debug)]
pub struct Params {
    pub dims: Vec3,
    pub wall: String,
    pub ceiling: String,
    pub floor: String,
    pub window_wall: String,
    pub glass: f64,
    pub curtains: f64,
    pub rug: Option<(f64, f64)>,
    pub panels: bool,
    pub cloud: bool,
    pub furniture: f64,
    pub spacing: f64,
    pub wall_distance: f64,
    pub listen_distance: f64,
    pub speaker_height: f64,
    pub ear_height: f64,
    pub toe: String,
    pub label: String,
}

impl Params {
    pub fn living(dims: Vec3, label: &str) -> Self {
        Params {
            dims,
            wall: "plaster".into(),
            ceiling: "plaster".into(),
            floor: "parquet".into(),
            window_wall: "right".into(),
            glass: 4.0,
            curtains: 2.0,
            rug: None,
            panels: false,
            cloud: false,
            furniture: 6.0,
            spacing: 2.4,
            wall_distance: 0.6,
            listen_distance: 3.0,
            speaker_height: 1.0,
            ear_height: 1.1,
            toe: "aim".into(),
            label: label.into(),
        }
    }
}

fn mix(parts: &[(&str, f64)]) -> [f64; 7] {
    let mut out = [0.0; 7];
    for (name, share) in parts {
        let m = material(name);
        for i in 0..7 {
            out[i] += share * m[i];
        }
    }
    out
}

/// One setup from its parameters.
pub fn build(p: &Params) -> Setup {
    let [lx, ly, lz] = p.dims;
    let mut absorption = [[0.0; 7]; 6];
    let mut scat = [[0.0; 7]; 6];
    for w in ["left", "right", "listener_wall", "speaker_wall"] {
        let i = surface_index(w);
        let area = if w == "left" || w == "right" { ly * lz } else { lx * lz };
        absorption[i] = if w == p.window_wall {
            let g = p.glass.min(0.6 * area) / area;
            let c = p.curtains.min(0.6 * area) / area;
            mix(&[(&p.wall, 1.0 - g - c), ("glass", g), ("curtain", c)])
        } else {
            material(&p.wall)
        };
        scat[i] = scattering("wall");
    }
    absorption[surface_index("ceiling")] = material(&p.ceiling);
    scat[surface_index("ceiling")] = scattering("ceiling");
    absorption[surface_index("floor")] = material(&p.floor);
    scat[surface_index("floor")] = scattering("floor");

    let source = [lx / 2.0 - p.spacing / 2.0, ly - p.wall_distance, p.speaker_height];
    let receiver = [lx / 2.0, ly - p.wall_distance - p.listen_distance, p.ear_height];
    let mut patches = Vec::new();
    if let Some((width, length)) = p.rug {
        let y1 = ly - p.wall_distance - 0.1;
        patches.push(Patch {
            surface: surface_index("floor"),
            rect: [lx / 2.0 - width / 2.0, lx / 2.0 + width / 2.0, y1 - length, y1],
            absorption: material("rug"),
            scattering: scattering("flat"),
            label: "rug".into(),
        });
    }
    if p.panels {
        let (xs, xr) = (source[0], receiver[0]);
        let t = xs / (xs + xr);
        let y_ref = source[1] + t * (receiver[1] - source[1]);
        for w in ["left", "right"] {
            patches.push(Patch {
                surface: surface_index(w),
                rect: [y_ref - 0.6, y_ref + 0.6, 0.5, 1.7],
                absorption: material("panel"),
                scattering: scattering("flat"),
                label: "panel".into(),
            });
        }
    }
    if p.cloud {
        let (hs, he) = (source[2], receiver[2]);
        let t = (lz - hs) / ((lz - hs) + (lz - he));
        let y_ref = source[1] + t * (receiver[1] - source[1]);
        patches.push(Patch {
            surface: surface_index("ceiling"),
            rect: [lx / 2.0 - 1.2, lx / 2.0 + 1.2, y_ref - 0.9, y_ref + 0.9],
            absorption: material("panel"),
            scattering: scattering("flat"),
            label: "cloud".into(),
        });
    }
    let fittings = FURNITURE_SHAPE.map(|s| s * p.furniture);
    let yaw_aim = aim(&source, &receiver);
    let yaw = match p.toe.as_str() {
        "aim" => yaw_aim,
        "straight" => 0.0,
        _ => 0.5 * yaw_aim,
    };
    let meta = Meta {
        dims: p.dims.to_vec(),
        wall: p.wall.clone(),
        ceiling: p.ceiling.clone(),
        floor: p.floor.clone(),
        rug: p.rug.is_some(),
        panels: p.panels,
        cloud: p.cloud,
        furniture: p.furniture,
        spacing: p.spacing,
        wall_distance: p.wall_distance,
        listen_distance: p.listen_distance,
        speaker_height: p.speaker_height,
        ear_height: p.ear_height,
        toe: p.toe.clone(),
    };
    Setup {
        room: Room { dims: p.dims, absorption, scattering: scat, patches, fittings, label: p.label.clone() },
        source,
        receiver,
        yaw,
        label: p.label.clone(),
        key: None,
        meta,
    }
}

/// ``n`` new random setups from the distributions of the module documentation.
pub fn sample(n: usize, seed: u64, treated_share: f64) -> Vec<Setup> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    while out.len() < n {
        let dims = [rng.range(3.4, 5.2), rng.range(4.5, 7.5), rng.range(2.4, 2.8)];
        let spacing = rng.range(1.8, 2.8).min(dims[0] - 0.9);
        let wall_distance = rng.range(0.4, 1.2);
        let listen = rng.range(2.0, 3.8);
        if dims[1] - wall_distance - listen < 0.5 {
            continue;
        }
        let treated = rng.uniform() < treated_share;
        let hard = rng.uniform() < 0.6;
        let floor = if hard { rng.choice(&["parquet", "tiles"]).to_string() } else { "carpet".into() };
        let has_rug = hard && (treated || rng.uniform() < 0.5);
        let rug = if has_rug { Some((rng.range(2.0, 3.0), rng.range(2.5, 3.5))) } else { None };
        let u = rng.uniform();
        let toe = if u < 0.5 { "aim" } else if u < 0.75 { "straight" } else { "half" };
        let p = Params {
            dims,
            wall: rng.choice(&["plaster", "gypsum"]).to_string(),
            ceiling: rng.choice(&["plaster", "gypsum"]).to_string(),
            floor,
            window_wall: rng.choice(&["left", "right", "listener_wall"]).to_string(),
            glass: rng.range(2.0, 6.0),
            curtains: rng.range(0.0, 4.0),
            rug,
            panels: treated,
            cloud: false,
            furniture: rng.range(6.0, 16.0),
            spacing,
            wall_distance,
            listen_distance: listen,
            speaker_height: rng.range(0.85, 1.15),
            ear_height: rng.range(1.0, 1.25),
            toe: toe.into(),
            label: format!("room {}", out.len() + 1),
        };
        out.push(build(&p));
    }
    out
}
