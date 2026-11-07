//! Every simulated quantity of the study, saved to ``results/study.bin``.

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::mat::Mat;
use crate::room::Meta;
use crate::Vec3;

/// One loudspeaker variant (layout, orientation, crossover).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VariantResults {
    pub name: String,
    /// Power at the 70 CTA-2034 directions, ``(70, n_freqs)``.
    pub cta: Mat<f32>,
    /// Sound power over the sphere (the mean of the quadrature).
    pub sp_full: Vec<f64>,
    /// Energy at the listener per room and category of path, ``(n_rooms * n_categories, n_freqs)``.
    pub rooms: Mat<f32>,
    /// Each direction's own deviation (dB): RMS residual of the straight line from 600 Hz to 8 kHz
    /// (``anom``), RMS within one octave of the crossover (``local``), deepest dip there (``notch``);
    /// at the 70 directions, the quadrature and every image of every room.
    pub anom_cta: Vec<f32>,
    pub anom_grid: Vec<f32>,
    pub anom_img: Vec<f32>,
    pub local_cta: Vec<f32>,
    pub local_grid: Vec<f32>,
    pub local_img: Vec<f32>,
    pub notch_cta: Vec<f32>,
    pub notch_grid: Vec<f32>,
    pub notch_img: Vec<f32>,
    /// Power averaged from 600 Hz to 8 kHz in the same directions.
    pub pband_cta: Vec<f32>,
    pub pband_grid: Vec<f32>,
    pub pband_img: Vec<f32>,
    /// Roughness (dB RMS, [`crate::analysis::roughness`]) in the same directions.
    pub rough_cta: Vec<f32>,
    pub rough_grid: Vec<f32>,
    pub rough_img: Vec<f32>,
    /// Sound power rebuilt from the two orbits driver by driver (upright only).
    pub sp_ways: Vec<f64>,
    /// With the paths taken from each driver: the same rooms with every path leaving the
    /// loudspeaker's reference point in the far field, steady state and early sound,
    /// ``(n_rooms * 2, n_freqs)``.
    pub rooms_far: Mat<f32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Results {
    pub freqs: Vec<f64>,
    pub grid: Vec<Vec3>,
    pub categories: Vec<String>,
    pub labels: Vec<String>,
    /// The named rooms' keys (``None`` for the random ensemble).
    pub keys: Vec<Option<String>>,
    pub n_random: usize,
    /// Energy of an omnidirectional source per room and category, ``(n_rooms * n_categories, n_freqs)``.
    pub omni: Mat<f32>,
    pub image_dirs: Vec<[f32; 3]>,
    pub image_starts: Vec<usize>,
    pub image_cat: Vec<u8>,
    pub image_delay: Vec<f32>,
    pub map_freqs: Vec<f64>,
    /// Image weights at ``map_freqs``, ``(n_images, n_map_freqs)``.
    pub image_weights: Mat<f32>,
    /// Image weights averaged from 600 Hz to 8 kHz.
    pub image_weights_band: Vec<f32>,
    /// Diffuse weights per room and direction, averaged from 600 Hz to 8 kHz, ``(n_rooms, n_grid)``.
    pub diffuse_band: Mat<f32>,
    /// Diffuse weights relative to the omnidirectional room total, mean of the random rooms,
    /// ``(n_grid, n_freqs)``.
    pub diffuse_mean: Mat<f32>,
    pub meta: Vec<Meta>,
    pub rt60: Vec<[f64; 7]>,
    pub variants: Vec<VariantResults>,
}

impl Results {
    pub fn save(&self, path: &Path) {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).unwrap();
        }
        let mut w = BufWriter::new(File::create(path).unwrap());
        bincode::serde::encode_into_std_write(self, &mut w, bincode::config::standard()).unwrap();
    }

    pub fn load(path: &Path) -> Self {
        let mut r = BufReader::new(File::open(path).unwrap_or_else(|e| panic!("{}: {e} (run the study first)", path.display())));
        bincode::serde::decode_from_std_read(&mut r, bincode::config::standard()).unwrap()
    }

    pub fn variant(&self, name: &str) -> &VariantResults {
        self.variants.iter().find(|v| v.name == name).unwrap_or_else(|| panic!("no variant {name}"))
    }
}
