//! Physical and numerical constants shared by the whole study.

/// Speed of sound (m/s), about 20 degC.
pub const C: f64 = 343.0;

/// Density of air (kg/m^3) at 20 degC.
pub const RHO: f64 = 1.204;

/// Air temperature (degC), relative humidity (%) and pressure (kPa) of the air absorption.
pub const AIR_TEMPERATURE: f64 = 20.0;
pub const AIR_HUMIDITY: f64 = 50.0;
pub const AIR_PRESSURE: f64 = 101.325;

/// Analysis band (Hz) and resolution (points per octave). 1/96 octave resolves the band where
/// the two drivers of a brickwall crossover overlap (about 300 Hz wide) with several points.
pub const F_LO: f64 = 150.0;
pub const F_HI: f64 = 20_000.0;
pub const PER_OCTAVE: usize = 96;

/// Octave-band centres (Hz) at which absorption and scattering coefficients are given.
pub const OCTAVES: [f64; 7] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0];
