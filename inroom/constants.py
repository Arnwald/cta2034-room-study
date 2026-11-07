"""Physical and numerical constants shared by the whole study."""

#: Speed of sound (m/s), about 20 degC.
C = 343.0

#: Density of air (kg/m^3) at 20 degC.
RHO = 1.204

#: Air temperature (degC), relative humidity (%) and pressure (kPa) of the air absorption.
AIR_TEMPERATURE = 20.0
AIR_HUMIDITY = 50.0
AIR_PRESSURE = 101.325

#: Analysis band (Hz) and resolution (points per octave). 1/96 octave resolves the band where the
#: two drivers of a brickwall crossover overlap (about 300 Hz wide) with several points.
F_LO = 150.0
F_HI = 20_000.0
PER_OCTAVE = 96

#: Octave-band centers (Hz) at which absorption and scattering coefficients are given.
OCTAVES = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0)
