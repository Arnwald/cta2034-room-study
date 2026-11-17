# Do gentle crossover slopes harm what a listener receives in a room? Protocol

This protocol was written and committed before any of the computations it describes were run.
The commit that adds it comes before the commits holding the code's results. Nothing below changes
once results exist. Any deviation would be reported as a deviation.

## The question

The study compared Linkwitz-Riley crossovers of order 2, 4 and 8 with a brickwall linear-phase FIR.
Different measures ranked them differently.

- **Each reflection read along its own path** puts the brickwall first. Its notches are as deep as a
  Linkwitz-Riley's but ten to thirteen times narrower.
- **The paths added with their phases at one point, early sound**, puts LR2 first.
- **The 1-octave roughness of the rooms' mean response** also puts LR2 first.

Two views are tested here.

- **View 1, the owner's intuition.** A gentle slope makes the drivers interfere over a wide band. It
  changes what the listener receives broadly in frequency, and broad changes are what hearing
  detects most easily. It removes energy toward the floor and the ceiling only around the crossover,
  while everything else is still reflected.
- **View 2.** By notching the floor and ceiling bounces near its frequency, a Linkwitz-Riley crossover
  removes their comb at the ear, and the result is smoother.

Both measures that put LR2 first have a known bias toward it:

- the 1-octave roughness cannot see a deviation wider than an octave;
- the early sound at one point is judged against flatness. A crossover that removes reflections
  scores better for that alone, and the omnidirectional source scores worst of all.

This protocol replaces both references with one that has neither bias, and it uses an instrument
calibrated on published detection data.

## What is compared

- **Loudspeakers and crossovers.** The seven loudspeakers of the library, upright, each with LR2,
  LR4, LR8 and the brickwall FIR, filtered exactly as in the study (`crossover::filters`).
- **The rooms.** The 300 random rooms of `data/rooms.json`. The named rooms are left out of every
  statistic. The towers stand on the floor, as in the library.
- **The far field.** Every path leaves the loudspeaker's reference point in the far field (as in the
  study's main computation and in `coherent.rs`), towers included.
  - The per-driver paths of the library cannot define the twin below.
  - The towers' far-field error (0.2 to 0.7 dB) applies to the variant and its twin alike.
- **The reference is each variant's interference-free twin.** It is the same loudspeaker with the
  same filters, every driver's radiation, its own diffraction included, moved rigidly to the baffle
  centre: `H_twin,d(u) = H_d(u) exp(-j k u.c_d)`, with `c_d` the driver's centre.
  - The twin equals the variant on the reference axis and in any direction where the drivers' path
    difference is zero.
  - Elsewhere it shows each driver's own directivity, blended by the crossover, without the drivers
    interfering.
  - The difference between variant and twin is therefore what the crossover does through the
    spacing of the drivers: the lobing.
  - Where a crossover removes a reflection's energy near its frequency, that counts as a difference
    from the twin, like any other.
- **The FIR's slope depends on the crossover frequency.** Its transition band is fixed in hertz
  (about ±300 Hz around the crossover). At the low crossovers of the monitor and the towers
  (250–450 Hz) it is therefore much less steep in octaves than at 2–3 kHz. The loudspeakers are
  counted as they are, and the results are also given crossover by crossover.

## The instrument: how far apart two responses are for the ear

**Excitation pattern.** This is a response's power spectrum smoothed by the rounded-exponential
auditory filters of Glasberg and Moore (1990), one ERB wide.
- Centres every 0.25 Cam from 200 Hz to 10 kHz.
- Any frequency grid, each sample weighted by its bandwidth.
- At each centre, the difference `Δ(c) = 10 log10(E_variant / E_twin)` in dB.

**Two candidate summaries.**

- **A**, `sqrt(∫ Δ(c)² dc)` over ERB number (dB·√Cam). The channels add as independent observations
  (multichannel integration), so a broad difference counts more than a narrow one of the same depth.
- **B**, `max_c |Δ(c)|`. The largest change of excitation in any auditory channel (Zwicker's
  criterion).

**Step 0, calibration, run and committed before any loudspeaker is computed.**

1. The stimuli are resonances: analog peaking filters (RBJ form) at 1, 2.2 and 5 kHz, with Q = 1, 10
   and 50. Peaks run from +0.25 to +20 dB, dips from −0.25 to −40 dB.
2. **The published anchors are for peaks with music.** Toole and Olive (1988), as Toole summarizes
   them: about 1–2 dB at threshold for Q = 1, about 10 dB for Q = 50. Their exact values were not
   accessible, which makes this calibration coarse.
3. **The statistic.** `L50` is the level of a Q = 50 peak that gives the same value as a +1.5 dB Q = 1
   peak at the same frequency, averaged in dB over the three frequencies.
4. **The choice.** The candidate whose `L50` is closest to 10 dB (in ratio) is kept. It is accepted if
   `L50` lies between 5 and 20 dB. If neither candidate is accepted, the protocol stops there and
   says so.
5. **The unit.** T, "one threshold", is the kept summary's value for a +1.5 dB, Q = 1 peak at 2.2 kHz.
   All results are given in T. It is an approximate threshold for one resonance in music, and it is
   used as a scale.
6. **Dips** are reported against Bücklein (1981), for whom dips are less audible than peaks of the
   same size: the depth of a dip that gives one T. This is a check and plays no part in the choice.

## The pathways, each against the twin

For every loudspeaker, crossover and room:

| | Pathway | Response compared | Summary over the room |
|---|---|---|---|
| P1s | direct sound, seated | the loudspeaker's response toward the listener's head (ears 1.0–1.25 m, the rooms' own) | one value |
| P1d | direct sound, standing | the same toward a head at 1.65 m, same place | one value |
| P2 | each first reflection's spectrum | the response in the direction each of the six first-order paths leaves in | mean of the six, weighted by the energy the path brings (room weight × the twin's power in its direction, both averaged over 600 Hz–8 kHz, and over the four crossovers' twins) |
| P3 | early sound at the ears | direct sound and first reflections added with their phases at the 14 points of `coherent.rs` (two ears, seven head positions), on a 4 Hz grid | mean over the 14 points of each point's distance |
| P4 | steady state | the paths up to the third order and the diffuse field added in power at the reference point (the study's in-room response) | one value |

Each value is a distance in T. A pathway's figure for a loudspeaker and a crossover is its median
over the 300 rooms.

## The decision rule, fixed now

For each pathway and each of the six loudspeakers other than the coaxial, take the room-by-room
paired difference `δ = d(LR2) − d(FIR)` and its median over the rooms.

| Outcome on the pathway | Condition |
|---|---|
| View 1 supported | median δ ≥ +0.5 T for at least 5 of the 6 loudspeakers |
| View 1 contradicted | median δ ≤ −0.5 T for at least 5 of the 6 |
| No general answer | anything else; reported loudspeaker by loudspeaker |

- **Below audibility.** If no variant of any loudspeaker reaches a median of 0.5 T on a pathway, the
  pathway is reported as below audibility by this instrument, whatever the signs.
- **Reported in every case:**
  - LR4 − FIR, LR8 − FIR and LR2 − LR4, by the same rule;
  - every median with its 10–90 % range over the rooms.

## Checks and controls

- **The coaxial** shares one centre for its drivers. Its distance to the twin must be zero to
  rounding. This checks the code.
- **Positive control.** The two-way laid on its side puts the lobing into the direct sound. The study
  found the early sound of a quarter of its listeners more than 3 dB below the median at the
  crossover. Its P1s must
  register this lobing clearly, above 1 T for LR4. If it does not, the instrument is blind to what it
  is meant to see, and the protocol stops.
- **Secondary, descriptive, outside the decision.** Each crossover's twin against the FIR's twin, P4.
  This is how the slope blends the drivers' directivities without lobing: the step of directivity a
  steep crossover makes when the drivers' directivities differ at the crossover (the monitor's
  midrange dome against its tweeter).

## Limits, stated in advance

- **The drivers are pistons.** Break-up, excursion, distortion and power handling are not modelled.
  These are a gentle slope's other costs in practice.
- **The instrument is an approximate model of discrimination, calibrated coarsely.** It says whether
  a difference is likely large enough to hear. It does not say which version listeners would
  prefer.
- **The twin measures departure from the same loudspeaker without interference.** Whether removing
  part of the floor bounce near the crossover sounds better or worse is not decided here.
- **The rooms are simulated.** Rectangular rooms, specular images to the third order, real and
  positive reflection factors for the phases, a diffuse field in energy.

## What will be published

The calibration, the decision table and every median, whichever way they go. The article
« Reading the Spinorama Differently » is corrected after this result.

## Amendment 1, after the first run (17 November 2026)

The first run (`crossover-slopes-results.md`, commit 70c9249) failed both checks above, and the protocol
stopped as it says. The owner of the study approved this amendment before the second run. Everything
above is kept as first written.

1. **The twin, in sums with phases.**
   - **What failed.** The coaxial differed from its twin in P3 (1.06 T), and by 10⁻¹⁵ in P1, P2 and P4.
   - **Why.** Moving every driver to the baffle centre changes nothing in a power. In a sum with
     phases, though, it also moves the source itself: by 2 cm for the coaxial, by 41 cm for a tower's
     tweeter. P3 then measured that displacement rather than the lobing.
   - **The new twin.** At each frequency, every driver is moved to the drivers' centre weighted by
     their power on the reference axis:
     `s0(f) = Σ_d |X_d H_d(axis)|² c_d / Σ_d |X_d H_d(axis)|²` and
     `H_twin,d(u) = H_d(u) exp(-j k u.(c_d - s0))`.
   - **What it changes.**
     - Outside the band where drivers overlap, `s0` is the playing driver's centre, so the twin is the
       variant. The coaxial is its twin exactly.
     - A power is unchanged by the common factor `exp(j k u.s0)`. P1, P2 and P4 must therefore come out
       as in the first run, which is a further check.
2. **The positive control.**
   - **What failed.** The median of P1s with LR4 was 0.57 T, under the 1 T threshold.
   - **Why.** In 145 of the 300 rooms the loudspeaker is aimed at the listener, so the direct sound
     arrives on its axis, where no roll makes it lobe. Those rooms give 0.00 T, against 3.64 T where
     the loudspeaker is parallel to the side walls.
   - **The new criterion.** It is the study's own statement, which concerns a quarter of the
     listeners: the 75th percentile of P1s with LR4 over the 300 rooms must exceed 1 T.
   - **This criterion was set after seeing the data**, where the 75th percentile is 1.99 T.

The decision rule, the instrument and the pathways are unchanged.
