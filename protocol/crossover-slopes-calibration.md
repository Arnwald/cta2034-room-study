# Step 0: calibration of the instrument

Output of `cargo run --release --bin slopes -- calibrate`, run before any loudspeaker was computed
(protocol: `crossover-slopes.md`).

Each table gives, for resonances (RBJ peaking filters) at 1000, 2200, 5000 Hz, the level that reaches the same value of the summary as a +1.5 dB, Q = 1 peak at the same frequency. Published anchors (peaks, music, Toole and Olive 1988): about 1-2 dB for Q = 1, about 10 dB for Q = 50. Dips: depth (dB) for the same value; "never" means not reached at 60 dB.

## Summary A, root sum of squares over ERB number

| | peak, Q = 1 | peak, Q = 10 | peak, Q = 50 | dip, Q = 1 | dip, Q = 10 | dip, Q = 50 |
|---|---|---|---|---|---|---|
| 1000 Hz | 1.5 | 5.0 | 11.4 | 1.5 | 6.1 | 26.4 |
| 2200 Hz | 1.5 | 4.9 | 11.3 | 1.5 | 5.9 | 25.5 |
| 5000 Hz | 1.5 | 4.8 | 11.1 | 1.5 | 5.8 | 24.8 |

L50 (Q = 50 peak, mean of the three frequencies): **11.2 dB**. Value of a +1.5 dB, Q = 1 peak at 2.2 kHz: 3.711.

## Summary B, largest change in one channel

| | peak, Q = 1 | peak, Q = 10 | peak, Q = 50 | dip, Q = 1 | dip, Q = 10 | dip, Q = 50 |
|---|---|---|---|---|---|---|
| 1000 Hz | 1.5 | 2.3 | 5.7 | 1.5 | 2.6 | 10.2 |
| 2200 Hz | 1.5 | 2.3 | 5.5 | 1.5 | 2.5 | 9.6 |
| 5000 Hz | 1.5 | 2.2 | 5.2 | 1.5 | 2.4 | 8.6 |

L50 (Q = 50 peak, mean of the three frequencies): **5.5 dB**. Value of a +1.5 dB, Q = 1 peak at 2.2 kHz: 1.479.

## Choice

Summary A, root sum of squares over ERB number is kept: its L50, 11.2 dB, is the closest to 10 dB and within 5-20 dB. The unit T is 3.711 (its value for a +1.5 dB, Q = 1 peak at 2.2 kHz).
