"""Crossovers: acoustic targets on the reference axis, and the filters that reach them.

Each driver is equalized to an acoustic target on the reference axis (the baffle normal, far
field): its filter is ``X_i = T_i E_i``, ``T_i`` the target and ``E_i`` the minimum-phase inverse of
the driver's 1/6-octave smoothed on-axis magnitude (which removes the baffle step and the smooth
part of the edge ripple). The drivers then sum on axis to the crossover's all-pass (IIR) or to a
pure delay (FIR); off axis, what is left is each driver's directivity, the edge diffraction and
the path differences between the drivers.

Targets, for a crossover frequency ``fc``:

* ``lr2``, ``lr4``, ``lr8``: Linkwitz-Riley of order 2, 4, 8 (squared Butterworth of order 1, 2,
  4), minimum phase. The high-pass of the LR2 has its polarity inverted, so that both outputs
  are in phase at ``fc`` (as with LR4 and LR8, whose outputs are in phase without inversion).
* ``fir``: the brickwall linear-phase FIR crossover of the four-way article: a low-pass with
  1501 taps at 93.75 kHz (16 ms), a windowed sinc with a Kaiser window (beta = 8), and the
  high-pass made as an impulse minus the low-pass, so that both paths add up to a pure delay.
  Its transition band is about 320 Hz wide (stopband 81 dB down). The common delay of 8 ms is
  left out (a bulk delay has no effect on any curve of this study).
"""

from __future__ import annotations

import numpy as np

from .signal import minimum_phase_at, smooth_power

KINDS = ("lr2", "lr4", "lr8", "fir")
LABELS = {"lr2": "LR2 (12 dB/oct)", "lr4": "LR4 (24 dB/oct)", "lr8": "LR8 (48 dB/oct)",
          "fir": "brickwall FIR"}

FIR_TAPS = 1501
FIR_FS = 93_750.0
FIR_BETA = 8.0


def butterworth(f, fc, order, kind):
    """Analog Butterworth low-pass or high-pass of any order at frequencies ``f``."""
    s = 1j * np.asarray(f, float) / fc
    if kind == "hp":
        s = 1.0 / s
    out = np.ones_like(s)
    for k in range(1, order // 2 + 1):
        theta = np.pi * (2 * k + order - 1) / (2 * order)
        out = out / (s**2 - 2.0 * np.cos(theta) * s + 1.0)
    if order % 2:
        out = out / (s + 1.0)
    return out


def linkwitz_riley(f, fc, order, kind):
    return butterworth(f, fc, order // 2, kind) ** 2


def fir_lowpass_amplitude(f, fc, taps=FIR_TAPS, fs=FIR_FS, beta=FIR_BETA):
    """Zero-phase amplitude of the brickwall low-pass (its bulk delay removed), at ``f``."""
    m = (taps - 1) // 2
    n = np.arange(-m, m + 1)
    h = 2.0 * fc / fs * np.sinc(2.0 * fc / fs * n) * np.kaiser(taps, beta)
    h = h / h.sum()
    f = np.asarray(f, float)
    return h[m] + 2.0 * np.cos(2.0 * np.pi * np.outer(f, n[m + 1:]) / fs) @ h[m + 1:]


def pair(kind: str, f, fc):
    """Low-pass and high-pass target of one crossover (complex)."""
    f = np.asarray(f, float)
    if kind == "fir":
        lp = fir_lowpass_amplitude(f, fc).astype(complex)
        return lp, 1.0 - lp
    order = {"lr2": 2, "lr4": 4, "lr8": 8}[kind]
    lp = linkwitz_riley(f, fc, order, "lp")
    hp = linkwitz_riley(f, fc, order, "hp")
    if order % 4 == 2:
        hp = -hp
    return lp, hp


def targets(kind: str, f, crossovers) -> list[np.ndarray]:
    """Acoustic targets of the drivers of a layout, lowest first."""
    n = len(crossovers) + 1
    out = [np.ones(len(f), complex) for _ in range(n)]
    for i, fc in enumerate(crossovers):
        lp, hp = pair(kind, f, fc)
        out[i] = out[i] * lp
        out[i + 1] = out[i + 1] * hp
    return out


def equalizers(freqs, on_axis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Minimum-phase inverse of each driver's 1/6-octave smoothed on-axis magnitude."""
    out = {}
    for name, h in on_axis.items():
        smooth = np.sqrt(smooth_power(freqs, np.abs(h) ** 2, 6))
        out[name] = minimum_phase_at(freqs, 1.0 / np.maximum(smooth, 1e-9))
    return out


def filters(kind: str, freqs, layout, on_axis: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``{driver: X_i}`` for the layout's drivers (lowest first) and crossover ``kind``."""
    eq = equalizers(freqs, on_axis)
    t = targets(kind, freqs, layout.crossovers)
    return {name: t[i] * eq[name] for i, name in enumerate(layout.names)}


def system(filters_: dict, spectra: dict) -> np.ndarray:
    """``sum_i X_i H_i`` (any leading shape)."""
    return sum(filters_[n] * spectra[n] for n in filters_)
