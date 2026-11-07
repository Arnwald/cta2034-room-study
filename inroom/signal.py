"""Frequency grids, minimum-phase reconstruction, smoothing."""

from __future__ import annotations

import numpy as np

from .constants import F_HI, F_LO, PER_OCTAVE


def log_grid(f_lo: float = F_LO, f_hi: float = F_HI, per_octave: int = PER_OCTAVE) -> np.ndarray:
    """Logarithmically spaced grid from ``f_lo`` to (at least) ``f_hi``."""
    n = int(np.ceil(np.log2(f_hi / f_lo) * per_octave)) + 1
    return f_lo * 2.0 ** (np.arange(n) / per_octave)


def rfft_freqs(n: int, fs: float) -> np.ndarray:
    return np.arange(n // 2 + 1) * fs / n


def interp_log(f_src, values, f_dst):
    """Linear interpolation in log frequency, end values held (``values`` on the last axis)."""
    x_src = np.log2(np.asarray(f_src, float))
    x_dst = np.log2(np.asarray(f_dst, float))
    values = np.asarray(values, float)
    if values.ndim == 1:
        return np.interp(x_dst, x_src, values)
    flat = values.reshape(-1, values.shape[-1])
    out = np.stack([np.interp(x_dst, x_src, v) for v in flat])
    return out.reshape(values.shape[:-1] + (len(x_dst),))


def minimum_phase(magnitude: np.ndarray) -> np.ndarray:
    """Minimum-phase spectrum with the given magnitude, on the bins of an even-length real FFT
    (homomorphic method: the real cepstrum of ``ln|H|`` folded onto positive quefrencies)."""
    mag = np.maximum(np.asarray(magnitude, float), 1e-12 * np.max(magnitude))
    n = 2 * (len(mag) - 1)
    cep = np.fft.irfft(np.log(mag), n)
    fold = np.zeros(n)
    fold[0] = cep[0]
    fold[1 : n // 2] = 2.0 * cep[1 : n // 2]
    fold[n // 2] = cep[n // 2]
    return np.exp(np.fft.rfft(fold))


def minimum_phase_at(freqs, magnitude, fs: float = 192_000.0, n: int = 1 << 20, reach: float = 2.0):
    """Minimum-phase response with the magnitude ``magnitude`` given on the log grid ``freqs``.

    The log-magnitude is interpolated on a fine real-FFT grid, extended beyond the grid with the
    slopes of its last sixth of an octave for ``reach`` octaves and then held, its minimum-phase
    version computed there and sampled back at ``freqs``."""
    freqs = np.asarray(freqs, float)
    log_mag = np.log(np.maximum(np.asarray(magnitude, float), 1e-300))
    fine = rfft_freqs(n, fs)
    octaves = np.log2(np.maximum(fine, 1e-9) / freqs[0])
    x = np.log2(freqs / freqs[0])
    top = x[-1]
    lo = x <= 1.0 / 6.0
    hi = x >= top - 1.0 / 6.0
    below = np.polyfit(x[lo], log_mag[lo], 1)[0]
    above = np.polyfit(x[hi], log_mag[hi], 1)[0]
    lm = np.interp(octaves, x, log_mag)
    lm = np.where(octaves < 0, log_mag[0] + below * np.maximum(octaves, -reach), lm)
    lm = np.where(octaves > top, log_mag[-1] + above * np.minimum(octaves - top, reach), lm)
    h = minimum_phase(np.exp(lm - lm.max())) * np.exp(lm.max())
    phase = np.interp(freqs, fine[1:], np.unwrap(np.angle(h[1:])))
    return np.asarray(magnitude, float) * np.exp(1j * phase)


def smooth_power(freqs, power, fraction: float, out_freqs=None):
    """Fractional-octave smoothing of a power spectrum (rectangular window in log frequency).

    ``out[i]`` is the mean of ``power`` over the points within ``out_freqs[i] * 2**(+-1/(2*fraction))``.
    On a log grid every point stands for the same fraction of an octave, so the mean is the
    log-frequency average. ``power`` may be N-D with frequency on the last axis."""
    freqs = np.asarray(freqs, float)
    power = np.asarray(power, float)
    out_freqs = freqs if out_freqs is None else np.asarray(out_freqs, float)
    half = 2.0 ** (1.0 / (2.0 * fraction))
    lo = np.searchsorted(freqs, out_freqs / half, side="left")
    hi = np.searchsorted(freqs, out_freqs * half, side="right")
    nearest = np.clip(np.searchsorted(freqs, out_freqs), 0, len(freqs) - 1)
    empty = hi <= lo
    lo = np.where(empty, nearest, lo)
    hi = np.where(empty, nearest + 1, hi)
    csum = np.concatenate([np.zeros(power.shape[:-1] + (1,)), np.cumsum(power, axis=-1)], axis=-1)
    return (csum[..., hi] - csum[..., lo]) / (hi - lo)


def db10(p, floor=1e-30):
    """10 log10 p."""
    return 10.0 * np.log10(np.maximum(p, floor))


def db20(x, floor=1e-15):
    """20 log10 |x|."""
    return 20.0 * np.log10(np.maximum(np.abs(x), floor))
