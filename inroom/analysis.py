"""Curves and metrics computed from results/study.npz.

In-room responses are given relative to an omnidirectional source at the same place
(:mod:`inroom.study`). Two of them:

* **steady state**: all the energy (direct sound, images up to order 3, diffuse field);
* **early**: the direct sound and the six first-order reflections, what CTA-2034's Listening
  Window and Early Reflections stand for.

Anomaly metric: the curve, in dB on the 1/96-octave grid, minus the straight line (dB against
log frequency) that best fits it from 600 Hz to 8 kHz, the three octaves around the crossover;
the RMS of that residual over the same band. A straight line is what a loudspeaker whose
directivity index rises evenly would show; a crossover adds a broad dip, a notch or a step.

Olive's narrow-band deviation (NBD, the mean absolute deviation from the band's mean in
half-octave bands from 100 Hz to 12 kHz) is also given, on the part of that range the study covers
(from 150 Hz).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import cta2034
from .signal import db10

RESULTS = Path(__file__).resolve().parent.parent / "results" / "study.npz"

#: Band of the anomaly metric (Hz).
ANOMALY_BAND = (600.0, 8000.0)
#: One octave either side of the 2.2 kHz crossover (Hz): where the crossover's defects lie.
CROSSOVER_BAND = (1100.0, 4400.0)
#: Band of the prediction errors (Hz): above the rooms' transition frequency.
ERROR_BAND = (300.0, 16000.0)
#: First (training) half of the random rooms, used to fit the updated weights; the second half
#: tests them.
SPLIT = 150

EARLY = slice(0, 7)


@dataclass
class Study:
    freqs: np.ndarray
    n_random: int
    omni: np.ndarray
    variants: list
    data: dict
    meta: dict
    names: list
    named: list
    raw: object

    def rooms(self, variant: str, which: str = "steady", rooms=slice(None)) -> np.ndarray:
        """Normalized in-room responses (power), shape ``(n_rooms, n_freqs)``."""
        e = self.data[variant]["rooms"][rooms]
        g = self.omni[rooms]
        if which == "steady":
            return e.sum(axis=1) / g.sum(axis=1)
        if which == "early":
            return e[:, EARLY].sum(axis=1) / g[:, EARLY].sum(axis=1)
        raise ValueError(which)

    def random(self) -> slice:
        return slice(0, self.n_random)

    def train(self) -> slice:
        return slice(0, SPLIT)

    def test(self) -> slice:
        return slice(SPLIT, self.n_random)

    def curves(self, variant: str) -> dict:
        c = cta2034.curves(self.data[variant]["cta"])
        c["sp_sphere"] = self.data[variant]["sp_full"]
        return c


def load(path: Path = RESULTS) -> Study:
    d = np.load(path)
    variants = [str(v) for v in d["variants"]]
    data = {v: {k: d[f"{v}:{k}"] for k in ("cta", "sp_full", "rooms")} for v in variants}
    meta = {k[len("meta_"):]: d[k] for k in d.files if k.startswith("meta_")}
    return Study(d["freqs"], int(d["n_random"]), d["omni"].astype(float), variants, data, meta,
                 [str(n) for n in d["names"]], [str(n) for n in d["named"]], d)


def band(freqs, lo_hi) -> np.ndarray:
    return (freqs >= lo_hi[0]) & (freqs <= lo_hi[1])


def line_residual(freqs, power, lo_hi=ANOMALY_BAND):
    """Residual (dB) of the straight-line fit over ``lo_hi``, shape ``(..., n_band)``, and its RMS."""
    m = band(freqs, lo_hi)
    x = np.log2(freqs[m])
    y = db10(np.asarray(power, float))[..., m]
    a = np.stack([x, np.ones_like(x)], axis=1)
    flat = y.reshape(-1, y.shape[-1])
    coef, *_ = np.linalg.lstsq(a, flat.T, rcond=None)
    r = (flat - (a @ coef).T).reshape(y.shape)
    return r, np.sqrt(np.mean(r**2, axis=-1))


def anomaly(freqs, power, lo_hi=ANOMALY_BAND):
    """RMS residual of the straight-line fit (dB)."""
    return line_residual(freqs, power, lo_hi)[1]


def local_residual(freqs, power, fit=ANOMALY_BAND, lo_hi=CROSSOVER_BAND):
    """Residual of the straight line fitted over ``fit``, kept within ``lo_hi`` (dB)."""
    r, _ = line_residual(freqs, power, fit)
    f_fit = freqs[band(freqs, fit)]
    return r[..., (f_fit >= lo_hi[0]) & (f_fit <= lo_hi[1])]


def local_anomaly(freqs, power, fit=ANOMALY_BAND, lo_hi=CROSSOVER_BAND):
    """RMS of the residual within one octave of the crossover (dB)."""
    r = local_residual(freqs, power, fit, lo_hi)
    return np.sqrt(np.mean(r**2, axis=-1))


def notch_depth(freqs, power, fit=ANOMALY_BAND, lo_hi=CROSSOVER_BAND):
    """Depth of the deepest dip below the straight line within one octave of the crossover (dB,
    positive)."""
    return -np.min(local_residual(freqs, power, fit, lo_hi), axis=-1)


def nbd(freqs, power, lo=150.0, hi=12000.0):
    """Olive's narrow-band deviation over half-octave bands from 100 Hz (those within ``lo``-``hi``)."""
    y = db10(np.asarray(power, float))
    edges = 100.0 * 2.0 ** (np.arange(0, 16) / 2.0)
    out = []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        if e0 < lo * 0.999 or e1 > hi * 1.001:
            continue
        m = (freqs >= e0) & (freqs < e1)
        yb = y[..., m]
        out.append(np.mean(np.abs(yb - yb.mean(axis=-1, keepdims=True)), axis=-1))
    return np.mean(out, axis=0)


def pir(curves: dict, weights=(0.12, 0.44, 0.44), sp: str = "sp") -> np.ndarray:
    """``w_LW LW + w_ER ER + w_SP SP``; ``sp`` = ``"sp"`` (two orbits) or ``"sp_sphere"``."""
    return weights[0] * curves["lw"] + weights[1] * curves["er"] + weights[2] * curves[sp]


def fit_weights(study: Study, sp: str = "sp_sphere", rooms=None, which: str = "steady",
                lo_hi=ERROR_BAND) -> np.ndarray:
    """Weights ``(w_LW, w_ER, w_SP)``, summing to 1, that best predict (least squares on the level
    ratio) the mean in-room response of ``rooms`` for every variant."""
    rooms = study.train() if rooms is None else rooms
    m = band(study.freqs, lo_hi)
    rows, rhs = [], []
    for v in study.variants:
        c = study.curves(v)
        target = study.rooms(v, which, rooms).mean(axis=0)[m]
        lw, er, s = c["lw"][m], c["er"][m], c[sp][m]
        # target ~ w1 lw + w2 er + (1 - w1 - w2) s, relative errors
        rows.append(np.stack([(lw - s) / target, (er - s) / target], axis=1))
        rhs.append(1.0 - s / target)
    a, b = np.concatenate(rows), np.concatenate(rhs)
    (w1, w2), *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.array([w1, w2, 1.0 - w1 - w2])


def errors(study: Study, variant: str, prediction: np.ndarray, rooms=None, which: str = "steady",
           lo_hi=ERROR_BAND):
    """Per-room error ``prediction - room`` (dB) over ``lo_hi``: its RMS per room, and the mean error
    against frequency (the bias)."""
    rooms = study.test() if rooms is None else rooms
    m = band(study.freqs, lo_hi)
    e = db10(prediction)[None, :] - db10(study.rooms(variant, which, rooms))
    return np.sqrt(np.mean(e[:, m] ** 2, axis=1)), e.mean(axis=0)


def shares(study: Study, rooms=None) -> np.ndarray:
    """Share of each category of path in the energy at the listener, omnidirectional source,
    shape ``(n_rooms, n_categories, n_freqs)``."""
    rooms = study.random() if rooms is None else rooms
    o = study.omni[rooms]
    return o / o.sum(axis=1, keepdims=True)


# --------------------------------------------------------------------------------------------
# Colouration of the paths themselves
# --------------------------------------------------------------------------------------------

#: Groups of paths: every path, the direct sound and first reflections, the first reflections,
#: the floor and ceiling reflections.
GROUPS = {"early": lambda c: c <= 6, "reflections": lambda c: (c >= 1) & (c <= 6),
          "vertical": lambda c: (c == 5) | (c == 6)}


def _common_power(study: Study, variant: str, key: str) -> np.ndarray:
    """Band-averaged power of the four crossovers of the same loudspeaker, averaged: weights that do
    not depend on the crossover, so that a notch does not lower the weight of the path it colours."""
    layout, orient, _ = variant.split("-")
    kinds = [v.split("-")[2] for v in study.variants if v.startswith(f"{layout}-{orient}-")]
    return np.mean([study.raw[f"{layout}-{orient}-{k}:{key}"] for k in kinds], axis=0)


def path_anomaly(study: Study, variant: str, which: str = "steady", rooms=None, metric: str = "local",
                 common: bool = True) -> np.ndarray:
    """Colouration of the paths, per room: each path's own deviation, weighted by the energy it brings.

    Every path (the direct sound, each image, each direction of the diffuse field) carries the
    loudspeaker's response in its own direction. Its deviation (``metric``: ``"local"``, the RMS
    residual within one octave of the crossover, :func:`local_anomaly`; ``"notch"``, the deepest dip
    there, :func:`notch_depth`; ``"anom"``, the RMS residual from 600 Hz to 8 kHz) is weighted by the
    room's weight of the path times the loudspeaker's power in its direction, both averaged over
    600 Hz--8 kHz; with ``common`` the power is that of the four crossovers averaged, the same for
    all of them. ``which``: ``"steady"`` (every path), ``"early"``, ``"reflections"`` or ``"vertical"``
    (the floor and ceiling reflections)."""
    d = study.raw
    rooms = study.random() if rooms is None else rooms
    idx = range(study.omni.shape[0])[rooms]
    starts = d["image_starts"]
    cat = d["image_cat"]
    p_img = _common_power(study, variant, "pband_img") if common else d[f"{variant}:pband_img"]
    p_grid = _common_power(study, variant, "pband_grid") if common else d[f"{variant}:pband_grid"]
    w_img = d["image_weights_band"] * p_img
    a_img = d[f"{variant}:{metric}_img"]
    a_grid = d[f"{variant}:{metric}_grid"]
    out = []
    for r in idx:
        sl = slice(starts[r], starts[r + 1])
        c, w, a = cat[sl], w_img[sl], a_img[sl]
        if which == "steady":
            wd = d["diffuse_band"][r] * p_grid
            out.append((np.sum(w * a) + np.sum(wd * a_grid)) / (np.sum(w) + np.sum(wd)))
        else:
            m = GROUPS[which](c)
            out.append(np.sum(w[m] * a[m]) / np.sum(w[m]))
    return np.array(out)


def cta_path_anomaly(study: Study, variant: str, weights=(0.12, 0.44, 0.44), sp: str = "sp",
                     metric: str = "local", common: bool = True) -> float:
    """The same colouration measure weighted by a CTA-2034-style kernel (``w_LW`` over the listening
    window, ``w_ER`` over the early reflections' directions, ``w_SP`` over the 70 directions with the
    standard's weights, ``sp = "sp"``, or uniformly over the sphere, ``"sp_sphere"``), each direction
    also weighted by the loudspeaker's power there."""
    d = study.raw
    a = d[f"{variant}:{metric}_cta"]
    p = _common_power(study, variant, "pband_cta") if common else d[f"{variant}:pband_cta"]
    k = weights[0] * cta2034.kernel("lw") + weights[1] * cta2034.kernel("er")
    num, den = np.sum(k * p * a), np.sum(k * p)
    if sp == "sp":
        num += weights[2] * np.sum(cta2034.kernel("sp") * p * a)
        den += weights[2] * np.sum(cta2034.kernel("sp") * p)
    else:
        pg = _common_power(study, variant, "pband_grid") if common else d[f"{variant}:pband_grid"]
        ag = d[f"{variant}:{metric}_grid"]
        num += weights[2] * np.mean(pg * ag)
        den += weights[2] * np.mean(pg)
    return float(num / den)
