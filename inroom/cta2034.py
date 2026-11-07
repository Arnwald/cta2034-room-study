"""The spatial averages of CTA-2034-A, and the weights they give to each direction.

Two orbits of 10-degree steps around the reference axis (speaker frame of :mod:`inroom.speaker`):

* horizontal, angle ``h`` positive toward +x: ``u = (sin h, 0, cos h)``;
* vertical, angle ``v`` positive upward: ``u = (0, sin v, cos v)`` (beyond 90 degrees the direction
  points backward, over the top or under the bottom).

Curves (power spectra; every average is an RMS average of pressures, a mean of ``|p|^2``):

* Listening Window (LW): 0, +-10 degrees vertical, +-10, +-20, +-30 degrees horizontal;
* Early Reflections (ER): the mean of five bounces, each the mean of its directions: floor
  (-20, -30, -40 vertical), ceiling (40, 50, 60 vertical), front wall (0, +-10, +-20, +-30
  horizontal), side walls (+-40 ... +-80 horizontal), rear wall (+-90, 180 horizontal);
* Sound Power (SP): all 70 directions, each weighted by the solid angle it stands for (Appendix C
  of the standard): the band between the "spherical squares" of half-width ``theta - 5`` and
  ``theta + 5`` degrees, ``w(theta) = S(theta + 5) - S(theta - 5)``, ``S(alpha) = 4 C - 2 pi`` with
  ``C`` the corner angle of the square bounded by the great circles at +-alpha horizontally and
  vertically; symmetric about 90 degrees, the 90-degree band counted twice;
* Predicted In-Room Response: ``PIR = 0.12 LW + 0.44 ER + 0.44 SP``;
* directivity indices ``SPDI = LW / SP`` and ``ERDI = LW / ER`` (dB).

Every curve is a weighted mean of the 70 power responses, with weights that sum to 1: :func:`kernel`
gives them. ``PIR`` is the same for every loudspeaker with the same directivity in all directions.
"""

from __future__ import annotations

import math

import numpy as np

STEP = 10
H_ANGLES = tuple(range(-170, 181, STEP))
V_ANGLES = tuple(range(-170, 181, STEP))

LISTENING_WINDOW_H = (0, 10, -10, 20, -20, 30, -30)
LISTENING_WINDOW_V = (10, -10)
FRONT_WALL = (0, 10, -10, 20, -20, 30, -30)
SIDE_WALL = (40, -40, 50, -50, 60, -60, 70, -70, 80, -80)
REAR_WALL = (90, -90, 180)
FLOOR = (-20, -30, -40)
CEILING = (40, 50, 60)
BOUNCES = {"floor": ("v", FLOOR), "ceiling": ("v", CEILING), "front": ("h", FRONT_WALL),
           "side": ("h", SIDE_WALL), "rear": ("h", REAR_WALL)}
PIR_WEIGHTS = {"lw": 0.12, "er": 0.44, "sp": 0.44}


def _square_solid_angle(alpha_deg: float) -> float:
    a = math.radians(alpha_deg)
    gamma = math.acos(math.cos(a) * math.cos(a))
    corner_a = math.atan(math.sin(a) / math.tan(a))
    corner = math.acos(-math.cos(corner_a) ** 2 + math.sin(corner_a) ** 2 * math.cos(gamma))
    return 4.0 * corner - 2.0 * math.pi


def sound_power_weights() -> dict[int, float]:
    """Weight of every angle of the 10-degree orbit, 0 ... 180 (symmetric about 90)."""
    edges = [i * 10 + 5 for i in range(9)] + [90]
    areas = [_square_solid_angle(e) for e in edges]
    w = [areas[0]] + [areas[i] - areas[i - 1] for i in range(1, 10)]
    w[9] *= 2.0
    out = {}
    for i, weight in enumerate(w):
        out[10 * i] = weight
        out[180 - 10 * i] = weight
    return out


SP_WEIGHTS = sound_power_weights()


def direction(plane: str, angle: float) -> np.ndarray:
    a = math.radians(angle)
    if plane == "h":
        return np.array([math.sin(a), 0.0, math.cos(a)])
    return np.array([0.0, math.sin(a), math.cos(a)])


#: The 70 directions, as ``(plane, angle)`` keys: the whole horizontal orbit, and the vertical one
#: without 0 and 180 degrees (shared with the horizontal orbit).
KEYS = tuple([("h", a) for a in H_ANGLES] + [("v", a) for a in V_ANGLES if a not in (0, 180)])
DIRECTIONS = np.array([direction(p, a) for p, a in KEYS])


def _key(plane: str, angle: float):
    a = int(round(angle)) % 360
    a = a - 360 if a > 180 else a
    if a in (0, 180):
        plane = "h"
    return plane, a


def kernel(curve: str) -> np.ndarray:
    """Weights of the 70 directions (in the order of :data:`KEYS`) of a curve: ``lw``, ``er``,
    ``sp``, ``pir``, or one bounce of the early reflections (``floor``, ``ceiling``, ``front``,
    ``side``, ``rear``)."""
    index = {k: i for i, k in enumerate(KEYS)}
    w = np.zeros(len(KEYS))
    if curve == "lw":
        keys = [_key("h", a) for a in LISTENING_WINDOW_H] + [_key("v", a) for a in LISTENING_WINDOW_V]
        for k in keys:
            w[index[k]] += 1.0 / len(keys)
    elif curve in BOUNCES:
        plane, angles = BOUNCES[curve]
        for a in angles:
            w[index[_key(plane, a)]] += 1.0 / len(angles)
    elif curve == "er":
        for b in BOUNCES:
            w += kernel(b) / len(BOUNCES)
    elif curve == "sp":
        for i, (_, a) in enumerate(KEYS):
            w[i] = SP_WEIGHTS[abs(a)]
        w /= w.sum()
    elif curve == "pir":
        for c, weight in PIR_WEIGHTS.items():
            w += weight * kernel(c)
    else:
        raise ValueError(curve)
    return w


def curves(power: np.ndarray) -> dict[str, np.ndarray]:
    """Every curve from the power responses at the 70 directions, ``power`` of shape
    ``(70, ..., n_freqs)``; ``spdi`` and ``erdi`` in dB."""
    power = np.asarray(power, float)
    out = {"on_axis": power[KEYS.index(("h", 0))]}
    for c in ("lw", "er", "sp", "pir") + tuple(BOUNCES):
        out[c] = np.tensordot(kernel(c), power, axes=(0, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["spdi"] = 10.0 * np.log10(out["lw"] / out["sp"])
        out["erdi"] = 10.0 * np.log10(out["lw"] / out["er"])
    return out
