"""Domestic listening rooms: materials, a random ensemble and a few named rooms.

Each setup holds one loudspeaker, the left one of a stereo pair (the right one is its mirror
image), and one listener on the room's middle line, facing the loudspeakers. Distributions
(uniform unless stated):

* room: width 3.4-5.2 m, length 4.5-7.5 m, height 2.4-2.8 m;
* loudspeakers on the short wall, their baffles 0.4-1.2 m from it, 1.8-2.8 m apart (at least
  0.45 m from the side walls), the reference axis (tweeter) 0.85-1.15 m above the floor;
* listener 2.0-3.8 m from the baffles' line (at least 0.5 m in front of the wall behind), ears
  1.0-1.25 m above the floor;
* loudspeaker aimed at the listener (half the setups), parallel to the side walls (a quarter), or
  halfway (a quarter);
* walls of painted plaster on masonry or of plasterboard on studs (half each), one of them with a
  window (2-6 m^2 of glass) and curtains (0-4 m^2); ceiling of plaster or plasterboard;
* floor: parquet or tiles (60 %; half of them with a rug, 2.0-3.0 m wide, 2.5-3.5 m long, in
  front of the loudspeakers), or wall-to-wall carpet (40 %);
* furniture: 6-16 m^2 of equivalent absorption at 1 kHz (half of it at 125 Hz);
* a quarter of the rooms is treated: absorbing panels at the side walls' first reflection points
  and a rug, the ceiling left bare.

The coefficients are typical values from textbook tables (octave bands 125 Hz to 8 kHz).
"""

from __future__ import annotations

import numpy as np

from .room import Patch, Room, Setup, aim

MATERIALS = {
    "plaster": (0.02, 0.02, 0.03, 0.04, 0.05, 0.05, 0.05),     # painted plaster on masonry
    "gypsum": (0.28, 0.12, 0.07, 0.05, 0.06, 0.08, 0.08),      # plasterboard on studs
    "glass": (0.35, 0.25, 0.18, 0.12, 0.07, 0.04, 0.04),       # window
    "curtain": (0.07, 0.31, 0.49, 0.75, 0.70, 0.60, 0.60),     # draped velour
    "parquet": (0.15, 0.11, 0.10, 0.07, 0.06, 0.07, 0.07),
    "tiles": (0.01, 0.01, 0.015, 0.02, 0.02, 0.02, 0.02),
    "carpet": (0.08, 0.24, 0.57, 0.69, 0.71, 0.73, 0.73),      # on underlay
    "rug": (0.05, 0.10, 0.25, 0.45, 0.55, 0.60, 0.60),         # on a hard floor
    "panel": (0.15, 0.55, 0.95, 1.00, 0.97, 0.95, 0.95),       # 50 mm mineral wool on the wall
}

SCATTERING = {
    "wall": (0.05, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30),        # shelves, pictures, furniture along it
    "floor": (0.10, 0.15, 0.25, 0.35, 0.40, 0.45, 0.50),       # furniture standing on it
    "ceiling": (0.05, 0.05, 0.05, 0.08, 0.10, 0.12, 0.15),
    "flat": (0.05, 0.05, 0.10, 0.10, 0.15, 0.15, 0.15),        # a panel or a rug
}

#: Frequency shape of the furniture's absorption (times its value at 1 kHz).
FURNITURE_SHAPE = (0.50, 0.70, 0.90, 1.0, 1.0, 1.0, 1.0)

WALLS = ("left", "right", "listener_wall", "speaker_wall")


def mix(*parts):
    """Area-weighted mixture ``[(material, fraction), ...]`` of octave-band coefficients."""
    out = np.zeros(7)
    for name, share in parts:
        out += share * np.asarray(MATERIALS[name])
    return tuple(out)


def build(dims, *, wall="plaster", ceiling="plaster", floor="parquet", window_wall="right",
          glass=4.0, curtains=2.0, rug=None, panels=False, cloud=False, furniture=6.0,
          spacing=2.4, wall_distance=0.6, listen_distance=3.0, speaker_height=1.0, ear_height=1.1,
          toe="aim", label="") -> Setup:
    """One setup. ``rug = (width, length)`` in front of the loudspeakers; ``panels``: absorbers at the
    side walls' first reflection points; ``cloud``: an absorber on the ceiling between the
    loudspeakers and the listener; ``toe``: ``"aim"``, ``"straight"`` or ``"half"``."""
    lx, ly, lz = dims
    absorption, scattering = {}, {}
    for w in WALLS:
        area = (ly if w in ("left", "right") else lx) * lz
        if w == window_wall:
            g = min(glass, 0.6 * area) / area
            c = min(curtains, 0.6 * area) / area
            absorption[w] = mix((wall, 1.0 - g - c), ("glass", g), ("curtain", c))
        else:
            absorption[w] = MATERIALS[wall]
        scattering[w] = SCATTERING["wall"]
    absorption["ceiling"] = MATERIALS[ceiling]
    scattering["ceiling"] = SCATTERING["ceiling"]
    absorption["floor"] = MATERIALS[floor]
    scattering["floor"] = SCATTERING["floor"]

    source = np.array([lx / 2 - spacing / 2, ly - wall_distance, speaker_height])
    receiver = np.array([lx / 2, ly - wall_distance - listen_distance, ear_height])
    patches = []
    if rug is not None:
        width, length = rug
        y1 = ly - wall_distance - 0.1
        patches.append(Patch("floor", (lx / 2 - width / 2, lx / 2 + width / 2, y1 - length, y1),
                             MATERIALS["rug"], SCATTERING["flat"], "rug"))
    if panels:
        # First reflection point of the left loudspeaker on the left wall, mirrored for the right
        # loudspeaker on the right wall.
        xs, xr = source[0], receiver[0]
        t = xs / (xs + xr)
        y_ref = source[1] + t * (receiver[1] - source[1])
        for w in ("left", "right"):
            patches.append(Patch(w, (y_ref - 0.6, y_ref + 0.6, 0.5, 1.7), MATERIALS["panel"],
                                 SCATTERING["flat"], "panel"))
    if cloud:
        hs, he = source[2], receiver[2]
        t = (lz - hs) / ((lz - hs) + (lz - he))
        y_ref = source[1] + t * (receiver[1] - source[1])
        patches.append(Patch("ceiling", (lx / 2 - 1.2, lx / 2 + 1.2, y_ref - 0.9, y_ref + 0.9),
                             MATERIALS["panel"], SCATTERING["flat"], "cloud"))
    room = Room((lx, ly, lz), absorption, scattering, patches,
                tuple(furniture * np.asarray(FURNITURE_SHAPE)), label)
    yaw_aim = aim(source, receiver)
    yaw = {"aim": yaw_aim, "straight": 0.0, "half": 0.5 * yaw_aim}[toe]
    meta = dict(dims=dims, wall=wall, ceiling=ceiling, floor=floor, rug=rug is not None, panels=panels,
                cloud=cloud, furniture=furniture, spacing=spacing, wall_distance=wall_distance,
                listen_distance=listen_distance, speaker_height=speaker_height, ear_height=ear_height, toe=toe)
    return Setup(room, source, receiver, yaw, label, meta)


def sample(n: int, seed: int = 2034, treated_share: float = 0.25) -> list[Setup]:
    """``n`` random setups from the distributions of the module docstring."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        lx, ly, lz = rng.uniform(3.4, 5.2), rng.uniform(4.5, 7.5), rng.uniform(2.4, 2.8)
        spacing = min(rng.uniform(1.8, 2.8), lx - 0.9)
        wall_distance = rng.uniform(0.4, 1.2)
        listen = rng.uniform(2.0, 3.8)
        if ly - wall_distance - listen < 0.5:
            continue
        treated = rng.uniform() < treated_share
        hard = rng.uniform() < 0.6
        floor = str(rng.choice(["parquet", "tiles"])) if hard else "carpet"
        has_rug = hard and (treated or rng.uniform() < 0.5)
        rug = (rng.uniform(2.0, 3.0), rng.uniform(2.5, 3.5)) if has_rug else None
        u = rng.uniform()
        toe = "aim" if u < 0.5 else ("straight" if u < 0.75 else "half")
        setup = build(
            (lx, ly, lz), wall=str(rng.choice(["plaster", "gypsum"])), ceiling=str(rng.choice(["plaster", "gypsum"])),
            floor=floor, window_wall=str(rng.choice(["left", "right", "listener_wall"])),
            glass=rng.uniform(2.0, 6.0), curtains=rng.uniform(0.0, 4.0), rug=rug, panels=treated,
            furniture=rng.uniform(6.0, 16.0), spacing=spacing, wall_distance=wall_distance,
            listen_distance=listen, speaker_height=rng.uniform(0.85, 1.15), ear_height=rng.uniform(1.0, 1.25),
            toe=toe, label=f"room {len(out) + 1}")
        out.append(setup)
    return out


def named() -> dict[str, Setup]:
    """Three illustrative rooms."""
    return {
        "living": build((4.2, 6.0, 2.5), floor="parquet", rug=(2.4, 3.0), furniture=10.0,
                        label="living room, 4.2 x 6.0 x 2.5 m, parquet and rug"),
        "treated": build((4.2, 6.0, 2.5), floor="parquet", rug=(2.4, 3.0), panels=True, furniture=10.0,
                         label="the same, side-wall panels at the first reflections"),
        "small": build((3.5, 4.4, 2.45), floor="carpet", wall="gypsum", furniture=7.0, spacing=2.0,
                       wall_distance=0.5, listen_distance=2.2, label="small room, 3.5 x 4.4 x 2.45 m, carpet"),
    }
