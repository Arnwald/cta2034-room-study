"""Shoebox rooms: image sources for the early sound, a diffuse field for the rest.

Room frame (metres): ``x`` across the room, toward the listener's right; ``y`` along it, from the
wall behind the listener (``y = 0``) to the wall behind the loudspeakers (``y = Ly``); ``z`` up from
the floor. The listener faces +y, the loudspeakers face the listener.

The sound reaching the listener is split three ways, in energy (pressures squared, every path's
phase taken as random, which is what a spatial average over a few tenths of a metre gives above
the room's transition frequency):

1. **Specular images** up to order ``N`` (3 by default). The image ``u = 2 n L + (1 - 2 q) s`` of the
   source ``s`` on each axis (``n`` integer, ``q`` 0 or 1) has bounced ``|n - q|`` times on the wall at
   0 and ``|n|`` times on the wall at ``L``. The loudspeaker emits that path in the direction
   ``P (r - u) / |r - u|``, ``P`` flipping the axes with an odd number of bounces. Its weight is

       W = prod (1 - alpha)(1 - s) * exp(-m |r - u|) / |r - u|^2,

   one factor per bounce: ``alpha`` the absorption, ``s`` the scattering coefficient (the share of
   the reflected energy sent in all directions instead of the specular one), ``m`` the energy
   attenuation per metre of the air, and of the furniture beyond the first order.
2. **Diffuse field.** At each bounce of a ray launched in a direction ``Omega``, the scattered share
   ``(1 - alpha) s`` of its energy joins a diffuse field; after the ``N``-th bounce, all of what is
   left does. Each direction thus feeds the diffuse field with a share ``g(Omega)`` of what it
   carries (rays traced exactly in the box, absorbing patches included). In steady state that
   field gives everywhere

       p^2 = 4 / A * integral g(Omega) |H(Omega)|^2 dOmega,
       A = -S ln(1 - mean alpha) + 4 m V        (Eyring, with air and furniture),

   which is the Hopkins-Stryker room term ``4 / R`` when ``g = 1 - alpha`` everywhere.
3. The **direct sound**, the image of order 0.

Every term is a weight times the loudspeaker's far-field power ``|H|^2`` (at 1 m) in one direction,
so the energy at the listener is ``E(f) = sum_d K_d(f) |H(Omega_d, f)|^2``: the room acts on the
loudspeaker through a set of weights over the directions, a *kernel*, as the CTA-2034 curves do.

Absorbing patches (a rug on the floor, panels on the walls, a ceiling cloud) replace a surface's
coefficients where a ray or a first-order reflection lands on them; the images of higher order
and the diffuse field use each surface's area-weighted mean.

Furniture (sofas, shelves, beds: everything that is not a surface) is an equivalent absorption
area ``A_f``, spread over the volume: an attenuation ``m_f = A_f / (4 V)`` per metre of path, applied to
the images of order 2 and more, to the rays and to the diffuse field (the direct sound and the
first reflections of a listening setup are assumed unobstructed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .constants import AIR_HUMIDITY, AIR_PRESSURE, AIR_TEMPERATURE, C, OCTAVES
from .signal import interp_log

SURFACES = ("left", "right", "listener_wall", "speaker_wall", "floor", "ceiling")
#: (axis, side) of each surface: side 0 is the plane at 0, side 1 the plane at L.
PLANES = {"left": (0, 0), "right": (0, 1), "listener_wall": (1, 0), "speaker_wall": (1, 1),
          "floor": (2, 0), "ceiling": (2, 1)}
SURFACE_INDEX = {(a, s): i for i, (a, s) in enumerate(PLANES.values())}


def bands(values, freqs) -> np.ndarray:
    """Octave-band coefficients (125 Hz ... 8 kHz) interpolated in log frequency, ends held."""
    return interp_log(OCTAVES, np.asarray(values, float), freqs)


def air_attenuation(freqs, temperature=AIR_TEMPERATURE, humidity=AIR_HUMIDITY, pressure=AIR_PRESSURE):
    """Energy attenuation of air per metre, ``m`` (1/m), ISO 9613-1."""
    f = np.asarray(freqs, float)
    t = temperature + 273.15
    t0, t01 = 293.15, 273.16
    pa = pressure / 101.325
    c_sat = -6.8346 * (t01 / t) ** 1.261 + 4.6151
    h = humidity * 10.0**c_sat / pa
    fr_o = pa * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
    fr_n = pa * (t / t0) ** -0.5 * (9.0 + 280.0 * h * np.exp(-4.170 * ((t / t0) ** (-1.0 / 3.0) - 1.0)))
    alpha_db = 8.686 * f**2 * (1.84e-11 / pa * (t / t0) ** 0.5 + (t / t0) ** -2.5 * (
        0.01275 * np.exp(-2239.1 / t) / (fr_o + f**2 / fr_o) + 0.1068 * np.exp(-3352.0 / t) / (fr_n + f**2 / fr_n)))
    return alpha_db / (10.0 * np.log10(np.e))


@dataclass
class Patch:
    """A rectangle of another material on one surface: ``rect = (u0, u1, v0, v1)`` in the surface's
    two in-plane coordinates, in increasing axis order (floor: x, y; side walls: y, z; end walls:
    x, z)."""

    surface: str
    rect: tuple[float, float, float, float]
    absorption: tuple
    scattering: tuple
    label: str = ""

    @property
    def area(self) -> float:
        u0, u1, v0, v1 = self.rect
        return (u1 - u0) * (v1 - v0)


@dataclass
class Room:
    dims: tuple[float, float, float]
    absorption: dict
    scattering: dict
    patches: list = field(default_factory=list)
    #: Equivalent absorption area of the furniture (m^2), octave bands.
    fittings: tuple = (0.0,) * 7
    label: str = ""

    @property
    def volume(self) -> float:
        return float(np.prod(self.dims))

    def area(self, surface: str) -> float:
        axis, _ = PLANES[surface]
        other = [self.dims[i] for i in range(3) if i != axis]
        return other[0] * other[1]

    @property
    def total_area(self) -> float:
        return sum(self.area(s) for s in SURFACES)

    def mean_coefficients(self, surface: str, freqs):
        """Area-weighted absorption and scattering of a surface, patches included."""
        a = self.area(surface)
        alpha = bands(self.absorption[surface], freqs) * a
        scat = bands(self.scattering[surface], freqs) * a
        for p in self.patches:
            if p.surface == surface:
                alpha += (bands(p.absorption, freqs) - bands(self.absorption[surface], freqs)) * p.area
                scat += (bands(p.scattering, freqs) - bands(self.scattering[surface], freqs)) * p.area
        return alpha / a, scat / a

    def furniture_attenuation(self, freqs):
        return bands(self.fittings, freqs) / (4.0 * self.volume)

    def mean_absorption(self, freqs):
        return sum(self.mean_coefficients(s, freqs)[0] * self.area(s) for s in SURFACES) / self.total_area

    def absorption_area(self, freqs):
        """Eyring equivalent absorption area with air and furniture, ``A`` (m^2)."""
        m = air_attenuation(freqs) + self.furniture_attenuation(freqs)
        return -self.total_area * np.log(1.0 - self.mean_absorption(freqs)) + 4.0 * m * self.volume

    def rt60(self, freqs):
        return 24.0 * np.log(10.0) / C * self.volume / self.absorption_area(freqs)

    # Materials at given points: index 0..5 for the surfaces, 6 + i for patch i.
    def material_tables(self, freqs):
        alpha = [bands(self.absorption[s], freqs) for s in SURFACES] + [bands(p.absorption, freqs) for p in self.patches]
        scat = [bands(self.scattering[s], freqs) for s in SURFACES] + [bands(p.scattering, freqs) for p in self.patches]
        return np.array(alpha), np.array(scat)

    def material_at(self, surface_index: np.ndarray, points: np.ndarray) -> np.ndarray:
        """Material index of the hit points ``points`` (n, 3) on the surfaces ``surface_index``."""
        out = np.asarray(surface_index, int).copy()
        for i, p in enumerate(self.patches):
            axis, side = PLANES[p.surface]
            sidx = SURFACE_INDEX[(axis, side)]
            uv = [j for j in range(3) if j != axis]
            u0, u1, v0, v1 = p.rect
            inside = ((surface_index == sidx) & (points[:, uv[0]] >= u0) & (points[:, uv[0]] <= u1)
                      & (points[:, uv[1]] >= v0) & (points[:, uv[1]] <= v1))
            out[inside] = 6 + i
        return out


@dataclass
class Setup:
    """A loudspeaker (its reference point ``source``, its axis turned by ``yaw`` toward +x) and a
    listener's head (``receiver``) in a room."""

    room: Room
    source: np.ndarray
    receiver: np.ndarray
    yaw: float = 0.0
    label: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def axes(self) -> np.ndarray:
        """Rows: the speaker frame's x (right as seen from the listener), y (up), z (forward),
        in room coordinates."""
        s, c = np.sin(self.yaw), np.cos(self.yaw)
        return np.array([[c, s, 0.0], [0.0, 0.0, 1.0], [s, -c, 0.0]])

    def to_speaker(self, d_room):
        return np.asarray(d_room, float) @ self.axes.T

    def to_room(self, d_speaker):
        return np.asarray(d_speaker, float) @ self.axes


def aim(source, receiver) -> float:
    """Yaw that points the loudspeaker's axis at the receiver (in the horizontal plane)."""
    d = np.asarray(receiver, float) - np.asarray(source, float)
    return float(np.arctan2(d[0], -d[1]))


# --------------------------------------------------------------------------------------------
# Specular images
# --------------------------------------------------------------------------------------------

def image_sources(setup: Setup, max_order: int = 3) -> dict:
    """Every image of order <= ``max_order``: order, bounces per surface (n, 6), distance to the
    receiver, emission direction (speaker frame), delay after the direct sound (s), and for the
    first order the surface and the material at the reflection point."""
    L = np.asarray(setup.room.dims, float)
    s = np.asarray(setup.source, float)
    r = np.asarray(setup.receiver, float)
    rng = np.arange(-max_order, max_order + 1)
    rows = []
    for nx in rng:
        for qx in (0, 1):
            cx = (abs(nx - qx), abs(nx))
            if sum(cx) > max_order:
                continue
            for ny in rng:
                for qy in (0, 1):
                    cy = (abs(ny - qy), abs(ny))
                    if sum(cx) + sum(cy) > max_order:
                        continue
                    for nz in rng:
                        for qz in (0, 1):
                            cz = (abs(nz - qz), abs(nz))
                            order = sum(cx) + sum(cy) + sum(cz)
                            if order > max_order:
                                continue
                            rows.append((nx, qx, ny, qy, nz, qz) + cx + cy + cz + (order,))
    rows = np.array(rows, int)
    n = rows[:, [0, 2, 4]]
    q = rows[:, [1, 3, 5]]
    counts = rows[:, 6:12]
    order = rows[:, 12]
    u = 2.0 * n * L + (1 - 2 * q) * s
    v = r - u
    dist = np.linalg.norm(v, axis=1)
    d_room = (1 - 2 * (q % 2)) * v / dist[:, None]
    direct = float(np.linalg.norm(r - s))

    first = np.full(len(rows), -1)
    point = np.full((len(rows), 3), np.nan)
    for i in np.flatnonzero(order == 1):
        k = int(np.flatnonzero(counts[i])[0])
        axis, side = divmod(k, 2)
        wall = side * L[axis]
        t = (wall - u[i, axis]) / (r[axis] - u[i, axis])
        point[i] = u[i] + t * (r - u[i])
        first[i] = SURFACE_INDEX[(axis, side)]
    material = first.copy()
    m1 = order == 1
    material[m1] = setup.room.material_at(first[m1], point[m1])
    sort = np.argsort(dist, kind="stable")
    return {"order": order[sort], "counts": counts[sort], "distance": dist[sort],
            "direction": setup.to_speaker(d_room[sort]), "delay": (dist[sort] - direct) / C,
            "first": first[sort], "material": material[sort], "point": point[sort]}


def specular_weights(setup: Setup, images: dict, freqs) -> np.ndarray:
    """Energy weight of every image at every frequency, shape ``(n_images, n_freqs)``."""
    room = setup.room
    freqs = np.asarray(freqs, float)
    alpha_tab, scat_tab = room.material_tables(freqs)
    mean = [room.mean_coefficients(s, freqs) for s in SURFACES]
    keep = np.array([(1.0 - a) * (1.0 - sc) for a, sc in mean])
    m_air = air_attenuation(freqs)
    m_fit = room.furniture_attenuation(freqs)
    d = images["distance"]
    w = np.exp(-np.outer(d, m_air)) / d[:, None] ** 2
    higher = images["order"] >= 2
    w[higher] *= np.exp(-np.outer(d[higher], m_fit))
    for i in range(len(d)):
        o = images["order"][i]
        if o == 1:
            mat = images["material"][i]
            w[i] *= (1.0 - alpha_tab[mat]) * (1.0 - scat_tab[mat])
        elif o >= 2:
            for k, c in enumerate(images["counts"][i]):
                if c:
                    w[i] *= keep[SURFACE_INDEX[divmod(k, 2)]] ** c
    return w


# --------------------------------------------------------------------------------------------
# Rays and the diffuse field
# --------------------------------------------------------------------------------------------

def trace(setup: Setup, directions_speaker, bounces: int):
    """Rays from the source in the given directions (speaker frame): for each bounce, the material
    hit (n_rays,) and the path length from the source (n_rays,)."""
    L = np.asarray(setup.room.dims, float)
    d = setup.to_room(directions_speaker).copy()
    p = np.broadcast_to(np.asarray(setup.source, float), d.shape).copy()
    length = np.zeros(len(d))
    materials, lengths = [], []
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(bounces):
            target = np.where(d > 0, L, 0.0)
            t = np.where(d != 0, (target - p) / d, np.inf)
            t = np.where(t < 0, 0.0, t)
            axis = np.argmin(t, axis=1)
            tmin = t[np.arange(len(d)), axis]
            p = p + tmin[:, None] * d
            length = length + tmin
            side = (d[np.arange(len(d)), axis] > 0).astype(int)
            surface = np.array([SURFACE_INDEX[(a, s)] for a, s in zip(axis, side)])
            materials.append(setup.room.material_at(surface, p))
            lengths.append(length.copy())
            d[np.arange(len(d)), axis] *= -1.0
    return np.array(materials), np.array(lengths)


def diffuse_share(setup: Setup, directions_speaker, freqs, bounces: int = 3) -> np.ndarray:
    """``g(Omega, f)``: the share of the energy emitted in each direction that joins the diffuse
    field, shape ``(n_directions, n_freqs)``."""
    room = setup.room
    alpha_tab, scat_tab = room.material_tables(freqs)
    m = air_attenuation(freqs) + room.furniture_attenuation(freqs)
    materials, lengths = trace(setup, directions_speaker, bounces)
    carried = np.ones((materials.shape[1], len(freqs)))
    g = np.zeros_like(carried)
    for j in range(bounces):
        a = alpha_tab[materials[j]]
        sc = scat_tab[materials[j]]
        arriving = carried * np.exp(-np.outer(lengths[j] - (lengths[j - 1] if j else 0.0), m))
        if j < bounces - 1:
            g += arriving * (1.0 - a) * sc
            carried = arriving * (1.0 - a) * (1.0 - sc)
        else:
            g += arriving * (1.0 - a)
    return g


def diffuse_weights(setup: Setup, directions_speaker, solid_angle, freqs, bounces: int = 3) -> np.ndarray:
    """Weights of the diffuse field over a quadrature of the sphere: ``4 / A * dOmega * g``."""
    g = diffuse_share(setup, directions_speaker, freqs, bounces)
    return 4.0 / setup.room.absorption_area(freqs)[None, :] * np.asarray(solid_angle)[..., None] * g
