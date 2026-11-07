"""The simulated loudspeakers and their far field over the whole sphere.

Speaker frame (metres), as in the turntable study: origin at the center of the front baffle, ``x``
horizontal and positive to the right as seen from the listening position, ``y`` up, ``z`` normal to
the baffle and positive toward the listener. The cabinet occupies ``-depth <= z <= 0``; the
drivers sit on the baffle plane ``z = 0``.

Each driver is a rigid radiator (a flat piston or a dome, :mod:`inroom.directivity`) with a flat
on-axis half-space response, and the cabinet diffracts it through the Distributed Edge Dipole
(DED) model of the turntable study (Urban et al., JAES 52(10), 2004): a line of dipoles along the
edges of the front baffle, driven by the pressure the driver produces there. Here it is evaluated
in the far field, in any direction ``u`` (unit vector), relative to the origin and normalized to
1 m:

    H(u, f) = D(theta, k) (1 + cos theta) / 2 exp(+j k u.s) - u_z sum_e Q_e(f) exp(+j k u.E_e)

``s`` the driver center, ``E_e`` the edge elements, ``cos theta = u_z``; the edge dipoles change
sign behind the baffle plane, the direct term vanishes behind the cabinet. At low frequency the
loudspeaker radiates 6 dB below its half-space response (the baffle step); at high frequency it
reaches it, with the edge ripple. The edge strengths ``Q_e`` are those of the turntable study.

Every driver is equalized later to an acoustic target on the reference axis
(:mod:`inroom.crossover`), so only its directivity, relative to its own on-axis response, matters.
"""

from __future__ import annotations

import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import C
from .directivity import Radiator, Waveguide

CACHE = Path(__file__).resolve().parent.parent / "results" / "cache"


@dataclass(frozen=True)
class Cabinet:
    width: float
    height: float
    depth: float


@dataclass(frozen=True)
class DriverSpec:
    """A driver on the baffle: a piston (``height = 0``) or a dome of base radius ``radius``. With
    ``waveguide = (radius, ka_max)`` the dome sits in a waveguide: it radiates as the constant-
    directivity source of :class:`inroom.directivity.Waveguide`, and the waveguide shields the
    baffle edges: their strengths are the dome's times the waveguide's directivity at 90 degrees."""

    name: str
    label: str
    x: float
    y: float
    radius: float
    height: float = 0.0
    waveguide: tuple[float, float] | None = None

    def radiator(self) -> Radiator:
        """The radiator whose surface drives the edges (the dome itself)."""
        return Radiator(self.radius, self.height)

    def far_radiator(self) -> Radiator:
        """The radiator whose directivity the far field follows."""
        if self.waveguide is not None:
            return Waveguide(*self.waveguide)
        return self.radiator()


@dataclass(frozen=True)
class Layout:
    name: str
    label: str
    cabinet: Cabinet
    drivers: tuple[DriverSpec, ...]
    #: Crossover frequency between consecutive drivers (Hz), lowest first.
    crossovers: tuple[float, ...] = field(default=())

    @property
    def names(self):
        return [d.name for d in self.drivers]


#: The main loudspeaker: an upright two-way bookshelf, 200 x 360 x 280 mm (20 L outside): a
#: 6.5-inch cone (flat piston of 129 cm^2) below a 1.1-inch dome (the turntable study's 28 mm dome,
#: 7.5 mm high), 140 mm apart, crossed over at 2.2 kHz.
TWO_WAY = Layout(
    name="two_way",
    label="two-way, 6.5 in cone and 1 in dome, 140 mm apart",
    cabinet=Cabinet(0.200, 0.360, 0.280),
    drivers=(
        DriverSpec("woofer", "6.5 in cone", 0.0, -0.030, 0.064),
        DriverSpec("tweeter", "1.1 in dome", 0.0, 0.110, 0.014, 0.0075),
    ),
    crossovers=(2200.0,),
)

#: The same two-way with its dome in a waveguide whose directivity matches the cone's at the
#: crossover (``k a = 2.58`` at 2.2 kHz for both) and stays constant above: a beamwidth of about
#: 115 degrees between the -6 dB points.
TWO_WAY_WG = Layout(
    name="two_way_wg",
    label="two-way, 6.5 in cone and 1 in dome in a waveguide matched at the crossover",
    cabinet=TWO_WAY.cabinet,
    drivers=(
        TWO_WAY.drivers[0],
        DriverSpec("tweeter", "1.1 in dome in a waveguide", 0.0, 0.110, 0.014, 0.0075,
                   waveguide=(0.064, 2.0 * np.pi * 2200.0 / C * 0.064)),
    ),
    crossovers=(2200.0,),
)

LAYOUTS = {TWO_WAY.name: TWO_WAY, TWO_WAY_WG.name: TWO_WAY_WG}


# --------------------------------------------------------------------------------------------
# Distributed edge dipoles (strengths as in the turntable study)
# --------------------------------------------------------------------------------------------

def front_edges(cabinet: Cabinet, step: float = 0.002):
    """Midpoints and lengths of the elements along the front baffle perimeter (z = 0)."""
    w, h = cabinet.width / 2, cabinet.height / 2
    corners = [(-w, -h), (w, -h), (w, h), (-w, h)]
    points, lengths = [], []
    for (x0, y0), (x1, y1) in zip(corners, corners[1:] + corners[:1]):
        length = np.hypot(x1 - x0, y1 - y0)
        n = int(np.ceil(length / step))
        t = (np.arange(n) + 0.5) / n
        points.append(np.stack([x0 + t * (x1 - x0), y0 + t * (y1 - y0), np.zeros(n)], axis=-1))
        lengths.append(np.full(n, length / n))
    return np.concatenate(points), np.concatenate(lengths)


def edge_strengths(driver: DriverSpec, cabinet: Cabinet, freqs, step: float = 0.002,
                   coarse_df: float = 50.0) -> np.ndarray:
    """Edge strengths ``Q_e(f)`` of one driver, shape ``(n_edges, n_freqs)``.

    ``Q_e = C dl_e G(rho_e, f) / A0(f)``: ``G`` the driving pressure at the edge point (a quadrature
    of the radiating surface, each point weighted by its obliquity ``(1 + cos)/2``), ``A0`` the
    radiator's on-axis factor, ``C`` the normalization that makes the direct and edge terms sum to
    1/2 on axis at low frequency (full-space radiation below the baffle step)."""
    freqs = np.asarray(freqs, float)
    points, lengths = front_edges(cabinet, step)
    rho_e = np.hypot(points[:, 0] - driver.x, points[:, 1] - driver.y)
    radiator = driver.radiator()
    a = radiator.radius

    spread = 2.0 * np.pi * freqs[-1] / C * 2.0 * a
    n_r = int(np.ceil(0.7 * spread + 16))
    n_phi = int(np.ceil(0.5 * spread + 16))
    x, wx = np.polynomial.legendre.leggauss(n_r)
    r = 0.5 * a * (x + 1.0)
    wr = 0.5 * a * wx
    z = radiator.surface(r)
    phi = (np.arange(n_phi) + 0.5) * np.pi / n_phi
    rr, pp = np.meshgrid(r, phi, indexing="ij")
    zz = np.broadcast_to(z[:, None], rr.shape)
    w = (wr[:, None] * rr) * (np.pi / n_phi) * 2.0 / (np.pi * a**2)
    rr, pp, zz, w = rr.ravel(), pp.ravel(), zz.ravel(), w.ravel()

    rho_grid = np.arange(rho_e.min() - 0.002, rho_e.max() + 0.003, 0.001)
    f_coarse = np.arange(0.0, freqs[-1] + coarse_df, coarse_df)
    k_coarse = 2.0 * np.pi * f_coarse / C
    g_tab = np.empty((len(rho_grid), len(f_coarse)), complex)
    for j, rho in enumerate(rho_grid):
        r1 = np.sqrt((rho - rr * np.cos(pp)) ** 2 + (rr * np.sin(pp)) ** 2 + zz**2)
        amp = w * 0.5 * (1.0 - zz / r1) / r1
        g_tab[j] = np.exp(-1j * np.outer(k_coarse, r1 - rho)) @ amp

    idx = np.clip(np.searchsorted(rho_grid, rho_e) - 1, 0, len(rho_grid) - 2)
    t = (rho_e - rho_grid[idx]) / (rho_grid[idx + 1] - rho_grid[idx])
    g_edges = (1.0 - t)[:, None] * g_tab[idx] + t[:, None] * g_tab[idx + 1]
    normalization = 0.5 / np.sum(lengths * g_edges[:, 0].real)
    re = CubicSpline(f_coarse, g_edges.real, axis=1)(freqs)
    im = CubicSpline(f_coarse, g_edges.imag, axis=1)(freqs)
    k = 2.0 * np.pi * freqs / C
    q = (normalization * lengths)[:, None] * (re + 1j * im) * np.exp(-1j * np.outer(rho_e, k))
    q = q / radiator.on_axis(k)[None, :]
    if driver.waveguide is not None:
        q = q * driver.far_radiator().directivity(k, np.full_like(k, np.pi / 2))[None, :].real
    return q.astype(np.complex64)


# --------------------------------------------------------------------------------------------
# Far field
# --------------------------------------------------------------------------------------------

def _workers() -> int:
    return max(1, min(12, (os.cpu_count() or 2) - 2))


class FarField:
    """Far-field responses of the drivers of a layout, in any direction of the speaker frame."""

    def __init__(self, layout: Layout, freqs, edge_step: float = 0.002, diffraction: bool = True):
        self.layout = layout
        self.freqs = np.asarray(freqs, float)
        self.k = 2.0 * np.pi * self.freqs / C
        self.diffraction = diffraction
        self.edge_points, self.edge_lengths = front_edges(layout.cabinet, edge_step)
        self.edge_step = edge_step
        self.strengths = {d.name: self._strengths(d) for d in layout.drivers} if diffraction else {}

    def _strengths(self, driver: DriverSpec) -> np.ndarray:
        key = f"v2-{driver}-{self.layout.cabinet}-{self.edge_step}-{hashlib.sha1(self.freqs.tobytes()).hexdigest()}"
        path = CACHE / f"edges-{driver.name}-{hashlib.sha1(key.encode()).hexdigest()[:12]}.npy"
        if path.exists():
            return np.load(path)
        q = edge_strengths(driver, self.layout.cabinet, self.freqs, self.edge_step)
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, q)
        return q

    def _direct(self, driver: DriverSpec, u: np.ndarray, sl: slice) -> np.ndarray:
        k = self.k[sl]
        cos_t = np.clip(u[:, 2], -1.0, 1.0)
        theta = np.arccos(cos_t)
        radiator = driver.far_radiator()
        s = np.array([driver.x, driver.y, 0.0])
        d = radiator.directivity(k[None, :], theta[:, None])
        phase = np.exp(1j * np.outer(u @ s, k))
        return (d * (0.5 * (1.0 + cos_t))[:, None] * phase).astype(np.complex64)

    def _edges(self, driver: DriverSpec, u: np.ndarray, sl: slice) -> np.ndarray:
        q = self.strengths[driver.name][:, sl]
        proj = (u @ self.edge_points.T).astype(np.float32)
        out = np.empty((len(u), q.shape[1]), np.complex64)
        for j, kj in enumerate(self.k[sl]):
            out[:, j] = np.exp(1j * np.float32(kj) * proj) @ q[:, j]
        return out * (-u[:, 2]).astype(np.float32)[:, None]

    def responses(self, directions, block: int = 24) -> dict[str, np.ndarray]:
        """``{driver: (n_directions, n_freqs) complex64}`` for unit vectors ``directions``."""
        u = np.asarray(directions, float)
        u = u / np.linalg.norm(u, axis=-1, keepdims=True)
        nf = len(self.freqs)
        slices = [slice(i, min(i + block, nf)) for i in range(0, nf, block)]
        out = {}
        for driver in self.layout.drivers:
            def work(sl, driver=driver):
                h = self._direct(driver, u, sl)
                if self.diffraction:
                    h = h + self._edges(driver, u, sl)
                return sl, h

            res = np.empty((len(u), nf), np.complex64)
            with ThreadPoolExecutor(_workers()) as pool:
                for sl, h in pool.map(work, slices):
                    res[:, sl] = h
            out[driver.name] = res
        return out


def roll(directions, angle_deg: float) -> np.ndarray:
    """Directions of a loudspeaker rolled by ``angle_deg`` about its axis (positive: its top toward
    +x), expressed in the frame of the unrolled one: ``H_rolled(u) = H(R u)``."""
    a = np.radians(angle_deg)
    u = np.asarray(directions, float)
    x, y, z = u[..., 0], u[..., 1], u[..., 2]
    # Rolling maps a point (x, y) of the cabinet to R (x, y), R = [[cos a, sin a], [-sin a, cos a]]
    # (the top (0, 1) lands on (sin a, cos a)); a direction u is seen by the rolled loudspeaker as
    # R^T u in its own frame.
    return np.stack([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z], axis=-1)
