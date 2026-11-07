"""Far-field directivity of the radiators.

* Cone: rigid flat circular piston of radius ``a`` in an infinite baffle,
  ``D(theta) = 2 J1(k a sin theta) / (k a sin theta)``.
* Dome: rigid convex spherical cap (base radius ``a``, height ``h``) moving axially. The far
  field is the Rayleigh integral over the projected disc, each ring of radius ``r`` being
  advanced by the cap height ``z(r)`` above the rim plane:

      A(theta, k) = (2 / a^2) * int_0^a J0(k r sin theta) exp(j k z(r) cos theta) r dr,
      z(r) = sqrt(R^2 - r^2) - (R - h),   R = (a^2 + h^2) / (2 h),

  and ``D(theta) = A(theta, k) / A(0, k)``: unity on axis, complex off axis because the
  protruding center moves the apparent source. ``h = 0`` gives the flat piston.
* Any other axisymmetric radiator (the recessed cone of the boundary-element model) is given by
  its profile ``z(r)``, piecewise linear, negative below the rim plane, in the same integral.

* Waveguide: a phenomenological constant-directivity source, the piston's directivity with an
  effective radius that stops growing in wavelengths above the frequency where ``k a = ka_max``:
  ``D(theta) = 2 J1(x) / x``, ``x = min(k a, ka_max) sin theta``. Below that frequency it radiates as
  a piston of radius ``a``; above, its beamwidth stays constant.

Both are normalized to 1 on their own axis; the on-axis response is the driver model's job.
"""

from __future__ import annotations

import numpy as np
from scipy.special import j0, j1


def piston(x):
    """``2 J1(x) / x`` with its limit 1 at ``x = 0``."""
    x = np.asarray(x, float)
    out = np.ones_like(x)
    m = np.abs(x) > 1e-8
    out[m] = 2.0 * j1(x[m]) / x[m]
    return out


class Radiator:
    """A piston (``height = 0``), a dome, or a ``profile`` ``[(r, z), ...]`` (m, ``z`` from the rim
    plane, negative when recessed), with Gauss-Legendre nodes over its radius."""

    def __init__(self, radius: float, height: float = 0.0, nodes: int = 96, profile=None):
        self.radius = float(radius)
        self.height = float(height)
        self.profile = None if profile is None else tuple(sorted((float(r), float(z)) for r, z in profile))
        x, w = np.polynomial.legendre.leggauss(nodes)
        self.r = 0.5 * radius * (x + 1.0)
        self.w = 0.5 * radius * w
        self.z = self.surface(self.r)

    def surface(self, r):
        """Axial position ``z(r)`` of the radiating surface above the rim plane (m)."""
        r = np.asarray(r, float)
        if self.profile is not None:
            rs, zs = zip(*self.profile)
            return np.interp(r, rs, zs)
        if self.height > 0.0:
            sphere = (self.radius**2 + self.height**2) / (2.0 * self.height)
            return np.sqrt(sphere**2 - r**2) - (sphere - self.height)
        return np.zeros_like(r)

    @property
    def is_piston(self) -> bool:
        return not np.any(self.z)

    def aperture(self, k, theta):
        """Unnormalized far-field factor ``A(theta, k)``; ``k`` and ``theta`` broadcast."""
        k = np.asarray(k, float)[..., None]
        theta = np.asarray(theta, float)[..., None]
        if self.is_piston:
            return piston(k[..., 0] * self.radius * np.sin(theta[..., 0])).astype(complex)
        terms = j0(k * self.r * np.sin(theta)) * np.exp(1j * k * self.z * np.cos(theta))
        return (2.0 / self.radius**2) * np.sum(terms * self.r * self.w, axis=-1)

    def on_axis(self, k):
        """``A(0, k)``: 1 for the piston, the phase-spread loss of the dome otherwise."""
        return self.aperture(k, np.zeros_like(np.asarray(k, float)))

    def directivity(self, k, theta):
        """``D(theta) = A(theta, k) / A(0, k)``."""
        if self.is_piston:
            return self.aperture(k, theta)
        return self.aperture(k, theta) / self.on_axis(k)


class Waveguide(Radiator):
    """Constant-directivity source: a piston of radius ``radius`` up to ``k a = ka_max``, then a fixed
    beamwidth."""

    def __init__(self, radius: float, ka_max: float, nodes: int = 96):
        super().__init__(radius, 0.0, nodes)
        self.ka_max = float(ka_max)

    def aperture(self, k, theta):
        k = np.asarray(k, float)
        theta = np.asarray(theta, float)
        return piston(np.minimum(k * self.radius, self.ka_max) * np.sin(theta)).astype(complex)
