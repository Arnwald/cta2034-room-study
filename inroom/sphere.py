"""Directions on the sphere: conventions, quadrature, angles.

Speaker frame (see :mod:`inroom.speaker`): ``x`` to the right as seen from the listener, ``y`` up,
``z`` forward. Azimuth ``az = atan2(x, z)`` (0 on axis, +90 to the right), elevation
``el = asin(y)`` (positive upward).
"""

from __future__ import annotations

import numpy as np


def fibonacci(n: int) -> np.ndarray:
    """``n`` nearly equally spaced unit vectors (Fibonacci lattice), each standing for ``4 pi / n``
    steradians. The lattice's pole is the ``y`` axis, so that its latitude rings are elevations."""
    i = np.arange(n) + 0.5
    y = 1.0 - 2.0 * i / n
    r = np.sqrt(1.0 - y**2)
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    return np.stack([r * np.sin(phi), y, r * np.cos(phi)], axis=-1)


def from_angles(az_deg, el_deg) -> np.ndarray:
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(el) * np.sin(az), np.sin(el) * np.ones_like(az), np.cos(el) * np.cos(az)], axis=-1)


def to_angles(u) -> tuple[np.ndarray, np.ndarray]:
    """Azimuth and elevation (degrees) of unit vectors."""
    u = np.asarray(u, float)
    az = np.degrees(np.arctan2(u[..., 0], u[..., 2]))
    el = np.degrees(np.arcsin(np.clip(u[..., 1], -1.0, 1.0)))
    return az, el


def off_axis_angle(u) -> np.ndarray:
    """Angle from the reference axis (degrees)."""
    return np.degrees(np.arccos(np.clip(np.asarray(u, float)[..., 2], -1.0, 1.0)))
