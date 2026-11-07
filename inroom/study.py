"""The computation: room kernels, loudspeaker responses, curves.

For every setup the room is reduced to weights over directions (:mod:`inroom.room`): the images
(direct sound and specular reflections up to order 3) and the diffuse field on a quadrature of the
sphere. For every loudspeaker variant (layout, orientation, crossover) the far-field power
``|H|^2`` is computed in all these directions, at the 70 directions of CTA-2034 and on the
quadrature; the energy at the listener is the weighted sum.

Every in-room response is given *relative to an omnidirectional source* in the same place (the
energy of the loudspeaker divided by that of a source radiating 1 in every direction): the room's
own colouration (its absorption rising with frequency) is divided out, and what is left is how
the room weighs the loudspeaker's directions, the quantity the Predicted In-Room Response of
CTA-2034 estimates. A loudspeaker radiating the same everywhere has the same in-room response
in every room, equal to its CTA curves.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import crossover, cta2034, room as rm, sphere
from .speaker import FarField, Layout, roll

#: Categories of energy at the listener.
CATEGORIES = ("direct",) + tuple(f"first_{s}" for s in rm.SURFACES) + ("second", "third", "diffuse")


@dataclass
class RoomKernel:
    """One setup reduced to weights: images (directions in the speaker frame, weights per
    frequency, category) and the diffuse field over the quadrature."""

    setup: rm.Setup
    directions: np.ndarray
    weights: np.ndarray
    category: np.ndarray
    delay: np.ndarray
    diffuse: np.ndarray


def room_kernel(setup: rm.Setup, freqs, grid, solid_angle, max_order: int = 3) -> RoomKernel:
    images = rm.image_sources(setup, max_order)
    w = rm.specular_weights(setup, images, freqs)
    cat = np.empty(len(w), int)
    for i, (o, first) in enumerate(zip(images["order"], images["first"])):
        cat[i] = 0 if o == 0 else (1 + first if o == 1 else (7 if o == 2 else 8))
    diffuse = rm.diffuse_weights(setup, grid, solid_angle, freqs, bounces=max_order)
    return RoomKernel(setup, images["direction"], w.astype(np.float32), cat, images["delay"],
                      diffuse.astype(np.float32))


def category_sums(kernel: RoomKernel, power_images: np.ndarray, power_grid: np.ndarray) -> np.ndarray:
    """Energy at the listener per category, shape ``(len(CATEGORIES), n_freqs)``."""
    out = np.zeros((len(CATEGORIES), power_images.shape[-1]))
    e = kernel.weights * power_images
    for c in range(len(CATEGORIES) - 1):
        m = kernel.category == c
        if m.any():
            out[c] = e[m].sum(axis=0)
    out[-1] = np.einsum("kf,kf->f", kernel.diffuse, power_grid)
    return out


class Variant:
    """A loudspeaker: a layout, rolled by ``roll_deg`` about its axis, with a crossover ``kind``."""

    def __init__(self, layout: Layout, kind: str, roll_deg: float = 0.0, label: str | None = None):
        self.layout, self.kind, self.roll_deg = layout, kind, roll_deg
        self.name = f"{layout.name}-{'side' if roll_deg else 'upright'}-{kind}"
        self.label = label or f"{crossover.LABELS[kind]}, {'on its side' if roll_deg else 'upright'}"


def driver_responses(far: FarField, directions, roll_deg: float) -> dict[str, np.ndarray]:
    u = roll(directions, roll_deg) if roll_deg else np.asarray(directions, float)
    return far.responses(u)


def power(filters: dict, responses: dict) -> np.ndarray:
    """``|sum_i X_i H_i|^2``, float32."""
    h = sum(filters[n][None, :].astype(np.complex64) * responses[n] for n in filters)
    return (h.real**2 + h.imag**2).astype(np.float32)
