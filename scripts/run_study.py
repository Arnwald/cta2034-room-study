"""Every simulated quantity of the study, saved to results/study.npz.

For the two-way loudspeaker (a dome on the baffle, or in a waveguide), upright and on its side, with
each crossover (LR2, LR4, LR8, brickwall FIR): its far-field power at the 70 CTA-2034 directions, on a 3000-point quadrature of the sphere,
and in the directions of every path of every room; the energy each room brings to the listener,
per category of path; the same for an omnidirectional source.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inroom import crossover, cta2034, ensemble, sphere  # noqa: E402
from inroom.analysis import anomaly, local_anomaly, notch_depth  # noqa: E402
from inroom import room as rm  # noqa: E402
from inroom.signal import log_grid  # noqa: E402
from inroom.speaker import TWO_WAY, TWO_WAY_WG, FarField  # noqa: E402
from inroom.study import CATEGORIES, Variant, category_sums, driver_responses, power, room_kernel  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results" / "study.npz"
N_ROOMS = 300
N_GRID = 3000
ROLLS = (0.0, 90.0)
LAYOUTS = (TWO_WAY, TWO_WAY_WG)


def main():
    t0 = time.time()
    freqs = log_grid()
    grid = sphere.fibonacci(N_GRID)
    solid_angle = np.full(N_GRID, 4.0 * np.pi / N_GRID)
    named = ensemble.named()
    setups = ensemble.sample(N_ROOMS) + list(named.values())
    print(f"{len(setups)} setups, {len(freqs)} frequencies, {N_GRID} directions", flush=True)

    kernels = [room_kernel(s, freqs, grid, solid_angle) for s in setups]
    print(f"room kernels: {time.time() - t0:.0f} s", flush=True)
    counts = [len(k.directions) for k in kernels]
    starts = np.concatenate([[0], np.cumsum(counts)])
    image_dirs = np.concatenate([k.directions for k in kernels])

    omni_img = np.ones((len(image_dirs), len(freqs)), np.float32)
    omni_grid = np.ones((N_GRID, len(freqs)), np.float32)
    omni = np.array([category_sums(k, omni_img[starts[i]:starts[i + 1]], omni_grid) for i, k in enumerate(kernels)])

    # Weights and powers of the paths averaged over the anomaly band, to weigh each path's colouration.
    band = (freqs >= 600.0) & (freqs <= 8000.0)
    results = {}
    for layout in LAYOUTS:
        far = FarField(layout, freqs)
        on_axis = {n: h[0] for n, h in far.responses(np.array([[0.0, 0.0, 1.0]])).items()}
        for roll_deg in ROLLS:
            resp_cta = driver_responses(far, cta2034.DIRECTIONS, roll_deg)
            resp_grid = driver_responses(far, grid, roll_deg)
            resp_img = driver_responses(far, image_dirs, roll_deg)
            print(f"{layout.name}, far field, roll {roll_deg:g}: {time.time() - t0:.0f} s", flush=True)
            for kind in crossover.KINDS:
                v = Variant(layout, kind, roll_deg)
                x = crossover.filters(kind, freqs, layout, on_axis)
                p_cta, p_grid, p_img = power(x, resp_cta), power(x, resp_grid), power(x, resp_img)
                sums = np.array([category_sums(k, p_img[starts[i]:starts[i + 1]], p_grid)
                                 for i, k in enumerate(kernels)])
                # Tonal anomaly of every direction's own response (RMS residual of a straight line,
                # 600 Hz - 8 kHz), for the off-axis colouration of each path.
                floor = 1e-12
                results[v.name] = dict(cta=p_cta, sp_full=p_grid.mean(axis=0), rooms=sums.astype(np.float32),
                                       grid=p_grid.astype(np.float16),
                                       anom_cta=anomaly(freqs, np.maximum(p_cta, floor)).astype(np.float32),
                                       anom_grid=anomaly(freqs, np.maximum(p_grid, floor)).astype(np.float32),
                                       anom_img=anomaly(freqs, np.maximum(p_img, floor)).astype(np.float32),
                                       local_cta=local_anomaly(freqs, np.maximum(p_cta, floor)).astype(np.float32),
                                       local_grid=local_anomaly(freqs, np.maximum(p_grid, floor)).astype(np.float32),
                                       local_img=local_anomaly(freqs, np.maximum(p_img, floor)).astype(np.float32),
                                       notch_cta=notch_depth(freqs, np.maximum(p_cta, floor)).astype(np.float32),
                                       notch_grid=notch_depth(freqs, np.maximum(p_grid, floor)).astype(np.float32),
                                       notch_img=notch_depth(freqs, np.maximum(p_img, floor)).astype(np.float32),
                                       pband_cta=p_cta[:, band].mean(axis=1), pband_grid=p_grid[:, band].mean(axis=1),
                                       pband_img=p_img[:, band].mean(axis=1))
            del resp_cta, resp_grid, resp_img

    # The rooms' kernels at a few frequencies, for the maps: image directions and weights.
    map_freqs = np.array([500.0, 1000.0, 2200.0, 4000.0, 8000.0])
    fi = np.array([int(np.argmin(np.abs(freqs - f))) for f in map_freqs])
    meta = {k: [] for k in ("dims", "rt60", "listen_distance", "spacing", "wall_distance", "speaker_height",
                            "ear_height", "toe", "floor", "rug", "panels", "furniture", "wall", "ceiling")}
    for s in setups:
        for k in meta:
            meta[k].append(s.room.rt60(np.array([125, 250, 500, 1000, 2000, 4000, 8000.0])) if k == "rt60" else s.meta[k])

    save = dict(freqs=freqs, grid=grid, categories=np.array(CATEGORIES), names=np.array([s.label for s in setups]),
                named=np.array(list(named)), n_random=N_ROOMS, cta_keys=np.array([f"{p}{a}" for p, a in cta2034.KEYS]),
                omni=omni.astype(np.float32), image_dirs=image_dirs.astype(np.float32), image_starts=starts,
                image_cat=np.concatenate([k.category for k in kernels]),
                image_delay=np.concatenate([k.delay for k in kernels]).astype(np.float32),
                image_weights=np.concatenate([k.weights[:, fi] for k in kernels]), map_freqs=map_freqs,
                image_weights_band=np.concatenate([k.weights[:, band].mean(axis=1) for k in kernels]).astype(np.float32),
                diffuse_band=np.array([k.diffuse[:, band].mean(axis=1) for k in kernels]).astype(np.float32),
                diffuse_mean=np.mean([k.diffuse / omni[i].sum(axis=0)[None, :] for i, k in enumerate(kernels[:N_ROOMS])],
                                     axis=0).astype(np.float32),
                variants=np.array(list(results)))
    for k, v in meta.items():
        save[f"meta_{k}"] = np.array(v, dtype=float if k in ("dims", "rt60") else None)
    for name, r in results.items():
        for k, v in r.items():
            save[f"{name}:{k}"] = v
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, **save)
    print(f"saved {OUT} ({OUT.stat().st_size / 1e6:.0f} MB), {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
