import numpy as np

from inroom import crossover, sphere
from inroom.signal import log_grid
from inroom.speaker import TWO_WAY, TWO_WAY_WG, FarField, roll

F = log_grid(150.0, 20000.0, 24)


def test_baffle_step_and_symmetry():
    far = FarField(TWO_WAY, F)
    u = sphere.from_angles(np.array([0.0, 40.0, -40.0]), np.array([0.0, 10.0, 10.0]))
    h = far.responses(u)
    for name, r in h.items():
        # Full space below the baffle step, half space well above it.
        assert abs(abs(r[0, 0]) - 0.5) < 0.15
        assert abs(20 * np.log10(np.mean(np.abs(r[0, F > 10000.0])))) < 1.0
        # Drivers on the vertical middle line of a symmetric baffle: left and right are equal.
        np.testing.assert_allclose(np.abs(r[1]), np.abs(r[2]), rtol=1e-4, atol=1e-6)


def test_roll_maps_directions():
    u = sphere.fibonacci(50)
    r = roll(u, 90.0)
    # Looking straight up at the rolled loudspeaker is looking at the unrolled one's left side.
    np.testing.assert_allclose(roll(np.array([[0.0, 1.0, 0.0]]), 90.0), [[-1.0, 0.0, 0.0]], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(r, axis=1), 1.0)


def test_equalized_sum_is_flat_on_axis():
    for layout in (TWO_WAY, TWO_WAY_WG):
        far = FarField(layout, F)
        on = {n: v[0] for n, v in far.responses(np.array([[0.0, 0.0, 1.0]])).items()}
        for kind in crossover.KINDS:
            x = crossover.filters(kind, F, layout, on)
            s = crossover.system(x, on)
            m = (F > 300.0) & (F < 16000.0)
            assert np.max(np.abs(20 * np.log10(np.abs(s[m])))) < 0.5


def test_waveguide_holds_its_beamwidth():
    far = FarField(TWO_WAY_WG, F)
    u = sphere.from_angles(np.array([0.0, 60.0]), np.array([0.0, 0.0]))
    h = far.responses(u)["tweeter"]
    rel = 20 * np.log10(np.abs(h[1]) / np.abs(h[0]))
    hi = (F > 4000.0) & (F < 16000.0)
    assert np.ptp(rel[hi]) < 3.0
