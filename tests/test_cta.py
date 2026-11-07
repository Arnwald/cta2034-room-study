import numpy as np

from inroom import cta2034, sphere
from inroom.constants import C


def test_kernels_sum_to_one():
    for c in ("lw", "er", "sp", "pir", "floor", "ceiling", "front", "side", "rear"):
        assert abs(cta2034.kernel(c).sum() - 1.0) < 1e-12


def test_a_source_radiating_the_same_everywhere_has_equal_curves():
    p = np.ones((70, 5)) * 3.0
    c = cta2034.curves(p)
    for k in ("lw", "er", "sp", "pir"):
        np.testing.assert_allclose(c[k], 3.0)
    np.testing.assert_allclose(c["spdi"], 0.0, atol=1e-12)


def test_listening_window_and_bounces():
    w = cta2034.kernel("lw")
    keys = [cta2034.KEYS[i] for i in np.flatnonzero(w)]
    assert sorted(keys) == sorted([("h", a) for a in (0, 10, -10, 20, -20, 30, -30)] + [("v", 10), ("v", -10)])
    floor = cta2034.kernel("floor")
    assert sorted(cta2034.KEYS[i] for i in np.flatnonzero(floor)) == [("v", -40), ("v", -30), ("v", -20)]
    # 0 and 180 degrees are shared by both orbits: 70 directions.
    assert len(cta2034.KEYS) == 70
    np.testing.assert_allclose(np.linalg.norm(cta2034.DIRECTIONS, axis=1), 1.0)


def test_sound_power_of_an_axisymmetric_source_from_two_orbits():
    """For a source symmetric about its axis, the two orbits are exact when it radiates the same
    everywhere and overestimate a directional one a little: the response on an orbit stands for a
    whole 'spherical square', whose corners lie farther off axis."""
    from inroom.directivity import piston as p

    grid = sphere.fibonacci(20000)
    biases = []
    for ka in (0.3, 3.0):
        def power(u):
            s = np.sqrt(np.maximum(1.0 - u[:, 2] ** 2, 0.0))
            return (p(ka * s) * 0.5 * (1.0 + u[:, 2])) ** 2
        exact = power(grid).mean()
        two = np.sum(cta2034.kernel("sp") * power(cta2034.DIRECTIONS))
        biases.append(10 * np.log10(two / exact))
    assert abs(biases[0]) < 0.1
    assert 0.2 < biases[1] < 0.6


def test_two_point_sources_two_orbits_miss_the_interference():
    """Two in-phase point sources 140 mm apart vertically: the sphere gives the textbook power,
    the two orbits overestimate it near the first null."""
    d, f = 0.14, 2200.0
    k = 2 * np.pi * f / C
    def p(u):
        return np.abs(1.0 + np.exp(1j * k * d * u[:, 1])) ** 2 / 4.0
    grid = sphere.fibonacci(40000)
    analytic = 0.5 * (1.0 + np.sinc(k * d / np.pi))
    assert abs(p(grid).mean() / analytic - 1.0) < 1e-3
    two = np.sum(cta2034.kernel("sp") * p(cta2034.DIRECTIONS))
    assert 10 * np.log10(two / analytic) > 1.0
