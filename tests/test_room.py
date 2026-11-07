import numpy as np

from inroom import ensemble, sphere
from inroom import room as rm


def uniform(alpha, s, dims=(4.3, 6.0, 2.6)):
    a = {k: (alpha,) * 7 for k in rm.SURFACES}
    sc = {k: (s,) * 7 for k in rm.SURFACES}
    return rm.Room(dims, a, sc, [], (0.0,) * 7)


SETUP = rm.Setup(uniform(0.2, 0.0), np.array([1.0, 5.2, 1.0]), np.array([2.15, 2.2, 1.15]), yaw=0.3)


def test_image_count_and_first_order_geometry():
    im = rm.image_sources(SETUP, 3)
    assert np.bincount(im["order"]).tolist() == [1, 6, 18, 38]
    s, r = SETUP.source, SETUP.receiver
    first = np.flatnonzero(im["order"] == 1)
    for i in first:
        surf = rm.SURFACES[im["first"][i]]
        axis, side = rm.PLANES[surf]
        p = im["point"][i]
        # The reflection point lies on the surface, and both legs add up to the image distance.
        assert abs(p[axis] - side * SETUP.room.dims[axis]) < 1e-9
        assert abs(np.linalg.norm(p - s) + np.linalg.norm(r - p) - im["distance"][i]) < 1e-9
        # The emission direction points from the source to the reflection point.
        d_room = SETUP.to_room(im["direction"][i])
        np.testing.assert_allclose(d_room, (p - s) / np.linalg.norm(p - s), atol=1e-9)
    assert abs(im["distance"][0] - np.linalg.norm(r - s)) < 1e-12


def test_speaker_frame_is_right_handed_and_aims():
    st = ensemble.named()["living"]
    a = st.axes
    np.testing.assert_allclose(a @ a.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(a[0], a[1]), a[2], atol=1e-12)
    d = st.receiver - st.source
    d[2] = 0.0
    np.testing.assert_allclose(a[2], d / np.linalg.norm(d), atol=1e-12)


def test_air_attenuation_iso_9613():
    # About 4.7 dB/km at 1 kHz and 105 dB/km at 8 kHz (20 degC, 50 %).
    m = rm.air_attenuation(np.array([1000.0, 8000.0])) * 10 * np.log10(np.e) * 1000
    assert 4.0 < m[0] < 5.5
    assert 90 < m[1] < 120


def test_hybrid_matches_hopkins_stryker_when_every_reflection_scatters():
    room = uniform(0.2, 1.0)
    st = rm.Setup(room, SETUP.source, SETUP.receiver, SETUP.yaw)
    f = np.array([1000.0])
    grid = sphere.fibonacci(3000)
    im = rm.image_sources(st, 3)
    total = rm.specular_weights(st, im, f).sum() + rm.diffuse_weights(st, grid, 4 * np.pi / 3000, f).sum()
    a = room.absorption_area(f)[0]
    r = np.linalg.norm(st.receiver - st.source)
    expected = 1 / r**2 + 16 * np.pi * 0.8 / a * np.exp(-rm.air_attenuation(f)[0] * 2.5)
    assert abs(total / expected - 1.0) < 0.03


def test_hybrid_against_a_full_image_method():
    """Purely specular room: the hybrid's energy within 10 % of 25 000 images."""
    f = np.array([1000.0])
    grid = sphere.fibonacci(3000)
    im3 = rm.image_sources(SETUP, 3)
    hybrid = rm.specular_weights(SETUP, im3, f).sum() + rm.diffuse_weights(SETUP, grid, 4 * np.pi / 3000, f).sum()
    full = rm.image_sources(SETUP, 26)
    w = 0.8 ** full["order"] * np.exp(-full["distance"] * rm.air_attenuation(f)[0]) / full["distance"] ** 2
    assert abs(hybrid / w.sum() - 1.0) < 0.1


def test_rays_and_patches():
    st = ensemble.named()["treated"]
    # A ray toward the left wall's first reflection point lands on the panel.
    im = rm.image_sources(st, 1)
    i = [j for j in range(len(im["order"])) if im["order"][j] == 1 and rm.SURFACES[im["first"][j]] == "left"][0]
    assert im["material"][i] >= 6
    mats, lengths = rm.trace(st, im["direction"][i:i + 1], 1)
    assert mats[0, 0] == im["material"][i]
    np.testing.assert_allclose(lengths[0, 0], np.linalg.norm(im["point"][i] - st.source), rtol=1e-9)


def test_ensemble_is_reproducible_and_sane():
    a = ensemble.sample(20, seed=1)
    b = ensemble.sample(20, seed=1)
    assert [s.label for s in a] == [s.label for s in b]
    for s in a:
        lx, ly, lz = s.room.dims
        assert 0 < s.source[0] < lx and 0 < s.receiver[1] < s.source[1] < ly
        assert 0.2 < s.room.rt60(np.array([1000.0]))[0] < 1.2
