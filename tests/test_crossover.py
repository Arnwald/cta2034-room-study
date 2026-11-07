import numpy as np

from inroom import crossover
from inroom.signal import log_grid

F = log_grid(100.0, 20000.0, 48)


def test_linkwitz_riley_outputs_sum_to_an_all_pass():
    for kind in ("lr2", "lr4", "lr8"):
        lp, hp = crossover.pair(kind, F, 2200.0)
        np.testing.assert_allclose(np.abs(lp + hp), 1.0, atol=1e-9)
        j = np.argmin(np.abs(F - 2200.0))
        # In phase at the crossover frequency, each 6 dB down.
        assert abs(np.angle(lp[j] / hp[j])) < 0.05
        assert abs(20 * np.log10(abs(lp[j])) + 6.0) < 0.2


def test_fir_outputs_sum_to_a_delay_and_are_steep():
    lp, hp = crossover.pair("fir", F, 2200.0)
    np.testing.assert_allclose(lp + hp, 1.0, atol=1e-12)
    j = np.argmin(np.abs(F - 2200.0))
    assert abs(lp[j] - 0.5) < 0.03
    # 81 dB stopband about 160 Hz beyond the cutoff.
    assert np.all(np.abs(lp[F > 2500.0]) < 10 ** (-70 / 20))
    assert np.all(np.abs(hp[F < 1900.0]) < 10 ** (-70 / 20))


def test_butterworth_is_maximally_flat():
    for n in (1, 2, 3, 4):
        h = crossover.butterworth(F, 1000.0, n, "lp")
        np.testing.assert_allclose(np.abs(h) ** 2, 1.0 / (1.0 + (F / 1000.0) ** (2 * n)), rtol=1e-9)
