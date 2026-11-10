import numpy as np

from helpers import smooth_table
from possession.proxy import camera_centre, measure_proxy, simulate_proxy


def test_camera_centre_inverts_the_homography():
    H0 = np.array([[1920.0, 0, 0], [0, 1080.0, 0], [0, 0, 1]])
    H1 = np.array([[1920.0, 0, -480.0], [0, 1080.0, 0], [0, 0, 1]])
    out = camera_centre(np.stack([H0, H1, H0]), np.array([1, 1, 0]), pitch=(105.0, 68.0))
    assert np.allclose(out[0], [0.0, 0.0], atol=1e-6)
    assert np.allclose(out[1], [26.25, 0.0], atol=1e-6)
    assert np.isnan(out[2]).all()


def test_measure_recovers_lag_from_a_quiet_proxy():
    ft = smooth_table(60000)
    quiet = {"best_lag": 10, "err_std_x": 0.5, "err_std_y": 0.5, "err_corr_frames": 5}
    assert abs(measure_proxy(simulate_proxy(ft, quiet, seed=0), ft)["best_lag"] - 10) <= 3


def test_measure_recovers_noise_from_a_noisy_proxy():
    ft = smooth_table(60000)
    noisy = {"best_lag": 0, "err_std_x": 6.0, "err_std_y": 4.0, "err_corr_frames": 50}
    proxy = simulate_proxy(ft, noisy, seed=0)
    m = measure_proxy(proxy, ft)
    assert 4.5 <= m["err_std_x"] <= 8.5 and 3.0 <= m["err_std_y"] <= 6.0
    assert m["corr_x"] > 0.8
    assert np.array_equal(proxy, simulate_proxy(ft, noisy, seed=0), equal_nan=True)
