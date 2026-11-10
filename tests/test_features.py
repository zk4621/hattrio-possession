import numpy as np

from helpers import (perturb_period, random_table, scene_ball_at, scene_home_on_ball,
                     scene_with_duplicates, scene_with_gaps)
from possession.features import BASE_NAMES, PROXY_NAMES, PROXY_WINDOWED, WINDOWED, build_features


def col(F, names, n):
    return F[:, names.index(n)]


def test_nearest_player_distances():
    F, names = build_features(scene_home_on_ball())
    assert np.allclose(col(F, names, "d1_home"), 0.0, atol=1e-5)
    assert np.allclose(col(F, names, "d1_away"), 10.0, atol=1e-4)
    assert np.allclose(col(F, names, "d_diff"), 10.0, atol=1e-4)
    assert (col(F, names, "near_team") == 1).all()
    assert np.isnan(col(F, names, "d2_home")).all()


def test_short_gap_is_interpolated_long_gap_is_not():
    F, names = build_features(scene_with_gaps([(10, 20), (60, 160)]))
    vis = col(F, names, "ball_vis")
    assert (vis[10:20] == 1).all() and (vis[60:160] == 0).all()
    assert (col(F, names, "gap_len")[60:160] == 100).all() and col(F, names, "gap_len")[59] == 0
    assert col(F, names, "since_seen")[60:160].tolist() == list(range(1, 101))
    assert col(F, names, "until_seen")[60:160].tolist() == list(range(100, 0, -1))
    assert np.isnan(col(F, names, "d1_home")[60:160]).all()


def test_long_blackout_keeps_player_features():
    F, names = build_features(scene_with_gaps([(100, 8900)], T=9000))
    inside = slice(100, 8900)
    assert col(F, names, "gap_len")[inside].max() == 3000
    assert col(F, names, "since_seen")[inside].max() == 3000
    assert np.isfinite(col(F, names, "cx_home")[inside]).all()
    assert np.isfinite(col(F, names, "mean_speed_m250")[inside]).all()
    assert np.isfinite(col(F, names, "last_out")[inside]).all()
    assert np.isnan(col(F, names, "d_diff_m50")[4000:5000]).all()


def test_out_margin_sign():
    F, names = build_features(scene_ball_at(0.0, 35.0))
    assert np.allclose(col(F, names, "out_margin"), 1.0, atol=1e-5)


def test_swap_symmetry():
    ft = random_table(seed=0, T=600, proxy=True)
    F, names = build_features(ft)
    G, _ = build_features(ft.swapped())
    for a, b in [("d1_home", "d1_away"), ("n3_home", "n3_away"), ("n_home", "n_away"),
                 ("spread_home", "spread_away"), ("len_home", "len_away"),
                 ("speed_home", "speed_away")]:
        assert np.allclose(col(G, names, a), col(F, names, b), equal_nan=True, atol=1e-4)
    for a, b in [("cx_home", "cx_away"), ("cvx_home", "cvx_away"), ("line_home", "line_away"),
                 ("proxy_rel_home", "proxy_rel_away")]:
        assert np.allclose(col(G, names, a), -col(F, names, b), equal_nan=True, atol=1e-4)
    for n in ["ball_x", "d_diff", "near_team", "last_near", "next_near", "d_diff_m50",
              "proxy_x", "proxy_vx"]:
        assert np.allclose(col(G, names, n), -col(F, names, n), equal_nan=True, atol=1e-4)
    for n in ["ball_vis", "gap_len", "out_margin", "ball_speed", "mean_speed", "frac_slow",
              "cx_gap"]:
        assert np.allclose(col(G, names, n), col(F, names, n), equal_nan=True, atol=1e-4)


def test_periods_are_independent_and_outside_is_nan():
    ft = random_table(seed=1, T=600, periods=[(1, 0, 250), (2, 350, 600)])
    F, names = build_features(ft)
    assert np.isnan(F[250:350]).all()
    F2, _ = build_features(perturb_period(ft, 2))
    assert np.array_equal(F[:250], F2[:250], equal_nan=True)


def test_duplicate_and_extra_players():
    F, names = build_features(scene_with_duplicates())
    assert np.isfinite(col(F, names, "d1_home")).all()
    assert (col(F, names, "n_home") == 14).all()


def test_shape_names_and_optional_proxy():
    F, names = build_features(random_table(seed=2, T=300))
    assert F.dtype == np.float32 and F.shape == (300, len(names))
    assert len(names) == len(set(names)) == len(BASE_NAMES) + 9 * len(WINDOWED)
    assert not any(n.startswith("proxy") for n in names)
    G, gnames = build_features(random_table(seed=2, T=300, proxy=True))
    assert len(gnames) == len(names) + len(PROXY_NAMES) + 9 * len(PROXY_WINDOWED)
    assert gnames[:len(names)] == names
