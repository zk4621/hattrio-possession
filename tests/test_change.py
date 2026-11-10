import numpy as np

from helpers import random_table
from possession.change import CHANGE_NAMES, change_features, change_targets
from possession.features import build_features
from possession.labels import A, D, H, X


def test_change_targets_mark_frames_near_each_change_inside_periods():
    y = np.array([X] * 5 + [H] * 10 + [A] * 10 + [D] * 5 + [X] * 5 + [A] * 5, np.int8)
    period = np.array([0] * 5 + [1] * 25 + [0] * 5 + [2] * 5, np.int8)
    t = change_targets(y, period, radius=2)
    assert t.dtype == np.int8 and len(t) == len(y)
    assert np.flatnonzero(t).tolist() == [13, 14, 15, 16, 17, 23, 24, 25, 26, 27]


def test_change_features_shape_edges_and_swap():
    ft = random_table(seed=6, T=800, periods=[(1, 0, 400), (2, 450, 800)])
    F, names = build_features(ft)
    G, gnames = change_features(ft, F, names)
    assert gnames == CHANGE_NAMES and G.shape == (800, len(CHANGE_NAMES)) and G.dtype == np.float32
    assert np.isnan(G[400:450]).all()
    Fs, _ = build_features(ft.swapped())
    Gs, _ = change_features(ft.swapped(), Fs, names)
    i = gnames.index("near_jump_5")
    assert np.allclose(Gs[:, i], -G[:, i], equal_nan=True)
    j = gnames.index("ball_turn")
    assert np.allclose(Gs[:, j], G[:, j], equal_nan=True)
