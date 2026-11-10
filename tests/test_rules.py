import numpy as np

from helpers import scene_handover
from possession.labels import A, D, H
from possession.rules import baseline_codes, rule_proba


def test_rule_proba_follows_nearest_player_and_abstains_without_ball():
    names = ["ball_vis", "d1_home", "d1_away", "d_diff", "out_margin", "frac_slow_m50"]
    F = np.array([[1, 0.5, 6.0, 5.5, -10, 0.1],
                  [1, 6.0, 0.5, -5.5, -10, 0.1],
                  [0, np.nan, np.nan, np.nan, np.nan, 0.1],
                  [1, 9.0, 9.0, 0.0, 2.0, 0.1],
                  [0, np.nan, np.nan, np.nan, np.nan, 0.9]], dtype=np.float32)
    P = rule_proba(F, names)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[0].argmax() == H and P[1].argmax() == A
    assert np.allclose(P[2], 1 / 3)
    assert P[3].argmax() == D and P[4].argmax() == D


def test_baseline_codes_matches_starter_logic():
    codes = baseline_codes(scene_handover(), window=50)
    assert (codes[:150] == H).all() and (codes[250:] == A).all()
