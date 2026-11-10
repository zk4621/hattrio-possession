import numpy as np

from helpers import broadcast_from, smooth_table
from possession.sync import fit_mapping, merge_ball, pair_cost


def test_pair_cost_is_small_only_for_the_same_instant():
    t = smooth_table(3000)
    b, truth = broadcast_from(t, [(1000, 1001)])
    cost = pair_cost(b.home[0], b.away[0], t.home, t.away)
    assert abs(int(np.argmin(cost)) - 1000) <= 3 and cost[1000] < 0.6
    assert np.median(cost[np.isfinite(cost)]) > 2.5


def test_mapping_recovers_live_pieces_across_a_cut_and_a_replay():
    t = smooth_table(6000)
    b, truth = broadcast_from(t, [(1000, 3000), (1500, 1800), (3500, 5000)])
    mapping = fit_mapping(b, t)
    live = np.r_[0:2000, 2300:3800]
    ok = np.abs(mapping[live] - truth[live]) <= 3
    assert ok.mean() >= 0.95
    wrong = (mapping >= 0) & (np.abs(mapping - truth) > 3)
    assert wrong.mean() <= 0.01


def test_merge_fills_only_blackouts_and_reports():
    t = smooth_table(6000)
    truth_ball = t.ball.copy()
    t.ball[2000:2600] = np.nan
    b, truth = broadcast_from(t, [(1000, 5000)])
    b.ball[:] = truth_ball[1000:5000]
    merged, stats = merge_ball(t, b, fit_mapping(b, t))
    assert np.isnan(merged.ball[2000:2600, 0]).mean() <= 0.05
    assert np.array_equal(merged.ball[:2000], t.ball[:2000])
    assert stats["filled_share"] >= 0.95 and stats["agree_median_m"] < 0.5
