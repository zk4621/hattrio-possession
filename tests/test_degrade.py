import numpy as np

from helpers import run_lengths, smooth_table
from possession.degrade import DEFAULT_PROFILE, DegradeProfile, degrade, measure_profile


def test_smooth_table_meets_its_spec():
    ft = smooth_table(20000)
    d2 = np.abs(np.diff(ft.home[..., 0], n=2, axis=0)).max()
    assert d2 < 0.01 and np.abs(np.diff(ft.ball[:, 0], n=2)).max() < 0.01
    assert (np.ptp(ft.home[..., 0], axis=0) >= 20).all() and ft.ball[:, 0].std() > 15


def test_default_profile_gives_long_blackouts_on_about_41_percent():
    out = degrade(smooth_table(400000), DEFAULT_PROFILE, seed=1)
    missing = np.isnan(out.ball[:, 0])
    assert 0.30 <= missing.mean() <= 0.52
    lengths = run_lengths(missing)
    assert 150 <= np.median(lengths) <= 400
    assert (lengths >= 126).sum() / len(lengths) >= 0.6


def test_deficit_drops_whole_blocks_of_far_players():
    out = degrade(smooth_table(100000), DEFAULT_PROFILE, seed=4)
    tracked = (~np.isnan(out.home[:, :, 0])).sum(axis=1)
    assert 0.5 <= (tracked == 11).mean() <= 0.7
    assert tracked.min() >= 6
    block = tracked[:250]
    assert block.min() == block.max()


def test_degrade_is_deterministic_and_pure():
    ft = smooth_table(20000)
    before = ft.ball.copy()
    a, b, c = (degrade(ft, DEFAULT_PROFILE, 7), degrade(ft, DEFAULT_PROFILE, 7),
               degrade(ft, DEFAULT_PROFILE, 8))
    assert np.array_equal(a.ball, b.ball, equal_nan=True)
    assert np.array_equal(a.home, b.home, equal_nan=True)
    assert not np.array_equal(a.ball, c.ball, equal_nan=True)
    assert np.array_equal(ft.ball, before, equal_nan=True)


def test_measure_recovers_what_degrade_applied():
    p = measure_profile(degrade(smooth_table(600000), DEFAULT_PROFILE, seed=3))
    pooled = np.mean(p.blackout_hazard)
    assert 1 / 900 <= pooled <= 1 / 450
    assert 180 <= p.blackout_quantiles[0][50] <= 380
    assert 0.5 <= p.deficit_pmf[0] <= 0.7
    assert 0.05 <= p.jitter_std_m <= 0.20


def test_profile_json_roundtrip(tmp_path):
    DEFAULT_PROFILE.to_json(tmp_path / "p.json")
    assert DegradeProfile.from_json(tmp_path / "p.json") == DEFAULT_PROFILE


def test_blackouts_do_not_cross_periods():
    ft = smooth_table(40000, periods=[(1, 0, 19000), (2, 21000, 40000)])
    out = degrade(ft, DEFAULT_PROFILE, seed=2)
    assert np.isnan(out.ball[19000:21000]).all() and out.period.tolist() == ft.period.tolist()
    assert not np.isnan(out.ball[[0, 21000]]).any()
