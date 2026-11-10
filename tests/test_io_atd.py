import numpy as np

from possession.io_atd import load_atd

META = """<main><Metadata><Teams>
<Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
<Players><Player id="a" teamId="T1"/><Player id="b" teamId="T2"/>
<Player id="c" teamId="T1"/><Player id="d" teamId="T2"/></Players></Metadata></main>"""
RAW = ("1:0.5,0.5,1.0;NaN,NaN,NaN;0.25,0.5,0.0;1.0,1.0,2.5:0.5,0.5,3.0\n"
       "2:0.5,0.5,1.0;0.0,0.0,0.5;NaN,NaN,NaN;1.0,1.0,2.5:NaN,NaN,NaN\n"
       "3:0.5,0.5,1.0;0.0,0.0,0.5;0.25,0.5,0.0;1.0,1.0,2.5:0.75,0.25\n")


def _load(tmp_path):
    (tmp_path / "m.xml").write_text(META)
    (tmp_path / "r.txt").write_text(RAW)
    return load_atd(tmp_path / "m.xml", tmp_path / "r.txt", n_frames=4,
                    pitch=(100.0, 60.0), period_blocks=[(1, 1, 2)])


def test_first_team_is_home(tmp_path):
    ft = _load(tmp_path)
    assert ft.home.shape == (4, 2, 2) and ft.away.shape == (4, 2, 2)
    assert ft.home[0, 0].tolist() == [0.0, 0.0]
    assert ft.home[0, 1].tolist() == [-25.0, 0.0]
    assert ft.away[0, 1].tolist() == [50.0, 30.0]


def test_missing_observations_and_frame_indexing(tmp_path):
    ft = _load(tmp_path)
    assert np.isnan(ft.away[0, 0]).all()
    assert np.isnan(ft.ball[1]).all()
    assert ft.ball[2].tolist() == [25.0, -15.0]
    assert np.isnan(ft.ball[3]).all()
    assert ft.home_speed[0].tolist() == [1.0, 0.0]
    assert ft.period.tolist() == [0, 1, 1, 0]
    assert ft.proxy is None


def test_swapped_and_period_slices(tmp_path):
    ft = _load(tmp_path)
    sw = ft.swapped()
    assert np.array_equal(sw.home, ft.away, equal_nan=True)
    assert np.array_equal(sw.away_speed, ft.home_speed, equal_nan=True)
    assert ft.period_slices() == [(1, 1, 3)]
