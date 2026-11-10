import numpy as np
import pytest

from possession.io_dfl import HALFTIME_GAP, load_dfl
from possession.labels import A, D, H, X

META = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest><MatchInformation><General MatchId="DFL-MAT-TEST"/>
<Environment PitchX="105.00" PitchY="68.00"/>
<Teams><Team TeamId="CLU-A" Role="guest"/><Team TeamId="CLU-H" Role="home"/></Teams>
</MatchInformation></PutDataRequest>"""


def _frames(ns, fmt):
    return "".join(fmt.format(n=n, i=i) for i, n in enumerate(ns))


def write_fixture(tmp_path):
    first, second = [10000, 10001, 10002], [100000, 100001]
    ball1 = [(1, 0), (1, 1), (2, 1)]
    ball2 = [(2, 1), (1, 0)]

    def ball(section, ns, flags):
        rows = "".join(
            f'<Frame N="{n}" X="1.0" Y="2.0" Z="0.1" S="10.0" BallPossession="{p}" BallStatus="{s}"/>'
            for n, (p, s) in zip(ns, flags))
        return f'<FrameSet GameSection="{section}" TeamId="BALL" PersonId="B1">{rows}</FrameSet>'

    def player(section, team, pid, ns):
        rows = "".join(f'<Frame N="{n}" X="-10.0" Y="5.0" S="7.2"/>' for n in ns)
        return f'<FrameSet GameSection="{section}" TeamId="{team}" PersonId="{pid}">{rows}</FrameSet>'

    body = (ball("firstHalf", first, ball1) + ball("secondHalf", second, ball2)
            + player("firstHalf", "CLU-H", "p1", first) + player("secondHalf", "CLU-H", "p1", second)
            + player("firstHalf", "CLU-A", "p2", [10001])
            + player("firstHalf", "referee", "r1", first))
    pos = f'<?xml version="1.0" encoding="utf-8"?><PutDataRequest><Positions>{body}</Positions></PutDataRequest>'
    (tmp_path / "meta.xml").write_text(META)
    (tmp_path / "pos.xml").write_text(pos)
    return tmp_path / "meta.xml", tmp_path / "pos.xml"


def test_load_dfl_maps_labels_units_and_gap(tmp_path):
    ft, y = load_dfl(*write_fixture(tmp_path))
    assert ft.n_frames == 3 + HALFTIME_GAP + 2
    assert y[:3].tolist() == [D, H, A]
    assert set(y[3:3 + HALFTIME_GAP].tolist()) == {X}
    assert y[-2:].tolist() == [A, D]
    assert ft.period[:3].tolist() == [1, 1, 1] and ft.period[3] == 0 and ft.period[-1] == 2
    assert ft.pitch == (105.0, 68.0)
    assert ft.home.shape[1] == 1 and ft.away.shape[1] == 1
    assert ft.home[0, 0].tolist() == [-10.0, 5.0]
    assert ft.home_speed[0, 0] == pytest.approx(2.0)
    assert np.isnan(ft.away[0, 0]).all() and not np.isnan(ft.away[1, 0]).any()
    assert np.isnan(ft.ball[3]).all()


def test_load_dfl_match_keeps_short_spells(tmp_path, monkeypatch):
    """The challenge's public ground truth counts possessions of 0.8 s and 1.3 s, so the DFL
    possession flag is used as it is: a one-frame opponent spell stays a possession."""
    import possession.io_dfl as io

    flags = [(1, 1), (1, 1), (2, 1), (1, 1), (1, 1)]
    rows = "".join(f'<Frame N="{10000 + i}" X="1.0" Y="2.0" S="10.0" BallPossession="{p}" '
                   f'BallStatus="{s}"/>' for i, (p, s) in enumerate(flags))
    ball = f'<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="B1">{rows}</FrameSet>'
    pos = ('<?xml version="1.0" encoding="utf-8"?><PutDataRequest><Positions>'
           f'{ball}</Positions></PutDataRequest>')
    ext = tmp_path / "external"
    (ext / "dfl").mkdir(parents=True)
    (ext / "dfl" / "DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-TEST01.xml").write_text(META)
    (ext / "dfl" / "DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-TEST01.xml"
     ).write_text(pos)
    monkeypatch.setattr(io, "EXTERNAL", ext)
    monkeypatch.setattr(io, "CACHE", tmp_path / "cache")
    _, y = io.load_dfl_match("TEST01")
    assert y.tolist() == [H, H, A, H, H]
