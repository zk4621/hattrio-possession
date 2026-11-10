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


def test_controlled_labels_absorb_short_touches():
    from possession.io_dfl import controlled_labels
    y = np.array([X] + [H] * 100 + [A] * 10 + [H] * 100 + [A] * 10 + [D] * 30
                 + [A] * 100 + [H] * 60 + [A] * 40 + [X], dtype=np.int8)
    out = controlled_labels(y, min_frames=50)
    # short A between H and H, and short A before a stoppage, become H
    assert out[1:221].tolist() == [H] * 220
    assert out[221:251].tolist() == [D] * 30
    # a long H spell, and a spell that is not followed by the previous team or D, stay
    assert out[251:351].tolist() == [A] * 100
    assert out[351:411].tolist() == [H] * 60
    assert out[411:451].tolist() == [A] * 40
    assert out[0] == X and out[-1] == X
