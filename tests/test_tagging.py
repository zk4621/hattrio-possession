import numpy as np

from helpers import random_table
from possession.features import build_features
from possession.tagging import (TAG_NAMES, Tag, competition_tags, detect_restart,
                                dfl_set_pieces_from_files, simulate_set_piece_tags)


def col(F, names, n):
    return F[:, names.index(n)]


def test_set_piece_features_count_frames_to_and_from_tags():
    ft = random_table(seed=3, T=1000)
    ft.tags = [Tag("SET PIECES", "throw", 200, 1075, 1), Tag("SET PIECES", "corner", 600, 1475, 0)]
    F, names = build_features(ft)
    assert col(F, names, "sp_since")[199] == 3000 and col(F, names, "sp_since")[250] == 50
    assert col(F, names, "sp_until")[100] == 100 and col(F, names, "sp_until")[650] == 3000
    assert col(F, names, "sp_throw_since")[650] == 450
    assert col(F, names, "sp_corner_since")[650] == 50
    assert col(F, names, "sp_corner_until")[650] == 3000
    assert col(F, names, "sp_team_last")[250] == 1 and col(F, names, "sp_team_last")[650] == 0
    assert col(F, names, "sp_team_next")[100] == 1 and col(F, names, "sp_team_next")[300] == 0


def test_tags_are_optional_and_swap_negates_team():
    ft = random_table(seed=4, T=600)
    F, names = build_features(ft)
    assert not any(n.startswith("sp_") for n in names)
    ft.tags = [Tag("SET PIECES", "goalkick", 100, 900, -1)]
    G, gnames = build_features(ft)
    assert gnames[:len(names)] == names and len(gnames) == len(names) + len(TAG_NAMES)
    S, _ = build_features(ft.swapped())
    assert np.array_equal(col(S, gnames, "sp_team_last"), -col(G, gnames, "sp_team_last"))
    assert np.array_equal(col(S, gnames, "sp_since"), col(G, gnames, "sp_since"))


def test_detect_restart_finds_the_kick_after_a_still_ball():
    T = 600
    ball = np.zeros((T, 2), np.float32)
    ball[:, 0] = 10.0
    ball[300:, 0] = 10.0 + 0.4 * np.arange(T - 300)
    assert detect_restart(ball, center=250, lo=-250, hi=375) == 300
    ball[280:295] = np.nan                       # ball not seen while it was still: no restart
    assert detect_restart(ball, center=250, lo=-250, hi=375) is None


def test_simulated_set_piece_tags_follow_the_profile():
    events = [("throw", 1000 + 300 * i, 1 if i % 2 else -1) for i in range(400)]
    profile = {"duration_q": [875.0] * 101,
               "kinds": {"throw": {"keep": 0.5, "team_known": 0.4,
                                   "offset_q": list(np.linspace(0, 125, 101))}}}
    tags = simulate_set_piece_tags(events, profile, seed=0)
    assert 0.4 <= len(tags) / 400 <= 0.6
    known = [g for g in tags if g.team != 0]
    assert 0.3 <= len(known) / len(tags) <= 0.5
    for g in tags:
        assert g.code == "SET PIECES" and g.kind == "throw" and g.end - g.start == 875
        row, team = next((r, t) for _, r, t in events if r - 125 <= g.start <= r)
        assert g.team in (0, team)
    assert tags == simulate_set_piece_tags(events, profile, seed=0)


META_DFL = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest><MatchInformation><Environment PitchX="105.00" PitchY="68.00"/>
<Teams><Team TeamId="CLU-A" Role="guest"/><Team TeamId="CLU-H" Role="home"/></Teams>
</MatchInformation></PutDataRequest>"""

POS_DFL = """<?xml version="1.0" encoding="UTF-8" standalone="yes" ?>
<PutDataRequest><Positions>
<FrameSet GameSection="firstHalf" MatchId="M" TeamId="CLU-H" PersonId="p1">
<Frame N="10000" T="2023-05-27T13:30:12.000+00:00" X="1" Y="1" S="0"/></FrameSet>
<FrameSet GameSection="firstHalf" MatchId="M" TeamId="BALL" PersonId="B">
<Frame N="10000" T="2023-05-27T13:30:12.600+00:00" X="0" Y="0" S="0"/></FrameSet>
<FrameSet GameSection="secondHalf" MatchId="M" TeamId="BALL" PersonId="B">
<Frame N="100000" T="2023-05-27T14:35:00.000+00:00" X="0" Y="0" S="0"/></FrameSet>
</Positions></PutDataRequest>"""

EVENTS_DFL = """<?xml version="1.0" encoding="utf-8"?><PutDataRequest>
<Event EventTime="2023-05-27T15:30:12.600+02:00"><KickOff GameSection="firstHalf"><Play Team="CLU-A"/></KickOff></Event>
<Event EventTime="2023-05-27T15:30:22.600+02:00"><ThrowIn Team="CLU-H"><Play Team="CLU-H"/></ThrowIn></Event>
<Event EventTime="2023-05-27T15:31:00.000+02:00"><Play Team="CLU-H"/></Event>
<Event EventTime="2023-05-27T15:31:10.000+02:00"><Delete/></Event>
<Event EventTime="2023-05-27T16:35:04.000+02:00"><CornerKick Team="CLU-A"><Play Team="CLU-A"/></CornerKick></Event>
</PutDataRequest>"""


def test_dfl_set_pieces_are_placed_on_the_table_rows(tmp_path):
    for name, text in (("m.xml", META_DFL), ("p.xml", POS_DFL), ("e.xml", EVENTS_DFL)):
        (tmp_path / name).write_text(text)
    period = np.array([1] * 400 + [0] * 250 + [2] * 400, np.int8)
    events = dfl_set_pieces_from_files(tmp_path / "m.xml", tmp_path / "p.xml",
                                       tmp_path / "e.xml", period)
    assert events == [("kickoff", 0, -1), ("throw", 250, 1), ("corner", 750, -1)]


SMART = """<file><ALL_INSTANCES>
<instance><ID>1</ID><start>4.0</start><end>39.0</end><code>SET PIECES</code>
<label><group>Type</group><text>Throw-in</text></label>
<label><group>Team</group><text>Away FC</text></label></instance>
<instance><ID>2</ID><start>10.0</start><end>27.6</end><code>PROGRESSION</code>
<label><group>Team</group><text>Home FC</text></label></instance>
<instance><ID>3</ID><start>20.0</start><end>55.0</end><code>SET PIECES</code>
<label><group>Type</group><text>Corner Kick</text></label>
<label><group>Team</group><text>N/A</text></label></instance>
</ALL_INSTANCES></file>"""

META_ATD = """<main><Metadata><Teams>
<Team id="T1"><Name>Home FC</Name></Team><Team id="T2"><Name>Away FC</Name></Team></Teams>
</Metadata></main>"""


def test_competition_tags_map_teams_by_metadata_name(tmp_path, monkeypatch):
    (tmp_path / "tactical").mkdir()
    (tmp_path / "tactical" / "BAR-ATH_tactical_smart_tagging.xml").write_text(SMART)
    (tmp_path / "tactical" / "BAR-ATH_tactical_atd.xml").write_text(META_ATD)
    monkeypatch.setenv("MTAH_DATA_DIR", str(tmp_path))
    assert competition_tags() == [Tag("SET PIECES", "throw", 100, 975, -1),
                                  Tag("PROGRESSION", "", 250, 690, 1),
                                  Tag("SET PIECES", "corner", 500, 1375, 0)]


def test_detect_restart_accepts_a_smoothed_kick():
    """Smoothed tracking: the ball speeds up over a few frames instead of jumping."""
    T = 600
    t = np.arange(T, dtype=np.float64)
    x = 10.0 + np.where(t >= 300, 0.5 * 0.05 * (t - 300) ** 2, 0.0)
    ball = np.stack([x, np.zeros(T)], axis=1).astype(np.float32)
    assert detect_restart(ball, center=250, lo=-250, hi=375) == 302


def test_phase_features_sum_active_tags_and_negate_on_swap():
    from possession.tagging import PH_NAMES
    ft = random_table(seed=5, T=1000)
    ft.tags = [Tag("PROGRESSION", "", 100, 500, 1), Tag("BALL IN THE BOX", "", 300, 700, -1),
               Tag("SET PIECES", "throw", 800, 1675, 0)]
    F, names = build_features(ft)
    assert all(n in names for n in PH_NAMES)
    assert col(F, names, "ph_progression")[150] == 1 and col(F, names, "ph_progression")[600] == 0
    assert col(F, names, "ph_active")[400] == 0 and col(F, names, "ph_count")[400] == 2
    assert col(F, names, "ph_active")[650] == -1
    assert col(F, names, "ph_since")[350] == 50 and col(F, names, "ph_team_last")[350] == -1
    assert col(F, names, "ph_until")[50] == 50 and col(F, names, "ph_team_next")[50] == 1
    S, _ = build_features(ft.swapped())
    for n in ("ph_active", "ph_progression", "ph_team_last", "ph_team_next"):
        assert np.array_equal(col(S, names, n), -col(F, names, n))
    for n in ("ph_count", "ph_since", "ph_until"):
        assert np.array_equal(col(S, names, n), col(F, names, n))


def test_dfl_phase_candidates_follow_possession_and_geometry():
    from helpers import make_table
    from possession.labels import A, D, H
    from possession.tagging import dfl_phase_candidates
    T = 1000
    ball = np.zeros((T, 2), np.float32)
    ball[:, 0] = np.linspace(-40, 50, T)           # home carries the ball from its own third
    home = np.zeros((T, 11, 2), np.float32)
    home[:, :, 0] = np.linspace(-45, 45, T)[:, None] - np.arange(11)[None, :] * 3
    away = np.zeros((T, 11, 2), np.float32)
    away[:, :, 0] = 30.0                             # away stands on the right: home attacks +x
    ft = make_table(T, ball, home, away)
    y = np.full(T, H, np.int8)
    y[:100] = A                                      # home regains the ball at frame 100
    y[950:] = D
    c = dfl_phase_candidates(ft, y)
    assert c["ATTACKING TRANSITION"] == [(100, 1)]
    assert [t for _, t in c["BUILD UP"]] == [1]
    (f3, _), = c["BALL IN FINAL THIRD"]
    assert abs(ball[f3, 0] - 105 / 6) < 0.2      # enters the attacking third at x = L/6
    (bb, _), = c["BALL IN THE BOX"]
    assert abs(ball[bb, 0] - (52.5 - 16.5)) < 0.2


def test_simulated_phase_tags_keep_count_and_start_before_the_anchor():
    from possession.tagging import simulate_phase_tags
    cands = {"PROGRESSION": [(1000 + 400 * i, 1 if i % 3 else -1) for i in range(300)]}
    profile = {"pre_frames": [50, 100],
               "codes": {"PROGRESSION": {"keep": 0.5, "team_known": 1.0,
                                         "duration_q": [440.0] * 101}}}
    tags = simulate_phase_tags(cands, profile, seed=1)
    assert 0.4 <= len(tags) / 300 <= 0.6
    rows = dict(cands["PROGRESSION"])
    for g in tags:
        anchor = next(r for r in rows if 50 <= r - g.start <= 100)
        assert g.team == rows[anchor] and g.end - g.start == 440 and g.kind == ""
    assert tags == simulate_phase_tags(cands, profile, seed=1)


def test_simulated_phase_tags_can_name_the_wrong_team():
    from possession.tagging import simulate_phase_tags
    cands = {"LONG BALL": [(1000 + 400 * i, 1) for i in range(200)]}
    profile = {"pre_frames": [50, 100],
               "codes": {"LONG BALL": {"keep": 1.0, "team_known": 1.0, "team_flip": 0.25,
                                       "duration_q": [300.0] * 101}}}
    tags = simulate_phase_tags(cands, profile, seed=2)
    wrong = sum(g.team == -1 for g in tags) / len(tags)
    assert 0.15 <= wrong <= 0.35


def test_entry_moments_skip_frames_where_the_ball_is_not_seen():
    from helpers import make_table
    from possession.tagging import entry_moments
    T = 300
    ball = np.zeros((T, 2), np.float32)
    ball[:, 0] = np.where(np.arange(T) < 150, 30.0, 45.0)    # outside, then inside the right box
    ft = make_table(T, ball, np.zeros((11, 2)), np.zeros((11, 2)))
    assert entry_moments("BALL IN THE BOX", ft, 0, T, 1, 1.0).tolist() == [150]
    ft.ball[140:160] = np.nan                                 # entry happens while unseen
    assert entry_moments("BALL IN THE BOX", ft, 0, T, 1, 1.0).tolist() == [160]
    ft.ball[:140] = [45.0, 0.0]                               # inside before the gap: no entry
    assert entry_moments("BALL IN THE BOX", ft, 0, T, 1, 1.0).tolist() == []
    assert entry_moments("BALL IN THE BOX", ft, 0, T, -1, -1.0).tolist() == []
