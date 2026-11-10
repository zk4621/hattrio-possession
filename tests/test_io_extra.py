import json

import numpy as np

from possession.io_metrica import metrica_labels
from possession.io_skillcorner import load_skillcorner
from possession.labels import A, D, H, X


def _sc_fixture(tmp_path):
    match = {"home_team": {"id": 1}, "away_team": {"id": 2}, "pitch_length": 104,
             "pitch_width": 68,
             "players": [{"id": 11, "team_id": 1}, {"id": 12, "team_id": 1},
                         {"id": 21, "team_id": 2}]}
    (tmp_path / "7_match.json").write_text(json.dumps(match))
    lines = []
    for f in range(100, 111):                        # 11 frames at 10 fps = 1.0 s, period 1
        lines.append({"frame": f, "period": 1,
                      "ball_data": {"x": float(f - 100), "y": 0.0, "z": 0.0,
                                    "is_detected": f != 105},
                      "possession": {"player_id": None,
                                     "group": "home team" if f < 106 else
                                     (None if f < 108 else "away team")},
                      "player_data": [{"player_id": 11, "x": 1.0, "y": 2.0, "is_detected": True},
                                      {"player_id": 12, "x": 5.0, "y": 5.0,
                                       "is_detected": False},
                                      {"player_id": 21, "x": -3.0, "y": 0.0,
                                       "is_detected": True}]})
    lines.append({"frame": 111, "period": None, "ball_data": {}, "possession": {},
                  "player_data": []})
    (tmp_path / "7_tracking_extrapolated.jsonl").write_text(
        "\n".join(json.dumps(x) for x in lines))
    (tmp_path / "7_phases_of_play.csv").write_text(
        "frame_start,frame_end,period,team_in_possession_id\n100,108,1,1\n")
    (tmp_path / "7_dynamic_events.csv").write_text(
        "frame_start,period,team_id,event_type,game_interruption_before\n"
        "104,1,2,player_possession,throw_in_for\n")
    return tmp_path


def test_skillcorner_resamples_to_25_fps_and_labels(tmp_path):
    ft, y, set_pieces = load_skillcorner(_sc_fixture(tmp_path), "7")
    assert ft.n_frames == 26                         # 1.0 s at 25 fps, both ends included
    assert ft.pitch == (104.0, 68.0)
    assert ft.home.shape[1] == 2 and ft.away.shape[1] == 1
    assert np.allclose(ft.ball[:3, 0], [0.0, 0.4, 0.8])
    assert np.isnan(ft.ball[12:13, 0]).all()         # 10 fps frame 105 (row 12.5) not seen
    assert np.isnan(ft.home[:, 1]).all()             # an extrapolated player is not an observation
    assert np.allclose(ft.away[:, 0, 0], -3.0)
    assert ft.period.tolist() == [1] * 26
    # live 10-fps frames 100..108; home until 105, nobody at 106-107 (stays home), away at 108;
    # each 25-fps row takes the nearest 10-fps frame
    assert y[:19].tolist() == [H] * 19
    assert y[19:22].tolist() == [A] * 3 and (y[22:] == D).all()
    assert set_pieces == [("kickoff", 0, 1), ("throw", 10, -1)]


def test_metrica_labels_follow_events():
    # frames 1..40 of one period; kick-off by home at 3, away recovery at 10, ball out at 20,
    # throw-in by away at 30
    events = [("SET PIECE", "KICK OFF", 1, 3, 3), ("PASS", "", 1, 5, 8),
              ("CHALLENGE", "", -1, 9, 9), ("RECOVERY", "", -1, 10, 10),
              ("BALL OUT", "", -1, 20, 20), ("SET PIECE", "THROW IN", -1, 30, 30)]
    period = np.ones(40, np.int8)
    y = metrica_labels(events, period)
    assert y[:2].tolist() == [D, D]                  # before the kick-off (rows 0-1 = frames 1-2)
    assert y[2:9].tolist() == [H] * 7                # frames 3..9: home (a challenge changes nothing)
    assert y[9:19].tolist() == [A] * 10              # frames 10..19: away
    assert y[19:29].tolist() == [D] * 10             # ball out until the throw-in
    assert y[29:].tolist() == [A] * 11
    assert X not in y.tolist()
