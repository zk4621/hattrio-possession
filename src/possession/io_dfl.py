"""Public DFL tracking (Bassek et al. 2025) -> FrameTable + per-frame labels."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .frames import FrameTable
from .labels import A, D, H, X
from .paths import CACHE, EXTERNAL

DFL_MATCH_IDS = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
HALFTIME_GAP = 250
SECTIONS = {"firstHalf": 1, "secondHalf": 2}


def _meta(meta_path):
    root = ET.parse(meta_path).getroot()
    home = guest = None
    for t in root.iter("Team"):
        if t.get("Role") == "home":
            home = t.get("TeamId")
        elif t.get("Role") == "guest":
            guest = t.get("TeamId")
    env = next(root.iter("Environment"), None)
    px = float(env.get("PitchX", 105.0)) if env is not None else 105.0
    py = float(env.get("PitchY", 68.0)) if env is not None else 68.0
    return home, guest, (px, py)


def _read_sets(positions_path, home, guest):
    """{(section, team, person): (N, X, Y, S, poss, status)} for teams and ball."""
    sets = {}
    root = None
    for event, elem in ET.iterparse(positions_path, events=("start", "end")):
        if root is None and event == "start":
            root = elem
            continue
        if event != "end" or elem.tag != "FrameSet":
            continue
        section = SECTIONS.get(elem.get("GameSection"))
        team = elem.get("TeamId")
        if section is not None and team in (home, guest, "BALL"):
            frames = elem.findall("Frame")
            n = np.array([int(f.get("N")) for f in frames], dtype=np.int64)
            x = np.array([float(f.get("X")) for f in frames])
            y = np.array([float(f.get("Y")) for f in frames])
            s = np.array([float(f.get("S", "nan")) for f in frames]) / 3.6
            if team == "BALL":
                poss = np.array([int(f.get("BallPossession", "0")) for f in frames], dtype=np.int8)
                stat = np.array([int(f.get("BallStatus", "0")) for f in frames], dtype=np.int8)
            else:
                poss = stat = None
            sets[(section, team, elem.get("PersonId"))] = (n, x, y, s, poss, stat)
        elem.clear()
        if root is not None:
            root.clear()
    return sets


def load_dfl(meta_path, positions_path) -> tuple[FrameTable, np.ndarray]:
    home, guest, pitch = _meta(meta_path)
    sets = _read_sets(positions_path, home, guest)
    balls = {sec: v for (sec, team, _), v in sets.items() if team == "BALL"}
    halves = sorted(balls)
    starts, lengths = {}, {}
    offset = 0
    for i, sec in enumerate(halves):
        n = balls[sec][0]
        starts[sec] = (offset, int(n.min()))
        lengths[sec] = int(n.max() - n.min() + 1)
        offset += lengths[sec] + (HALFTIME_GAP if i < len(halves) - 1 else 0)
    T = offset
    persons = {team: sorted({pid for (_, t, pid) in sets if t == team}) for team in (home, guest)}

    ball = np.full((T, 2), np.nan, np.float32)
    y = np.full(T, X, np.int8)
    period = np.zeros(T, np.int8)
    pos = {team: np.full((T, len(persons[team]), 2), np.nan, np.float32) for team in (home, guest)}
    spd = {team: np.full((T, len(persons[team])), np.nan, np.float32) for team in (home, guest)}

    for sec in halves:
        row0, n0 = starts[sec]
        period[row0:row0 + lengths[sec]] = sec
        n, bx, by, _, poss, stat = balls[sec]
        r = row0 + (n - n0)
        ball[r, 0], ball[r, 1] = bx, by
        lab = np.where(stat != 1, D, np.where(poss == 1, H, A)).astype(np.int8)
        y[r] = lab
        # rows of the half without a ball frame: dead ball
        half_rows = np.arange(row0, row0 + lengths[sec])
        missing = np.setdiff1d(half_rows, r)
        y[missing] = D

    for (sec, team, pid), (n, px, py, s, _, _) in sets.items():
        if team == "BALL" or sec not in starts:
            continue
        row0, n0 = starts[sec]
        r = row0 + (n - n0)
        keep = (r >= row0) & (r < row0 + lengths[sec])
        j = persons[team].index(pid)
        pos[team][r[keep], j, 0] = px[keep]
        pos[team][r[keep], j, 1] = py[keep]
        spd[team][r[keep], j] = s[keep]

    ft = FrameTable(pitch=pitch, ball=ball, home=pos[home], away=pos[guest],
                    home_speed=spd[home], away_speed=spd[guest], period=period)
    return ft, y


def _find(pattern: str) -> Path:
    hits = sorted((EXTERNAL / "dfl").glob(pattern))
    if not hits:
        raise IOError(f"external data not found: {EXTERNAL / 'dfl' / pattern}\n"
                      "Run: uv run python scripts/fetch_external.py")
    return hits[0]


def load_dfl_match(match_id: str) -> tuple[FrameTable, np.ndarray]:
    """The match and its labels, the DFL possession flag used as it is."""
    cache = CACHE / f"dfl_{match_id}.npz"
    if cache.exists():
        z = np.load(cache)
        ft = FrameTable(pitch=tuple(float(v) for v in z["pitch"]), ball=z["ball"], home=z["home"],
                        away=z["away"], home_speed=z["home_speed"], away_speed=z["away_speed"],
                        period=z["period"])
        return ft, z["y"]
    meta = _find(f"DFL_02_01_matchinformation_*_DFL-MAT-{match_id}.xml")
    positions = _find(f"DFL_04_03_positions_raw_observed_*_DFL-MAT-{match_id}.xml")
    ft, y = load_dfl(meta, positions)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pitch=np.array(ft.pitch), ball=ft.ball, home=ft.home, away=ft.away,
             home_speed=ft.home_speed, away_speed=ft.away_speed, period=ft.period, y=y)
    return ft, y
