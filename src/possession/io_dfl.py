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
# The challenge counts a change of possession only on *controlled* possession: a touch, a
# deflection or a won duel without control does not change it. DFL's flag flips on such touches,
# so an opponent spell shorter than this that ends with the ball back with the previous team, or
# with a stoppage, is given to the previous team. 2 s comes from the challenge's own proxy ("a
# player keeps the ball for a few seconds"); it was not tuned on any ground truth.
STICKY_FRAMES = 50
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


def controlled_labels(y: np.ndarray, min_frames: int = STICKY_FRAMES) -> np.ndarray:
    """Absorb short uncontrolled opponent spells into the previous team's possession."""
    y = np.asarray(y, dtype=np.int8).copy()
    while True:
        breaks = np.flatnonzero(y[1:] != y[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(y)]))
        segs = [(int(a), int(b), int(y[a])) for a, b in zip(bounds[:-1], bounds[1:])]
        changed = False
        for i in range(1, len(segs) - 1):
            a, b, k = segs[i]
            prev, nxt = segs[i - 1][2], segs[i + 1][2]
            if (k in (H, A) and b - a < min_frames and prev in (H, A) and prev != k
                    and nxt in (prev, D)):
                y[a:b] = prev
                changed = True
                break
        if not changed:
            return y


def _find(pattern: str) -> Path:
    hits = sorted((EXTERNAL / "dfl").glob(pattern))
    if not hits:
        raise IOError(f"external data not found: {EXTERNAL / 'dfl' / pattern}\n"
                      "Run: uv run python scripts/fetch_external.py")
    return hits[0]


def load_dfl_match(match_id: str, raw: bool = False) -> tuple[FrameTable, np.ndarray]:
    """The match and its labels; `controlled_labels` applied unless raw=True."""
    ft, y = _load_dfl_match(match_id)
    return ft, (y if raw else controlled_labels(y))


def _load_dfl_match(match_id: str) -> tuple[FrameTable, np.ndarray]:
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
