"""Metrica Sports sample games (tracking 25 fps + events) -> FrameTable + H/A/D labels.

github.com/metrica-sports/sample-data ("please be responsible with the use of this data; if you
use it for anything public, please acknowledge the source"). Games 1 and 2 are CSV, game 3 is
FIFA EPTS like the provided match. The ball is missing on 32-41 % of frames, as in the provided
tracking, so these games are used as they are, without simulated blackouts.

Labels come from Metrica's own events: a team has the ball from its pass, carry, recovery, shot,
lost ball or set piece onwards; a challenge changes nothing; the ball is dead from a "BALL OUT",
a "FAULT RECEIVED" or the end of a goal until the next set piece, and before each kick-off.

The events miss many stoppages (a shot wide, a foul recorded as a challenge, an offside): the
labels stayed with a team while play stood still. Where the tracking has no ball for at least
2 s (DEAD_GAP) inside a stretch labelled with a team, the ball is dead there (`dead_in_gaps`);
on those frames the players move as slowly as in labelled stoppages.
"""
from __future__ import annotations

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from .frames import FrameTable
from .io_skillcorner import speeds
from .labels import A, D, H
from .paths import CACHE, EXTERNAL

PITCH = (105.0, 68.0)
GAMES = ["1", "2", "3"]
POSSESSION_EVENTS = {"PASS", "CARRY", "RECOVERY", "SHOT", "BALL LOST", "SET PIECE"}
DEAD_EVENTS = {"BALL OUT", "FAULT RECEIVED"}
DEAD_GAP = 50          # frames (2 s) without a ball that make a stoppage
KIND = {"KICK OFF": "kickoff", "CORNER KICK": "corner", "FREE KICK": "freekick",
        "THROW IN": "throw", "GOAL KICK": "goalkick"}


def metrica_dir() -> Path:
    return EXTERNAL / "metrica" / "data"


def metrica_labels(events, period: np.ndarray) -> np.ndarray:
    """events: [(type, subtype, team +1/-1, start_frame, end_frame)], frames counted from 1."""
    T = len(period)
    changes = []                                       # (row, order, state)
    for p in sorted(set(int(v) for v in period) - {0}):
        changes.append((int(np.flatnonzero(period == p)[0]), 0, D))
    for typ, sub, team, start, end in events:
        state = H if team == 1 else A
        if typ in POSSESSION_EVENTS:
            changes.append((start - 1, 1, state))
        if typ in DEAD_EVENTS:
            changes.append((start - 1, 2, D))
        if typ == "SHOT" and "GOAL" in sub and "OUT" not in sub:
            changes.append((max(end, start) - 1, 2, D))
    changes.sort()
    y = np.full(T, D, np.int8)
    for (row, _, state), nxt in zip(changes, changes[1:] + [(T, 0, D)]):
        a, b = max(row, 0), min(nxt[0], T)
        if b > a:
            y[a:b] = state
    return y


def dead_in_gaps(y: np.ndarray, ball: np.ndarray, period: np.ndarray,
                 min_len: int = DEAD_GAP) -> np.ndarray:
    """`y` with D on every run of at least `min_len` frames inside a half where the ball is
    missing; frames labelled with a team elsewhere are unchanged."""
    out = np.array(y, copy=True)
    gap = (period > 0) & ~np.isfinite(ball[:, 0])
    m = np.concatenate(([0], gap.astype(np.int8), [0]))
    d = np.diff(m)
    for a, z in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if z - a >= min_len:
            seg = out[a:z]
            seg[(seg == H) | (seg == A)] = D
    return out


def _set_pieces(events) -> list[tuple[str, int, int]]:
    out = []
    for typ, sub, team, start, _ in events:
        if typ == "SET PIECE":
            kind = KIND.get(sub.split("-")[0].strip())
            if kind:
                out.append((kind, start - 1, team))
    return out


def _csv_game(g: str):
    root = metrica_dir() / f"Sample_Game_{g}"
    sides = {}
    for side in ("Home", "Away"):
        df = pd.read_csv(root / f"Sample_Game_{g}_RawTrackingData_{side}_Team.csv", skiprows=2)
        cols = list(df.columns)
        player_cols = [c for c in cols if c.startswith("Player")]
        xy = np.stack([np.stack([df[c].to_numpy(float), df[cols[cols.index(c) + 1]].to_numpy(float)],
                                axis=-1) for c in player_cols], axis=1)
        bi = cols.index("Ball")
        ball = np.stack([df[cols[bi]].to_numpy(float), df[cols[bi + 1]].to_numpy(float)], axis=-1)
        sides[side] = (df["Period"].to_numpy(np.int8), df["Frame"].to_numpy(np.int64), xy, ball)
    period, frame, home, ball = sides["Home"]
    away = sides["Away"][2]
    events = []
    with open(root / f"Sample_Game_{g}_RawEventsData.csv") as f:
        for r in csv.DictReader(f):
            team = 1 if r["Team"] == "Home" else -1
            events.append((r["Type"], r["Subtype"] or "", team, int(r["Start Frame"]),
                           int(r["End Frame"])))
    T = int(frame.max())
    return _table(T, frame, period, home, away, ball), events


def _epts_game():
    root = metrica_dir() / "Sample_Game_3"
    meta = ET.parse(root / "Sample_Game_3_metadata.xml").getroot()
    teams = [t.get("id") for t in meta.iter("Team")]
    team_of = {p.get("id"): (1 if p.get("teamId") == teams[0] else -1) for p in meta.iter("Player")}
    slots = {t: sorted(pid for pid, tt in team_of.items() if tt == t) for t in (1, -1)}
    col = {t: {pid: j for j, pid in enumerate(slots[t])} for t in slots}
    chan = {c.get("id"): c.get("playerId") for c in meta.iter("PlayerChannel")}
    blocks = []
    for d in meta.iter("DataFormatSpecification"):
        order = []
        for c in d.iter("PlayerChannelRef"):
            pid = chan.get(c.get("playerChannelId"))
            if pid is not None and (not order or order[-1] != pid):
                order.append(pid)
        blocks.append((int(d.get("startFrame")), int(d.get("endFrame")), order))
    lines = (root / "Sample_Game_3_tracking.txt").read_text().splitlines()
    T = len(lines)
    home = np.full((T, len(slots[1]), 2), np.nan)
    away = np.full((T, len(slots[-1]), 2), np.nan)
    ball = np.full((T, 2), np.nan)
    frame = np.zeros(T, np.int64)
    for i, line in enumerate(lines):
        fid, players, b = line.split(":")
        fid = int(fid)
        frame[i] = fid
        order = next((o for s, e, o in blocks if s <= fid <= e), [])
        for pid, g in zip(order, players.split(";")):
            vals = g.split(",")
            if len(vals) < 2 or vals[0] == "NaN":
                continue
            t = team_of[pid]
            (home if t == 1 else away)[i, col[t][pid]] = (float(vals[0]), float(vals[1]))
        bv = b.split(",")
        if len(bv) >= 2 and bv[0] != "NaN":
            ball[i] = (float(bv[0]), float(bv[1]))
    data = json.loads((root / "Sample_Game_3_events.json").read_text())["data"]
    events = []
    for e in data:
        subs = e.get("subtypes")
        subs = [subs] if isinstance(subs, dict) else (subs or [])
        sub = "-".join(s["name"] for s in subs if s)
        team = 1 if (e.get("team") or {}).get("id") == teams[0] else -1
        start = int(e["start"]["frame"])
        end = int((e.get("end") or {}).get("frame") or start)
        events.append((e["type"]["name"], sub, team, start, end))
    # the period of each frame: from the first to the last event of that period
    period = _periods_from_events(data, T)
    return _table(T, frame, period, home, away, ball), events


def _periods_from_events(data, T: int) -> np.ndarray:
    first = {}
    last = {}
    for e in data:
        p = int(e.get("period") or 0)
        s, z = int(e["start"]["frame"]), int((e.get("end") or {}).get("frame") or e["start"]["frame"])
        first[p] = min(first.get(p, s), s)
        last[p] = max(last.get(p, z), z)
    period = np.zeros(T, np.int8)
    for p in sorted(first):
        if p > 0:
            period[first[p] - 1:min(last[p], T)] = p
    return period


def _table(T, frame, period_rows, home, away, ball) -> FrameTable:
    L, W = PITCH
    rows = frame - 1
    H_, A_ = np.full((T, home.shape[1], 2), np.nan), np.full((T, away.shape[1], 2), np.nan)
    B = np.full((T, 2), np.nan)
    P = np.zeros(T, np.int8)
    H_[rows], A_[rows], B[rows] = home, away, ball
    P[rows] = period_rows if len(period_rows) == len(rows) else period_rows[rows]

    def metres(v):
        out = np.empty_like(v)
        out[..., 0] = (v[..., 0] - 0.5) * L
        out[..., 1] = (v[..., 1] - 0.5) * W
        return out.astype(np.float32)

    h, a = metres(H_), metres(A_)
    return FrameTable(pitch=PITCH, ball=metres(B), home=h, away=a, home_speed=speeds(h),
                      away_speed=speeds(a), period=P)


def load_metrica_game(g: str):
    """(FrameTable, labels, set pieces) for sample game g, cached under cache/."""
    cache = CACHE / f"metrica_{g}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        ft = FrameTable(pitch=tuple(float(v) for v in z["pitch"]), ball=z["ball"],
                        home=z["home"], away=z["away"], home_speed=z["home_speed"],
                        away_speed=z["away_speed"], period=z["period"])
        sp = [(str(k), int(r), int(t)) for k, r, t in zip(z["sp_kind"], z["sp_row"], z["sp_team"])]
        return ft, dead_in_gaps(z["y"], ft.ball, ft.period), sp
    ft, events = _epts_game() if g == "3" else _csv_game(g)
    y = metrica_labels(events, ft.period)
    from .labels import X
    y[ft.period == 0] = X
    sp = _set_pieces(events)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pitch=np.array(ft.pitch), ball=ft.ball, home=ft.home, away=ft.away,
             home_speed=ft.home_speed, away_speed=ft.away_speed, period=ft.period, y=y,
             sp_kind=np.array([k for k, _, _ in sp], dtype="U12"),
             sp_row=np.array([r for _, r, _ in sp], dtype=np.int64),
             sp_team=np.array([t for _, _, t in sp], dtype=np.int64))
    return ft, dead_in_gaps(y, ft.ball, ft.period), sp
