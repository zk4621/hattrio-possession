"""SkillCorner open data (broadcast tracking, 10 fps) -> FrameTable at 25 fps + H/A/D labels.

github.com/SkillCorner/opendata, MIT licence. Players and ball count as observed only where
SkillCorner flags them as detected; extrapolated positions become NaN, so the table carries the
real gaps of broadcast tracking. Labels: dead ball outside the phases of play (SkillCorner defines
phases only while the ball is in play); inside, the team SkillCorner names as in possession on
each frame, carried forward over frames where it names nobody.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .frames import FPS, FrameTable
from .labels import A, D, H
from .paths import CACHE, EXTERNAL

SRC_FPS = 10
HALFTIME_GAP = 250
SET_PIECE_KIND = {"throw_in": "throw", "free_kick": "freekick", "goal_kick": "goalkick",
                  "corner": "corner", "goal": "kickoff"}


def skillcorner_dir() -> Path:
    return EXTERNAL / "skillcorner" / "data" / "matches"


def match_ids() -> list[str]:
    root = skillcorner_dir()
    return sorted(p.name for p in root.iterdir() if p.is_dir()) if root.exists() else []


def _interp(a: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Values of a (n, ...) at fractional indices u; NaN where a needed neighbour is NaN."""
    i0 = np.floor(u).astype(np.int64)
    i1 = np.minimum(i0 + 1, len(a) - 1)
    w = (u - i0).reshape((-1,) + (1,) * (a.ndim - 1))
    out = (1.0 - w) * a[i0] + w * a[i1]
    exact = (u == i0)
    out[exact] = a[i0[exact]]
    return out


def load_skillcorner(folder, match_id: str):
    """(FrameTable at 25 fps, labels, set pieces [(kind, row, team)]) for one match folder."""
    folder = Path(folder)
    meta = json.loads((folder / f"{match_id}_match.json").read_text())
    home_id = meta["home_team"]["id"]
    team_of = {p["id"]: (1 if p["team_id"] == home_id else -1) for p in meta["players"]}
    slots = {1: sorted(i for i, t in team_of.items() if t == 1),
             -1: sorted(i for i, t in team_of.items() if t == -1)}
    col = {t: {pid: j for j, pid in enumerate(ids)} for t, ids in slots.items()}
    pitch = (float(meta.get("pitch_length") or 105.0), float(meta.get("pitch_width") or 68.0))

    frames = {}
    with open(folder / f"{match_id}_tracking_extrapolated.jsonl") as f:
        for line in f:
            x = json.loads(line)
            if x.get("period") in (1, 2):
                frames[int(x["frame"])] = x

    phases = []
    with open(folder / f"{match_id}_phases_of_play.csv") as f:
        for r in csv.DictReader(f):
            phases.append((int(r["frame_start"]), int(r["frame_end"]), int(r["period"]),
                           1 if int(float(r["team_in_possession_id"])) == home_id else -1))

    parts, labels, offsets, row0 = [], [], {}, 0
    for p in (1, 2):
        fs = sorted(f for f, x in frames.items() if x["period"] == p)
        if not fs:
            continue
        f0, f1 = fs[0], fs[-1]
        n10 = f1 - f0 + 1
        ball = np.full((n10, 2), np.nan)
        pos = {t: np.full((n10, len(slots[t]), 2), np.nan) for t in (1, -1)}
        group = np.zeros(n10, np.int8)               # +1 home, -1 away, 0 nobody
        for f in fs:
            x = frames[f]
            i = f - f0
            b = x.get("ball_data") or {}
            if b.get("is_detected") and b.get("x") is not None and b.get("y") is not None:
                ball[i] = (b["x"], b["y"])
            for pl in x.get("player_data") or []:
                t = team_of.get(pl.get("player_id"))
                if t is not None and pl.get("is_detected"):
                    pos[t][i, col[t][pl["player_id"]]] = (pl["x"], pl["y"])
            g = (x.get("possession") or {}).get("group")
            group[i] = 1 if g == "home team" else -1 if g == "away team" else 0
        live = np.zeros(n10, bool)
        phase_team = np.zeros(n10, np.int8)
        for a, z, pp, team in phases:
            if pp != p:
                continue
            lo, hi = max(a - f0, 0), min(z - f0 + 1, n10)
            if hi > lo:
                live[lo:hi] = True
                phase_team[lo:hi] = team
        lab10 = np.full(n10, D, np.int8)
        current = 0
        for i in range(n10):
            if not live[i]:
                current = 0
                continue
            if group[i] != 0:
                current = group[i]
            elif current == 0:
                current = phase_team[i]
            lab10[i] = H if current == 1 else A
        n25 = int(np.floor((n10 - 1) * FPS / SRC_FPS + 1e-9)) + 1
        u = np.arange(n25) * SRC_FPS / FPS
        nearest = np.minimum(np.floor(u + 0.5).astype(np.int64), n10 - 1)
        if parts:
            parts.append(None)                       # halftime gap
            row0 += HALFTIME_GAP
        parts.append((p, _interp(ball, u), {t: _interp(pos[t], u) for t in pos}))
        labels.append(lab10[nearest])
        offsets[p] = (row0, f0)
        row0 += n25

    T = row0
    ball = np.full((T, 2), np.nan, np.float32)
    home = np.full((T, len(slots[1]), 2), np.nan, np.float32)
    away = np.full((T, len(slots[-1]), 2), np.nan, np.float32)
    period = np.zeros(T, np.int8)
    y = np.full(T, D, np.int8)
    k = 0
    for part in parts:
        if part is None:
            continue
        p, b, pos = part
        r0, _ = offsets[p]
        n = len(b)
        ball[r0:r0 + n], home[r0:r0 + n], away[r0:r0 + n] = b, pos[1], pos[-1]
        period[r0:r0 + n] = p
        y[r0:r0 + n] = labels[k]
        k += 1
    from .labels import X
    y[period == 0] = X
    ft = FrameTable(pitch=pitch, ball=ball, home=home, away=away,
                    home_speed=speeds(home), away_speed=speeds(away), period=period)

    set_pieces = []
    for p, (r0, f0) in offsets.items():               # kick-off at the start of each period
        rows = np.flatnonzero((period == p) & (y != D))
        if len(rows):
            set_pieces.append(("kickoff", int(rows[0]), 1 if y[rows[0]] == H else -1))
    with open(folder / f"{match_id}_dynamic_events.csv") as f:
        for r in csv.DictReader(f):
            gib = r.get("game_interruption_before") or ""
            if r.get("event_type") != "player_possession" or not gib:
                continue
            base, _, side = gib.rpartition("_")
            kind = SET_PIECE_KIND.get(base)
            p = int(float(r["period"])) if r.get("period") else 0
            if kind is None or p not in offsets:
                continue
            r0, f0 = offsets[p]
            row = r0 + int(np.floor((int(float(r["frame_start"])) - f0) * FPS / SRC_FPS + 0.5))
            team = 1 if int(float(r["team_id"])) == home_id else -1
            set_pieces.append((kind, row, team if side == "for" else -team))
    set_pieces.sort(key=lambda e: e[1])
    return ft, y, set_pieces


def speeds(pos: np.ndarray) -> np.ndarray:
    """Centred speed in m/s of (T, P, 2) positions; NaN where a neighbour is missing."""
    out = np.full(pos.shape[:2], np.nan, np.float32)
    if len(pos) > 2:
        out[1:-1] = np.linalg.norm(pos[2:] - pos[:-2], axis=2) * FPS / 2
    return out


def load_skillcorner_match(match_id: str):
    """One SkillCorner match from external/skillcorner, cached under cache/."""
    cache = CACHE / f"sc_{match_id}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        ft = FrameTable(pitch=tuple(float(v) for v in z["pitch"]), ball=z["ball"],
                        home=z["home"], away=z["away"], home_speed=z["home_speed"],
                        away_speed=z["away_speed"], period=z["period"])
        sp = [(str(k), int(r), int(t)) for k, r, t in zip(z["sp_kind"], z["sp_row"], z["sp_team"])]
        return ft, z["y"], sp
    ft, y, sp = load_skillcorner(skillcorner_dir() / match_id, match_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, pitch=np.array(ft.pitch), ball=ft.ball, home=ft.home, away=ft.away,
             home_speed=ft.home_speed, away_speed=ft.away_speed, period=ft.period, y=y,
             sp_kind=np.array([k for k, _, _ in sp], dtype="U12"),
             sp_row=np.array([r for _, r, _ in sp], dtype=np.int64),
             sp_team=np.array([t for _, _, t in sp], dtype=np.int64))
    return ft, y, sp
