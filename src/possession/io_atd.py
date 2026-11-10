"""Provided tracking (FIFA/EPTS raw text + metadata XML) -> FrameTable.

Never prints frame content: the competition data must not leave this machine.
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from .frames import FrameTable
from .paths import CACHE, data_dir, vendor_import

GAME = "BAR-ATH"


def _slots(meta_path) -> list[int]:
    root = ET.parse(meta_path).getroot()
    teams = [t.get("id") for t in root.iter("Team")]
    if len(teams) < 2:
        raise ValueError("metadata lists fewer than two teams")
    home_id, away_id = teams[0], teams[1]
    out = []
    for p in root.iter("Player"):
        tid = p.get("teamId")
        out.append(0 if tid == home_id else 1 if tid == away_id else -1)
    return out


def _num(tok: str) -> float:
    try:
        v = float(tok)
    except ValueError:
        return math.nan
    return v if math.isfinite(v) else math.nan


def _parse_group(group: str, n: int) -> list[float]:
    vals = [_num(t) for t in group.split(",")] if group else []
    if len(vals) < 2:
        return [math.nan] * n
    if n == 3 and len(vals) == 2:
        vals = vals + [math.nan]
    vals = vals[:n] + [math.nan] * (n - len(vals[:n]))
    if math.isnan(vals[0]) or math.isnan(vals[1]):
        return [math.nan] * n
    return vals


def load_atd(meta_path, raw_path, n_frames: int, pitch: tuple[float, float],
             period_blocks: list[tuple[int, int, int]]) -> FrameTable:
    slot_team = _slots(meta_path)
    n_slots = len(slot_team)
    length, width = float(pitch[0]), float(pitch[1])
    players = np.full((n_frames, n_slots, 3), np.nan, dtype=np.float64)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float64)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\r\n").split(":")
            if len(parts) < 2:
                continue
            try:
                row = int(parts[0]) - 1
            except ValueError:
                continue
            if not 0 <= row < n_frames:
                continue
            groups = parts[1].split(";") if parts[1] else []
            for s, g in enumerate(groups[:n_slots]):
                players[row, s] = _parse_group(g, 3)
            if len(parts) >= 3:
                b = _parse_group(parts[2], 2)
                ball[row] = b[:2]

    def to_m(xy):
        out = np.empty_like(xy)
        out[..., 0] = (xy[..., 0] - 0.5) * length
        out[..., 1] = (xy[..., 1] - 0.5) * width
        return out

    team = np.array(slot_team)
    hs, as_ = np.flatnonzero(team == 0), np.flatnonzero(team == 1)
    pos = to_m(players[..., :2])
    spd = players[..., 2]
    spd = np.where(np.isnan(pos[..., 0]), np.nan, spd)
    period = np.zeros(n_frames, dtype=np.int8)
    for p, first, last in period_blocks:
        period[max(first, 0):min(last + 1, n_frames)] = p
    return FrameTable(
        pitch=(length, width),
        ball=to_m(ball).astype(np.float32),
        home=pos[:, hs].astype(np.float32),
        away=pos[:, as_].astype(np.float32),
        home_speed=spd[:, hs].astype(np.float32),
        away_speed=spd[:, as_].astype(np.float32),
        period=period,
    )


def _save(path: Path, ft: FrameTable) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, pitch=np.array(ft.pitch), ball=ft.ball, home=ft.home, away=ft.away,
             home_speed=ft.home_speed, away_speed=ft.away_speed, period=ft.period)


def _load_cached(path: Path) -> FrameTable:
    z = np.load(path)
    return FrameTable(pitch=tuple(float(v) for v in z["pitch"]), ball=z["ball"], home=z["home"],
                      away=z["away"], home_speed=z["home_speed"], away_speed=z["away_speed"],
                      period=z["period"])


def load_competition(source: str = "tactical") -> FrameTable:
    games = vendor_import("games")
    meta, raw = games.require(["atd_meta", "atd_raw"], GAME, data_dir=str(data_dir()),
                              source=source)
    cache = CACHE / f"atd_{source}.npz"
    if cache.exists() and cache.stat().st_mtime > Path(raw).stat().st_mtime:
        return _load_cached(cache)
    blocks = games.half_video_blocks(GAME) if source == "tactical" else []
    ft = load_atd(meta, raw, games.source_frames(source, GAME), games.GAMES[GAME]["pitch"], blocks)
    _save(cache, ft)
    return ft



HALF_LABELS = {"1st Half": 1, "2nd Half": 2}


def kickoff_tag_frames() -> dict[int, int]:
    """First frame of the Smart Tagging 'Kick Off (Start)' set piece of each half."""
    st = vendor_import("load_smart_tagging")
    events = st.load(data_dir=str(data_dir()))
    out: dict[int, int] = {}
    for e in events:
        tags = {str(k).strip(): str(v or "").strip() for k, v in (e.get("tags") or {}).items()}
        if (str(e.get("code") or "").strip().lower() == "set pieces"
                and tags.get("Type", "").lower() == "kick off (start)"
                and tags.get("Half") in HALF_LABELS):
            p = HALF_LABELS[tags["Half"]]
            out[p] = min(out.get(p, e["start_frame"]), e["start_frame"])
    return out
