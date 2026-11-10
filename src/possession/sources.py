"""Optional extra ball sources, switched on by models/sources.json (Tasks 9-11).

Format: {"camera_proxy": bool, "proxy_stats": {...}, "broadcast_ball": bool,
"smart_tagging": bool}. An absent models/sources.json, or an absent key, means that source is off.
"""
from __future__ import annotations

import json

from .frames import FrameTable
from .paths import MODELS

SOURCES_PATH = MODELS / "sources.json"


def read_sources() -> dict:
    d = json.loads(SOURCES_PATH.read_text()) if SOURCES_PATH.exists() else {}
    for key in ("camera_proxy", "broadcast_ball", "smart_tagging", "smart_tagging_phases"):
        d.setdefault(key, False)
    return d


def write_sources(**updates) -> dict:
    d = read_sources()
    d.update(updates)
    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(json.dumps(d, indent=1))
    return d


def load_flags() -> dict:
    d = read_sources()
    return {"proxy": bool(d["camera_proxy"]), "broadcast": bool(d["broadcast_ball"]),
            "tags": bool(d["smart_tagging"]), "phases": bool(d["smart_tagging_phases"]),
            "proxy_stats": d.get("proxy_stats")}


def attach_sources(ft: FrameTable) -> FrameTable:
    """Apply the adopted sources to the competition's tactical table."""
    flags = load_flags()
    if flags["broadcast"]:
        from .sync import merged_competition_ball
        ft.ball = merged_competition_ball(ft)
    if flags["proxy"]:
        from .proxy import competition_proxy
        ft.proxy = competition_proxy(ft)
    if flags["tags"] or flags["phases"]:
        from .tagging import PH_KEYS, SP_CODE, competition_tags
        ft.tags = [g for g in competition_tags()
                   if (flags["tags"] and g.code == SP_CODE) or (flags["phases"] and g.code in PH_KEYS)]
    return ft


TAG_SEED = 200


def attach_training_tags(ft: FrameTable, y, set_pieces, index: int) -> FrameTable:
    """Simulated counterparts of the adopted sources on a training match (before degradation).

    set_pieces: [(kind, row, team)] restarts of that match, from its own event data.
    """
    flags = load_flags()
    if flags["proxy"]:
        from .proxy import simulate_proxy
        ft.proxy = simulate_proxy(ft, flags["proxy_stats"], seed=index)
    if flags["tags"] or flags["phases"]:
        from .tagging import dfl_phase_candidates, simulate_phase_tags, simulate_set_piece_tags
        profile = json.loads((MODELS / "tag_profile.json").read_text())
        ft.tags = []
        if flags["tags"]:
            ft.tags += simulate_set_piece_tags(set_pieces, profile, seed=TAG_SEED + index)
        if flags["phases"]:
            ft.tags += simulate_phase_tags(dfl_phase_candidates(ft, y), profile,
                                           seed=TAG_SEED + 50 + index)
    return ft


def attach_dfl_sources(ft: FrameTable, y, match_id: str, index: int) -> FrameTable:
    """attach_training_tags for a DFL match, its restarts read from the DFL event data."""
    flags = load_flags()
    set_pieces = []
    if flags["tags"]:
        from .tagging import dfl_set_pieces
        set_pieces = dfl_set_pieces(match_id, ft.period)
    return attach_training_tags(ft, y, set_pieces, index)
