"""Optional extra ball sources, switched on by models/sources.json (Tasks 9-11).

Format: {"camera_proxy": bool, "proxy_stats": {...}, "broadcast_ball": bool}.
An absent models/sources.json means both sources are off.
"""
from __future__ import annotations

import json

from .frames import FrameTable
from .paths import MODELS

SOURCES_PATH = MODELS / "sources.json"


def read_sources() -> dict:
    if not SOURCES_PATH.exists():
        return {"camera_proxy": False, "broadcast_ball": False}
    d = json.loads(SOURCES_PATH.read_text())
    d.setdefault("camera_proxy", False)
    d.setdefault("broadcast_ball", False)
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
    return ft
