"""Optional extra ball sources, switched on by models/sources.json (Tasks 9-11).

An absent models/sources.json means both sources are off.
"""
from __future__ import annotations

import json

from .frames import FrameTable
from .paths import MODELS


def load_flags() -> dict:
    path = MODELS / "sources.json"
    if not path.exists():
        return {"proxy": False, "broadcast": False}
    flags = json.loads(path.read_text())
    return {"proxy": bool(flags.get("proxy")), "broadcast": bool(flags.get("broadcast"))}


def attach_sources(ft: FrameTable) -> FrameTable:
    flags = load_flags()
    if flags["broadcast"]:
        from .sync import merged_competition_ball
        ft.ball = merged_competition_ball(ft)
    if flags["proxy"]:
        from .proxy import competition_proxy
        ft.proxy = competition_proxy(ft)
    return ft
