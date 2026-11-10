"""The common frame table every tracking source is loaded into."""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

FPS = 25


@dataclass
class FrameTable:
    pitch: tuple[float, float]   # (length_m, width_m)
    ball: np.ndarray             # (T, 2) float32, NaN where the ball is not tracked
    home: np.ndarray             # (T, Ph, 2) float32, NaN where a slot is empty
    away: np.ndarray             # (T, Pa, 2)
    home_speed: np.ndarray       # (T, Ph) float32, m/s
    away_speed: np.ndarray       # (T, Pa)
    period: np.ndarray           # (T,) int8: 1, 2, or 0 outside play
    proxy: np.ndarray | None = None   # (T, 2) coarse ball position, NaN where unknown
    tags: list | None = None          # Smart Tagging (possession.tagging.Tag), None if unused

    @property
    def n_frames(self) -> int:
        return int(len(self.period))

    def swapped(self) -> "FrameTable":
        tags = None if self.tags is None else [replace(g, team=-g.team) for g in self.tags]
        return replace(self, home=self.away.copy(), away=self.home.copy(),
                       home_speed=self.away_speed.copy(), away_speed=self.home_speed.copy(),
                       tags=tags)

    def period_slices(self) -> list[tuple[int, int, int]]:
        """(period, start, stop) half-open, for each maximal run of one non-zero period."""
        p = np.asarray(self.period)
        out = []
        if not len(p):
            return out
        breaks = np.flatnonzero(p[1:] != p[:-1]) + 1
        bounds = np.concatenate(([0], breaks, [len(p)]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            if p[a] != 0:
                out.append((int(p[a]), int(a), int(b)))
        return out
