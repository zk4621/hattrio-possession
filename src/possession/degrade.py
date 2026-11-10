"""Blackout and player-loss profile: measure it on a table, apply it to a clean table.

The provided tracking loses the ball in long blackouts and tracks fewer than 11 players per team
part of the time. The profile captures that in aggregate numbers only; `degrade` imposes the same
corruption on clean DFL tracking so a model trained there has seen it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace

import numpy as np

from .frames import FrameTable

BOUNDARY_MARGIN = 2.0      # a ball is "at the boundary" if out_margin > -2.0
DEFICIT_BLOCK = 250        # frames
MIN_BLACKOUTS = 20
N_PLAYERS = 11


@dataclass
class DegradeProfile:
    blackout_hazard: list[float]            # [2]: in field, at boundary
    blackout_quantiles: list[list[float]]   # [2][101]
    deficit_pmf: list[float]                # [12]
    jitter_std_m: float
    ball_jitter_std_m: float

    def to_json(self, path) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=1)

    @classmethod
    def from_json(cls, path) -> "DegradeProfile":
        with open(path) as f:
            d = json.load(f)
        return cls(blackout_hazard=[float(v) for v in d["blackout_hazard"]],
                   blackout_quantiles=[[float(v) for v in row] for row in d["blackout_quantiles"]],
                   deficit_pmf=[float(v) for v in d["deficit_pmf"]],
                   jitter_std_m=float(d["jitter_std_m"]),
                   ball_jitter_std_m=float(d["ball_jitter_std_m"]))


def _default_quantiles() -> list[float]:
    pts = [(0, 1), (10, 22), (25, 153), (50, 268), (75, 529), (90, 836), (95, 1278), (99, 3466),
           (100, 5537)]
    q, v = zip(*pts)
    return [float(x) for x in np.interp(np.arange(101), q, v)]


DEFAULT_PROFILE = DegradeProfile(
    blackout_hazard=[1 / 640, 1 / 640],
    blackout_quantiles=[_default_quantiles(), _default_quantiles()],
    deficit_pmf=[0.60, 0.20, 0.10, 0.05, 0.02, 0.03, 0, 0, 0, 0, 0, 0],
    jitter_std_m=0.10,
    ball_jitter_std_m=0.15,
)


# -- helpers -----------------------------------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _context(ft: FrameTable) -> np.ndarray:
    """1 where the ball is within BOUNDARY_MARGIN of a line or outside, else 0."""
    L, W = ft.pitch
    b = ft.ball.astype(np.float64)
    with np.errstate(invalid="ignore"):
        margin = np.maximum(np.abs(b[:, 0]) - L / 2, np.abs(b[:, 1]) - W / 2)
        return (margin > -BOUNDARY_MARGIN).astype(np.int8)


def _second_diffs(x: np.ndarray) -> np.ndarray:
    """Second differences along axis 0 where three consecutive frames are observed."""
    d2 = x[2:] - 2 * x[1:-1] + x[:-2]
    return d2[np.isfinite(d2)]


def _robust_jitter(d2: np.ndarray) -> float:
    if len(d2) == 0:
        return 0.0
    mad = np.median(np.abs(d2 - np.median(d2)))
    return float(1.4826 * mad / np.sqrt(6.0))


def _jitter(arrays: list[np.ndarray]) -> float:
    return _robust_jitter(np.concatenate([_second_diffs(a.astype(np.float64)) for a in arrays]))


def _player_arrays(ft: FrameTable, a: int, z: int) -> list[np.ndarray]:
    return [ft.home[a:z, :, 0], ft.home[a:z, :, 1], ft.away[a:z, :, 0], ft.away[a:z, :, 1]]


# -- measure -----------------------------------------------------------------------------------
def measure_profile(ft: FrameTable) -> DegradeProfile:
    ctx = _context(ft)
    lengths = {0: [], 1: []}
    visible = np.zeros(2)
    pdiffs, bdiffs, ks = [], [], []
    for _, a, z in ft.period_slices():
        vis = np.isfinite(ft.ball[a:z, 0])
        c = ctx[a:z]
        visible += [np.sum(vis & (c == 0)), np.sum(vis & (c == 1))]
        for r0, r1 in _runs(~vis):
            if r0 == 0 or r1 == z - a:
                continue
            lengths[int(c[r0 - 1])].append(r1 - r0)
        for arr in _player_arrays(ft, a, z):
            pdiffs.append(_second_diffs(arr.astype(np.float64)))
        for j in range(2):
            bdiffs.append(_second_diffs(ft.ball[a:z, j].astype(np.float64)))
        for b0 in range(a, z, DEFICIT_BLOCK):
            b1 = min(b0 + DEFICIT_BLOCK, z)
            for team in (ft.home, ft.away):
                n = np.isfinite(team[b0:b1, :, 0]).sum(1).mean()
                ks.append(int(np.clip(round(N_PLAYERS - n), 0, N_PLAYERS)))
    pooled_len = lengths[0] + lengths[1]
    pooled_h = len(pooled_len) / max(visible.sum(), 1)
    pooled_q = (np.percentile(pooled_len, np.arange(101)).tolist() if pooled_len
                else _default_quantiles())
    hazard, quant = [], []
    for c in (0, 1):
        if len(lengths[c]) >= MIN_BLACKOUTS:
            hazard.append(len(lengths[c]) / max(visible[c], 1))
            quant.append(np.percentile(lengths[c], np.arange(101)).tolist())
        else:
            hazard.append(pooled_h)
            quant.append(list(pooled_q))
    pmf = np.bincount(np.array(ks, dtype=np.int64), minlength=N_PLAYERS + 1)[:N_PLAYERS + 1]
    pmf = pmf / max(pmf.sum(), 1)
    return DegradeProfile(
        blackout_hazard=[float(h) for h in hazard],
        blackout_quantiles=[[float(v) for v in row] for row in quant],
        deficit_pmf=[float(v) for v in pmf],
        jitter_std_m=_robust_jitter(np.concatenate(pdiffs) if pdiffs else np.array([])),
        ball_jitter_std_m=_robust_jitter(np.concatenate(bdiffs) if bdiffs else np.array([])),
    )


# -- degrade -----------------------------------------------------------------------------------
def _draw_length(rng, quantiles: list[float]) -> int:
    u = rng.random() * 100.0
    return max(1, int(round(float(np.interp(u, np.arange(101), quantiles)))))


def degrade(ft: FrameTable, profile: DegradeProfile, seed: int) -> FrameTable:
    rng = np.random.default_rng(seed)
    ball = ft.ball.copy()
    home, away = ft.home.copy(), ft.away.copy()
    hs, as_ = ft.home_speed.copy(), ft.away_speed.copy()
    ctx = _context(ft)
    hazard = np.asarray(profile.blackout_hazard)
    slices = ft.period_slices()

    # 1. ball blackouts
    for _, a, z in slices:
        vis = np.isfinite(ft.ball[a:z, 0])
        u = rng.random(z - a)
        trig = np.flatnonzero(vis & (u < hazard[ctx[a:z]]))
        free_from = 0
        for t in trig.tolist():
            if t < free_from:
                continue
            n = _draw_length(rng, profile.blackout_quantiles[int(ctx[a + t])])
            ball[a + t + 1: a + min(t + 1 + n, z - a)] = np.nan
            free_from = t + 1 + n

    # 2. players: blank the k farthest along x from the ball, per block and team
    pmf = np.asarray(profile.deficit_pmf, dtype=np.float64)
    pmf = pmf / pmf.sum()
    for _, a, z in slices:
        for b0 in range(a, z, DEFICIT_BLOCK):
            b1 = min(b0 + DEFICIT_BLOCK, z)
            for pos, spd in ((home, hs), (away, as_)):
                k = int(rng.choice(len(pmf), p=pmf))
                if k == 0:
                    continue
                x0 = ft.home[b0] if pos is home else ft.away[b0]
                tracked = np.flatnonzero(np.isfinite(x0[:, 0]))
                if not len(tracked):
                    continue
                ref = ft.ball[b0, 0]
                if not np.isfinite(ref):
                    ref = np.nanmean(x0[tracked, 0])
                far = tracked[np.argsort(-np.abs(x0[tracked, 0] - ref), kind="stable")][:k]
                pos[b0:b1, far] = np.nan
                spd[b0:b1, far] = np.nan

    # 3. jitter up to the profile's level
    own_p = _jitter([arr for _, a, z in slices for arr in _player_arrays(ft, a, z)]) \
        if slices else 0.0
    own_b = _jitter([ft.ball[a:z, j] for _, a, z in slices for j in range(2)]) if slices else 0.0
    sp = float(np.sqrt(max(profile.jitter_std_m ** 2 - own_p ** 2, 0.0)))
    sb = float(np.sqrt(max(profile.ball_jitter_std_m ** 2 - own_b ** 2, 0.0)))
    if sp > 0:
        home += rng.normal(0, sp, home.shape).astype(np.float32)
        away += rng.normal(0, sp, away.shape).astype(np.float32)
    if sb > 0:
        ball += rng.normal(0, sb, ball.shape).astype(np.float32)

    return replace(ft, ball=ball, home=home, away=away, home_speed=hs, away_speed=as_,
                   proxy=ft.proxy)
