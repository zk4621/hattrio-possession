"""Broadcast timeline -> tactical timeline by matching player positions; ball merge."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from .frames import FrameTable

MATCH_MAX_COST = 2.0      # metres
MIN_PLAYERS = 6
MIN_RUN = 25              # frames
CLUSTER_TOL = 3
MIN_CLUSTER = 5
CHUNK = 4000


def _rowwise_cost(bh, ba, th, ta) -> np.ndarray:
    """Cost of aligned frame pairs: b* (n, Pb, 2), t* (n, Pt, 2) -> (n,)."""
    nearest = []
    for b, t in ((bh, th), (ba, ta)):
        d = np.linalg.norm(b[:, :, None, :] - t[:, None, :, :], axis=-1)   # (n, Pb, Pt)
        d = np.where(np.isnan(d), np.inf, d)
        near = d.min(axis=2)
        near[np.isnan(b[..., 0])] = np.nan
        nearest.append(near)
    allnear = np.concatenate(nearest, axis=1)
    nb = np.isfinite(bh[..., 0]).sum(1) + np.isfinite(ba[..., 0]).sum(1)
    nt = np.isfinite(th[..., 0]).sum(1) + np.isfinite(ta[..., 0]).sum(1)
    cost = np.full(len(allnear), np.inf)
    ok = (nb >= MIN_PLAYERS) & (nt >= MIN_PLAYERS)
    if ok.any():
        with np.errstate(invalid="ignore"):
            cost[ok] = np.nanmedian(allnear[ok], axis=1)
    cost[~np.isfinite(cost)] = np.inf
    return cost


def pair_cost(b_home, b_away, t_home, t_away) -> np.ndarray:
    """Cost of one broadcast frame against every tactical frame."""
    out = np.empty(len(t_home))
    b_home = np.asarray(b_home, np.float64)
    b_away = np.asarray(b_away, np.float64)
    if np.isfinite(b_home[:, 0]).sum() + np.isfinite(b_away[:, 0]).sum() < MIN_PLAYERS:
        out[:] = np.inf
        return out
    # empty broadcast slots never contribute; dropping them changes nothing but the speed
    bh = b_home[np.isfinite(b_home[:, 0])][None]
    ba = b_away[np.isfinite(b_away[:, 0])][None]
    for a in range(0, len(t_home), CHUNK):
        z = min(a + CHUNK, len(t_home))
        n = z - a
        out[a:z] = _rowwise_cost(np.broadcast_to(bh, (n,) + bh.shape[1:]),
                                 np.broadcast_to(ba, (n,) + ba.shape[1:]),
                                 t_home[a:z].astype(np.float64), t_away[a:z].astype(np.float64))
    return out


def _tracked(ft: FrameTable) -> np.ndarray:
    return np.isfinite(ft.home[..., 0]).sum(1) + np.isfinite(ft.away[..., 0]).sum(1)


_POOL_DATA = None


def _anchor_one(i):
    bft, tft = _POOL_DATA
    c = pair_cost(bft.home[i], bft.away[i], tft.home, tft.away)
    j = int(np.argmin(c))
    return j - int(i), float(c[j])


def anchor_offsets(bft: FrameTable, tft: FrameTable, n_anchors: int = 400, workers: int = 8):
    global _POOL_DATA
    good = np.flatnonzero(_tracked(bft) >= MIN_PLAYERS)
    if not len(good):
        return np.array([], int), np.array([], int), np.array([])
    pick = good[np.unique(np.linspace(0, len(good) - 1, n_anchors).round().astype(int))]
    _POOL_DATA = (bft, tft)
    try:
        if workers > 1 and len(pick) > 50:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(workers) as pool:
                res = pool.map(_anchor_one, pick.tolist())
        else:
            res = [_anchor_one(i) for i in pick.tolist()]
    finally:
        _POOL_DATA = None
    return pick, np.array([r[0] for r in res]), np.array([r[1] for r in res])


def candidate_offsets(offsets: np.ndarray, costs: np.ndarray) -> list[int]:
    o = np.sort(offsets[costs < MATCH_MAX_COST])
    if not len(o):
        return []
    clusters, cur = [], [o[0]]
    for v in o[1:]:
        if v - cur[-1] <= CLUSTER_TOL:
            cur.append(v)
        else:
            clusters.append(cur)
            cur = [v]
    clusters.append(cur)
    return [int(np.median(c)) for c in clusters if len(c) >= MIN_CLUSTER]


def _costs_at_offset(bft: FrameTable, tft: FrameTable, d: int) -> np.ndarray:
    Tb, Tt = bft.n_frames, tft.n_frames
    out = np.full(Tb, np.inf)
    idx = np.arange(Tb)
    valid = (idx + d >= 0) & (idx + d < Tt)
    rows = idx[valid]
    for a in range(0, len(rows), CHUNK):
        r = rows[a:a + CHUNK]
        out[r] = _rowwise_cost(bft.home[r].astype(np.float64), bft.away[r].astype(np.float64),
                               tft.home[r + d].astype(np.float64), tft.away[r + d].astype(np.float64))
    return out


def _mapping_and_cost(bft: FrameTable, tft: FrameTable, anchors=None):
    _, offsets, costs = anchors if anchors is not None else anchor_offsets(bft, tft)
    cands = candidate_offsets(offsets, costs)
    Tb = bft.n_frames
    mapping = np.full(Tb, -1, np.int32)
    best = np.full(Tb, np.inf)
    if not cands:
        return mapping, best, cands
    all_costs = np.stack([_costs_at_offset(bft, tft, d) for d in cands])   # (C, Tb)
    k = np.argmin(all_costs, axis=0)
    best = all_costs[k, np.arange(Tb)]
    off = np.array(cands)[k]
    ok = best < MATCH_MAX_COST
    mapping[ok] = (np.arange(Tb) + off)[ok]
    # drop runs of one offset shorter than MIN_RUN
    key = np.where(ok, off, np.iinfo(np.int64).min)
    breaks = np.flatnonzero(key[1:] != key[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [Tb]))
    for a, z in zip(bounds[:-1], bounds[1:]):
        if ok[a] and z - a < MIN_RUN:
            mapping[a:z] = -1
    best[mapping < 0] = np.inf
    return mapping, best, cands


def fit_mapping(bft: FrameTable, tft: FrameTable) -> np.ndarray:
    return _mapping_and_cost(bft, tft)[0]


def merge_ball(tft: FrameTable, bft: FrameTable, mapping: np.ndarray):
    Tt = tft.n_frames
    mapping = np.asarray(mapping)
    mapped = np.flatnonzero(mapping >= 0)
    cost = np.full(len(mapping), np.inf)
    if len(mapped):
        m = mapping[mapped]
        cost[mapped] = _rowwise_cost(bft.home[mapped].astype(np.float64),
                                     bft.away[mapped].astype(np.float64),
                                     tft.home[m].astype(np.float64), tft.away[m].astype(np.float64))
    # best broadcast ball per tactical frame
    bball = np.full((Tt, 2), np.nan)
    bcost = np.full(Tt, np.inf)
    has_ball = mapped[np.isfinite(bft.ball[mapped, 0])]
    order = has_ball[np.argsort(-cost[has_ball], kind="stable")]   # lowest cost written last
    tgt = mapping[order]
    bball[tgt] = bft.ball[order]
    bcost[tgt] = cost[order]
    tball = tft.ball.astype(np.float64)
    tvis = np.isfinite(tball[:, 0])
    both = tvis & np.isfinite(bball[:, 0])
    agree = np.linalg.norm(tball[both] - bball[both], axis=1)
    merged = np.where(tvis[:, None], tball, bball)
    inplay = tft.period > 0 if (tft.period > 0).any() else np.ones(Tt, bool)
    before = int((~tvis & inplay).sum())
    after = int((~np.isfinite(merged[:, 0]) & inplay).sum())
    stats = {
        "mapped_share": float(len(mapped) / max(len(mapping), 1)),
        "agree_n": int(both.sum()),
        "agree_median_m": float(np.median(agree)) if len(agree) else float("nan"),
        "agree_p90_m": float(np.percentile(agree, 90)) if len(agree) else float("nan"),
        "blackout_frames_before": before,
        "blackout_frames_after": after,
        "filled_share": float((before - after) / before) if before else 0.0,
    }
    out = replace(tft, ball=merged.astype(np.float32))
    return out, stats


def competition_mapping(tft: FrameTable, bft: FrameTable) -> np.ndarray:
    """Broadcast -> tactical mapping for the competition match, cached under cache/."""
    from .paths import CACHE
    path = CACHE / "broadcast_mapping.npy"
    if path.exists():
        return np.load(path)
    mapping = fit_mapping(bft, tft)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, mapping)
    return mapping


def merged_competition_ball(tft: FrameTable) -> np.ndarray:
    """The tactical ball with tactical blackouts filled from the aligned broadcast ball."""
    from .io_atd import load_competition
    bft = load_competition("broadcast")
    merged, _ = merge_ball(tft, bft, competition_mapping(tft, bft))
    return merged.ball
