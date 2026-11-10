"""FrameTable -> per-frame feature matrix.

Two families: ball features for the stretches where the ball is seen, and player-only features
plus blackout counters for the long stretches where it is not. Everything is computed per period;
no window reads across a period boundary. Rows outside the periods are NaN.
"""
from __future__ import annotations

import warnings

import numpy as np

from .frames import FrameTable
from .tagging import PH_NAMES, SP_NAMES, tag_features

CLIP = 3000
INTERP_MAX = 12
NEAR_M = 2.0
SLOW_MS = 1.0
VEL_LAG = 12

BALL_NAMES = ["ball_x", "ball_y_abs", "out_margin", "ball_speed", "ball_acc",
              "d1_home", "d2_home", "d1_away", "d2_away", "d_diff", "n3_home", "n3_away"]
BLACKOUT_NAMES = ["ball_vis", "gap_len", "since_seen", "until_seen", "near_team",
                  "last_near", "next_near", "since_near", "until_near", "last_out", "next_out"]
PLAYER_NAMES = ["n_home", "n_away", "cx_home", "cx_away", "cx_gap", "cvx_home", "cvx_away",
                "spread_home", "spread_away", "len_home", "len_away", "wid_home", "wid_away",
                "line_home", "line_away", "speed_home", "speed_away", "mean_speed", "frac_slow"]
BASE_NAMES = BALL_NAMES + BLACKOUT_NAMES + PLAYER_NAMES
WINDOWED = ["ball_vis", "d1_home", "d1_away", "d_diff", "ball_speed", "out_margin", "near_team",
            "mean_speed", "frac_slow", "cvx_home", "cvx_away", "cx_home", "cx_away",
            "spread_home", "spread_away", "speed_home", "speed_away"]
PROXY_NAMES = ["proxy_x", "proxy_y_abs", "proxy_rel_home", "proxy_rel_away", "proxy_vx"]
PROXY_WINDOWED = ["proxy_vx", "proxy_rel_home", "proxy_rel_away"]
WINDOW_SUFFIXES = (["m12", "m50", "m250"] + ["p25", "f25", "p125", "f125", "p500", "f500"])


def window_names(names: list[str]) -> list[str]:
    return [f"{n}_{s}" for n in names for s in WINDOW_SUFFIXES]


# -- small vectorised helpers ------------------------------------------------------------------
def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open runs where mask is True."""
    m = np.concatenate(([0], mask.astype(np.int8), [0]))
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _interp_short(b: np.ndarray, max_len: int) -> np.ndarray:
    b = b.copy()
    miss = np.isnan(b[:, 0])
    T = len(b)
    for a, z in _runs(miss):
        if a == 0 or z == T or z - a > max_len:
            continue
        w = (np.arange(a, z) - (a - 1)) / (z - (a - 1))
        b[a:z] = b[a - 1] + w[:, None] * (b[z] - b[a - 1])
    return b


def _last_index(valid: np.ndarray) -> np.ndarray:
    """Index of the most recent True at or before t, -1 if none."""
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx) if len(idx) else idx


def _next_index(valid: np.ndarray) -> np.ndarray:
    """Index of the next True at or after t, -1 if none."""
    T = len(valid)
    idx = np.where(valid, np.arange(T), T)
    nxt = np.minimum.accumulate(idx[::-1])[::-1] if T else idx
    return np.where(nxt == T, -1, nxt)


def _shift_diff(x: np.ndarray, lag: int) -> np.ndarray:
    """(x[t+lag] - x[t-lag]) with NaN at the edges."""
    out = np.full_like(x, np.nan)
    if len(x) > 2 * lag:
        out[lag:-lag] = x[2 * lag:] - x[:-2 * lag]
    return out


def _window_means(x: np.ndarray) -> list[np.ndarray]:
    """Nine NaN-ignoring window means of a 1-D series, clipped at the series edges."""
    T = len(x)
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(T)

    def mean(lo, hi):  # inclusive [lo, hi], clipped; empty -> NaN
        lo = np.clip(lo, 0, T)
        hi = np.clip(hi + 1, 0, T)
        hi = np.maximum(hi, lo)
        n = cn[hi] - cn[lo]
        s = cs[hi] - cs[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(n > 0, s / np.maximum(n, 1), np.nan)

    out = [mean(t - w, t + w) for w in (12, 50, 250)]
    for w in (25, 125, 500):
        out.append(mean(t - w, t - 1))
        out.append(mean(t + 1, t + w))
    return out


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _nanvar(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanvar(a, axis=axis)


def _nanmax(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmax(a, axis=axis)


def _nanmin(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(a, axis=axis)


def _kth(vals: np.ndarray, k: int) -> np.ndarray:
    """k-th smallest (0-based) per row ignoring NaN; NaN if fewer than k+1 values."""
    s = np.sort(vals, axis=1)  # NaN last
    if vals.shape[1] <= k:
        return np.full(len(vals), np.nan)
    return s[:, k]


# -- one period --------------------------------------------------------------------------------
def _period_features(ft: FrameTable, a: int, z: int, with_proxy: bool) -> dict[str, np.ndarray]:
    L, W = ft.pitch
    home = ft.home[a:z].astype(np.float64)
    away = ft.away[a:z].astype(np.float64)
    hs = ft.home_speed[a:z].astype(np.float64)
    as_ = ft.away_speed[a:z].astype(np.float64)
    T = z - a
    f: dict[str, np.ndarray] = {}

    hx_mean = _nanmean(home[..., 0], 1)
    ax_mean = _nanmean(away[..., 0], 1)
    gap_mean = _nanmean(ax_mean - hx_mean, 0) if T else np.nan
    s = -1.0 if np.isfinite(gap_mean) and gap_mean < 0 else 1.0

    # ball
    b = _interp_short(ft.ball[a:z].astype(np.float64), INTERP_MAX)
    vis = np.isfinite(b[:, 0])
    f["ball_x"] = s * b[:, 0]
    f["ball_y_abs"] = np.abs(b[:, 1])
    out_margin = np.maximum(np.abs(b[:, 0]) - L / 2, np.abs(b[:, 1]) - W / 2)
    f["out_margin"] = out_margin
    v = np.full_like(b, np.nan)
    if T > 2:
        v[1:-1] = (b[2:] - b[:-2]) * 12.5
    f["ball_speed"] = np.linalg.norm(v, axis=1)
    acc = np.full_like(b, np.nan)
    if T > 2:
        acc[1:-1] = (v[2:] - v[:-2]) * 12.5
    f["ball_acc"] = np.linalg.norm(acc, axis=1)

    dh = np.linalg.norm(home - b[:, None, :], axis=2)
    da = np.linalg.norm(away - b[:, None, :], axis=2)
    f["d1_home"], f["d2_home"] = _kth(dh, 0), _kth(dh, 1)
    f["d1_away"], f["d2_away"] = _kth(da, 0), _kth(da, 1)
    f["d_diff"] = f["d1_away"] - f["d1_home"]
    f["n3_home"] = np.where(vis, (dh < 3.0).sum(1), np.nan)
    f["n3_away"] = np.where(vis, (da < 3.0).sum(1), np.nan)

    # blackout counters
    f["ball_vis"] = vis.astype(np.float64)
    gap_len = np.zeros(T)
    for r0, r1 in _runs(~vis):
        gap_len[r0:r1] = r1 - r0
    f["gap_len"] = np.minimum(gap_len, CLIP)
    t = np.arange(T)
    last_vis, next_vis = _last_index(vis), _next_index(vis)
    f["since_seen"] = np.where(last_vis < 0, CLIP, np.minimum(t - last_vis, CLIP))
    f["until_seen"] = np.where(next_vis < 0, CLIP, np.minimum(next_vis - t, CLIP))
    d1h = np.where(np.isnan(f["d1_home"]), np.inf, f["d1_home"])
    d1a = np.where(np.isnan(f["d1_away"]), np.inf, f["d1_away"])
    near = np.zeros(T)
    near[vis & (d1h < NEAR_M) & (d1h < d1a)] = 1
    near[vis & (d1a < NEAR_M) & (d1a < d1h)] = -1
    f["near_team"] = near
    nz = near != 0
    li, ni = _last_index(nz), _next_index(nz)
    f["last_near"] = np.where(li < 0, 0.0, near[np.maximum(li, 0)])
    f["next_near"] = np.where(ni < 0, 0.0, near[np.maximum(ni, 0)])
    f["since_near"] = np.where(li < 0, CLIP, np.minimum(t - li, CLIP))
    f["until_near"] = np.where(ni < 0, CLIP, np.minimum(ni - t, CLIP))
    f["last_out"] = np.where(last_vis < 0, np.nan, out_margin[np.maximum(last_vis, 0)])
    f["next_out"] = np.where(next_vis < 0, np.nan, out_margin[np.maximum(next_vis, 0)])

    # players
    h_ok, a_ok = np.isfinite(home[..., 0]), np.isfinite(away[..., 0])
    nh, na = h_ok.sum(1).astype(np.float64), a_ok.sum(1).astype(np.float64)
    f["n_home"], f["n_away"] = nh, na
    f["cx_home"], f["cx_away"] = s * hx_mean, s * ax_mean
    f["cx_gap"] = f["cx_away"] - f["cx_home"]
    f["cvx_home"] = _shift_diff(f["cx_home"], VEL_LAG) * 25 / 24
    f["cvx_away"] = _shift_diff(f["cx_away"], VEL_LAG) * 25 / 24
    for name, pos in (("home", home), ("away", away)):
        f[f"spread_{name}"] = np.sqrt(_nanvar(pos[..., 0], 1) + _nanvar(pos[..., 1], 1))
        f[f"len_{name}"] = _nanmax(pos[..., 0], 1) - _nanmin(pos[..., 0], 1)
        f[f"wid_{name}"] = _nanmax(pos[..., 1], 1) - _nanmin(pos[..., 1], 1)
    f["line_home"] = _kth(s * home[..., 0], 1)
    f["line_away"] = -_kth(-s * away[..., 0], 1)
    hs = np.where(h_ok, hs, np.nan)
    as_ = np.where(a_ok, as_, np.nan)
    f["speed_home"] = _nanmean(hs, 1)
    f["speed_away"] = _nanmean(as_, 1)
    allspd = np.concatenate([hs, as_], axis=1)
    f["mean_speed"] = _nanmean(allspd, 1)
    known = np.isfinite(allspd)
    n_known = known.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        f["frac_slow"] = np.where(n_known > 0, (known & (allspd < SLOW_MS)).sum(1) / n_known,
                                  np.nan)

    for n in WINDOWED:
        for suffix, arr in zip(WINDOW_SUFFIXES, _window_means(f[n])):
            f[f"{n}_{suffix}"] = arr

    if with_proxy:
        q = ft.proxy[a:z].astype(np.float64)
        f["proxy_x"] = s * q[:, 0]
        f["proxy_y_abs"] = np.abs(q[:, 1])
        f["proxy_rel_home"] = f["proxy_x"] - f["cx_home"]
        f["proxy_rel_away"] = f["proxy_x"] - f["cx_away"]
        f["proxy_vx"] = _shift_diff(f["proxy_x"], VEL_LAG) * 25 / 24
        for n in PROXY_WINDOWED:
            for suffix, arr in zip(WINDOW_SUFFIXES, _window_means(f[n])):
                f[f"{n}_{suffix}"] = arr
    return f


def feature_names(with_proxy: bool, with_tags: bool = False) -> list[str]:
    names = BASE_NAMES + window_names(WINDOWED)
    if with_proxy:
        names = names + PROXY_NAMES + window_names(PROXY_WINDOWED)
    if with_tags:
        names = names + SP_NAMES + PH_NAMES
    return names


def build_features(ft: FrameTable) -> tuple[np.ndarray, list[str]]:
    with_proxy = ft.proxy is not None
    with_tags = ft.tags is not None
    names = feature_names(with_proxy, with_tags)
    F = np.full((ft.n_frames, len(names)), np.nan, dtype=np.float32)
    for _, a, z in ft.period_slices():
        f = _period_features(ft, a, z, with_proxy)
        if with_tags:
            f.update(tag_features(ft.tags, a, z))
        for j, n in enumerate(names):
            F[a:z, j] = f[n]
    return F, names
