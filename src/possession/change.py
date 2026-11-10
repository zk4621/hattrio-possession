"""Where does the label change? A second model scores every frame for "a change happens here".

The frame model says which state a frame is in; this one says on which frame the state changes,
from features that react on the very frame something happens: the ball turning or speeding up,
the nearest player switching team, the ball crossing a line, the ball appearing or vanishing.
The decoder turns its output into a switch cost per frame: cheap where a change is likely.
"""
from __future__ import annotations

import warnings

import numpy as np

from .frames import FrameTable

RADIUS = 2          # a frame counts as a change frame if a change happens within 2 frames
JUMP = 5            # frames on each side for the "before / after" differences
TURN = 3            # frames on each side for the ball's turn and kick

JUMP_OF = ["near_team", "d_diff", "d1_home", "d1_away", "ball_speed", "out_margin", "ball_vis",
           "mean_speed", "frac_slow"]
CHANGE_NAMES = ([f"{n.replace('_team', '')}_jump_{JUMP}" for n in JUMP_OF]
                + ["ball_turn", "ball_kick", "out_cross", "in_cross",
                   "near_switch_1s", "ddiff_1s", "speed_1s", "vis_1s"])


def change_targets(y: np.ndarray, period: np.ndarray, radius: int = RADIUS) -> np.ndarray:
    """1 on frames within `radius` of a label change inside a period, else 0."""
    y, period = np.asarray(y), np.asarray(period)
    T = len(y)
    out = np.zeros(T, np.int8)
    t = np.arange(1, T)
    change = t[(y[1:] != y[:-1]) & (period[1:] == period[:-1]) & (period[1:] > 0)]
    for c in change:
        a, b = max(c - radius, 0), min(c + radius + 1, T)
        same = period[a:b] == period[c]
        out[a:b][same] = 1
    return out


def _mean_after_minus_before(x: np.ndarray, w: int) -> np.ndarray:
    """mean(x[t+1 .. t+w]) - mean(x[t-w .. t-1]), NaN-ignoring, NaN at the edges."""
    T = len(x)
    ok = np.isfinite(x)
    cs = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    cn = np.concatenate(([0], np.cumsum(ok)))
    t = np.arange(T)
    lo_a, hi_a = t + 1, t + w + 1
    lo_b, hi_b = t - w, t
    valid = (lo_b >= 0) & (hi_a <= T)
    lo_a, hi_a = np.clip(lo_a, 0, T), np.clip(hi_a, 0, T)
    lo_b, hi_b = np.clip(lo_b, 0, T), np.clip(hi_b, 0, T)
    na, nb = cn[hi_a] - cn[lo_a], cn[hi_b] - cn[lo_b]
    with np.errstate(invalid="ignore", divide="ignore"):
        after = np.where(na > 0, (cs[hi_a] - cs[lo_a]) / np.maximum(na, 1), np.nan)
        before = np.where(nb > 0, (cs[hi_b] - cs[lo_b]) / np.maximum(nb, 1), np.nan)
    return np.where(valid, after - before, np.nan)


def _shift(x: np.ndarray, k: int) -> np.ndarray:
    """y[t] = x[t + k], NaN outside."""
    y = np.full_like(x, np.nan, dtype=np.float64)
    if k >= 0:
        y[:len(x) - k] = x[k:]
    else:
        y[-k:] = x[:len(x) + k]
    return y


def _period_change_features(ft: FrameTable, F: np.ndarray, names: list[str],
                            a: int, z: int) -> dict[str, np.ndarray]:
    col = {n: F[a:z, names.index(n)].astype(np.float64) for n in set(JUMP_OF) | {"d_diff"}}
    f = {}
    for n in JUMP_OF:
        f[f"{n.replace('_team', '')}_jump_{JUMP}"] = _mean_after_minus_before(col[n], JUMP)
    b = ft.ball[a:z].astype(np.float64)
    v1 = b - np.stack([_shift(b[:, 0], -TURN), _shift(b[:, 1], -TURN)], axis=1)
    v2 = np.stack([_shift(b[:, 0], TURN), _shift(b[:, 1], TURN)], axis=1) - b
    n1, n2 = np.linalg.norm(v1, axis=1), np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        cos = np.sum(v1 * v2, axis=1) / (n1 * n2)
        moving = (n1 > 0.05) & (n2 > 0.05)
        f["ball_turn"] = np.where(moving, np.arccos(np.clip(cos, -1.0, 1.0)), np.nan)
        f["ball_kick"] = (n2 - n1) * 25.0 / TURN
        om = col["out_margin"]
        before, after = _shift(om, -TURN), _shift(om, TURN)
        f["out_cross"] = np.where(np.isfinite(before) & np.isfinite(after),
                                  ((before <= 0) & (after > 0)).astype(float), np.nan)
        f["in_cross"] = np.where(np.isfinite(before) & np.isfinite(after),
                                 ((before > 0) & (after <= 0)).astype(float), np.nan)
    f["near_switch_1s"] = _mean_after_minus_before(col["near_team"], 25)
    f["ddiff_1s"] = _mean_after_minus_before(col["d_diff"], 25)
    f["speed_1s"] = _mean_after_minus_before(col["ball_speed"], 25)
    f["vis_1s"] = _mean_after_minus_before(col["ball_vis"], 25)
    return f


def change_features(ft: FrameTable, F: np.ndarray, names: list[str]):
    """(T, len(CHANGE_NAMES)) float32 change features, NaN outside the periods."""
    G = np.full((ft.n_frames, len(CHANGE_NAMES)), np.nan, dtype=np.float32)
    for _, a, z in ft.period_slices():
        f = _period_change_features(ft, F, names, a, z)
        for j, n in enumerate(CHANGE_NAMES):
            G[a:z, j] = f[n]
    return G, list(CHANGE_NAMES)


def change_logit(p: np.ndarray) -> np.ndarray:
    """log(p / (1 - p)), clipped away from 0 and 1."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1.0 - p))
