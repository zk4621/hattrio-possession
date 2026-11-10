"""Rule-based probabilities (no training) and the starter baseline on a FrameTable."""
from __future__ import annotations

import numpy as np

from .frames import FrameTable
from .labels import A, D, H, X


def rule_proba(F: np.ndarray, names: list[str]) -> np.ndarray:
    """Football priors, fixed, not tuned. A frame with no evidence is uniform."""
    c = {n: F[:, names.index(n)].astype(np.float64) for n in
         ("ball_vis", "d1_home", "d1_away", "d_diff", "out_margin", "frac_slow_m50")}
    logits = np.zeros((len(F), 3))
    with np.errstate(invalid="ignore"):
        vis = c["ball_vis"] == 1
        close = vis & (np.fmin(c["d1_home"], c["d1_away"]) < 3.0)
        m = np.clip(np.nan_to_num(c["d_diff"]), -5, 5)
        logits[close, H] += 0.8 * m[close]
        logits[close, A] -= 0.8 * m[close]
        logits[vis & (c["out_margin"] > 0.5), D] += 4.0
        logits[c["frac_slow_m50"] > 0.7, D] += 2.0
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def baseline_codes(ft: FrameTable, window: int = 50) -> np.ndarray:
    """Starter baseline: team of the nearest player, centred majority vote, D without votes."""
    b = ft.ball.astype(np.float64)
    dh = np.linalg.norm(ft.home - b[:, None, :], axis=2)
    da = np.linalg.norm(ft.away - b[:, None, :], axis=2)
    with np.errstate(invalid="ignore"):
        mh = np.where(np.isnan(dh).all(1), np.inf, np.nanmin(np.where(np.isnan(dh), np.inf, dh), 1))
        ma = np.where(np.isnan(da).all(1), np.inf, np.nanmin(np.where(np.isnan(da), np.inf, da), 1))
    seen = np.isfinite(b[:, 0]) & (np.isfinite(mh) | np.isfinite(ma))
    home_vote = seen & (mh <= ma)
    away_vote = seen & (ma < mh)
    hc = np.concatenate(([0], np.cumsum(home_vote)))
    ac = np.concatenate(([0], np.cumsum(away_vote)))
    i = np.arange(len(b))
    lo = np.maximum(0, i - window)
    hi = np.minimum(len(b), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    out = np.full(len(b), D, dtype=np.int8)
    any_vote = (h + a) > 0
    out[any_vote] = np.where(h[any_vote] >= a[any_vote], H, A)
    out[ft.period == 0] = X
    return out
