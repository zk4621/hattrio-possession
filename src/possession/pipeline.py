"""FrameTable -> label codes."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .change import change_features, change_logit
from .decode import DecoderConfig, decode_periods
from .features import build_features
from .frames import FrameTable
from .sequence import BLEND


def predict_codes(ft: FrameTable, proba_fn: Callable, cfg: DecoderConfig,
                  live_from: dict[int, int] | None = None,
                  change_fn: Callable | None = None, sequence=None) -> np.ndarray:
    """sequence: a SequenceNet whose probabilities are averaged with proba_fn's, or None."""
    F, names = build_features(ft)
    proba = proba_fn(F, names)
    if sequence is not None:
        Fs, _ = build_features(ft.swapped())
        slices = [(a, z) for p, a, z in ft.period_slices() if p > 0]
        proba = BLEND * proba + (1 - BLEND) * sequence.proba(F, Fs, slices)
    seen = F[:, names.index("ball_vis")] == 1
    logit = None
    if change_fn is not None and cfg.change_weight:
        G, gnames = change_features(ft, F, names)
        logit = change_logit(change_fn(np.hstack([F, G]), names + gnames))
    return decode_periods(proba, ft.period, cfg, live_from, seen=seen, change_logit=logit)
