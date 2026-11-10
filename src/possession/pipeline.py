"""FrameTable -> label codes."""
from __future__ import annotations

from typing import Callable

import numpy as np

from .decode import DecoderConfig, decode_periods
from .features import build_features
from .frames import FrameTable


def predict_codes(ft: FrameTable, proba_fn: Callable, cfg: DecoderConfig,
                  live_from: dict[int, int] | None = None) -> np.ndarray:
    F, names = build_features(ft)
    proba = proba_fn(F, names)
    return decode_periods(proba, ft.period, cfg, live_from)
