"""Label codes and the submission CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .paths import vendor_import

H, A, D, X = 0, 1, 2, 3
NAMES = np.array(["H", "A", "D", "X"])


def write_submission(path, codes: np.ndarray) -> None:
    codes = np.asarray(codes)
    if codes.min() < 0 or codes.max() > X:
        raise ValueError("codes outside 0..3")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = NAMES[codes.astype(np.int64)]
    lines = ["frame,label"] + [f"{i},{n}" for i, n in enumerate(names)]
    path.write_text("\n".join(lines) + "\n")


def read_codes(path) -> np.ndarray:
    pm = vendor_import("possession_metrics")
    return pm.label_codes(pm.read_label_csv(path)).astype(np.int8)


def swap_codes(codes) -> np.ndarray:
    codes = np.asarray(codes)
    out = codes.copy()
    out[codes == H] = A
    out[codes == A] = H
    return out
