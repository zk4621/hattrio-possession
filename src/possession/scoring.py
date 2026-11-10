"""Thin wrapper over the official scorer."""
from __future__ import annotations

import numpy as np

from .paths import vendor_import


def score(gt_codes, pred_codes) -> dict:
    pm = vendor_import("possession_metrics")
    r = pm.evaluate(np.asarray(gt_codes, dtype=np.int8), np.asarray(pred_codes, dtype=np.int8))
    return {
        "spell_f1": float(r["spell_f1_mean"]),
        "macro_f1": float(r["macro_f1"]),
        "n_gt_spells": int(r["n_gt_spells"]),
        "n_pred_spells": int(r["n_pred_spells"]),
        "team_acc": float(r["team_acc_gt_alive"]),
    }
