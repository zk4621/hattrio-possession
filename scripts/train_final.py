"""Train the submitted model on DFL + Metrica and pick its decoder on the Metrica games.

Uses the out-of-fold predictions of scripts/compare_training_sets.py (training set "both"): the
decoder setting with the best mean spellF1 over the three Metrica games, which are the closest
to the provided match (same tracking provider, the ball missing on 32-41 % of frames). Then the
frame model and the change model are trained on all 10 matches and written to models/.

    uv run python scripts/compare_training_sets.py <dir>
    uv run python scripts/train_final.py <dir>
"""
import importlib.util
import itertools
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from possession.decode import decode_periods
from possession.degrade import DegradeProfile
from possession.model import save, save_decoder, train, train_change
from possession.paths import MODELS, ROOT
from possession.scoring import score

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

_DATA = None


def _init(data):
    global _DATA
    _DATA = data


def _eval(t):
    cfg = tm._config(t)
    return t, [score(y, decode_periods(P, per, cfg, seen=seen, change_logit=lg))["spell_f1"]
               for P, per, y, seen, lg in _DATA]


def main() -> int:
    src = Path(sys.argv[1])
    z = np.load(src / "oof.npz")
    games = ["1", "2", "3"]
    data = [tuple(z[f"both|metrica:{g}|{i}"] for i in range(5)) for g in games]
    with Pool(8, initializer=_init, initargs=(data,)) as p:
        results = p.map(_eval, list(itertools.product(*tm.CHANGE_GRID)))
    table = sorted(((float(np.mean(v)), t) for t, v in results), key=lambda r: (-r[0], r[1]))
    best_mean, best = table[0]
    cfg = tm._config(best)
    print(f"decoder {cfg}: mean spellF1 {best_mean:.4f} on the Metrica games (out of fold)",
          flush=True)

    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = tm.registry(["dfl", "metrica"])
    rows = {k: tm.training_rows(*k, i, profile) for i, k in enumerate(keys)}
    names, gnames = rows[keys[0]]["names"], rows[keys[0]]["gnames"]
    Xf = np.concatenate([rows[k]["F"] for k in keys])
    yf = np.concatenate([rows[k]["y"] for k in keys])
    booster = train(Xf, yf, names)
    del Xf, yf
    Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in keys])
    yc = np.concatenate([rows[k]["ct"] for k in keys])
    cb = train_change(Xc, yc, names + gnames)
    save(booster, MODELS / "possession_lgb.txt")
    save(cb, MODELS / "change_lgb.txt")
    save_decoder(cfg, MODELS / "decoder.json")

    comparison = json.loads((src / "comparison.json").read_text())
    report = {"training_matches": [f"{d}:{m}" for d, m in keys],
              "decoder": {"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty,
                          "change_weight": cfg.change_weight,
                          "chosen_on": "mean spellF1 of the 3 Metrica games, out of fold",
                          "mean_on_metrica": best_mean},
              "comparison_of_training_sets": comparison,
              "top20_gain": [[names[j], float(g)] for j, g in
                             sorted(enumerate(booster.feature_importance("gain")),
                                    key=lambda r: -r[1])[:20]]}
    (MODELS / "cv_report.json").write_text(json.dumps(report, indent=1))
    print("models written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
