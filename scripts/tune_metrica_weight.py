"""More weight for the Metrica games, and stronger regularisation: does it beat DFL + Metrica?

Each Metrica game is predicted by a model trained on the 7 DFL matches and the other 2 Metrica
games, the Metrica rows weighted w times; the decoder is chosen on the other two games, as in
scripts/compare_training_sets.py, whose out-of-fold predictions give the baseline (w = 1).

Rule fixed before the run (2026-11-11): the variant with the best mean on the 3 Metrica games
replaces the submitted model only if it is better on all three games and the mean rises by at
least 0.005; the winner must then also keep the DFL mean within 0.02 of 0.592.

    uv run python scripts/tune_metrica_weight.py <compare_training_sets dir> <out dir>
"""
import importlib.util
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from possession.change import change_logit
from possession.decode import decode_periods
from possession.degrade import DegradeProfile
from possession.model import (CHANGE_PARAMS, LGB_PARAMS, change_fn, predict_proba, train,
                              train_change)
from possession.paths import MODELS, ROOT, vendor_import

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

REG = {"num_leaves": 31, "min_data_in_leaf": 400, "lambda_l2": 5.0}
VARIANTS = [("w2", 2.0, False), ("w3", 3.0, False), ("w5", 5.0, False),
            ("reg_w1", 1.0, True), ("reg_w3", 3.0, True)]
CONFIGS = list(itertools.product(*tm.CHANGE_GRID))


def _scores(item):
    key, (P, period, y, seen, lg) = item
    pm = vendor_import("possession_metrics")
    out = []
    for t in CONFIGS:
        r = pm.evaluate(y, decode_periods(P, period, tm._config(t), seen=seen, change_logit=lg))
        out.append((r["spell_f1_mean"], r["spell_f1"][0.7]["f1"]))
    return key, out


def main() -> int:
    base_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = tm.registry(["dfl", "metrica"])
    index = {k: i for i, k in enumerate(keys)}
    rows = {k: tm.training_rows(*k, index[k], profile) for k in keys}
    names, gnames = rows[keys[0]]["names"], rows[keys[0]]["gnames"]
    cnames = names + gnames
    games = [k for k in keys if k[0] == "metrica"]
    tables = {k: tm.build(*k, index[k], profile) for k in games}
    print(f"prepared ({time.time() - t0:.0f} s)", flush=True)

    z = np.load(base_dir / "oof.npz")
    oof = {("base", g): tuple(z[f"both|metrica:{g[1]}|{i}"] for i in range(5)) for g in games}
    for name, w, reg in VARIANTS:
        params = dict(LGB_PARAMS, **REG) if reg else LGB_PARAMS
        cparams = dict(CHANGE_PARAMS, **REG) if reg else CHANGE_PARAMS
        for g in games:
            ks = [k for k in keys if k != g]
            wt = np.concatenate([np.full(len(rows[k]["y"]), w if k[0] == "metrica" else 1.0)
                                 for k in ks])
            Xf = np.concatenate([rows[k]["F"] for k in ks])
            yf = np.concatenate([rows[k]["y"] for k in ks])
            booster = train(Xf, yf, names, params=params, weight=wt)
            del Xf, yf
            Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in ks])
            yc = np.concatenate([rows[k]["ct"] for k in ks])
            cb = train_change(Xc, yc, cnames, weight=wt, params=cparams)
            del Xc, yc
            ft, y, F, _, G, _ = tables[g]
            oof[(name, g)] = (predict_proba(booster, F), ft.period, y,
                              F[:, names.index("ball_vis")] == 1,
                              change_logit(change_fn(cb)(np.hstack([F, G]), cnames)))
        print(f"{name} done ({time.time() - t0:.0f} s)", flush=True)

    with Pool(8) as p:
        sc = dict(p.map(_scores, list(oof.items())))
    report = {}
    print("\nMetrica games, each predicted by a model that never saw it (decoder chosen on the others)")
    print("variant    game 1  game 2  game 3 |  mean | strict (70 % overlap) mean")
    for name in ["base"] + [v[0] for v in VARIANTS]:
        res = {}
        for g in games:
            others = [h for h in games if h != g]
            mean_others = np.mean([[a for a, _ in sc[(name, h)]] for h in others], axis=0)
            best = int(np.argmax(mean_others))
            res[g[1]] = {"spell_f1": sc[(name, g)][best][0], "strict": sc[(name, g)][best][1],
                         "decoder": list(CONFIGS[best])}
        report[name] = res
        m = [res[x]["spell_f1"] for x in ("1", "2", "3")]
        st = [res[x]["strict"] for x in ("1", "2", "3")]
        print(f"{name:<9} {m[0]:7.3f} {m[1]:7.3f} {m[2]:7.3f} | {np.mean(m):5.3f} | {np.mean(st):5.3f}")
    (out_dir / "weights.json").write_text(json.dumps(report, indent=1))
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
