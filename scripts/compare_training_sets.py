"""Which training set is best: DFL only, Metrica only, or both? Judged on the Metrica games.

Every match is predicted by a model that never saw it:
  dfl     - DFL matches: the other 6 DFL matches; Metrica games: all 7 DFL matches
  metrica - Metrica games: the other 2 games; DFL matches: all 3 games
  both    - any match: all other 9 matches
Frame model and change model are trained for each case, as in scripts/train_multi.py.

The decoder is chosen the same way for every training set: for each test match, the setting with
the best mean spellF1 on the other matches of its test set (the other 2 Metrica games, or the
other 6 DFL matches), then applied to that match.

Rule fixed before the run (2026-11-11): the training set with the best mean on the 3 Metrica
games wins; it replaces the submitted model (DFL only) only if it beats DFL only on at least 2 of
the 3 games.
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
from possession.model import change_fn, predict_proba, train, train_change
from possession.paths import MODELS, ROOT
from possession.scoring import score

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

REGIMES = ("dfl", "metrica", "both")


def train_set(regime, test, keys):
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    if regime == "dfl":
        pool = dfl
    elif regime == "metrica":
        pool = met
    else:
        pool = keys
    return tuple(k for k in pool if k != test)


_GRID = None


def _init(data):
    global _GRID
    _GRID = data


def _score_all(item):
    """spellF1 of one prediction under every decoder setting."""
    key, (P, period, y, seen, lg) = item
    out = []
    for t in itertools.product(*tm.CHANGE_GRID):
        cfg = tm._config(t)
        out.append(score(y, decode_periods(P, period, cfg, seen=seen, change_logit=lg))["spell_f1"])
    return key, out


def main() -> int:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = tm.registry(["dfl", "metrica"])
    index = {k: i for i, k in enumerate(keys)}
    rows = {k: tm.training_rows(*k, index[k], profile) for k in keys}
    names, gnames = rows[keys[0]]["names"], rows[keys[0]]["gnames"]
    cnames = names + gnames
    tables = {k: tm.build(*k, index[k], profile) for k in keys}
    print(f"prepared {len(keys)} matches ({time.time() - t0:.0f} s)", flush=True)

    models = {}

    def model_for(train_keys):
        if train_keys not in models:
            Xf = np.concatenate([rows[k]["F"] for k in train_keys])
            yf = np.concatenate([rows[k]["y"] for k in train_keys])
            booster = train(Xf, yf, names)
            del Xf, yf
            Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in train_keys])
            yc = np.concatenate([rows[k]["ct"] for k in train_keys])
            cb = train_change(Xc, yc, cnames)
            del Xc, yc
            models[train_keys] = (booster, cb)
            print(f"trained on {len(train_keys)} matches ({time.time() - t0:.0f} s)", flush=True)
        return models[train_keys]

    oof = {}
    for regime in REGIMES:
        for k in keys:
            booster, cb = model_for(train_set(regime, k, keys))
            ft, y, F, _, G, _ = tables[k]
            oof[(regime, k)] = (predict_proba(booster, F), ft.period, y,
                                F[:, names.index("ball_vis")] == 1,
                                change_logit(change_fn(cb)(np.hstack([F, G]), cnames)))
        models.clear()
        print(f"{regime}: predictions done ({time.time() - t0:.0f} s)", flush=True)

    with Pool(8, initializer=_init, initargs=(None,)) as p:
        scores = dict(p.map(_score_all, list(oof.items())))
    configs = list(itertools.product(*tm.CHANGE_GRID))

    report = {}
    for regime in REGIMES:
        report[regime] = {}
        for test_set in ("metrica", "dfl"):
            members = [k for k in keys if k[0] == test_set]
            for k in members:
                others = [m for m in members if m != k]
                mean_others = np.mean([scores[(regime, m)] for m in others], axis=0)
                best = int(np.argmax(mean_others))
                report[regime][f"{k[0]}:{k[1]}"] = {"spell_f1": scores[(regime, k)][best],
                                                    "decoder": list(configs[best])}
    print("\nspellF1 on each match, by a model that never saw it (decoder chosen on the others)")
    print("training set   metrica 1  metrica 2  metrica 3  | Metrica mean | DFL mean")
    for regime in REGIMES:
        m = [report[regime][f"metrica:{g}"]["spell_f1"] for g in ("1", "2", "3")]
        d = [v["spell_f1"] for kk, v in report[regime].items() if kk.startswith("dfl:")]
        print(f"{regime:<13} {m[0]:9.3f}  {m[1]:9.3f}  {m[2]:9.3f}  | {np.mean(m):12.3f} | "
              f"{np.mean(d):8.3f}")
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(out_dir / "oof.npz", **{f"{r}|{k[0]}:{k[1]}|{i}": a
                                               for (r, k), v in oof.items()
                                               for i, a in enumerate(v) if a is not None})
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
