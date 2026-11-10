"""Train on DFL plus the open SkillCorner and Metrica matches; cross-validate on all of them.

Folds: every DFL match and every Metrica game is held out on its own; the 20 SkillCorner matches
in 5 groups of 4. DFL tracking is clean and gets the measured blackouts; SkillCorner (broadcast)
and Metrica (the provided match's tracker) already lose the ball on ~40 % of frames and are used
as they are. The decoder is tuned on the pooled out-of-fold predictions of all matches.

With --compare, the model in models/ (trained on DFL only) is scored on the same SkillCorner
and Metrica matches, which it never saw, so the two can be compared match by match.

Adoption rule, fixed before the first run (2026-11-10):
  1. mean spellF1 over the held-out SkillCorner and Metrica matches beats the DFL-only model,
  2. on the Metrica games (closest to the provided match) it is better on at least 2 of 3,
  3. the DFL out-of-fold mean stays at or above 0.5885 (0.5935 - 0.005).
"""
import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from possession.change import change_features, change_logit, change_targets
from possession.decode import DecoderConfig, decode_periods
from possession.degrade import DegradeProfile, degrade
from possession.features import build_features
from possession.io_dfl import DFL_MATCH_IDS, load_dfl_match
from possession.io_metrica import GAMES, load_metrica_game
from possession.io_skillcorner import load_skillcorner_match, match_ids
from possession.labels import X, swap_codes
from possession.model import (N_ROUNDS, change_fn, load, load_decoder, predict_proba, proba_fn,
                              save, save_decoder, train, train_change)
from possession.paths import MODELS
from possession.scoring import score
from possession.sources import attach_training_tags, load_flags
from possession.tagging import dfl_set_pieces

SUBSAMPLE = 2
SC_GROUPS = 5
CHANGE_GRID = ([12, 25, 50], [25, 50, 75], [0, 2000, 5000], [0.0, 0.25, 0.5, 1.0])


def registry(datasets):
    out = []
    if "dfl" in datasets:
        out += [("dfl", m) for m in DFL_MATCH_IDS]
    if "metrica" in datasets:
        out += [("metrica", g) for g in GAMES]
    if "skillcorner" in datasets:
        out += [("skillcorner", m) for m in match_ids()]
    return out


def load_match(dataset, mid):
    if dataset == "dfl":
        ft, y = load_dfl_match(mid)
        sp = dfl_set_pieces(mid, ft.period) if load_flags()["tags"] else []
        return ft, y, sp
    if dataset == "metrica":
        return load_metrica_game(mid)
    return load_skillcorner_match(mid)


def build(dataset, mid, index, profile):
    """Tagged, (for DFL) degraded table, its features both ways, change features, labels."""
    ft, y, sp = load_match(dataset, mid)
    ft = attach_training_tags(ft, y, sp, index)
    if dataset == "dfl":
        ft = degrade(ft, profile, seed=index)
    F, names = build_features(ft)
    G, gnames = change_features(ft, F, names)
    return ft, y, F, names, G, gnames


def training_rows(dataset, mid, index, profile):
    ft, y, F, names, G, gnames = build(dataset, mid, index, profile)
    sw = ft.swapped()
    Fs, _ = build_features(sw)
    Gs, _ = change_features(sw, Fs, names)
    ct = change_targets(y, ft.period)
    keep = np.flatnonzero(y != X)[::SUBSAMPLE]
    ys = swap_codes(y)
    return {"F": np.concatenate([F[keep], Fs[keep]]), "y": np.concatenate([y[keep], ys[keep]]),
            "G": np.concatenate([G[keep], Gs[keep]]), "ct": np.concatenate([ct[keep], ct[keep]]),
            "names": names, "gnames": gnames}


def folds(keys):
    out = [[k] for k in keys if k[0] in ("dfl", "metrica")]
    sc = [k for k in keys if k[0] == "skillcorner"]
    out += [sc[i::SC_GROUPS] for i in range(SC_GROUPS) if sc[i::SC_GROUPS]]
    return out


_GRID = None


def _init(data):
    global _GRID
    _GRID = data


def _config(t):
    pmin, dmin, pen, w = t
    return DecoderConfig(min_len=(pmin, pmin, dmin), switch_penalty=pen, change_weight=w)


def _eval(t):
    cfg = _config(t)
    return t, [score(y, decode_periods(P, period, cfg, seen=seen, change_logit=lg))["spell_f1"]
               for P, period, y, seen, lg in _GRID]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["dfl", "metrica", "skillcorner"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--compare", action="store_true",
                    help="also score the DFL-only model in models/ on the held-out extra matches")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = registry(args.datasets)
    index = {k: i for i, k in enumerate(keys)}
    t0 = time.time()

    rows = {}
    for k in keys:
        rows[k] = training_rows(*k, index[k], profile)
    names, gnames = rows[keys[0]]["names"], rows[keys[0]]["gnames"]
    cnames = names + gnames
    print(f"prepared {len(keys)} matches ({time.time() - t0:.0f} s)", flush=True)

    def fit(exclude):
        """Frame model and change model on every match not in `exclude`, one after the other."""
        ks = [k for k in keys if k not in exclude]
        Xf = np.concatenate([rows[k]["F"] for k in ks])
        yf = np.concatenate([rows[k]["y"] for k in ks])
        booster = train(Xf, yf, names, n_rounds=args.rounds)
        del Xf, yf
        Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in ks])
        yc = np.concatenate([rows[k]["ct"] for k in ks])
        cb = train_change(Xc, yc, cnames)
        del Xc, yc
        return booster, cb

    oof = {}
    for fold in folds(keys):
        booster, cb = fit(set(fold))
        for k in fold:
            ft, y, F, _, G, _ = build(*k, index[k], profile)
            oof[k] = (predict_proba(booster, F), ft.period, y, F[:, names.index("ball_vis")] == 1,
                      change_logit(change_fn(cb)(np.hstack([F, G]), cnames)))
        print(f"fold {[f'{d}:{m}' for d, m in fold]} done ({time.time() - t0:.0f} s)", flush=True)

    grid = list(itertools.product(*CHANGE_GRID))
    data = [oof[k] for k in keys]
    with Pool(8, initializer=_init, initargs=(data,)) as p:
        results = p.map(_eval, grid)
    table = sorted(((float(np.mean(v)), t) for t, v in results), key=lambda r: (-r[0], r[1]))
    best = table[0][1]
    cfg = _config(best)
    save_decoder(cfg, out / "decoder.json")
    per = dict(zip(keys, dict(results)[best]))
    print(f"decoder {cfg}: pooled mean {table[0][0]:.4f} ({time.time() - t0:.0f} s)", flush=True)

    report = {"datasets": args.datasets, "decoder": {"min_len": list(cfg.min_len),
              "switch_penalty": cfg.switch_penalty, "change_weight": cfg.change_weight},
              "matches": {f"{d}:{m}": per[(d, m)] for d, m in keys}, "mean": {}}
    for d in args.datasets:
        v = [per[k] for k in keys if k[0] == d]
        report["mean"][d] = float(np.mean(v)) if v else None

    if args.compare:
        old_cfg = load_decoder()
        old = proba_fn(load(MODELS / "possession_lgb.txt"))
        oldc = change_fn(load(MODELS / "change_lgb.txt")) if old_cfg.change_weight else None
        report["dfl_only_model"] = {}
        for k in keys:
            if k[0] == "dfl":
                continue
            ft, y, F, _, G, _ = build(*k, index[k], profile)
            lg = (change_logit(oldc(np.hstack([F, G]), cnames)) if oldc else None)
            codes = decode_periods(old(F, names), ft.period, old_cfg,
                                   seen=F[:, names.index("ball_vis")] == 1, change_logit=lg)
            report["dfl_only_model"][f"{k[0]}:{k[1]}"] = score(y, codes)["spell_f1"]
        print("compared with the DFL-only model", flush=True)

    booster, cb = fit(set())
    save(booster, out / "possession_lgb.txt")
    save(cb, out / "change_lgb.txt")
    (out / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report["mean"], indent=1))
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
