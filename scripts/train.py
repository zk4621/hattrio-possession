"""Train the possession model on degraded DFL matches, leave-one-match-out CV, decoder tuning.

Writes models/possession_lgb.txt, models/decoder.json and models/cv_report.json.
Only public DFL data is used here; the competition match is never touched.
"""
import argparse
import itertools
import json
import sys
import time
from multiprocessing import Pool

import numpy as np

from possession.decode import DecoderConfig, decode_periods
from possession.degrade import DEFAULT_PROFILE, DegradeProfile, degrade
from possession.features import build_features
from possession.io_dfl import DFL_MATCH_IDS, load_dfl_match
from possession.labels import X, swap_codes
from possession.model import (LGB_PARAMS, N_ROUNDS, predict_proba, save, save_decoder, train)
from possession.paths import MODELS
from possession.rules import baseline_codes, rule_proba
from possession.scoring import score
from possession.sources import attach_dfl_sources, load_flags

POSS_MIN = [6, 12, 25, 38, 50]
DEAD_MIN = [25, 50, 75, 125]
PENALTIES = [0, 2000, 5000, 10000, 20000]
SUBSAMPLE = 2


def prepare(match_id, i, profile, flags, drop):
    ft, y = load_dfl_match(match_id)
    ft = attach_dfl_sources(ft, y, match_id, i)
    deg = degrade(ft, profile, seed=i)
    blacked = (ft.period > 0) & np.isfinite(ft.ball[:, 0]) & ~np.isfinite(deg.ball[:, 0])
    F, names = build_features(deg)
    Fs, _ = build_features(deg.swapped())
    keep = [j for j, n in enumerate(names) if n not in drop]
    names = [names[j] for j in keep]
    return {"id": match_id, "ft": deg, "y": y, "F": F[:, keep], "Fs": Fs[:, keep],
            "ys": swap_codes(y), "names": names, "blacked": blacked}


def training_rows(matches):
    Xs, ys = [], []
    for m in matches:
        for F, y in ((m["F"], m["y"]), (m["Fs"], m["ys"])):
            F, y = F[::SUBSAMPLE], y[::SUBSAMPLE]
            ok = y != X
            Xs.append(F[ok])
            ys.append(y[ok])
    return np.concatenate(Xs), np.concatenate(ys)


_GRID_DATA = None


def _init(data):
    global _GRID_DATA
    _GRID_DATA = data


def _eval_cfg(cfg_tuple):
    pmin, dmin, pen = cfg_tuple
    cfg = DecoderConfig(min_len=(pmin, pmin, dmin), switch_penalty=pen)
    out = []
    for P, period, y in _GRID_DATA:
        s = score(y, decode_periods(P, period, cfg))
        out.append((s["spell_f1"], s["macro_f1"]))
    return cfg_tuple, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["atd", "default"], default="atd")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    args = ap.parse_args()
    profile = (DegradeProfile.from_json(MODELS / "atd_profile.json") if args.profile == "atd"
               else DEFAULT_PROFILE)
    flags = load_flags()
    t0 = time.time()
    matches = [prepare(mid, i, profile, flags, set(args.drop)) for i, mid in enumerate(DFL_MATCH_IDS)]
    names = matches[0]["names"]
    print(f"prepared {len(matches)} matches, {len(names)} features, "
          f"proxy={flags['proxy']} broadcast={flags['broadcast']} tags={flags['tags']} phases={flags['phases']} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # leave-one-match-out
    oof = {}
    for k, held in enumerate(matches):
        Xtr, ytr = training_rows([m for j, m in enumerate(matches) if j != k])
        booster = train(Xtr, ytr, names, n_rounds=args.rounds)
        oof[held["id"]] = predict_proba(booster, held["F"])
        print(f"fold {held['id']}: trained on {len(ytr)} rows ({time.time() - t0:.0f} s)",
              flush=True)
        del Xtr, ytr, booster

    # decoder grid on out-of-fold probabilities
    data = [(oof[m["id"]], m["ft"].period, m["y"]) for m in matches]
    grid = list(itertools.product(POSS_MIN, DEAD_MIN, PENALTIES))
    with Pool(8, initializer=_init, initargs=(data,)) as pool:
        results = pool.map(_eval_cfg, grid)
    table = []
    for cfg_tuple, per in results:
        table.append((float(np.mean([a for a, _ in per])), float(np.mean([b for _, b in per])),
                      cfg_tuple))
    table.sort(key=lambda r: (-r[0], -r[1], r[2]))
    best_spell, best_macro, (pmin, dmin, pen) = table[0]
    cfg = DecoderConfig(min_len=(pmin, pmin, dmin), switch_penalty=pen)
    save_decoder(cfg)
    print(f"decoder: {cfg} mean spell_f1 {best_spell:.4f} macro_f1 {best_macro:.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # report
    report = {"profile": args.profile,
              "flags": {k: flags[k] for k in ("proxy", "broadcast", "tags", "phases")}, "dropped": args.drop,
              "decoder": {"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty},
              "grid_top10": [{"spell_f1": a, "macro_f1": b, "cfg": list(c)} for a, b, c in table[:10]],
              "matches": {}}
    for m in matches:
        y, ft = m["y"], m["ft"]
        codes = decode_periods(oof[m["id"]], ft.period, cfg)
        graded = y != X
        bl, vis = graded & m["blacked"], graded & ~m["blacked"]
        report["matches"][m["id"]] = {
            "baseline": score(y, baseline_codes(ft)),
            "rules": score(y, decode_periods(rule_proba(m["F"], names), ft.period,
                                             DecoderConfig())),
            "model": score(y, codes),
            "model_acc_blacked_out": float((codes[bl] == y[bl]).mean()),
            "model_acc_visible": float((codes[vis] == y[vis]).mean()),
            "blacked_out_share": float(bl.sum() / graded.sum()),
        }
    mean = {}
    for src in ("baseline", "rules", "model"):
        mean[src] = {k: float(np.mean([r[src][k] for r in report["matches"].values()]))
                     for k in ("spell_f1", "macro_f1", "team_acc")}
    for k in ("model_acc_blacked_out", "model_acc_visible"):
        mean[k] = float(np.mean([r[k] for r in report["matches"].values()]))
    report["mean"] = mean

    Xall, yall = training_rows(matches)
    booster = train(Xall, yall, names, n_rounds=args.rounds)
    save(booster, MODELS / "possession_lgb.txt")
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(-gain)[:20]
    report["top20_gain"] = [[names[j], float(gain[j])] for j in order]
    (MODELS / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"mean": mean, "decoder": report["decoder"],
                      "top20": [n for n, _ in report["top20_gain"]]}, indent=1))
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
