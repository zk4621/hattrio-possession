"""Train the possession model on degraded DFL matches, leave-one-match-out CV, decoder tuning.

Writes models/possession_lgb.txt, models/decoder.json and models/cv_report.json.
Only public DFL data is used here; the competition match is never touched.
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool

import numpy as np

from possession.change import change_features, change_logit, change_targets
from possession.decode import DecoderConfig, decode_periods
from possession.degrade import DEFAULT_PROFILE, DegradeProfile, degrade
from possession.features import build_features
from possession.io_dfl import DFL_MATCH_IDS, load_dfl_match
from possession.labels import X, swap_codes
from possession.model import (N_ROUNDS, change_fn, predict_proba, save, save_decoder, train,
                              train_change)
from possession.paths import MODELS
from possession.rules import baseline_codes, rule_proba
from possession.scoring import score
from possession.sources import attach_dfl_sources, load_flags

POSS_MIN = [6, 12, 25, 38, 50]
DEAD_MIN = [25, 50, 75, 125]
PENALTIES = [0, 2000, 5000, 10000, 20000]
# --seen-grid: a shorter possession minimum where the ball is seen. Tried on 2026-11-10: mean CV
# spell_f1 0.5873 -> 0.5880, better on 3 of 7 matches; not adopted.
SEEN_GRID = ([6, 12, 25, 38, 50], [25, 38, 50, 75], [25, 50, 75], [0, 2000, 5000])
# --change: a second model scores each frame for "the label changes here"; the decoder's switch
# cost then varies per frame. Grid: possession minimum, dead-ball minimum, base penalty, weight.
CHANGE_GRID = ([12, 25, 50], [25, 50, 75], [0, 2000, 5000], [0.0, 0.25, 0.5, 1.0])
SUBSAMPLE = 2


def prepare(match_id, i, profile, flags, drop, with_change=False):
    ft, y = load_dfl_match(match_id)
    ft = attach_dfl_sources(ft, y, match_id, i)
    deg = degrade(ft, profile, seed=i) if profile is not None else ft
    blacked = (ft.period > 0) & np.isfinite(ft.ball[:, 0]) & ~np.isfinite(deg.ball[:, 0])
    F, names = build_features(deg)
    Fs, _ = build_features(deg.swapped())
    seen = F[:, names.index("ball_vis")] == 1
    out = {"id": match_id, "ft": deg, "y": y, "ys": swap_codes(y), "blacked": blacked,
           "seen": seen}
    if with_change:
        out["G"], out["gnames"] = change_features(deg, F, names)
        out["Gs"], _ = change_features(deg.swapped(), Fs, names)
        out["ct"] = change_targets(y, deg.period)
    keep = [j for j, n in enumerate(names) if n not in drop]
    out.update(F=F[:, keep], Fs=Fs[:, keep], names=[names[j] for j in keep])
    return out


def change_rows(matches):
    Xs, ys = [], []
    for m in matches:
        for F, G in ((m["F"], m["G"]), (m["Fs"], m["Gs"])):
            ok = (m["y"] != X)[::SUBSAMPLE]
            Xs.append(np.hstack([F[::SUBSAMPLE], G[::SUBSAMPLE]])[ok])
            ys.append(m["ct"][::SUBSAMPLE][ok])
    return np.concatenate(Xs), np.concatenate(ys)


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


def _config(cfg_tuple) -> DecoderConfig:
    smin, pmin, dmin, pen, weight = cfg_tuple
    return DecoderConfig(min_len=(pmin, pmin, dmin), switch_penalty=pen, min_len_seen=smin,
                         change_weight=weight)


def _eval_cfg(cfg_tuple):
    cfg = _config(cfg_tuple)
    out = []
    for P, period, y, seen, logit in _GRID_DATA:
        s = score(y, decode_periods(P, period, cfg, seen=seen, change_logit=logit))
        out.append((s["spell_f1"], s["macro_f1"]))
    return cfg_tuple, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=["atd", "default", "none"], default="atd",
                    help="corruption applied to the DFL matches; none = clean tracking")
    ap.add_argument("--out", default=str(MODELS), help="where to write the model and reports")
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--rounds", type=int, default=N_ROUNDS)
    ap.add_argument("--seen-grid", action="store_true",
                    help="also tune a shorter possession minimum where the ball is seen")
    ap.add_argument("--change", action="store_true",
                    help="train the change model and tune its weight in the decoder")
    args = ap.parse_args()
    profile = {"atd": lambda: DegradeProfile.from_json(MODELS / "atd_profile.json"),
               "default": lambda: DEFAULT_PROFILE, "none": lambda: None}[args.profile]()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    flags = load_flags()
    t0 = time.time()
    matches = [prepare(mid, i, profile, flags, set(args.drop), args.change)
               for i, mid in enumerate(DFL_MATCH_IDS)]
    names = matches[0]["names"]
    print(f"prepared {len(matches)} matches, {len(names)} features, "
          f"proxy={flags['proxy']} broadcast={flags['broadcast']} tags={flags['tags']} phases={flags['phases']} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # leave-one-match-out
    oof, oof_logit = {}, {}
    cnames = names + matches[0]["gnames"] if args.change else None
    for k, held in enumerate(matches):
        rest = [m for j, m in enumerate(matches) if j != k]
        Xtr, ytr = training_rows(rest)
        booster = train(Xtr, ytr, names, n_rounds=args.rounds)
        oof[held["id"]] = predict_proba(booster, held["F"])
        del Xtr, ytr, booster
        oof_logit[held["id"]] = None
        if args.change:
            Xc, yc = change_rows(rest)
            cb = train_change(Xc, yc, cnames)
            p = change_fn(cb)(np.hstack([held["F"], held["G"]]), cnames)
            oof_logit[held["id"]] = change_logit(p)
            del Xc, yc, cb
        print(f"fold {held['id']} done ({time.time() - t0:.0f} s)", flush=True)

    # decoder grid on out-of-fold probabilities
    data = [(oof[m["id"]], m["ft"].period, m["y"], m["seen"], oof_logit[m["id"]])
            for m in matches]
    if args.change:
        grid = [(None,) + c for c in itertools.product(*CHANGE_GRID)]
    elif args.seen_grid:
        grid = [c + (0.0,) for c in itertools.product(*SEEN_GRID)]
    else:
        grid = [(None,) + c + (0.0,) for c in itertools.product(POSS_MIN, DEAD_MIN, PENALTIES)]
    with Pool(8, initializer=_init, initargs=(data,)) as pool:
        results = pool.map(_eval_cfg, grid)
    table = []
    for cfg_tuple, per in results:
        table.append((float(np.mean([a for a, _ in per])), float(np.mean([b for _, b in per])),
                      cfg_tuple))
    table.sort(key=lambda r: (-r[0], -r[1], tuple(-1 if v is None else v for v in r[2])))
    best_spell, best_macro, best_tuple = table[0]
    cfg = _config(best_tuple)
    save_decoder(cfg, out / "decoder.json")
    print(f"decoder: {cfg} mean spell_f1 {best_spell:.4f} macro_f1 {best_macro:.4f} "
          f"({time.time() - t0:.0f} s)", flush=True)

    # report
    report = {"profile": args.profile,
              "flags": {k: flags[k] for k in ("proxy", "broadcast", "tags", "phases")}, "dropped": args.drop,
              "decoder": {"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty,
                          "min_len_seen": cfg.min_len_seen, "change_weight": cfg.change_weight},
              "grid_top10": [{"spell_f1": a, "macro_f1": b, "cfg": list(c)} for a, b, c in table[:10]],
              "matches": {}}
    for m in matches:
        y, ft = m["y"], m["ft"]
        codes = decode_periods(oof[m["id"]], ft.period, cfg, seen=m["seen"],
                               change_logit=oof_logit[m["id"]])
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
    save(booster, out / "possession_lgb.txt")
    gain = booster.feature_importance(importance_type="gain")
    order = np.argsort(-gain)[:20]
    report["top20_gain"] = [[names[j], float(gain[j])] for j in order]
    del Xall, yall, booster
    if args.change:
        Xc, yc = change_rows(matches)
        cb = train_change(Xc, yc, cnames)
        save(cb, out / "change_lgb.txt")
        cgain = cb.feature_importance(importance_type="gain")
        report["change_top20_gain"] = [[cnames[j], float(cgain[j])]
                                       for j in np.argsort(-cgain)[:20]]
    (out / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"mean": mean, "decoder": report["decoder"],
                      "top20": [n for n, _ in report["top20_gain"]]}, indent=1))
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
