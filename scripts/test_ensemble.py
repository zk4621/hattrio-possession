"""Path A: an ensemble (several models averaged) against the single submitted model.

Every member is trained like the submitted model (7 DFL matches and the Metrica games, the
Metrica rows weighted 2); the members' frame probabilities and change probabilities are averaged
before the decoder. Three ways to make the 5 members differ:
  seeds     only LightGBM's random seed (it subsamples rows and features)
  resim     the seed and the simulated randomness of the training data: fresh ball blackouts on
            the DFL matches, fresh simulated Smart Tagging on every training match
  leaveout  the seed, and member i leaves DFL match i out (as proposed in the team summary)
Member 0 of seeds and of resim is the submitted model itself (seed 0, the same data).

Evaluation as in scripts/check_weight2.py: each Metrica game is predicted by members trained on
the DFL matches and the other two Metrica games; the decoder is chosen on the other two games.

Rule fixed before the run (2026-11-11): the variant with the best mean on the 3 Metrica games
replaces the single model only if it is better on all three games and the mean rises by at least
0.005. Then the DFL check: each DFL match predicted by the same kind of ensemble trained on the
other 6 DFL matches and the 3 Metrica games, decoder chosen on the other 6; the mean must stay at
or above 0.572. The final ensemble (all 10 matches) and its decoder go to <out>/, never to
models/; cv_report.json says whether both checks passed.

    uv run python scripts/test_ensemble.py <out dir>            # ~1 h, 2 h if the DFL check runs
    uv run python scripts/test_ensemble.py <out dir> --smoke    # checks the code, not the idea
"""
import argparse
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
from possession.model import (CHANGE_PARAMS, LGB_PARAMS, N_CHANGE_ROUNDS, N_ROUNDS, change_fn,
                              predict_proba, save, save_decoder, train, train_change)
from possession.paths import MODELS, ROOT, vendor_import

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

W_METRICA = 2.0
VARIANTS = ("seeds", "resim", "leaveout")
RESIM_STEP = 1000       # member m > 0 of resim simulates its training data with seed index + 1000 m
MIN_GAIN = 0.005
DFL_FLOOR = 0.572
CONFIGS = list(itertools.product(*tm.CHANGE_GRID))


def _scores(item):
    key, (P, period, y, seen, lg) = item
    pm = vendor_import("possession_metrics")
    out = []
    for t in CONFIGS:
        r = pm.evaluate(y, decode_periods(P, period, tm._config(t), seen=seen, change_logit=lg))
        out.append((r["spell_f1_mean"], r["spell_f1"][0.7]["f1"]))
    return key, out


def nested(tab, name, members):
    """Each match scored with the decoder that is best on the other matches of `members`."""
    res = {}
    for k in members:
        others = [m for m in members if m != k]
        mean_others = np.mean([[a for a, _ in tab[(name, m)]] for m in others], axis=0)
        b = int(np.argmax(mean_others))
        res[f"{k[0]}:{k[1]}"] = {"spell_f1": tab[(name, k)][b][0], "strict": tab[(name, k)][b][1],
                                "decoder": list(CONFIGS[b])}
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--smoke", action="store_true",
                    help="2 members, 10 rounds, every phase forced: checks the code, not the idea")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_members = 2 if args.smoke else 5
    rounds = (10, 10) if args.smoke else (N_ROUNDS, N_CHANGE_ROUNDS)
    t0 = time.time()

    def log(msg):
        print(f"{msg} ({time.time() - t0:.0f} s)", flush=True)

    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = tm.registry(["dfl", "metrica"])
    index = {k: i for i, k in enumerate(keys)}
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    base = {k: tm.training_rows(*k, index[k], profile) for k in keys}
    names, gnames = base[keys[0]]["names"], base[keys[0]]["gnames"]
    cnames = names + gnames
    log(f"prepared {len(keys)} matches")

    def resim_rows(m):
        return {k: tm.training_rows(*k, index[k] + RESIM_STEP * m, profile) for k in keys}

    def fit(rows, ks, seed):
        """Frame model and change model, as in check_weight2.py, with LightGBM seed `seed`."""
        wt = np.concatenate([np.full(len(rows[k]["y"]), W_METRICA if k[0] == "metrica" else 1.0)
                             for k in ks])
        Xf = np.concatenate([rows[k]["F"] for k in ks])
        yf = np.concatenate([rows[k]["y"] for k in ks])
        booster = train(Xf, yf, names, params=dict(LGB_PARAMS, seed=seed), n_rounds=rounds[0],
                        weight=wt)
        del Xf, yf
        Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in ks])
        yc = np.concatenate([rows[k]["ct"] for k in ks])
        cb = train_change(Xc, yc, cnames, n_rounds=rounds[1], weight=wt,
                          params=dict(CHANGE_PARAMS, seed=seed))
        return booster, cb

    def training_keys(variant, test, m):
        """Training matches of member m predicting `test` (None: the final ensemble)."""
        ks = [k for k in keys if k != test]
        if variant == "leaveout":
            ks.remove([d for d in dfl if d != test][m])
        return ks

    tables = {}

    def table(k):
        if k not in tables:
            ft, y, F, _, G, _ = tm.build(*k, index[k], profile)
            tables[k] = (ft.period, y, np.asarray(F, np.float32), np.asarray(G, np.float32))
        return tables[k]

    def predict(model, k):
        booster, cb = model
        _, _, F, G = table(k)
        return predict_proba(booster, F), change_fn(cb)(np.hstack([F, G]), cnames)

    sums = {}

    def add(name, k, pred):
        s = sums.setdefault((name, k), [0.0, 0.0, 0])
        s[0], s[1], s[2] = s[0] + pred[0], s[1] + pred[1], s[2] + 1

    def save_oof(path, names_, ks):
        """Averaged out-of-fold frame and change probabilities, for later blends (path B)."""
        arrays = {}
        for name in names_:
            for k in ks:
                sP, spc, n = sums[(name, k)]
                arrays[f"{name}|{k[0]}:{k[1]}|P"] = sP / n
                arrays[f"{name}|{k[0]}:{k[1]}|pc"] = spc / n
        np.savez_compressed(path, **arrays)

    def items(name, ks):
        res = []
        for k in ks:
            sP, spc, n = sums[(name, k)]
            period, y, F, _ = table(k)
            res.append(((name, k), (sP / n, period, y, F[:, names.index("ball_vis")] == 1,
                                    change_logit(spc / n))))
        return res

    # 1. the Metrica games, every variant, member by member
    for m in range(n_members):
        for g in met:
            pred = predict(fit(base, training_keys("seeds", g, m), m), g)
            add("seeds", g, pred)
            if m == 0:
                add("single", g, pred)
                add("resim", g, pred)
            add("leaveout", g, predict(fit(base, training_keys("leaveout", g, m), m), g))
        if m:
            rows = resim_rows(m)
            for g in met:
                add("resim", g, predict(fit(rows, training_keys("resim", g, m), m), g))
            del rows
        log(f"member {m + 1}/{n_members} of every variant")

    names_met = ("single",) + VARIANTS
    save_oof(out / "oof_metrica.npz", names_met, met)
    with Pool(8) as p:
        tab = dict(p.map(_scores, [it for v in names_met for it in items(v, met)]))
    report = {name: nested(tab, name, met) for name in names_met}

    def per_game(name, field="spell_f1"):
        return [report[name][f"metrica:{g[1]}"][field] for g in met]

    print("\nMetrica games, each predicted by models that never saw it "
          "(decoder chosen on the other two)")
    print("variant     game 1  game 2  game 3 |  mean | strict (70 % overlap) mean")
    for name in names_met:
        v, s = per_game(name), per_game(name, "strict")
        print(f"{name:<10} {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} | {np.mean(v):.4f} | {np.mean(s):.4f}")

    best = max(VARIANTS, key=lambda v: np.mean(per_game(v)))
    a, b = per_game("single"), per_game(best)
    passed = all(x > y for x, y in zip(b, a)) and np.mean(b) - np.mean(a) >= MIN_GAIN
    report.update(best_variant=best, metrica_gain=float(np.mean(b) - np.mean(a)),
                  metrica_rule_passed=bool(passed), members=n_members, smoke=args.smoke)
    log(f"best variant {best}: {np.mean(b):.4f} vs single {np.mean(a):.4f}, "
        f"better on {sum(x > y for x, y in zip(b, a))}/3; rule passed: {passed}")
    if not (passed or args.smoke):
        (out / "cv_report.json").write_text(json.dumps(report, indent=1))
        log("done, the single model stays")
        return 0

    # 2. DFL check and the final members, member by member (resim rebuilds each member's data once)
    for m in range(n_members):
        rows = base if (best != "resim" or m == 0) else resim_rows(m)
        for d in dfl:
            add(f"{best}_dfl", d, predict(fit(rows, training_keys(best, d, m), m), d))
        booster, cb = fit(rows, training_keys(best, None, m), m)
        save(booster, out / f"possession_lgb_{m}.txt")
        save(cb, out / f"change_lgb_{m}.txt")
        del rows
        log(f"DFL check and final model, member {m + 1}/{n_members}")

    save_oof(out / "oof_dfl.npz", (f"{best}_dfl",), dfl)
    with Pool(8) as p:
        tab.update(dict(p.map(_scores, items(f"{best}_dfl", dfl))))
    report["dfl"] = nested(tab, f"{best}_dfl", dfl)
    dfl_mean = float(np.mean([r["spell_f1"] for r in report["dfl"].values()]))
    means = np.mean([[x for x, _ in tab[(best, g)]] for g in met], axis=0)
    i = int(np.argmax(means))
    cfg = tm._config(CONFIGS[i])
    save_decoder(cfg, out / "decoder.json")
    report.update(dfl_mean=dfl_mean, dfl_check_passed=bool(dfl_mean >= DFL_FLOOR),
                  adopt=bool(passed and dfl_mean >= DFL_FLOOR),
                  training_matches={m: [f"{d}:{k}" for d, k in training_keys(best, None, m)]
                                    for m in range(n_members)},
                  decoder={"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty,
                           "change_weight": cfg.change_weight,
                           "chosen_on": "mean spellF1 of the 3 Metrica games, out of fold",
                           "mean_on_metrica": float(means[i])})
    (out / "cv_report.json").write_text(json.dumps(report, indent=1))
    log(f"DFL mean {dfl_mean:.4f} (must be >= {DFL_FLOOR}); final decoder {cfg}; "
        f"adopt: {report['adopt']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
