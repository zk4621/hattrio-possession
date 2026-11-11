"""Weight 2 for the Metrica games: DFL check, a fair SkillCorner test, and the final model.

1. Metrica games predicted by DFL + the other 2 Metrica games (weight 2): reproduces the winner of
   scripts/tune_metrica_weight.py and gives the predictions the final decoder is chosen on.
2. DFL matches predicted by the other 6 DFL + 3 Metrica games (weight 2): the DFL check (mean must
   stay at or above 0.572, i.e. within 0.02 of 0.592).
3. Metrica games predicted by DFL + the other 2 Metrica games (weight 2) + the 20 SkillCorner
   matches: SkillCorner is added only if better than (1) on all three games and the mean rises by
   at least 0.005 (rule fixed before the run, 2026-11-11).
The decoder is chosen as before: for each test match, on the other matches of its test set.
The final model (training set of the winner) goes to <out>/, not to models/.

    uv run python scripts/check_weight2.py <out dir>
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
from possession.model import (change_fn, predict_proba, save, save_decoder, train,
                              train_change)
from possession.paths import MODELS, ROOT, vendor_import

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

W_METRICA = 2.0
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
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = tm.registry(["dfl", "metrica", "skillcorner"])
    index = {k: i for i, k in enumerate(keys)}
    rows = {k: tm.training_rows(*k, index[k], profile) for k in keys}
    names, gnames = rows[keys[0]]["names"], rows[keys[0]]["gnames"]
    cnames = names + gnames
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    sc = [k for k in keys if k[0] == "skillcorner"]
    print(f"prepared {len(keys)} matches ({time.time() - t0:.0f} s)", flush=True)

    def fit(ks):
        wt = np.concatenate([np.full(len(rows[k]["y"]), W_METRICA if k[0] == "metrica" else 1.0)
                             for k in ks])
        Xf = np.concatenate([rows[k]["F"] for k in ks])
        yf = np.concatenate([rows[k]["y"] for k in ks])
        booster = train(Xf, yf, names, weight=wt)
        del Xf, yf
        Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in ks])
        yc = np.concatenate([rows[k]["ct"] for k in ks])
        cb = train_change(Xc, yc, cnames, weight=wt)
        return booster, cb

    def predict(model, k):
        booster, cb = model
        ft, y, F, _, G, _ = tm.build(*k, index[k], profile)
        return (predict_proba(booster, F), ft.period, y, F[:, names.index("ball_vis")] == 1,
                change_logit(change_fn(cb)(np.hstack([F, G]), cnames)))

    oof = {}
    for g in met:
        oof[("w2", g)] = predict(fit(dfl + [m for m in met if m != g]), g)
    print(f"Metrica, weight 2 ({time.time() - t0:.0f} s)", flush=True)
    for d in dfl:
        oof[("w2", d)] = predict(fit([m for m in dfl if m != d] + met), d)
    print(f"DFL check ({time.time() - t0:.0f} s)", flush=True)
    for g in met:
        oof[("w2_sc", g)] = predict(fit(dfl + [m for m in met if m != g] + sc), g)
    print(f"Metrica, weight 2 + SkillCorner ({time.time() - t0:.0f} s)", flush=True)

    with Pool(8) as p:
        sc_tab = dict(p.map(_scores, list(oof.items())))

    def nested(name, members):
        res = {}
        for k in members:
            others = [m for m in members if m != k]
            mean_others = np.mean([[a for a, _ in sc_tab[(name, m)]] for m in others], axis=0)
            b = int(np.argmax(mean_others))
            res[f"{k[0]}:{k[1]}"] = {"spell_f1": sc_tab[(name, k)][b][0],
                                    "strict": sc_tab[(name, k)][b][1]}
        return res

    report = {"w2_metrica": nested("w2", met), "w2_dfl": nested("w2", dfl),
              "w2_sc_metrica": nested("w2_sc", met)}
    for name, res in report.items():
        v = [r["spell_f1"] for r in res.values()]
        s = [r["strict"] for r in res.values()]
        print(f"{name:<14} " + "  ".join(f"{r['spell_f1']:.3f}" for r in res.values())
              + f" | mean {np.mean(v):.3f} | strict mean {np.mean(s):.3f}")

    a = [report["w2_metrica"][f"metrica:{g}"]["spell_f1"] for g in ("1", "2", "3")]
    b = [report["w2_sc_metrica"][f"metrica:{g}"]["spell_f1"] for g in ("1", "2", "3")]
    use_sc = all(x > y for x, y in zip(b, a)) and np.mean(b) - np.mean(a) >= 0.005
    dfl_mean = float(np.mean([r["spell_f1"] for r in report["w2_dfl"].values()]))
    winner = "w2_sc" if use_sc else "w2"
    means = np.mean([[x for x, _ in sc_tab[(winner, g)]] for g in met], axis=0)
    best = int(np.argmax(means))
    cfg = tm._config(CONFIGS[best])
    print(f"SkillCorner added: {use_sc}; DFL mean {dfl_mean:.3f} (check >= 0.572: "
          f"{dfl_mean >= 0.572}); final decoder {cfg} (Metrica mean {means[best]:.4f})", flush=True)

    final_keys = dfl + met + (sc if use_sc else [])
    booster, cb = fit(final_keys)
    save(booster, out / "possession_lgb.txt")
    save(cb, out / "change_lgb.txt")
    save_decoder(cfg, out / "decoder.json")
    report.update({"skillcorner_added": bool(use_sc), "dfl_mean": dfl_mean,
                   "metrica_weight": W_METRICA,
                   "training_matches": [f"{d}:{m}" for d, m in final_keys],
                   "decoder": {"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty,
                               "change_weight": cfg.change_weight,
                               "chosen_on": "mean spellF1 of the 3 Metrica games, out of fold",
                               "mean_on_metrica": float(means[best])}})
    (out / "cv_report.json").write_text(json.dumps(report, indent=1))
    print(f"done ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
