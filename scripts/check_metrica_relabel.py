"""The Metrica games relabelled (ball missing >= 2 s inside a team's stretch = dead ball, see
possession/io_metrica.py): retrain everything on the corrected labels and compare with the
submission.

The events of the Metrica sample games miss many stoppages (a shot wide, a foul recorded as a
challenge, an offside), so the old labels kept a team in possession for 8-14 minutes per game
while play stood still. The submitted model was trained on those labels (weight 2) and its decoder
and sequence network were chosen on them.

Steps (PyTorch and LightGBM never in one process):
  lgb      LightGBM frame and change models, as submitted (Metrica rows weighted 2): each Metrica
           game predicted by models trained on the 7 DFL matches and the other 2 games, each DFL
           match by models trained on the other 6 and the 3 games; then the final models on all
           10. Writes oof_metrica.npz and oof_dfl_single.npz in the format of
           scripts/test_ensemble.py, and possession_lgb.txt, change_lgb.txt.
  (network: scripts/test_sequence_model.py train <dir E'>, which loads the corrected labels)
  compare  the submitted pipeline (old saved out-of-fold predictions) against the retrained one,
           both scored against the current labels, decoder chosen on the other games of each test
           set; then the final decoder on the 3 Metrica games. Writes the report.

Rule fixed before the run (2026-11-13): the retrained pipeline replaces the submission if
  1. on the 7 DFL matches (labels unchanged) its mean spellF1 is at least the submission's minus
     0.005, and
  2. on the 3 Metrica games scored against the corrected labels its mean is at least the
     submission's.
Settings unchanged: Metrica weight 2, LightGBM and network settings, the decoder grid.

    uv run python scripts/check_metrica_relabel.py lgb <dir D'>                     # ~30 min
    PY=<python with torch>; PYTHONPATH=src $PY scripts/test_sequence_model.py train <dir E'>
    uv run python scripts/check_metrica_relabel.py compare <dir D'> --nn <dir E'> \
        --old-oof <dir D> --old-nn <dir E> --out <dir>
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
from possession.model import change_fn, predict_proba, save, save_decoder, train, train_change
from possession.paths import MODELS, ROOT, vendor_import
from possession.sequence import BLEND

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

W_METRICA = 2.0
TOLERANCE = 0.005
CONFIGS = list(itertools.product(*tm.CHANGE_GRID))
RULE = ("fixed before the run (2026-11-13): adopt if the DFL mean is at least the submission's "
        "minus 0.005 and the mean on the 3 Metrica games, scored against the corrected labels, "
        "is at least the submission's")


def name(k):
    return f"{k[0]}:{k[1]}"


def setup():
    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = tm.registry(["dfl", "metrica"])
    return profile, keys, {k: i for i, k in enumerate(keys)}


def run_lgb(out, smoke, log):
    profile, keys, index = setup()
    rounds = (10, 10) if smoke else (None, None)
    rows = {k: tm.training_rows(*k, index[k], profile) for k in keys}
    names, gnames = rows[keys[0]]["names"], rows[keys[0]]["gnames"]
    cnames = names + gnames
    log(f"prepared {len(keys)} matches")

    def fit(ks):
        wt = np.concatenate([np.full(len(rows[k]["y"]), W_METRICA if k[0] == "metrica" else 1.0)
                             for k in ks])
        kw = {} if rounds[0] is None else {"n_rounds": rounds[0]}
        Xf = np.concatenate([rows[k]["F"] for k in ks])
        yf = np.concatenate([rows[k]["y"] for k in ks])
        booster = train(Xf, yf, names, weight=wt, **kw)
        del Xf, yf
        Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in ks])
        yc = np.concatenate([rows[k]["ct"] for k in ks])
        kw = {} if rounds[1] is None else {"n_rounds": rounds[1]}
        cb = train_change(Xc, yc, cnames, weight=wt, **kw)
        return booster, cb

    met, dfl = {}, {}
    for k in keys:
        booster, cb = fit([m for m in keys if m != k])
        _, _, F, _, G, _ = tm.build(*k, index[k], profile)
        P, pc = predict_proba(booster, F), change_fn(cb)(np.hstack([F, G]), cnames)
        if k[0] == "metrica":
            met |= {f"single|{name(k)}|P": P, f"single|{name(k)}|pc": pc}
        else:
            dfl |= {f"single_dfl|{name(k)}|P": P, f"single_dfl|{name(k)}|pc": pc}
        log(f"{name(k)} predicted out of fold")
    np.savez_compressed(out / "oof_metrica.npz", **met)
    np.savez_compressed(out / "oof_dfl_single.npz", **dfl)
    booster, cb = fit(keys)
    save(booster, out / "possession_lgb.txt")
    save(cb, out / "change_lgb.txt")
    log("final models on all 10 matches saved")


def _scores(item):
    key, (P, period, y, seen, lg, configs) = item
    pm = vendor_import("possession_metrics")
    res = []
    for t in configs:
        r = pm.evaluate(y, decode_periods(P, period, tm._config(t), seen=seen, change_logit=lg))
        res.append((r["spell_f1_mean"], r["spell_f1"][0.7]["f1"]))
    return key, res


def nested(tab, label, members, configs):
    res = {}
    for k in members:
        others = [m for m in members if m != k]
        b = int(np.argmax(np.mean([[a for a, _ in tab[(label, m)]] for m in others], axis=0)))
        res[name(k)] = {"spell_f1": tab[(label, k)][b][0], "strict": tab[(label, k)][b][1],
                        "decoder": list(configs[b])}
    return res


def run_compare(new_oof, new_nn, old_oof, old_nn, out, smoke, log):
    profile, keys, index = setup()
    configs = CONFIGS[:2] if smoke else CONFIGS
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    data = {}
    for k in keys:
        ft, y, F, names, _, _ = tm.build(*k, index[k], profile)
        data[k] = (ft.period, y, F[:, names.index("ball_vis")] == 1)

    def preds(oof_dir, nn_dir):
        zb, zd = np.load(oof_dir / "oof_metrica.npz"), np.load(oof_dir / "oof_dfl_single.npz")
        zn = np.load(nn_dir / "oof_nn_s0.npz")
        res = {}
        for k in keys:
            z, tag = (zb, "single") if k[0] == "metrica" else (zd, "single_dfl")
            P = BLEND * z[f"{tag}|{name(k)}|P"] + (1 - BLEND) * zn[f"nn|{name(k)}|P"]
            res[k] = (P, z[f"{tag}|{name(k)}|pc"])
        return res

    pipes = {"submitted": preds(old_oof, old_nn), "relabelled": preds(new_oof, new_nn)}
    items = []
    for label, pr in pipes.items():
        for k in keys:
            period, y, seen = data[k]
            P, pc = pr[k]
            items.append(((label, k), (P, period, y, seen, change_logit(pc), configs)))
    with Pool(8) as p:
        tab = dict(p.map(_scores, items))
    log("scored")

    report = {"rule": RULE, "smoke": smoke}
    means = {}
    print("\nspellF1, decoder chosen on the other matches of each test set")
    for label in pipes:
        for part, members in (("metrica", met), ("dfl", dfl)):
            r = nested(tab, label, members, configs)
            report[f"{label}_{part}"] = r
            means[(label, part)] = float(np.mean([v["spell_f1"] for v in r.values()]))
            strict = float(np.mean([v["strict"] for v in r.values()]))
            print(f"{label:<11} {part:<8} " + " ".join(f"{v['spell_f1']:.4f}" for v in r.values())
                  + f" | mean {means[(label, part)]:.4f} | strict mean {strict:.4f}")
    dfl_ok = means[("relabelled", "dfl")] >= means[("submitted", "dfl")] - TOLERANCE
    met_ok = means[("relabelled", "metrica")] >= means[("submitted", "metrica")]
    m = np.mean([[a for a, _ in tab[("relabelled", g)]] for g in met], axis=0)
    i = int(np.argmax(m))
    cfg = tm._config(configs[i])
    save_decoder(cfg, out / "decoder.json")
    report |= {"means": {f"{a}_{b}": v for (a, b), v in means.items()},
               "dfl_check_passed": bool(dfl_ok), "metrica_check_passed": bool(met_ok),
               "adopt": bool(dfl_ok and met_ok),
               "decoder": {"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty,
                           "change_weight": cfg.change_weight,
                           "chosen_on": "mean spellF1 of the 3 Metrica games (corrected labels), "
                                        "out of fold", "mean_on_metrica": float(m[i])}}
    log(f"DFL check {dfl_ok}, Metrica check {met_ok}: adopt {report['adopt']}; "
        f"final decoder {cfg}")
    (out / "relabel_report.json").write_text(json.dumps(report, indent=1))
    if not smoke:
        (MODELS / "relabel_report.json").write_text(json.dumps(report, indent=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["lgb", "compare"])
    ap.add_argument("dir", help="lgb: output dir; compare: the lgb step's dir")
    ap.add_argument("--nn", help="compare: dir with the retrained networks' oof_nn_s0.npz")
    ap.add_argument("--old-oof", help="compare: the submission's oof dir (test_ensemble.py)")
    ap.add_argument("--old-nn", help="compare: the submission's network dir")
    ap.add_argument("--out", help="compare: where the report and decoder go")
    ap.add_argument("--smoke", action="store_true", help="checks the code, not the idea")
    args = ap.parse_args()
    t0 = time.time()

    def log(msg):
        print(f"{msg} ({time.time() - t0:.0f} s)", flush=True)

    d = Path(args.dir)
    d.mkdir(parents=True, exist_ok=True)
    if args.step == "lgb":
        run_lgb(d, args.smoke, log)
    else:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        run_compare(d, Path(args.nn), Path(args.old_oof), Path(args.old_nn), out, args.smoke, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
