"""Self-training on the match itself: the model's own labels as training data, judged on the
three Metrica games.

Why: the frame model scores 0.89 spellF1 on DFL matches it was trained on and 0.59 on matches it
never saw, so knowing the match is worth a lot. On the match to be labelled, the model names the
right team on most frames where the ball is visible; those frames, labelled by the model itself,
can teach a model this match: how these two teams look when one or the other has the ball, also
when the ball is lost. The challenge rules allow it: pseudo-labels produced by an algorithm,
provided no person decided any of them (third_party/mtah2027/RULES.md, rule 1).

For each test match k, everything comes from models that never saw k's labels:
  1. Pseudo-labels: the submitted pipeline's out-of-fold labels for k. LightGBM trained on the
     other matches (Metrica rows weighted 2; retrained here, and it must reproduce the saved
     out-of-fold prediction), averaged with the sequence network (its saved out-of-fold
     prediction, weight 0.5), the change model's saved out-of-fold probabilities, and the decoder
     chosen on the other games of k's test set.
  2. Rows from k: a copy of k with extra ball blackouts, drawn from the measured profile where the
     ball is visible (players and jitter untouched). The rows are the frames where the ball is
     visible in k itself, in play, every 2nd, both ways round, with the copy's features and the
     pseudo-labels as targets: this match with the ball lost, labelled where the labels are most
     reliable.
  3. Two candidates for the LightGBM frame model (network, change model and decoder grid stay):
       pool   retrained from scratch on its usual matches plus these rows, weighted 2 (as a
              Metrica game)
       tune   the out-of-fold model plus 100 more rounds on these rows alone
  4. k predicted again with the new frame model, averaged with the network as before, decoded
     with the decoder chosen on the other games of its test set (from the candidate's scores).
The labels of k are used only to score.

Rule fixed before the run (2026-11-13): the candidate with the better mean on the 3 Metrica games
is adopted only if it is better than the submitted pipeline on all three games and the mean rises
by at least 0.005; then the DFL check (the same procedure on each DFL match, decoder chosen on the
other six) must keep the DFL mean at or above 0.572. Fixed before the run, not tuned on the games:
the blackout profile, weight 2, 100 rounds, one round of self-training. The DFL check runs only if
the Metrica part passes (or with --smoke).

    uv run python scripts/test_self_training.py <out dir> --oof <dir D> --nn <dir E>
    # ~11 min; ~25 min more if the DFL check runs
    uv run python scripts/test_self_training.py <out dir> --oof <dir D> --nn <dir E> --smoke

<dir D> holds oof_metrica.npz and oof_dfl_single.npz (scripts/test_ensemble.py,
scripts/test_sequence_model.py evaluate), <dir E> oof_nn_s0.npz (test_sequence_model.py train).
"""
import argparse
import importlib.util
import itertools
import json
import sys
import time
from dataclasses import replace
from multiprocessing import Pool
from pathlib import Path

import lightgbm as lgb
import numpy as np

from possession.change import change_logit
from possession.decode import decode_periods
from possession.degrade import DegradeProfile, degrade
from possession.features import build_features
from possession.labels import X, swap_codes
from possession.model import LGB_PARAMS, N_ROUNDS, predict_proba, train
from possession.paths import MODELS, ROOT, vendor_import
from possession.sequence import BLEND

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

W_METRICA = 2.0
W_SELF = 2.0            # the pseudo-labelled rows weigh as much as a Metrica game
TUNE_ROUNDS = 100
COPY_SEED = 5000        # match i's copy gets its extra blackouts from seed 5000 + i
MIN_GAIN = 0.005
DFL_FLOOR = 0.572
CONFIGS = list(itertools.product(*tm.CHANGE_GRID))
CANDIDATES = ("pool", "tune")
RULE = ("fixed before the run (2026-11-13): the better candidate on the 3 Metrica games is adopted "
        "only if better than the submitted pipeline on all three and the mean rises by >= 0.005; "
        "then the DFL mean must stay >= 0.572")


def name(k):
    return f"{k[0]}:{k[1]}"


def _scores(item):
    key, (P, period, y, seen, lg, configs) = item
    pm = vendor_import("possession_metrics")
    out = []
    for t in configs:
        r = pm.evaluate(y, decode_periods(P, period, tm._config(t), seen=seen, change_logit=lg))
        out.append((r["spell_f1_mean"], r["spell_f1"][0.7]["f1"]))
    return key, out


def chosen(tab, label, k, members):
    """The decoder setting with the best mean spellF1 on the other members of the test set."""
    others = [m for m in members if m != k]
    return int(np.argmax(np.mean([[a for a, _ in tab[(label, m)]] for m in others], axis=0)))


def nested(tab, label, members, configs):
    res = {}
    for k in members:
        b = chosen(tab, label, k, members)
        res[name(k)] = {"spell_f1": tab[(label, k)][b][0], "strict": tab[(label, k)][b][1],
                        "decoder": list(configs[b])}
    return res


def ball_only(profile):
    """The measured profile without its player losses and jitter: ball blackouts only."""
    return replace(profile, deficit_pmf=[1.0] + [0.0] * (len(profile.deficit_pmf) - 1),
                   jitter_std_m=0.0, ball_jitter_std_m=0.0)


def both_ways(ft, F, labels, keep):
    """Rows `keep` of F and of the swapped match (home and away exchanged), as in
    train_multi.training_rows."""
    Fs, _ = build_features(ft.swapped())
    return (np.concatenate([F[keep], Fs[keep]]).astype(np.float32),
            np.concatenate([labels[keep], swap_codes(labels[keep])]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--oof", required=True, help="dir with oof_metrica.npz and oof_dfl_single.npz")
    ap.add_argument("--nn", required=True, help="dir with oof_nn_s0.npz")
    ap.add_argument("--smoke", action="store_true",
                    help="10 rounds, 2 decoder settings, DFL check forced on 2 matches: checks "
                         "the code, not the idea")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rounds = 10 if args.smoke else N_ROUNDS
    tune_rounds = 5 if args.smoke else TUNE_ROUNDS
    configs = CONFIGS[:2] if args.smoke else CONFIGS
    t0 = time.time()

    def log(msg):
        print(f"{msg} ({time.time() - t0:.0f} s)", flush=True)

    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    extra = ball_only(profile)
    keys = tm.registry(["dfl", "metrica"])
    index = {k: i for i, k in enumerate(keys)}
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]

    zb = np.load(Path(args.oof) / "oof_metrica.npz")
    zd = np.load(Path(args.oof) / "oof_dfl_single.npz")
    zn = np.load(Path(args.nn) / "oof_nn_s0.npz")
    saved = {g: (zb[f"single|{name(g)}|P"], zb[f"single|{name(g)}|pc"]) for g in met}
    saved |= {d: (zd[f"single_dfl|{name(d)}|P"], zd[f"single_dfl|{name(d)}|pc"]) for d in dfl}
    nn = {k: zn[f"nn|{name(k)}|P"] for k in keys}

    tables, rows = {}, {}
    names = None
    for k in keys:
        ft, y, F, names, _, _ = tm.build(*k, index[k], profile)
        rows[k] = both_ways(ft, F, y, np.flatnonzero(y != X)[::tm.SUBSAMPLE])
        tables[k] = (ft, y, np.asarray(F, np.float32))
    vis = names.index("ball_vis")
    log(f"prepared {len(keys)} matches")

    def fit(ks, own=None):
        """LightGBM frame model as submitted (Metrica rows weighted 2), plus `own` rows."""
        parts = [rows[k] for k in ks] + ([own] if own is not None else [])
        wt = [np.full(len(rows[k][1]), W_METRICA if k[0] == "metrica" else 1.0) for k in ks]
        if own is not None:
            wt.append(np.full(len(own[1]), W_SELF))
        Xf = np.concatenate([p[0] for p in parts])
        yf = np.concatenate([p[1] for p in parts])
        return train(Xf, yf, names, n_rounds=rounds, weight=np.concatenate(wt))

    def fit_tune(booster, own):
        ds = lgb.Dataset(own[0], label=own[1].astype(np.int32), feature_name=list(names),
                         free_raw_data=True)
        return lgb.train(dict(LGB_PARAMS), ds, num_boost_round=tune_rounds, init_model=booster)

    def item(label, k, P_lgb):
        ft, y, F = tables[k]
        P = BLEND * P_lgb + (1 - BLEND) * nn[k]
        return (label, k), (P, ft.period, y, F[:, vis] == 1, change_logit(saved[k][1]), configs)

    def run_part(members, cands):
        """Baseline and candidates on one test set; returns the score table and statistics."""
        boosters, P_lgb, stats = {}, {}, {}
        for k in members:
            boosters[k] = fit([m for m in keys if m != k])
            P_lgb[("base", k)] = predict_proba(boosters[k], tables[k][2])
            stats[name(k)] = {"base_vs_saved_max_abs_diff":
                              float(np.abs(P_lgb[("base", k)] - saved[k][0]).max())}
        log(f"{members[0][0]}: out-of-fold models retrained, max |P - saved| "
            f"{max(s['base_vs_saved_max_abs_diff'] for s in stats.values()):.2e}")
        with Pool(8) as p:
            tab = dict(p.map(_scores, [item("base", k, P_lgb[("base", k)]) for k in members]))

        for k in members:
            ft, y, F = tables[k]
            cfg = tm._config(configs[chosen(tab, "base", k, members)])
            _, (P, period, _, seen, lg, _) = item("base", k, P_lgb[("base", k)])
            pseudo = decode_periods(P, period, cfg, seen=seen, change_logit=lg)
            inplay = (period > 0) & (pseudo != X)
            copy = degrade(ft, extra, seed=COPY_SEED + index[k])
            lost = seen & ~np.isfinite(copy.ball[:, 0])
            Fc, _ = build_features(copy)
            own = both_ways(copy, Fc, pseudo, np.flatnonzero(seen & inplay)[::tm.SUBSAMPLE])
            del Fc
            live = (y == 0) | (y == 1)

            def right(m):
                return 100 * float((pseudo[m] == y[m]).mean())
            stats[name(k)] |= {
                # aggregates only; the labels of k are used here to describe, never to train
                "pseudo_right_seen_pct": right(seen & inplay),
                "pseudo_right_unseen_pct": right(~seen & inplay),
                "team_right_seen_live_pct": right(seen & live),
                "seen_pct_of_inplay": 100 * float(seen[inplay].mean()),
                "copy_hides_pct_of_seen": 100 * float(lost[seen & inplay].mean()),
                "own_rows": int(len(own[1]))}
            if "pool" in cands:
                b = fit([m for m in keys if m != k], own)
                P_lgb[("pool", k)] = predict_proba(b, F)
            if "tune" in cands:
                b = fit_tune(boosters[k], own)
                assert b.num_trees() == 3 * (rounds + tune_rounds), b.num_trees()
                P_lgb[("tune", k)] = predict_proba(b, F)
            del own
            log(f"{name(k)}: pseudo-labels right on "
                f"{stats[name(k)]['pseudo_right_seen_pct']:.1f} % of in-play frames with the "
                "ball seen; candidates trained")
        with Pool(8) as p:
            tab.update(dict(p.map(_scores, [item(c, k, P_lgb[(c, k)])
                                            for c in cands for k in members])))
        return tab, stats

    report = {"rule": RULE, "smoke": args.smoke,
              "settings": {"self_weight": W_SELF, "tune_rounds": tune_rounds, "rounds": rounds,
                           "copy_seed": COPY_SEED, "blend": BLEND,
                           "decoder_settings": len(configs)}}
    tab, stats = run_part(met, CANDIDATES)
    res = {c: nested(tab, c, met, configs) for c in ("base",) + CANDIDATES}

    def per_game(c, members, field="spell_f1"):
        return [res[c][name(k)][field] for k in members]

    print("\nspellF1 on each Metrica game, decoder chosen on the other two games")
    print("candidate   game 1  game 2  game 3 |  mean | strict (70 % overlap) mean")
    for c in ("base",) + CANDIDATES:
        v, s = per_game(c, met), per_game(c, met, "strict")
        print(f"{c:<10} {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} | {np.mean(v):.4f} | {np.mean(s):.4f}")
    best = max(CANDIDATES, key=lambda c: np.mean(per_game(c, met)))
    a, b = per_game("base", met), per_game(best, met)
    passed = all(x > y for x, y in zip(b, a)) and np.mean(b) - np.mean(a) >= MIN_GAIN
    log(f"best candidate {best}: {np.mean(b):.4f} vs submitted {np.mean(a):.4f}, better on "
        f"{sum(x > y for x, y in zip(b, a))}/3; Metrica rule passed: {passed}")
    report |= {"metrica": res, "metrica_stats": stats, "best_candidate": best,
               "metrica_gain": float(np.mean(b) - np.mean(a)), "metrica_rule_passed": bool(passed)}

    if passed or args.smoke:
        members = dfl[:2] if args.smoke else dfl
        tab_d, stats_d = run_part(members, (best,))
        res_d = {c: nested(tab_d, c, members, configs) for c in ("base", best)}
        base_d = float(np.mean([r["spell_f1"] for r in res_d["base"].values()]))
        cand_d = float(np.mean([r["spell_f1"] for r in res_d[best].values()]))
        report |= {"dfl": res_d, "dfl_stats": stats_d, "dfl_base_mean": base_d,
                   "dfl_mean": cand_d, "dfl_check_passed": bool(cand_d >= DFL_FLOOR)}
        log(f"DFL mean {cand_d:.4f} (submitted {base_d:.4f}, must be >= {DFL_FLOOR})")
    report["adopt"] = bool(passed and report.get("dfl_check_passed", False))
    log(f"adopt: {report['adopt']}")
    (out / "self_training_report.json").write_text(json.dumps(report, indent=1))
    if not args.smoke:
        (MODELS / "self_training_report.json").write_text(json.dumps(report, indent=1))
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
