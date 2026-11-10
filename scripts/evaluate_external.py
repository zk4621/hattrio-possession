"""Score a probability source on the DFL matches with the official scorer."""
import argparse
import json
import sys

import numpy as np

from possession.decode import DecoderConfig
from possession.io_dfl import DFL_MATCH_IDS, load_dfl_match
from possession.paths import OUTPUTS
from possession.pipeline import predict_codes
from possession.rules import baseline_codes, rule_proba
from possession.scoring import score
from possession.sources import attach_dfl_sources

def get_profile(name):
    if name == "none":
        return None
    from possession.degrade import DEFAULT_PROFILE, DegradeProfile
    if name == "default":
        return DEFAULT_PROFILE
    from possession.paths import MODELS
    return DegradeProfile.from_json(MODELS / "atd_profile.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba", choices=["baseline", "rules", "model"], required=True)
    ap.add_argument("--degrade", choices=["none", "default", "atd"], default="none")
    ap.add_argument("--matches", nargs="*", default=DFL_MATCH_IDS)
    ap.add_argument("--consistency", action="store_true",
                    help="degraded prediction against clean-ball prediction")
    args = ap.parse_args()
    if args.consistency:
        return consistency(args)

    profile = get_profile(args.degrade)
    if args.proba == "model":
        from possession.model import load_decoder, load_proba_fn
        proba_fn, cfg = load_proba_fn(), load_decoder()
    else:
        proba_fn, cfg = rule_proba, DecoderConfig()

    rows = {}
    print("match   spell_f1  macro_f1  n_gt  n_pred")
    for mid in args.matches:
        ft, y = load_dfl_match(mid)
        ft = attach_dfl_sources(ft, y, mid, DFL_MATCH_IDS.index(mid))
        if profile is not None:
            from possession.degrade import degrade
            ft = degrade(ft, profile, seed=DFL_MATCH_IDS.index(mid))
        if args.proba == "baseline":
            codes = baseline_codes(ft)
        else:
            codes = predict_codes(ft, proba_fn, cfg)
        s = score(y, codes)
        rows[mid] = s
        print(f"{mid}  {s['spell_f1']:8.4f}  {s['macro_f1']:8.4f}  {s['n_gt_spells']:4d}  "
              f"{s['n_pred_spells']:6d}", flush=True)
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in ("spell_f1", "macro_f1")}
    print(f"mean    {mean['spell_f1']:8.4f}  {mean['macro_f1']:8.4f}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUTS / f"external_{args.proba}_{args.degrade}.json"
    out.write_text(json.dumps({"matches": rows, "mean": mean}, indent=2))
    return 0


def consistency(args) -> int:
    from possession.degrade import degrade
    from possession.model import load_decoder, load_proba_fn
    profile = get_profile("atd" if args.degrade == "none" else args.degrade)
    if args.proba == "model":
        proba_fn, cfg = load_proba_fn(), load_decoder()
    else:
        proba_fn, cfg = rule_proba, DecoderConfig()
    rows = {}
    print("match   agree_new  spell(deg vs clean)  spell_true_clean  spell_true_deg")
    for mid in args.matches:
        ft, y = load_dfl_match(mid)
        ft = attach_dfl_sources(ft, y, mid, DFL_MATCH_IDS.index(mid))
        deg = degrade(ft, profile, seed=DFL_MATCH_IDS.index(mid))
        new = (ft.period > 0) & np.isfinite(ft.ball[:, 0]) & ~np.isfinite(deg.ball[:, 0])
        clean = predict_codes(ft, proba_fn, cfg)
        dcodes = predict_codes(deg, proba_fn, cfg)
        r = {"agree_new": float((clean[new] == dcodes[new]).mean()),
             "spell_deg_vs_clean": score(clean, dcodes)["spell_f1"],
             "spell_true_clean": score(y, clean)["spell_f1"],
             "spell_true_deg": score(y, dcodes)["spell_f1"]}
        rows[mid] = r
        print(f"{mid}  {r['agree_new']:9.4f}  {r['spell_deg_vs_clean']:19.4f}  "
              f"{r['spell_true_clean']:16.4f}  {r['spell_true_deg']:14.4f}", flush=True)
    mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))}
    print("mean    " + "  ".join(f"{k}={v:.4f}" for k, v in mean.items()))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"consistency_{args.proba}.json").write_text(
        json.dumps({"matches": rows, "mean": mean}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
