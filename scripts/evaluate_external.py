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

DEGRADE_SEED = 1000


def get_profile(name):
    if name == "none":
        return None
    from possession.degrade import DEFAULT_PROFILE, DegradeProfile
    if name == "default":
        return DEFAULT_PROFILE
    from possession.paths import MODELS
    return DegradeProfile.load(MODELS / "atd_profile.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba", choices=["baseline", "rules", "model"], required=True)
    ap.add_argument("--degrade", choices=["none", "default", "atd"], default="none")
    ap.add_argument("--matches", nargs="*", default=DFL_MATCH_IDS)
    args = ap.parse_args()

    profile = get_profile(args.degrade)
    if args.proba == "model":
        from possession.model import load_decoder, load_proba_fn
        proba_fn, cfg = load_proba_fn(), load_decoder()
    else:
        proba_fn, cfg = rule_proba, DecoderConfig()

    rows = {}
    print("match   spell_f1  macro_f1  n_gt  n_pred")
    for i, mid in enumerate(args.matches):
        ft, y = load_dfl_match(mid)
        if profile is not None:
            from possession.degrade import degrade
            ft = degrade(ft, profile, seed=DEGRADE_SEED + i)
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


if __name__ == "__main__":
    sys.exit(main())
