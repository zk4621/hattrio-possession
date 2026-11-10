"""Write outputs/<team>.csv for the provided match. Prints aggregates only."""
import argparse
import hashlib
import sys

import numpy as np

from possession.decode import DecoderConfig
from possession.frames import FPS
from possession.labels import A, D, H, X, write_submission
from possession.paths import OUTPUTS, vendor_import


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--proba", choices=["rules", "model"], default="model")
    args = ap.parse_args()

    from possession.io_atd import kickoff_tag_frames, load_competition
    from possession.pipeline import predict_codes
    from possession.rules import rule_proba
    try:
        ft = load_competition("tactical")
        tags = kickoff_tag_frames()
        from possession.sources import attach_sources
        ft = attach_sources(ft)
    except IOError as err:
        print(err, file=sys.stderr)
        return 2

    if args.proba == "model":
        from possession.model import load_decoder, load_proba_fn
        proba_fn, cfg = load_proba_fn(), load_decoder()
    else:
        proba_fn, cfg = rule_proba, DecoderConfig()
    live_from = {p: max(start, tags[p]) for p, start, stop in ft.period_slices() if p in tags}
    codes = predict_codes(ft, proba_fn, cfg, live_from)

    path = OUTPUTS / f"{args.team}.csv"
    write_submission(path, codes)
    pm = vendor_import("possession_metrics")
    problem = pm.validate_submission(pm.read_label_csv(path), len(codes))
    print("valid" if problem is None else f"INVALID: {problem}")
    graded = ft.period > 0
    counts = {n: int((codes[graded] == k).sum()) for n, k in (("H", H), ("A", A), ("D", D), ("X", X))}
    runs = [(a, b) for p, a, b in ft.period_slices()]
    spells = pm.possession_spells(codes, runs)
    dur = [(b - a) / FPS for a, b, _ in spells]
    print(f"graded label counts: {counts}")
    print(f"possessions: {len(spells)}, mean duration {np.mean(dur):.1f} s")
    print(f"sha256: {hashlib.sha256(path.read_bytes()).hexdigest()}")
    return 0 if problem is None else 1


if __name__ == "__main__":
    sys.exit(main())
