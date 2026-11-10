"""Compare feature distributions: competition match vs the seven degraded DFL matches.

Prints and writes aggregates only (mean, std, NaN share per feature and domain).
Rule fixed in advance (plan, Task 11): drop features with |SMD| > 1.0 or NaN-share
difference > 0.25.
"""
import json
import sys

import numpy as np

from possession.degrade import DegradeProfile, degrade
from possession.features import build_features
from possession.io_dfl import DFL_MATCH_IDS, load_dfl_match
from possession.paths import MODELS, OUTPUTS
from possession.sources import attach_dfl_sources

SMD_MAX = 1.0
NAN_DIFF_MAX = 0.25


class Acc:
    def __init__(self, n):
        self.s = np.zeros(n)
        self.ss = np.zeros(n)
        self.c = np.zeros(n)
        self.rows = 0

    def add(self, F):
        F = F.astype(np.float64)
        ok = np.isfinite(F)
        Z = np.where(ok, F, 0.0)
        self.s += Z.sum(0)
        self.ss += (Z * Z).sum(0)
        self.c += ok.sum(0)
        self.rows += len(F)

    def stats(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.s / self.c
            var = np.maximum(self.ss / self.c - mean ** 2, 0)
        return mean, np.sqrt(var), 1 - self.c / max(self.rows, 1)


def main() -> int:
    from possession.io_atd import load_competition
    from possession.sources import attach_sources
    try:
        ft = attach_sources(load_competition("tactical"))
    except IOError as err:
        print(err, file=sys.stderr)
        return 2
    F, names = build_features(ft)
    atd = Acc(len(names))
    atd.add(F[ft.period > 0])
    del F
    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    dfl = Acc(len(names))
    for i, mid in enumerate(DFL_MATCH_IDS):
        d, y = load_dfl_match(mid)
        d = degrade(attach_dfl_sources(d, y, mid, i), profile, seed=i)
        G, gnames = build_features(d)
        assert gnames == names
        dfl.add(G[d.period > 0])
    ma, sa, na = atd.stats()
    md, sd, nd = dfl.stats()
    pooled = np.sqrt((sa ** 2 + sd ** 2) / 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        smd = np.where(pooled > 0, (ma - md) / pooled, 0.0)
    rows = []
    for j, n in enumerate(names):
        rows.append({"feature": n, "mean_atd": ma[j], "mean_dfl": md[j], "std_atd": sa[j],
                     "std_dfl": sd[j], "nan_atd": na[j], "nan_dfl": nd[j],
                     "smd": float(np.nan_to_num(smd[j]))})
    rows.sort(key=lambda r: -abs(r["smd"]))
    drop = [r["feature"] for r in rows
            if abs(r["smd"]) > SMD_MAX or abs(r["nan_atd"] - r["nan_dfl"]) > NAN_DIFF_MAX]
    print("feature                    smd    mean_atd  mean_dfl   nan_atd nan_dfl")
    for r in rows[:40]:
        print(f"{r['feature']:<24} {r['smd']:6.2f}  {r['mean_atd']:9.3f} {r['mean_dfl']:9.3f}  "
              f"{r['nan_atd']:6.3f}  {r['nan_dfl']:6.3f}")
    for n in ("n_home", "n_away", "len_home", "len_away", "wid_home", "wid_away"):
        r = next(r for r in rows if r["feature"] == n)
        print(f"check {n}: smd {r['smd']:.2f} atd {r['mean_atd']:.2f} dfl {r['mean_dfl']:.2f}")
    print(f"features to drop ({len(drop)}): {' '.join(drop)}")
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "domain_report.json").write_text(json.dumps(
        {"rows": rows, "drop": drop, "smd_max": SMD_MAX, "nan_diff_max": NAN_DIFF_MAX},
        indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
