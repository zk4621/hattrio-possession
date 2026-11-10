"""Possession statistics of the DFL matches: does their possession flag look like the challenge's?"""
import sys

import numpy as np

from possession.frames import FPS
from possession.io_dfl import DFL_MATCH_IDS, load_dfl_match
from possession.labels import D
from possession.paths import vendor_import


def main() -> int:
    pm = vendor_import("possession_metrics")
    means = []
    print("match   frames_in_play  D_share  n_poss  mean_poss_s  p10_poss_s  ball_tracked")
    for mid in DFL_MATCH_IDS:
        ft, y = load_dfl_match(mid)
        play = ft.period > 0
        runs = pm.valid_runs(y)
        spells = pm.possession_spells(y, runs)
        dur = np.array([(b - a) / FPS for a, b, _ in spells])
        means.append(dur.mean())
        print(f"{mid}  {play.sum():14d}  {np.mean(y[play] == D):7.3f}  {len(spells):6d}  "
              f"{dur.mean():11.1f}  {np.percentile(dur, 10):10.1f}  "
              f"{np.mean(~np.isnan(ft.ball[play, 0])):12.3f}", flush=True)
    print(f"median of mean possession durations: {np.median(means):.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
