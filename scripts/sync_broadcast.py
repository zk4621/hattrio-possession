"""Align the broadcast tracking to the tactical timeline and measure the merged ball.

Prints aggregate statistics only. Adoption rule fixed in advance (plan, Task 10):
agree_median_m <= 3.0, agree_p90_m <= 10.0, filled_share >= 0.10.
"""
import json
import sys
import time

import numpy as np

from possession.paths import CACHE, OUTPUTS


def main() -> int:
    from possession.io_atd import load_competition
    from possession.sources import write_sources
    from possession.sync import _mapping_and_cost, anchor_offsets, candidate_offsets, merge_ball
    try:
        tft = load_competition("tactical")
        bft = load_competition("broadcast")
    except IOError as err:
        print(err, file=sys.stderr)
        return 2
    t0 = time.time()
    anchors = anchor_offsets(bft, tft)
    _, offsets, costs = anchors
    cands = candidate_offsets(offsets, costs)
    print(f"anchors: {len(costs)}, below cost limit: {int((costs < 2.0).sum())}, "
          f"anchor cost quantiles p10/p50/p90: "
          f"{np.round(np.percentile(costs[np.isfinite(costs)], [10, 50, 90]), 2).tolist()} "
          f"({time.time() - t0:.0f} s)", flush=True)
    mapping, _, cands = _mapping_and_cost(bft, tft, anchors)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(CACHE / "broadcast_mapping.npy", mapping)
    merged, stats = merge_ball(tft, bft, mapping)
    stats = {"n_candidates": len(cands), "candidate_offsets": cands, **stats}
    print(json.dumps(stats, indent=1))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "sync_stats.json").write_text(json.dumps(stats, indent=1))
    adopt = (len(cands) > 0 and stats["agree_median_m"] <= 3.0 and stats["agree_p90_m"] <= 10.0
             and stats["filled_share"] >= 0.10)
    write_sources(broadcast_ball=bool(adopt))
    print(f"broadcast ball {'ADOPTED' if adopt else 'not adopted'} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
