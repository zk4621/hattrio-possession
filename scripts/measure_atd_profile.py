"""Measure the blackout / player-loss profile of the provided tracking -> models/atd_profile.json.

Prints only the profile's aggregate numbers.
"""
import sys

import numpy as np

from possession.degrade import measure_profile
from possession.paths import MODELS


def main() -> int:
    from possession.io_atd import load_competition
    from possession.sources import attach_sources, load_flags
    try:
        ft = load_competition("tactical")
    except IOError as err:
        print(err, file=sys.stderr)
        return 2
    if load_flags()["broadcast"]:
        ft = attach_sources(ft)
        print("measuring the tactical ball merged with the broadcast ball")
    p = measure_profile(ft)
    MODELS.mkdir(parents=True, exist_ok=True)
    p.to_json(MODELS / "atd_profile.json")
    q = np.array(p.blackout_quantiles)
    print(f"blackout hazard (in field, boundary): 1/{1 / p.blackout_hazard[0]:.0f}, "
          f"1/{1 / p.blackout_hazard[1]:.0f}")
    for c, name in enumerate(("in field", "boundary")):
        print(f"blackout length quantiles {name} (p10/p25/p50/p75/p90/p99/max): "
              f"{[int(round(q[c][i])) for i in (10, 25, 50, 75, 90, 99, 100)]}")
    print(f"deficit pmf k=0..11: {[round(v, 3) for v in p.deficit_pmf]}")
    print(f"jitter players {p.jitter_std_m:.3f} m, ball {p.ball_jitter_std_m:.3f} m")
    for c, name in enumerate(("in field", "boundary")):
        mean_len = float(np.mean(np.interp(np.linspace(0, 100, 1001), np.arange(101), q[c])))
        share = mean_len / (1 / p.blackout_hazard[c] + mean_len)
        print(f"implied blackout share if every frame were '{name}': {100 * share:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
