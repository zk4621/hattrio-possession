"""Measure the tactical camera's view centre against the tactical ball; adopt it or not.

Prints aggregate statistics only. Adoption rule fixed in advance (plan, Task 9):
corr_x >= 0.8, med_abs_dx <= 10.0 m, std_x >= 5.0 m.
"""
import json
import sys

from possession.paths import OUTPUTS


def main() -> int:
    from possession.io_atd import load_competition
    from possession.proxy import competition_proxy, measure_proxy
    from possession.sources import write_sources
    try:
        ft = load_competition("tactical")
        proxy = competition_proxy(ft)
    except IOError as err:
        print(err, file=sys.stderr)
        return 2
    stats = measure_proxy(proxy, ft)
    print(json.dumps(stats, indent=1))
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "proxy_stats.json").write_text(json.dumps(stats, indent=1))
    adopt = stats["corr_x"] >= 0.8 and stats["med_abs_dx"] <= 10.0 and stats["std_x"] >= 5.0
    if adopt:
        write_sources(camera_proxy=True, proxy_stats=stats)
    else:
        write_sources(camera_proxy=False)
    print(f"camera proxy {'ADOPTED' if adopt else 'not adopted'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
