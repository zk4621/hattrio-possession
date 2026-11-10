"""Load the provided tactical tracking and check it against the measured aggregates.

Prints aggregates only; never frame content.
"""
import sys

import numpy as np

from possession.io_atd import load_competition


def main() -> int:
    try:
        ft = load_competition("tactical")
    except IOError as err:
        print(err, file=sys.stderr)
        return 2
    graded = ft.period > 0
    ball_vis = ~np.isnan(ft.ball[:, 0])
    tracked = (~np.isnan(ft.home[..., 0])).sum(1) + (~np.isnan(ft.away[..., 0])).sum(1)
    stats = {
        "records": ft.n_frames,
        "slots": (ft.home.shape[1], ft.away.shape[1]),
        "ball_visible_all_pct": 100 * ball_vis.mean(),
        "ball_visible_graded_pct": 100 * ball_vis[graded].mean(),
        "tracked_players_graded": tracked[graded].mean(),
    }
    for k, v in stats.items():
        print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
    assert stats["records"] == 199319
    assert stats["slots"] == (24, 22)
    assert abs(stats["ball_visible_all_pct"] - 56.94) <= 0.05
    assert abs(stats["ball_visible_graded_pct"] - 59.31) <= 0.05
    assert abs(stats["tracked_players_graded"] - 20.49) <= 0.05
    print("check_atd: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
