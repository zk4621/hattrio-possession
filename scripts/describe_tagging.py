"""What the Smart Tagging contains, and how it lines up with a prediction file.

Aggregates only: counts per code and tag value, duration quantiles, coverage of the graded frames,
and the share of frames on which a prediction agrees with the tagged team. Never prints an
instance, a time or a frame.

    uv run python scripts/describe_tagging.py [--pred outputs/HatTrio.csv]
"""
import argparse
import sys
from collections import Counter, defaultdict

import numpy as np

from possession.frames import FPS
from possession.labels import A, D, H, read_codes
from possession.paths import data_dir, vendor_import


def team_code(name):
    n = (name or "").strip().lower()
    if "barcelona" in n:
        return H
    if "athletic" in n or "atletic" in n:
        return A
    return None


def q(values, ps=(10, 50, 90)):
    return [round(float(np.percentile(values, p)), 1) for p in ps] if len(values) else []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=None)
    args = ap.parse_args()
    games = vendor_import("games")
    st = vendor_import("load_smart_tagging")
    try:
        events = st.load(data_dir=str(data_dir()))
    except IOError as err:
        print(err, file=sys.stderr)
        return 2
    n = games.source_frames()
    graded = np.zeros(n, bool)
    for _, a, b in games.half_video_blocks("BAR-ATH"):
        graded[a:b + 1] = True

    by_code = defaultdict(list)
    for e in events:
        by_code[(e["code"] or "").strip()].append(e)
    print(f"{len(events)} instances, {len(by_code)} codes")
    for code, evs in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
        dur = [(e["end_frame"] - e["start_frame"]) / FPS for e in evs]
        cover = np.zeros(n, bool)
        for e in evs:
            cover[max(e["start_frame"], 0):min(e["end_frame"], n)] = True
        groups = defaultdict(Counter)
        for e in evs:
            for g, v in (e["tags"] or {}).items():
                groups[(g or "").strip()][(v or "").strip()] += 1
        print(f"\n[{code}] {len(evs)} instances, duration p10/p50/p90 {q(dur)} s, "
              f"covers {cover[graded].mean():.1%} of graded frames")
        for g, c in groups.items():
            vals = dict(c.most_common(12))
            more = f" (+{len(c) - 12} more)" if len(c) > 12 else ""
            print(f"  {g}: {vals}{more}")

    # how much of the graded match carries a team, and do tags contradict each other?
    team_votes = np.zeros((n, 2), np.int32)
    for e in events:
        t = team_code((e["tags"] or {}).get("Team"))
        if t is not None:
            team_votes[max(e["start_frame"], 0):min(e["end_frame"], n), t] += 1
    any_team = team_votes.sum(1) > 0
    both = (team_votes > 0).all(1)
    print(f"\ngraded frames inside an instance with a team tag: {any_team[graded].mean():.1%}; "
          f"inside instances of both teams at once: {both[graded].mean():.1%}")

    if args.pred:
        pred = read_codes(args.pred)
        print(f"\nagreement with {args.pred} on graded frames inside tagged instances:")
        print("code                         frames   same_team  other_team  dead")
        for code, evs in sorted(by_code.items(), key=lambda kv: -len(kv[1])):
            same = other = dead = tot = 0
            for e in evs:
                t = team_code((e["tags"] or {}).get("Team"))
                if t is None:
                    continue
                sl = slice(max(e["start_frame"], 0), min(e["end_frame"], n))
                m = graded[sl]
                p = pred[sl][m]
                tot += len(p)
                same += int((p == t).sum())
                other += int((p == (A if t == H else H)).sum())
                dead += int((p == D).sum())
            if tot:
                print(f"{code[:28]:<28} {tot:7d}   {same / tot:9.1%}  {other / tot:10.1%}  "
                      f"{dead / tot:5.1%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
