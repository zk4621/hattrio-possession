"""Measure what the provided Smart Tagging's set-piece tags look like -> models/tag_profile.json.

Aggregates only: counts per kind, share with a team, tag length quantiles, and when the restart
happens relative to the tag start. The restart is taken where a prediction made without tags
(--pred, the no-tag submission) switches from dead ball to a team; a tracking-based detector (a
still ball that is suddenly played) is reported alongside, and checked on the DFL matches, where
the true restart time is known.

    uv run python scripts/measure_tag_profile.py --pred outputs/HatTrio.csv
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from possession.frames import FPS
from possession.labels import A, D, H, read_codes
from possession.io_dfl import DFL_MATCH_IDS, load_dfl_match
from possession.paths import MODELS
from possession.tagging import (PH_KEYS, SP_CODE, SP_KINDS, ball_speed, competition_tags,
                                detect_restart, dfl_phase_candidates, dfl_set_pieces)

LO, HI = -250, 375          # search window around the tag start / the true restart (frames)
MIN_OFFSETS = 5
# How long before its moment a phase tag starts. Measured on the provided match: inside
# "ATTACKING TRANSITION" tags the prediction made without tags switches to the tagged team
# between 2 and 4 s after the tag start (scratch analysis reported in the technical document).
PRE_FRAMES = [50, 100]
# Early part of a phase tag where its team is compared: the first 4 s, or 4-8 s for regains
# (whose moment comes 2-4 s after the tag start).
EARLY = {"ATTACKING TRANSITION": (100, 200)}
EARLY_DEFAULT = (0, 100)
# Frame accuracy of the no-tag model on frames with the ball seen (cross-validation, 0.9405):
# the prediction used as reference on the provided match is itself wrong this often.
REF_ACC = 0.94


def q101(values):
    return [float(v) for v in np.percentile(values, np.arange(101))]


def show(values):
    v = np.asarray(values, dtype=float) / FPS
    return f"p10/p50/p90 {np.percentile(v, 10):+.1f} / {np.percentile(v, 50):+.1f} / " \
           f"{np.percentile(v, 90):+.1f} s" if len(v) else "-"


def first_live_after_dead(pred, start, lo, hi, min_live=50):
    """First frame in [start+lo, start+hi] where the prediction goes from D to a team for good."""
    a, z = max(start + lo, 1), min(start + hi, len(pred) - min_live)
    for f in range(a, z):
        if pred[f - 1] == D and pred[f] != D and (pred[f:f + min_live] != D).all():
            return f
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="a submission made without tags")
    args = ap.parse_args()
    from possession.io_atd import load_competition
    try:
        ft = load_competition("tactical")
        all_tags = competition_tags()
        tags = [g for g in all_tags if g.code == SP_CODE and g.kind]
    except IOError as err:
        print(err, file=sys.stderr)
        return 2
    pred = read_codes(args.pred)
    speed = ball_speed(ft.ball)
    count, known = Counter(), Counter()
    offsets, detected = defaultdict(list), defaultdict(list)
    for g in tags:
        count[g.kind] += 1
        known[g.kind] += g.team != 0
        f = first_live_after_dead(pred, g.start, LO, HI)
        if f is not None:
            offsets[g.kind].append(f - g.start)
        r = detect_restart(ft.ball, g.start, LO, HI, speed=speed)
        if r is not None:
            detected[g.kind].append(r - g.start)
    durations = [g.end - g.start for g in tags]

    graded = ft.period > 0
    real_early = {}
    for code in PH_KEYS:
        e0, e1 = EARLY.get(code, EARLY_DEFAULT)
        same = tot = 0
        for g in all_tags:
            if g.code != code or g.team == 0:
                continue
            sl = slice(max(g.start + e0, 0), min(g.start + e1, g.end, len(pred)))
            p = pred[sl][graded[sl]]
            live = p != D
            tot += int(live.sum())
            same += int((p[live] == (H if g.team == 1 else A)).sum())
        real_early[code] = same / tot if tot else 1.0

    dfl_count, errors, dfl_cands = Counter(), [], Counter()
    sim_same, sim_tot = Counter(), Counter()
    mid_pre = sum(PRE_FRAMES) // 2
    for mid in DFL_MATCH_IDS:
        d, y = load_dfl_match(mid)
        for code, rows in dfl_phase_candidates(d, y).items():
            dfl_cands[code] += len(rows)
            e0, e1 = EARLY.get(code, EARLY_DEFAULT)
            for row, team in rows:
                w = y[max(row - mid_pre + e0, 0):max(row - mid_pre + e1, 0)]
                live = (w == H) | (w == A)
                sim_tot[code] += int(live.sum())
                sim_same[code] += int((w[live] == (H if team == 1 else A)).sum())
        events = dfl_set_pieces(mid, d.period)
        s = ball_speed(d.ball)
        for kind, row, _ in events:
            dfl_count[kind] += 1
            r = detect_restart(d.ball, row, LO, HI, speed=s)
            if r is not None:
                errors.append(r - row)
    n_dfl = len(DFL_MATCH_IDS)

    pooled = [o for k in SP_KINDS for o in offsets[k]]
    profile = {"duration_q": q101(durations), "kinds": {},
               "restart_source": {"pred_sha256": hashlib.sha256(open(args.pred, "rb").read())
                                  .hexdigest()}}
    print("kind      real  with_team  dfl/match  keep   found  restart - tag start (prediction)"
          "   | tracking detector")
    for k in SP_KINDS:
        per_match = dfl_count[k] / n_dfl
        keep = min(1.0, count[k] / per_match) if per_match else 0.0
        own = offsets[k] if len(offsets[k]) >= MIN_OFFSETS else pooled
        profile["kinds"][k] = {
            "real_count": int(count[k]), "team_known": known[k] / count[k] if count[k] else 0.0,
            "dfl_per_match": per_match, "keep": keep, "n_offsets": len(offsets[k]),
            "offset_q": q101(own) if own else [0.0] * 101,
        }
        print(f"{k:<9} {count[k]:4d}  {known[k]:9d}  {per_match:9.1f}  {keep:4.2f}   "
              f"{len(offsets[k]):5d}  {show(offsets[k])}   | {len(detected[k])}: "
              f"{show(detected[k])}")
    print(f"tag length: {show(durations)}")
    e = np.abs(np.asarray(errors))
    print(f"restart detector on DFL: found {len(errors)} of {sum(dfl_count.values())}; "
          f"error {show(errors)}; within 1 s: {np.mean(e <= FPS):.0%}")
    profile["pre_frames"] = PRE_FRAMES
    profile["codes"] = {}
    print("\nphase code              real  with_team  dfl candidates/match  keep  early agreement "
          "real / sim  team_flip  length p10/p50/p90")
    for code in PH_KEYS:
        gs = [g for g in all_tags if g.code == code]
        per_match = dfl_cands[code] / n_dfl
        keep = min(1.0, len(gs) / per_match) if per_match else 0.0
        lengths = [g.end - g.start for g in gs]
        sim_early = sim_same[code] / sim_tot[code] if sim_tot[code] else 1.0
        flip = float(np.clip(1.0 - real_early[code] / (REF_ACC * sim_early), 0.0, 0.5))
        profile["codes"][code] = {
            "real_early_agreement": real_early[code], "sim_early_agreement": sim_early,
            "team_flip": flip,
            "real_count": len(gs), "team_known": (sum(g.team != 0 for g in gs) / len(gs)
                                                  if gs else 0.0),
            "dfl_per_match": per_match, "keep": keep,
            "duration_q": q101(lengths) if lengths else [0.0] * 101,
        }
        print(f"{code[:22]:<22} {len(gs):5d}  {sum(g.team != 0 for g in gs):9d}  "
              f"{per_match:20.1f}  {keep:4.2f}  {real_early[code]:15.0%} / {sim_early:3.0%}  "
              f"{flip:9.2f}  {show(lengths)}")
    MODELS.mkdir(parents=True, exist_ok=True)
    (MODELS / "tag_profile.json").write_text(json.dumps(profile, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
