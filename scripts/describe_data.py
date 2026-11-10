"""Aggregate description of the competition bundle. Prints summaries only, never frame content.

    python3.12 scripts/describe_data.py

Standard library only. The competition terms forbid handing the match data to a cloud LLM, so
everything printed here is a count, a rate or a quantile.
"""

import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("MTAH_DATA_DIR") or os.path.join(ROOT, "data")
PITCH = (105.3, 68.0)
GRADED = ((11185, 90651), (110808, 190197))  # published in games.py, inclusive video frames
N_TACTICAL = 199319
QS = (10, 25, 50, 75, 90, 95, 99, 100)


def quantiles(values, qs=QS):
    if not values:
        return "n=0"
    ordered = sorted(values)
    out = []
    for q in qs:
        index = min(len(ordered) - 1, int(round(q / 100.0 * (len(ordered) - 1))))
        out.append("p{}={:.4g}".format(q, ordered[index]))
    return "n={}  ".format(len(ordered)) + "  ".join(out)


def as_float(text):
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def runs(flags, spans):
    """Lengths of consecutive True runs and False runs inside each span; edge-touching runs kept
    separately so they can be excluded."""
    true_runs, false_runs, edge_false = [], [], 0
    for first, last in spans:
        current, length, start = None, 0, first
        for index in range(first, last + 1):
            flag = bool(flags[index])
            if flag == current:
                length += 1
                continue
            if current is not None:
                if current:
                    true_runs.append(length)
                elif start == first:
                    edge_false += length
                else:
                    false_runs.append(length)
            current, length, start = flag, 1, index
        if current:
            true_runs.append(length)
        elif current is not None:
            edge_false += length
    return true_runs, false_runs, edge_false


def describe_meta(path):
    root = ET.parse(path).getroot()
    teams = [(team.get("id"), team.findtext("Name")) for team in root.iter("Team")]
    slots = [player.get("teamId") for player in root.iter("Player")]
    print("  teams (file order, first = home): {}".format([name for _, name in teams]))
    print("  track slots: {} total, per team {}".format(
        len(slots), [sum(1 for s in slots if s == team_id) for team_id, _ in teams]))
    tags = Counter(element.tag for element in root.iter())
    print("  xml element tags: {}".format(dict(tags.most_common(25))))
    for element in root.iter():
        if element.tag in ("FrameRate", "FieldSize", "Width", "Height"):
            print("  <{}> {} {}".format(element.tag, (element.text or "").strip(), dict(element.attrib)))
    channels = [e.get("id") or e.get("channelId") for e in root.iter()
                if e.tag in ("PlayerChannel", "BallChannel", "PlayerChannelRef", "BallChannelRef")]
    print("  channel ids (first 12 of {}): {}".format(len(channels), channels[:12]))
    home_id = teams[0][0]
    return [0 if s == home_id else 1 for s in slots]


def describe_atd(raw_path, team_of_slot, n_frames, spans, label):
    n_slots = len(team_of_slot)
    ball_seen = bytearray(n_frames)
    slot_seen = [bytearray(n_frames) for _ in range(n_slots)]
    tracked = bytearray(n_frames)
    parts_count, groups_count = Counter(), Counter()
    player_fields, ball_fields, bad_tokens = Counter(), Counter(), Counter()
    ball_x, ball_y, ball_third, player_third = [], [], [], []
    px_min = py_min = float("inf")
    px_max = py_max = float("-inf")
    nearest = {}          # frame -> (distance to nearest player m, nearest team)
    ball_pos = {}
    first_id = last_id = None
    n_lines = 0
    length, width = PITCH

    with open(raw_path) as handle:
        for line in handle:
            n_lines += 1
            parts = line.rstrip("\n").split(":")
            parts_count[len(parts)] += 1
            if len(parts) != 3:
                continue
            frame_id = int(parts[0])
            first_id = frame_id if first_id is None else first_id
            last_id = frame_id
            row = frame_id - 1
            if not 0 <= row < n_frames:
                continue

            groups = parts[1].split(";")
            groups_count[len(groups)] += 1
            positions = []
            count = 0
            for slot, group in enumerate(groups):
                fields = group.split(",")
                player_fields[len(fields)] += 1
                x = as_float(fields[0]) if fields else None
                y = as_float(fields[1]) if len(fields) > 1 else None
                if x is None or y is None:
                    if fields and as_float(fields[0]) is None:
                        bad_tokens[fields[0][:6]] += 1
                    continue
                count += 1
                if slot < n_slots:
                    slot_seen[slot][row] = 1
                    positions.append((x * length, y * width, team_of_slot[slot]))
                px_min, px_max = min(px_min, x), max(px_max, x)
                py_min, py_max = min(py_min, y), max(py_max, y)
                if len(fields) > 2 and row % 50 == 0:
                    third = as_float(fields[2])
                    if third is not None:
                        player_third.append(third)
            tracked[row] = min(count, 255)

            fields = parts[2].split(",")
            ball_fields[len(fields)] += 1
            bx = as_float(fields[0]) if fields else None
            by = as_float(fields[1]) if len(fields) > 1 else None
            if bx is None or by is None:
                continue
            ball_seen[row] = 1
            ball_x.append(bx)
            ball_y.append(by)
            if len(fields) > 2:
                third = as_float(fields[2])
                if third is not None:
                    ball_third.append(third)
            ball_pos[row] = (bx * length, by * width)
            if positions:
                best = min(((px - bx * length) ** 2 + (py - by * width) ** 2, team)
                           for px, py, team in positions)
                nearest[row] = (math.sqrt(best[0]), best[1])

    print("\n[{}] raw tracking".format(label))
    print("  lines {}  frame_id {}..{}  contiguous {}".format(
        n_lines, first_id, last_id, last_id - first_id + 1 == n_lines))
    print("  ':'-parts per line {}".format(dict(parts_count)))
    print("  player groups per line {}".format(dict(groups_count)))
    print("  fields per player group {}".format(dict(player_fields)))
    print("  fields per ball group {}".format(dict(ball_fields)))
    print("  non-numeric first tokens {}".format(dict(bad_tokens.most_common(5))))
    print("  ball visible, all frames: {:.2f}%".format(100.0 * sum(ball_seen) / n_frames))
    print("  player x range [{:.3f}, {:.3f}]  y range [{:.3f}, {:.3f}]".format(
        px_min, px_max, py_min, py_max))
    print("  ball x: {}".format(quantiles(ball_x, (0, 1, 50, 99, 100))))
    print("  ball y: {}".format(quantiles(ball_y, (0, 1, 50, 99, 100))))
    outside = sum(1 for x, y in zip(ball_x, ball_y) if not (0 <= x <= 1 and 0 <= y <= 1))
    print("  ball observed outside the 0..1 pitch: {:.2f}% of ball frames".format(
        100.0 * outside / max(1, len(ball_x))))
    print("  player 3rd field (sampled): {}".format(quantiles(player_third, (0, 50, 90, 99, 100))))
    print("  ball 3rd field: {}".format(quantiles(ball_third, (0, 50, 90, 99, 100))))

    if not spans:
        return
    graded = [i for first, last in spans for i in range(first, last + 1)]
    n_graded = len(graded)
    seen_graded = sum(ball_seen[i] for i in graded)
    print("  -- graded frames ({}) --".format(n_graded))
    print("  ball visible: {:.2f}%   mean tracked players: {:.2f}".format(
        100.0 * seen_graded / n_graded, sum(tracked[i] for i in graded) / float(n_graded)))
    print("  tracked players per frame: {}".format(
        quantiles([tracked[i] for i in graded], (0, 1, 10, 50, 90, 99, 100))))
    for team, name in ((0, "home"), (1, "away")):
        per_frame = [sum(slot_seen[s][i] for s in range(n_slots) if team_of_slot[s] == team)
                     for i in graded[::25]]
        print("  tracked {} players (sampled): {}".format(name, quantiles(per_frame, (1, 10, 50, 90, 99))))

    visible_runs, gaps, edge_missing = runs(ball_seen, spans)
    missing = n_graded - seen_graded
    print("  ball gaps (frames, period-edge gaps excluded = {} frames): {}".format(
        edge_missing, quantiles(gaps)))
    print("  ball visible runs (frames): {}".format(quantiles(visible_runs)))
    print("  share of MISSING frames by the length of the gap they sit in:")
    bounds = ((1, 12), (13, 25), (26, 50), (51, 125), (126, 250), (251, 750), (751, 10 ** 9))
    for low, high in bounds:
        frames = sum(g for g in gaps if low <= g <= high)
        number = sum(1 for g in gaps if low <= g <= high)
        print("    gap {:>4}-{:<6} {:>6} gaps  {:5.1f}% of missing frames".format(
            low, high if high < 10 ** 9 else "inf", number, 100.0 * frames / max(1, missing)))

    # What brackets each gap: nearest team before and after, and how far the ball moved.
    same = different = unknown = 0
    same_by_len, total_by_len = Counter(), Counter()
    jumps, dist_before, dist_all = [], [], []
    for first, last in spans:
        index = first
        while index <= last:
            if ball_seen[index]:
                index += 1
                continue
            start = index
            while index <= last and not ball_seen[index]:
                index += 1
            before, after = start - 1, index
            if before < first or after > last:
                continue
            gap = after - start
            bucket = next(b for b in ((12, "<=12"), (50, "13-50"), (125, "51-125"),
                                      (10 ** 9, ">125")) if gap <= b[0])[1]
            a, b = nearest.get(before), nearest.get(after)
            jumps.append(math.dist(ball_pos[before], ball_pos[after]))
            if a:
                dist_before.append(a[0])
            if not a or not b or a[0] > 3.0 or b[0] > 3.0:
                unknown += 1
                continue
            total_by_len[bucket] += 1
            if a[1] == b[1]:
                same += 1
                same_by_len[bucket] += 1
            else:
                different += 1
    for row in graded[::10]:
        if row in nearest:
            dist_all.append(nearest[row][0])
    total = same + different + unknown
    print("  gaps bracketed by a player within 3 m on both sides: {} of {}".format(same + different, total))
    print("    same team before and after: {}   different team: {}   no clear owner on one side: {}".format(
        same, different, unknown))
    for bucket in ("<=12", "13-50", "51-125", ">125"):
        if total_by_len[bucket]:
            print("    gap {:>7}: same team {:5.1f}% of {} bracketed gaps".format(
                bucket, 100.0 * same_by_len[bucket] / total_by_len[bucket], total_by_len[bucket]))
    print("  ball displacement across a gap (m): {}".format(quantiles(jumps)))
    print("  nearest-player distance, last frame before a gap (m): {}".format(quantiles(dist_before)))
    print("  nearest-player distance, all visible graded frames (m): {}".format(quantiles(dist_all)))

    slot_runs, slot_gaps = [], []
    for slot in range(n_slots):
        seen, gone, _ = runs(slot_seen[slot], spans)
        slot_runs.extend(seen)
        slot_gaps.extend(gone)
    print("  track-slot lifetimes (frames): {}".format(quantiles(slot_runs)))
    print("  track-slot absences (frames): {}".format(quantiles(slot_gaps)))


def describe_tagging(path):
    codes, kickoffs, teams = Counter(), Counter(), Counter()
    for instance in ET.parse(path).getroot().iter("instance"):
        code = instance.findtext("code")
        codes[code] += 1
        tags = {label.findtext("group"): label.findtext("text") for label in instance.findall("label")}
        teams[tags.get("Team")] += 1
        if (code or "").lower() == "set pieces" and (tags.get("Type") or "").lower() == "kick off (start)":
            kickoffs[tags.get("Half")] += 1
    print("\n[smart tagging] {} instances".format(sum(codes.values())))
    print("  codes {}".format(dict(codes.most_common())))
    print("  team tag {}".format(dict(teams)))
    print("  'Kick Off (Start)' tags per half {}".format(dict(kickoffs)))


def describe_homography(path, label):
    rows = valid = 0
    errors = []
    with open(path) as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        for row in reader:
            rows += 1
            if row["valid"] == "1":
                valid += 1
                if rows % 20 == 0:
                    errors.append(float(row["reprojection_error"]))
    print("\n[{}] homography: columns {}".format(label, columns))
    print("  rows {}  valid {:.1f}%  reprojection error (sampled): {}".format(
        rows, 100.0 * valid / max(1, rows), quantiles(errors, (50, 90, 99, 100))))


if __name__ == "__main__":
    tactical = os.path.join(DATA, "tactical")
    broadcast = os.path.join(DATA, "broadcast")
    if not os.path.isfile(os.path.join(tactical, "BAR-ATH_tactical_atd.txt")):
        print("match data not found under {}".format(DATA))
        sys.exit(2)

    print("[tactical] metadata")
    teams = describe_meta(os.path.join(tactical, "BAR-ATH_tactical_atd.xml"))
    describe_atd(os.path.join(tactical, "BAR-ATH_tactical_atd.txt"), teams, N_TACTICAL, GRADED, "tactical")
    describe_tagging(os.path.join(tactical, "BAR-ATH_tactical_smart_tagging.xml"))
    describe_homography(os.path.join(tactical, "BAR-ATH_tactical_homography.csv"), "tactical")

    print("\n[broadcast] metadata")
    teams = describe_meta(os.path.join(broadcast, "BAR-ATH_broadcast_atd.xml"))
    describe_atd(os.path.join(broadcast, "BAR-ATH_broadcast_atd.txt"), teams, 164488, (), "broadcast")
    describe_homography(os.path.join(broadcast, "BAR-ATH_broadcast_homography.csv"), "broadcast")
