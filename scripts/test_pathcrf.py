"""A published ball-free model, PathCRF (Kim et al. 2026), against ours on DFL match J03WR9.

PathCRF (arXiv 2602.12080, KDD 2026; code github.com/hyunsungkim-ds/pathcrf, MPL-2.0) reads from
the 22 players' tracks alone, without the ball, which player has the ball or which pass is under
way, every 5th frame, and decodes the sequence with a CRF that allows only legal hand-overs. The
published model (trial 140, Dynamic Masked CRF) was trained on five DFL matches, with J03WQQ for
validation and J03WR9 as its test match; on J03WR9 the paper reports the right team on 92.4 % of
in-play frames.

Rule fixed before the run (2026-11-13): continue with PathCRF only if it names the right team on
at least 92 of 100 in-play frames where our simulated tracking has no ball, judged against the DFL
possession flag (our frame model, trained on the other six DFL matches: 89.3).

Runs with PathCRF's own environment (PyTorch, torch_geometric, kloppy, pandas < 3), cloned into
external/pathcrf (see the README). Three steps:
  prep     link the DFL files where PathCRF expects them and run its preprocessing on J03WR9
           (its labels: the team from the synchronised events, split half way between two
           events of different teams)
  infer    run the published model as the repository's tutorial does; prints the paper's
           numbers (edge accuracy, event F1, team accuracy) as a check
  compare  which team, frame by frame, against the DFL flag and against PathCRF's labels, for
           PathCRF and for our out-of-fold frame models (the more likely of H and A, before the
           decoder): the DFL-only model of scripts/compare_training_sets.py (--oof) and the
           submitted LightGBM of scripts/test_ensemble.py (--oof-single). 'No ball' = frames
           where our simulated blackouts on J03WR9 hide the ball. Writes models/pathcrf_report.json.

usage: external/pathcrf/.venv/bin/python scripts/test_pathcrf.py prep <dir>
       external/pathcrf/.venv/bin/python scripts/test_pathcrf.py infer <dir> [--device mps]
       external/pathcrf/.venv/bin/python scripts/test_pathcrf.py compare <dir> \
           --oof <dir A>/oof.npz --oof-single <dir D>/oof_dfl_single.npz
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "external" / "pathcrf"
DFL = ROOT / "external" / "dfl"
MATCH = "J03WR9"
H, A, D, X = 0, 1, 2, 3  # as in src/possession/labels.py
RULE_MIN_NO_BALL = 92.0


def _enter_repo():
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))


def prep(out: Path):
    for sub, prefix in (("metadata", "DFL_02_01_"), ("event", "DFL_03_02_"), ("tracking", "DFL_04_03_")):
        d = REPO / "data" / "sportec" / sub
        d.mkdir(parents=True, exist_ok=True)
        for f in DFL.glob(f"{prefix}*.xml"):
            if not (d / f.name).exists():
                (d / f.name).symlink_to(f)
    _enter_repo()
    from datatools import utils
    from datatools.preprocess import Preprocessor
    from datatools.sportec_data import SportecData

    t0 = time.time()
    events = pd.read_parquet(f"data/sportec_synced/{MATCH}.parquet")
    tracking = SportecData(MATCH, load_tracking=True).tracking.reset_index(drop=True)
    tracking[["timestamp", "ball_x", "ball_y"]] = tracking[["timestamp", "ball_x", "ball_y"]].round(2)
    proc = Preprocessor(events, tracking)
    touches = proc.detect_touches()
    os.makedirs("data/sportec/event_processed", exist_ok=True)
    os.makedirs("data/sportec/tracking_processed", exist_ok=True)
    touches.to_parquet(f"data/sportec/event_processed/{MATCH}.parquet")
    tp = utils.calculate_running_features(proc.merge_tracking_poss(touches))
    tp.to_parquet(f"data/sportec/tracking_processed/{MATCH}.parquet")
    print(f"prep done: {len(tp)} frames, {len(touches)} events ({time.time() - t0:.0f} s)")


def infer(out: Path, device: str):
    _enter_repo()
    import torch

    from applications.team_poss import compute_true_pred_poss
    from datatools.evaluate_events import align_true_and_pred
    from datatools.postprocess import detect_events, simplify_events
    from inference import inference
    from models.utils import build_model, load_trial_args, resolve_model_path

    t0 = time.time()
    tracking = pd.read_parquet(f"data/sportec/tracking_processed/{MATCH}.parquet")
    events = pd.read_parquet(f"data/sportec/event_processed/{MATCH}.parquet")
    trial_args = load_trial_args("saved/140")
    model = build_model(trial_args, device=device)
    path = resolve_model_path("saved/140", "state_dict_best_acc.pt")
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    with torch.no_grad():
        _, _, micro_pred, stats = inference(
            model, tracking, use_crf=trial_args.get("crf_weight", 0) > 0, decode="indep",
            window_seconds=trial_args.get("window_seconds", None), correct_episode_lasts=True)

    pred_events = detect_events(tracking, micro_pred)
    ev, _ = align_true_and_pred(events, pred_events, threshold=1.0)
    frames = compute_true_pred_poss(simplify_events(events), pred_events, tracking)
    team_acc = float((frames["true_team"] == frames["pred_team"]).mean())
    summary = {"edge_acc_pct": 100 * stats["edge_acc"],
               "event_precision_pct": 100 * float(ev.loc["total", "precision"]),
               "event_recall_pct": 100 * float(ev.loc["total", "recall"]),
               "event_f1_pct": 100 * float(ev.loc["total", "f1"]),
               "events_true": int(ev.loc["total", "true_count"]),
               "events_predicted": int(ev.loc["total", "pred_count"]),
               "team_acc_their_measure_pct": 100 * team_acc, "team_frames": len(frames),
               "paper": {"edge_acc_pct": 69.64, "event_f1_pct": 75.69, "events_true": 1831,
                         "events_predicted": 1961, "team_acc_pct": 92.40, "team_frames": 85304}}
    print(json.dumps(summary, indent=1), f"\n({time.time() - t0:.0f} s)")

    out.mkdir(parents=True, exist_ok=True)
    keep = ["frame_id", "period_id", "episode_id", "ball_state", "ball_owning_team_id"]
    tracking[keep].to_parquet(out / "tracking_state.parquet")
    frames[["frame_id", "period_id", "true_team", "pred_team"]].to_parquet(out / "team_frames.parquet")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


def _team(P):
    return np.where(P[:, H] >= P[:, A], H, A)


def _changes_per_5min(lab, mask):
    """Team changes inside contiguous stretches of `mask`, per 5 minutes of `mask`."""
    n = 0
    idx = np.flatnonzero(mask)
    for seg in np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1):
        s = lab[seg]
        s = s[(s == H) | (s == A)]
        n += int((s[1:] != s[:-1]).sum())
    return n / (mask.sum() / (25 * 300))


def compare(out: Path, oof_path: str, single_path: str):
    key = f"dfl:{MATCH}"
    z = np.load(oof_path)
    P_dfl, period, y, seen = (z[f"dfl|{key}|{i}"] for i in range(4))
    seen = seen.astype(bool)
    P_single = np.load(single_path)[f"single_dfl|{key}|P"]
    assert len(P_single) == len(y)

    ts = pd.read_parquet(out / "tracking_state.parquet")
    tf = pd.read_parquet(out / "team_frames.parquet")

    # PathCRF's frame_id is the DFL frame number N; our rows count N from the half's first ball frame
    row = np.full(len(ts), -1, np.int64)
    for sec in (1, 2):
        ours = np.flatnonzero(period == sec)
        m = (ts["period_id"] == sec).to_numpy()
        r = ours[0] + ts.loc[m, "frame_id"].to_numpy() - int(ts.loc[m, "frame_id"].min())
        ok = r <= ours[-1]
        row[np.flatnonzero(m)[ok]] = r[ok]
    placed = row >= 0
    alive = (ts["ball_state"] == "alive").to_numpy()
    own = ts["ball_owning_team_id"].astype(str).to_numpy()
    home_id = pd.Series(own[placed & alive][y[row[placed & alive]] == H]).mode()[0]
    their = np.where(~alive, D, np.where(own == home_id, H, A))
    aligned = float((their[placed] == y[row[placed]]).mean())
    assert aligned > 0.995, f"frames not aligned ({aligned:.4f})"

    lookup = dict(zip(zip(ts["period_id"], ts["frame_id"]), row))
    tr = np.array([lookup.get(k, -1) for k in zip(tf["period_id"], tf["frame_id"])])
    code = {"home": H, "away": A}
    pc_pred = np.full(len(y), -1, np.int8)   # -1: no PathCRF output; 9: names no team ("out")
    pc_true = np.full(len(y), -1, np.int8)
    ok = tr >= 0
    pc_pred[tr[ok]] = [code.get(v, 9) for v in tf["pred_team"].to_numpy()[ok]]
    pc_true[tr[ok]] = [code.get(v, 9) for v in tf["true_team"].to_numpy()[ok]]

    models = {"PathCRF (published, trial 140)": pc_pred,
              "ours: LightGBM, other 6 DFL matches": _team(P_dfl),
              "ours: submitted LightGBM (+ Metrica, weight 2)": _team(P_single)}
    live = (y == H) | (y == A)

    def table(mask, truth):
        res = {}
        for name, pred in models.items():
            res[name] = [100 * float((pred[mask & s] == truth[mask & s]).mean())
                         for s in (np.ones_like(seen), seen, ~seen)]
        return res

    report = {"rule": f"PathCRF >= {RULE_MIN_NO_BALL} % right team on in-play frames without the "
                      "ball (our simulated blackouts), against the DFL flag",
              "columns": ["all", "ball seen", "no ball"],
              "reproduction": json.loads((out / "summary.json").read_text()),
              "frame_alignment_pct": 100 * aligned,
              "in_play_frames": int(live.sum()),
              "no_ball_pct_of_in_play": round(100 * float((~seen[live]).mean()), 2)}
    for variant, cov in (("all_frames", live & (pc_pred >= 0)),
                         ("pathcrf_names_a_team", live & ((pc_pred == H) | (pc_pred == A)))):
        named = cov & ((pc_true == H) | (pc_true == A))
        report[variant] = {
            "frames": int(cov.sum()),
            "pathcrf_names_no_team_pct": round(100 * float((pc_pred[cov] == 9).mean()), 2),
            "vs_dfl_flag": table(cov, y),
            "vs_pathcrf_labels": table(named, pc_true),
            "labels_agree_pct": round(100 * float((pc_true[named] == y[named]).mean()), 2)}
    cov = live & (pc_pred >= 0)
    report["team_changes_per_5min"] = {name: round(float(_changes_per_5min(lab, cov)), 1) for name, lab in
                                       (("DFL flag", y), ("PathCRF labels", pc_true),
                                        ("PathCRF prediction", pc_pred))}
    no_ball = report["all_frames"]["vs_dfl_flag"]["PathCRF (published, trial 140)"][2]
    report["passed"] = bool(no_ball >= RULE_MIN_NO_BALL)

    for variant in ("all_frames", "pathcrf_names_a_team"):
        r = report[variant]
        print(f"\n== {variant}: {r['frames']} in-play frames; PathCRF names no team on "
              f"{r['pathcrf_names_no_team_pct']} %; the two labels agree on {r['labels_agree_pct']} %")
        for truth in ("vs_dfl_flag", "vs_pathcrf_labels"):
            print(f"  {truth:<18} {'all':>6} {'seen':>6} {'no ball':>8}")
            for name, a in r[truth].items():
                print(f"  {name:<48} {a[0]:6.1f} {a[1]:6.1f} {a[2]:8.1f}")
    print("\nteam changes per 5 min of play:", report["team_changes_per_5min"])
    print(f"rule: PathCRF without the ball {no_ball:.1f} (needs {RULE_MIN_NO_BALL}) -> "
          f"{'passed' if report['passed'] else 'not passed'}")
    (ROOT / "models" / "pathcrf_report.json").write_text(json.dumps(report, indent=1) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("step", choices=["prep", "infer", "compare"])
    ap.add_argument("out", type=Path)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--oof")
    ap.add_argument("--oof-single")
    a = ap.parse_args()
    a.out = a.out.resolve()
    if a.step == "prep":
        prep(a.out)
    elif a.step == "infer":
        infer(a.out, a.device)
    else:
        compare(a.out, a.oof, a.oof_single)
    return 0


if __name__ == "__main__":
    sys.exit(main())
