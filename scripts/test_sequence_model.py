"""Path B: a sequence model (a small convolutional network over time) next to LightGBM.

LightGBM sees the past and the future of a frame only through averages over windows of up to
20 s. A network over the sequence of frames (5 per second) learns its own context of +-25 s: how
the players run and how the two teams spread out, which is what tells possession while the ball
is missing for 11 s or more.

Input: the same 223 features per frame, every 5th frame, standardised (a missing value becomes 0
plus a flag). Output: p(H), p(A), p(D) per step, averaged with the prediction for the swapped
match (home and away exchanged, then exchanged back), interpolated back to 25 frames per second.
The change model and the decoder stay as they are. Two candidates:
  nn      the network's frame probabilities alone, with the baseline's change probabilities
  blend   the average of the network's and the baseline's frame probabilities (weight 0.5)
The baseline is the submitted model, as out-of-fold predictions written by
scripts/test_ensemble.py (--base names the entry: single, or an ensemble variant).

Two steps, because PyTorch and LightGBM must not share a process (two OpenMP runtimes):
  train     every match predicted by a network that never saw it: for a Metrica game, trained on
            the other two (weighted 2) and 6 DFL matches; for a DFL match, on the 3 Metrica games
            and 5 other DFL matches. One more DFL match (the last one not tested) only says when
            to stop (lowest loss there, checked every 250 steps). Then the final network, on 9
            matches (DFL match 7 says when to stop). Needs PyTorch.
  evaluate  the candidates against the baseline, decoder chosen on the other games as in
            scripts/check_weight2.py. No PyTorch; for --base single, the baseline's DFL
            predictions are computed here if scripts/test_ensemble.py did not write them.

Rule fixed before the run (2026-11-11): the candidate with the better mean on the 3 Metrica games
replaces the baseline only if it is better on all three games and the mean rises by at least
0.005; then the DFL check (each DFL match, decoder chosen on the other six) must keep the mean at
or above 0.572. All network settings are fixed before the run; nothing is tuned on the games.

    uv run --with torch python scripts/test_sequence_model.py train <out dir>
    uv run python scripts/test_sequence_model.py evaluate <out dir> --oof <test_ensemble dir> \
        --base single
"""
import argparse
import importlib.util
import itertools
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from possession.change import change_logit
from possession.decode import decode_periods
from possession.degrade import DegradeProfile
from possession.features import build_features
from possession.labels import X, swap_codes
from possession.model import (N_CHANGE_ROUNDS, N_ROUNDS, change_fn, predict_proba, save_decoder,
                              train, train_change)
from possession.paths import MODELS, ROOT, vendor_import
from possession.sequence import BLEND, STEP, transform

spec = importlib.util.spec_from_file_location("train_multi", ROOT / "scripts" / "train_multi.py")
tm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tm)

CHUNK = 1024                    # steps per training window (~3.4 min)
BATCH = 16
MAX_ITERS = 3000
EVAL_EVERY = 250
LR = 1e-3
WEIGHT_DECAY = 1e-2
CHANNELS = 64
DILATIONS = (1, 2, 4, 8, 16, 32, 64)    # each step sees +-127 steps = +-25 s
DROPOUT = 0.2
IN_DROPOUT = 0.1
W_METRICA = 2.0
MIN_GAIN = 0.005
DFL_FLOOR = 0.572
CONFIGS = list(itertools.product(*tm.CHANGE_GRID))
IGNORE = -100


# ---- data ------------------------------------------------------------------------------------
def setup():
    profile = DegradeProfile.from_json(MODELS / "atd_profile.json")
    keys = tm.registry(["dfl", "metrica"])
    index = {k: i for i, k in enumerate(keys)}
    return profile, keys, index


def name(k):
    return f"{k[0]}:{k[1]}"


def prepare(k, index, profile):
    """Every 5th frame of each half, both ways round, and what scoring needs at 25 fps."""
    ft, y, F, names, _, _ = tm.build(*k, index, profile)
    Fs, _ = build_features(ft.swapped())
    sl = [(a, z) for p, a, z in ft.period_slices() if p > 0]
    return {"sl": sl, "period": ft.period, "y": y, "seen": F[:, names.index("ball_vis")] == 1,
            "x": [F[a:z:STEP].copy() for a, z in sl], "xs": [Fs[a:z:STEP].copy() for a, z in sl],
            "t": [y[a:z:STEP].copy() for a, z in sl],
            "ts": [swap_codes(y[a:z:STEP]) for a, z in sl]}


def stats(data, ks):
    x = np.concatenate([s for k in ks for s in data[k]["x"] + data[k]["xs"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    mu = np.nan_to_num(mu).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32)
    return mu, sd, np.flatnonzero(np.isnan(x).any(axis=0))


def targets(t):
    return np.where(t == X, IGNORE, t).astype(np.int64)


# ---- network (PyTorch only here) ------------------------------------------------------------
def torch_parts():
    import torch
    from torch import nn
    from torch.nn import functional as Fn

    class Block(nn.Module):
        def __init__(self, ch: int, dilation: int):
            super().__init__()
            self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation, padding=dilation)
            self.norm = nn.LayerNorm(ch)
            self.proj = nn.Conv1d(ch, ch, 1)
            self.drop = nn.Dropout(DROPOUT)

        def forward(self, h):                       # (batch, channels, time)
            z = Fn.gelu(self.norm(self.conv(h).transpose(1, 2))).transpose(1, 2)
            return h + self.proj(self.drop(z))

    class Net(nn.Module):
        def __init__(self, d_in: int):
            super().__init__()
            self.drop = nn.Dropout(IN_DROPOUT)
            self.inp = nn.Conv1d(d_in, CHANNELS, 1)
            self.blocks = nn.Sequential(*[Block(CHANNELS, d) for d in DILATIONS])
            self.out = nn.Conv1d(CHANNELS, 3, 1)

        def forward(self, x):                       # (batch, time, features) -> logits
            h = self.inp(self.drop(x).transpose(1, 2))
            return self.out(self.blocks(h)).transpose(1, 2)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch, Fn, Net, device


def train_net(T, data, ks, val, iters, every, seed=0):
    """The network on matches `ks`; the state with the lowest loss on match `val` is kept."""
    torch, Fn, Net, dev = T
    st = stats(data, ks)
    xs, ys, ws = [], [], []
    for k in ks:
        for x, t in zip(data[k]["x"] + data[k]["xs"], data[k]["t"] + data[k]["ts"]):
            xs.append(transform(x, *st))
            ys.append(targets(t))
            ws.append(W_METRICA if k[0] == "metrica" else 1.0)
    lens = np.array([len(t) for t in ys])
    assert lens.min() >= CHUNK, "a half shorter than one training window"
    offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
    prob = (lens - CHUNK + 1) / (lens - CHUNK + 1).sum()
    ws = np.asarray(ws, np.float32)
    Xd = torch.from_numpy(np.concatenate(xs)).to(dev)
    Yd = torch.from_numpy(np.concatenate(ys)).to(dev)
    del xs, ys
    vx = [torch.from_numpy(transform(x, *st)).to(dev)[None] for x in data[val]["x"] + data[val]["xs"]]
    vy = [torch.from_numpy(targets(t)).to(dev) for t in data[val]["t"] + data[val]["ts"]]

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Net(Xd.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=iters, pct_start=0.1)
    ar = torch.arange(CHUNK, device=dev)
    best = (np.inf, None, 0)
    for it in range(1, iters + 1):
        net.train()
        s = rng.choice(len(lens), BATCH, p=prob)
        start = offs[s] + rng.integers(0, lens[s] - CHUNK + 1)
        idx = torch.from_numpy(start).to(dev)[:, None] + ar
        yb = Yd[idx].reshape(-1)
        wb = torch.from_numpy(ws[s]).to(dev)[:, None].expand(-1, CHUNK).reshape(-1)
        wb = wb * (yb != IGNORE).float()
        loss = Fn.cross_entropy(net(Xd[idx]).reshape(-1, 3), yb, ignore_index=IGNORE,
                                reduction="none")
        loss = (loss * wb).sum() / wb.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % every == 0 or it == iters:
            net.eval()
            tot, n = 0.0, 0
            with torch.no_grad():
                for x, t in zip(vx, vy):
                    tot += float(Fn.cross_entropy(net(x)[0], t, ignore_index=IGNORE,
                                                  reduction="sum"))
                    n += int((t != IGNORE).sum())
            v = tot / max(n, 1)
            if v < best[0]:
                best = (v, {k: w.detach().cpu().clone() for k, w in net.state_dict().items()}, it)
    net.load_state_dict(best[1])
    net.eval()
    del Xd, Yd, vx, vy
    if dev == "mps":
        torch.mps.empty_cache()
    return net, st, {"val_loss": float(best[0]), "steps": int(best[2])}


def predict_net(T, net, st, d):
    """p(H), p(A), p(D) at 25 fps: both ways round, averaged, interpolated between steps."""
    torch, _, _, dev = T
    P = np.full((len(d["period"]), 3), 1.0 / 3.0)
    with torch.no_grad():
        for (a, z), x, xs in zip(d["sl"], d["x"], d["xs"]):
            p = torch.softmax(net(torch.from_numpy(transform(x, *st)).to(dev)[None])[0], -1)
            ps = torch.softmax(net(torch.from_numpy(transform(xs, *st)).to(dev)[None])[0], -1)
            p5 = (p.cpu().numpy().astype(np.float64) + ps.cpu().numpy()[:, [1, 0, 2]]) / 2
            idx, frames = np.arange(a, z, STEP), np.arange(a, z)
            P[a:z] = np.stack([np.interp(frames, idx, p5[:, c]) for c in range(3)], axis=1)
    return P


def run_train(out, smoke, log):
    T = torch_parts()
    iters, every = (40, 20) if smoke else (MAX_ITERS, EVAL_EVERY)
    profile, keys, index = setup()
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    data = {k: prepare(k, index[k], profile) for k in keys}
    log(f"prepared {len(keys)} matches, device {T[3]}")
    arrays, info = {}, {}
    for k in met + dfl:
        val = [d for d in dfl if d != k][-1]
        net, st, inf = train_net(T, data, [m for m in keys if m not in (k, val)], val, iters, every)
        arrays[f"nn|{name(k)}|P"] = predict_net(T, net, st, data[k])
        info[name(k)] = dict(inf, stop_on=name(val))
        log(f"network for {name(k)}: lowest loss on {name(val)} {inf['val_loss']:.4f} "
            f"after {inf['steps']} steps")
    np.savez_compressed(out / "oof_nn.npz", **arrays)
    val = dfl[-1]
    net, st, inf = train_net(T, data, [m for m in keys if m != val], val, iters, every)
    T[0].save(net.state_dict(), out / "sequence_net.pt")
    np.savez(out / "sequence_stats.npz", mu=st[0], sd=st[1], flags=st[2])
    info["final"] = dict(inf, stop_on=name(val))
    (out / "training.json").write_text(json.dumps(info, indent=1))
    log(f"final network: lowest loss on {name(val)} {inf['val_loss']:.4f} after {inf['steps']} steps")
    run_export(out)


def run_export(out):
    """sequence_net.pt + sequence_stats.npz -> sequence_net.npz, the NumPy form the submission
    reads (src/possession/sequence.py)."""
    import torch
    state = torch.load(out / "sequence_net.pt", map_location="cpu")
    st = np.load(out / "sequence_stats.npz")
    np.savez(out / "sequence_net.npz", **{k: v.numpy() for k, v in state.items()},
             mu=st["mu"], sd=st["sd"], flags=st["flags"], dilations=np.array(DILATIONS),
             step=np.array(STEP))


# ---- evaluation (no PyTorch) ----------------------------------------------------------------
def _scores(item):
    key, (P, period, y, seen, lg) = item
    pm = vendor_import("possession_metrics")
    out = []
    for t in CONFIGS:
        r = pm.evaluate(y, decode_periods(P, period, tm._config(t), seen=seen, change_logit=lg))
        out.append((r["spell_f1_mean"], r["spell_f1"][0.7]["f1"]))
    return key, out


def nested(tab, label, members):
    res = {}
    for k in members:
        others = [m for m in members if m != k]
        mean_others = np.mean([[a for a, _ in tab[(label, m)]] for m in others], axis=0)
        b = int(np.argmax(mean_others))
        res[name(k)] = {"spell_f1": tab[(label, k)][b][0], "strict": tab[(label, k)][b][1],
                        "decoder": list(CONFIGS[b])}
    return res


def single_dfl_oof(keys, index, profile, dfl, rounds):
    """The single LightGBM model's out-of-fold DFL predictions, as in check_weight2.py."""
    rows = {k: tm.training_rows(*k, index[k], profile) for k in keys}
    names, gnames = rows[keys[0]]["names"], rows[keys[0]]["gnames"]
    cnames = names + gnames
    arrays = {}
    for d in dfl:
        ks = [k for k in keys if k != d]
        wt = np.concatenate([np.full(len(rows[k]["y"]), W_METRICA if k[0] == "metrica" else 1.0)
                             for k in ks])
        Xf = np.concatenate([rows[k]["F"] for k in ks])
        yf = np.concatenate([rows[k]["y"] for k in ks])
        booster = train(Xf, yf, names, n_rounds=rounds[0], weight=wt)
        del Xf, yf
        Xc = np.concatenate([np.hstack([rows[k]["F"], rows[k]["G"]]) for k in ks])
        yc = np.concatenate([rows[k]["ct"] for k in ks])
        cb = train_change(Xc, yc, cnames, n_rounds=rounds[1], weight=wt)
        del Xc, yc
        _, _, F, _, G, _ = tm.build(*d, index[d], profile)
        arrays[f"single_dfl|{name(d)}|P"] = predict_proba(booster, F)
        arrays[f"single_dfl|{name(d)}|pc"] = change_fn(cb)(np.hstack([F, G]), cnames)
    return arrays


def run_evaluate(out, oof_dir, base_name, smoke, log):
    profile, keys, index = setup()
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    data = {}
    for k in keys:
        ft, y, F, names, _, _ = tm.build(*k, index[k], profile)
        data[k] = (ft.period, y, F[:, names.index("ball_vis")] == 1)
    zn = np.load(out / "oof_nn.npz")
    zb = np.load(oof_dir / "oof_metrica.npz")

    def item(label, k, P, pc):
        period, y, seen = data[k]
        return (label, k), (P, period, y, seen, change_logit(pc))

    def candidates(k, P0, pc0):
        Pn = zn[f"nn|{name(k)}|P"]
        return {"nn": (Pn, pc0), "blend": (BLEND * P0 + (1 - BLEND) * Pn, pc0)}

    its = []
    for g in met:
        P0, pc0 = zb[f"{base_name}|{name(g)}|P"], zb[f"{base_name}|{name(g)}|pc"]
        its.append(item("base", g, P0, pc0))
        its += [item(c, g, *v) for c, v in candidates(g, P0, pc0).items()]
    with Pool(8) as p:
        tab = dict(p.map(_scores, its))
    report = {c: nested(tab, c, met) for c in ("base", "nn", "blend")}

    def per_game(c, field="spell_f1"):
        return [report[c][name(g)][field] for g in met]

    print(f"\nMetrica games, baseline '{base_name}' (decoder chosen on the other two games)")
    print("candidate   game 1  game 2  game 3 |  mean | strict (70 % overlap) mean")
    for c in ("base", "nn", "blend"):
        v, s = per_game(c), per_game(c, "strict")
        print(f"{c:<10} {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} | {np.mean(v):.4f} | {np.mean(s):.4f}")
    best = max(("nn", "blend"), key=lambda c: np.mean(per_game(c)))
    a, b = per_game("base"), per_game(best)
    passed = all(x > y for x, y in zip(b, a)) and np.mean(b) - np.mean(a) >= MIN_GAIN
    report.update(base_name=base_name, best_candidate=best, smoke=smoke,
                  metrica_gain=float(np.mean(b) - np.mean(a)), metrica_rule_passed=bool(passed))
    log(f"best candidate {best}: {np.mean(b):.4f} vs baseline {np.mean(a):.4f}, better on "
        f"{sum(x > y for x, y in zip(b, a))}/3; rule passed: {passed}")

    if passed or smoke:
        if base_name == "single":
            path = oof_dir / "oof_dfl_single.npz"
            if not path.exists():
                rounds = (10, 10) if smoke else (N_ROUNDS, N_CHANGE_ROUNDS)
                np.savez_compressed(path, **single_dfl_oof(keys, index, profile, dfl, rounds))
                log("single model's DFL predictions computed")
        else:
            path = oof_dir / "oof_dfl.npz"
        zd = np.load(path)
        its = []
        for d in dfl:
            P0, pc0 = zd[f"{base_name}_dfl|{name(d)}|P"], zd[f"{base_name}_dfl|{name(d)}|pc"]
            its += [item(f"base_dfl", d, P0, pc0), item(f"{best}_dfl", d, *candidates(d, P0, pc0)[best])]
        with Pool(8) as p:
            tab.update(dict(p.map(_scores, its)))
        report["base_dfl"] = nested(tab, "base_dfl", dfl)
        report[f"{best}_dfl"] = nested(tab, f"{best}_dfl", dfl)
        base_dfl = float(np.mean([r["spell_f1"] for r in report["base_dfl"].values()]))
        dfl_mean = float(np.mean([r["spell_f1"] for r in report[f"{best}_dfl"].values()]))
        means = np.mean([[x for x, _ in tab[(best, g)]] for g in met], axis=0)
        i = int(np.argmax(means))
        cfg = tm._config(CONFIGS[i])
        save_decoder(cfg, out / "decoder.json")
        report.update(dfl_mean=dfl_mean, base_dfl_mean=base_dfl,
                      dfl_check_passed=bool(dfl_mean >= DFL_FLOOR),
                      adopt=bool(passed and dfl_mean >= DFL_FLOOR),
                      decoder={"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty,
                               "change_weight": cfg.change_weight,
                               "chosen_on": "mean spellF1 of the 3 Metrica games, out of fold",
                               "mean_on_metrica": float(means[i])})
        log(f"DFL mean {dfl_mean:.4f} (baseline {base_dfl:.4f}, must be >= {DFL_FLOOR}); "
            f"final decoder {cfg}; adopt: {report['adopt']}")
    (out / f"cv_report_{base_name}.json").write_text(json.dumps(report, indent=1))
    log("done")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "evaluate", "export"])
    ap.add_argument("out")
    ap.add_argument("--oof", help="output dir of scripts/test_ensemble.py (evaluate)")
    ap.add_argument("--base", default="single", help="baseline entry in its oof files")
    ap.add_argument("--smoke", action="store_true",
                    help="40 steps, every phase forced: checks the code, not the idea")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"{msg} ({time.time() - t0:.0f} s)", flush=True)

    if args.step == "train":
        run_train(out, args.smoke, log)
    elif args.step == "export":
        run_export(out)
    else:
        run_evaluate(out, Path(args.oof), args.base, args.smoke, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
