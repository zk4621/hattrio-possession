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
            matches (DFL match 7 says when to stop). Needs PyTorch. --seeds trains one network
            per seed for every match; --kind mt adds the change output (below).
  evaluate  the candidates against the baseline, decoder chosen on the other games as in
            scripts/check_weight2.py. No PyTorch; for --base single, the baseline's DFL
            predictions are computed here if scripts/test_ensemble.py did not write them.

Rule fixed before the run (2026-11-11): the candidate with the better mean on the 3 Metrica games
replaces the baseline only if it is better on all three games and the mean rises by at least
0.005; then the DFL check (each DFL match, decoder chosen on the other six) must keep the mean at
or above 0.572. All network settings are fixed before the run; nothing is tuned on the games.

Two follow-ups (--experiment), each with the same rule, fixed before its run (2026-11-11):
  avg       5 networks (seeds 0-4; seed 0 is the submitted one) averaged with each other, then
            with LightGBM as before. Baseline: the submission (LightGBM with network 0).
  boundary  networks with a fourth output, "the label changes within 2 frames of this step" (the
            change detector's target, on every 5th frame; loss = frame loss + change loss), as
            many seeds as the frame networks have after 'avg' (--frame-seeds). Baseline: the state
            after 'avg'. Candidates:
              change  frame probabilities as in the baseline; change probabilities the average of
                      the change detector's and these networks' (weight 0.5)
              both    as 'change', and the frame probabilities from these networks instead

A third follow-up, same rule, fixed before its run (2026-11-11), because 'boundary' failed with a
change output that saw only every 5th frame:
  fullrate  networks ('fr') with a change output on every frame (25 fps): the network's state,
            interpolated linearly to every frame, plus the change detector's inputs that are not
            window averages (70 of the 223 columns, and its 17 change features), through 5 dilated
            convolutions over +-31 frames (1.2 s). Target: the detector's own, "the label changes
            within 2 frames of this frame" (frames labelled X left out, as for the detector);
            loss = frame loss + change loss. One seed (0). Baseline: the submission. Candidates:
              change  change probabilities the average of the detector's and the network's (0.5)
              alone   the network's change probabilities in place of the detector's
            The DFL networks and the final network are trained only if the Metrica part passes.

    PY=.../.venv/bin/python     # a Python with torch (README)
    PYTHONPATH=src $PY scripts/test_sequence_model.py train <dir> --seeds 0
    uv run python scripts/test_sequence_model.py evaluate <dir> --oof <test_ensemble dir>
    PYTHONPATH=src $PY scripts/test_sequence_model.py train <dir> --seeds 1 2 3 4
    uv run python scripts/test_sequence_model.py evaluate <dir> --oof <test_ensemble dir> \
        --experiment avg --seeds 0 1 2 3 4
    PYTHONPATH=src $PY scripts/test_sequence_model.py train <dir> --kind mt --seeds 0 1 2 3 4
    uv run python scripts/test_sequence_model.py evaluate <dir> --oof <test_ensemble dir> \
        --experiment boundary --frame-seeds 0 1 2 3 4 --seeds 0 1 2 3 4
    PYTHONPATH=src $PY scripts/test_sequence_model.py train <dir> --kind fr --parts metrica
    uv run python scripts/test_sequence_model.py evaluate <dir> --oof <test_ensemble dir> \
        --experiment fullrate
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

from possession.change import change_features, change_logit, change_targets
from possession.decode import decode_periods
from possession.degrade import DegradeProfile
from possession.features import WINDOW_SUFFIXES, build_features
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
CHANGE_LOSS = 1.0               # weight of the change output's loss ('mt' and 'fr' networks)
CHANGE_BLEND = 0.5              # weight of the change detector in the average
FAST_DILATIONS = (1, 2, 4, 8, 16)   # 'fr': the change output on every frame sees +-31 frames
# frame outputs p(H), p(A), p(D); 'mt' adds "a change near here" on every 5th frame, 'fr' on
# every frame (a second head at 25 fps)
KINDS = {"nn": 3, "mt": 4, "fr": 3}
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


def single_frame_columns(names):
    """The columns that are not window averages: what each frame shows on its own."""
    return [i for i, n in enumerate(names)
            if not n.endswith(tuple(f"_{s}" for s in WINDOW_SUFFIXES))]


def prepare(k, index, profile, fast=False):
    """Every 5th frame of each half, both ways round, and what scoring needs at 25 fps. With
    `fast` ('fr' networks), also every frame the steps span (5 per step, padded past the end of
    the half): the change detector's inputs that are not window averages, its target, and which
    frames count (not X)."""
    ft, y, F, names, G, _ = tm.build(*k, index, profile)
    sw = ft.swapped()
    Fs, _ = build_features(sw)
    sl = [(a, z) for p, a, z in ft.period_slices() if p > 0]
    ct = change_targets(y, ft.period)       # a change within 2 frames: every change hits 1 step
    d = {"sl": sl, "period": ft.period, "y": y, "seen": F[:, names.index("ball_vis")] == 1,
         "x": [F[a:z:STEP].copy() for a, z in sl], "xs": [Fs[a:z:STEP].copy() for a, z in sl],
         "t": [y[a:z:STEP].copy() for a, z in sl],
         "ts": [swap_codes(y[a:z:STEP]) for a, z in sl],
         "c": [ct[a:z:STEP].astype(np.float32) for a, z in sl]}
    if fast:
        Gs, _ = change_features(sw, Fs, names)
        cols = single_frame_columns(names)
        g, gs = np.hstack([F[:, cols], G]), np.hstack([Fs[:, cols], Gs])

        def span(m, a, z, fill):
            out = np.full((STEP * len(range(a, z, STEP)),) + m.shape[1:], fill, np.float32)
            out[:z - a] = m[a:z]
            return out
        d.update(g=[span(g, a, z, np.nan) for a, z in sl],
                 gs=[span(gs, a, z, np.nan) for a, z in sl],
                 cf=[span(ct, a, z, 0.0) for a, z in sl],
                 mf=[span(y != X, a, z, 0.0) for a, z in sl])
    return d


def stats(data, ks, key="x", every=1):
    """Mean and spread of the training matches' inputs (`key`, both ways round; from every
    `every`-th row), and the columns that can be missing (from all rows)."""
    arrs = [s for k in ks for s in data[k][key] + data[k][key + "s"]]
    x = np.concatenate([s[::every] for s in arrs])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu, sd = np.nanmean(x, axis=0), np.nanstd(x, axis=0)
    mu = np.nan_to_num(mu).astype(np.float32)
    sd = np.where(np.isfinite(sd) & (sd > 1e-6), sd, 1.0).astype(np.float32)
    return mu, sd, np.flatnonzero(np.any([np.isnan(s).any(axis=0) for s in arrs], axis=0))


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
        """d_fast > 0: a second head on every frame ('fr'), fed the network's state interpolated
        between steps and d_fast inputs of the frame itself; it outputs one change logit."""

        def __init__(self, d_in: int, n_out: int = 3, d_fast: int = 0):
            super().__init__()
            self.drop = nn.Dropout(IN_DROPOUT)
            self.inp = nn.Conv1d(d_in, CHANNELS, 1)
            self.blocks = nn.Sequential(*[Block(CHANNELS, d) for d in DILATIONS])
            self.out = nn.Conv1d(CHANNELS, n_out, 1)
            if d_fast:
                self.fast_drop = nn.Dropout(IN_DROPOUT)
                self.fast_inp = nn.Conv1d(d_fast, CHANNELS, 1)
                self.fast_blocks = nn.Sequential(*[Block(CHANNELS, d) for d in FAST_DILATIONS])
                self.fast_out = nn.Conv1d(CHANNELS, 1, 1)

        def forward(self, x, g=None):
            """x (batch, steps, features) -> logits (batch, steps, n_out); with g (batch,
            5 * steps, fast features) also change logits on every frame (batch, 5 * steps)."""
            h = self.blocks(self.inp(self.drop(x).transpose(1, 2)))
            out = self.out(h).transpose(1, 2)
            if g is None:
                return out
            hn = torch.cat([h[..., 1:], h[..., -1:]], dim=-1)     # the next step (last: itself)
            u = torch.stack([h + (j / STEP) * (hn - h) for j in range(STEP)], dim=-1).flatten(2)
            f = self.fast_blocks(u + self.fast_inp(self.fast_drop(g).transpose(1, 2)))
            return out, self.fast_out(f)[:, 0]

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return torch, Fn, Net, device


def train_net(T, data, ks, val, iters, every, seed=0, kind="nn"):
    """The network on matches `ks`; the state with the lowest loss on match `val` is kept.
    'mt': a fourth output for "a change within 2 frames" on every 5th frame; 'fr': the same on
    every frame, from the second head. The change loss is added to the frame loss.
    Returns the network, the input statistics (frame inputs; every-frame inputs or None) and
    where it stopped."""
    torch, Fn, Net, dev = T
    n_out, fast = KINDS[kind], kind == "fr"
    st = stats(data, ks)
    sg = stats(data, ks, "g", STEP) if fast else None
    xs, ys, cs, ws, gs, cfs, mfs = [], [], [], [], [], [], []
    for k in ks:
        d = data[k]
        for x, t, c in zip(d["x"] + d["xs"], d["t"] + d["ts"], d["c"] + d["c"]):
            xs.append(transform(x, *st))
            ys.append(targets(t))
            cs.append(c)
            ws.append(W_METRICA if k[0] == "metrica" else 1.0)
        if fast:
            for g, cf, mf in zip(d["g"] + d["gs"], d["cf"] + d["cf"], d["mf"] + d["mf"]):
                gs.append(transform(g, *sg))
                cfs.append(cf)
                mfs.append(mf)
    lens = np.array([len(t) for t in ys])
    assert lens.min() >= CHUNK, "a half shorter than one training window"
    offs = np.concatenate(([0], np.cumsum(lens)[:-1]))
    prob = (lens - CHUNK + 1) / (lens - CHUNK + 1).sum()
    ws = np.asarray(ws, np.float32)
    Xd = torch.from_numpy(np.concatenate(xs)).to(dev)
    Yd = torch.from_numpy(np.concatenate(ys)).to(dev)
    Cd = torch.from_numpy(np.concatenate(cs)).to(dev)
    del xs, ys, cs
    if fast:
        Gd = torch.from_numpy(np.concatenate(gs)).to(dev)
        CFd = torch.from_numpy(np.concatenate(cfs)).to(dev)
        MFd = torch.from_numpy(np.concatenate(mfs)).to(dev)
        assert len(Gd) == STEP * len(Xd), "every step needs its 5 frames"
        del gs, cfs, mfs
    d = data[val]
    vx = [torch.from_numpy(transform(x, *st)).to(dev)[None] for x in d["x"] + d["xs"]]
    vy = [torch.from_numpy(targets(t)).to(dev) for t in d["t"] + d["ts"]]
    vc = [torch.from_numpy(c).to(dev) for c in d["c"] + d["c"]]
    if fast:
        vg = [torch.from_numpy(transform(g, *sg)).to(dev)[None] for g in d["g"] + d["gs"]]
        vf = [(torch.from_numpy(c).to(dev), torch.from_numpy(m).to(dev))
              for c, m in zip(d["cf"] + d["cf"], d["mf"] + d["mf"])]

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    net = Net(Xd.shape[1], n_out, Gd.shape[1] if fast else 0).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=iters, pct_start=0.1)
    ar = torch.arange(CHUNK, device=dev)
    arf = torch.arange(CHUNK * STEP, device=dev)
    best = (np.inf, None, 0)
    for it in range(1, iters + 1):
        net.train()
        s = rng.choice(len(lens), BATCH, p=prob)
        start = offs[s] + rng.integers(0, lens[s] - CHUNK + 1)
        idx = torch.from_numpy(start).to(dev)[:, None] + ar
        yb = Yd[idx].reshape(-1)
        wm = torch.from_numpy(ws[s]).to(dev)[:, None]          # match weights (batch, 1)
        w = wm.expand(-1, CHUNK).reshape(-1)
        wb = w * (yb != IGNORE).float()
        if fast:
            fidx = torch.from_numpy(start * STEP).to(dev)[:, None] + arf
            out, oc = net(Xd[idx], Gd[fidx])
        else:
            out = net(Xd[idx])
        loss = Fn.cross_entropy(out[..., :3].reshape(-1, 3), yb, ignore_index=IGNORE,
                                reduction="none")
        loss = (loss * wb).sum() / wb.sum().clamp(min=1.0)
        if n_out == 4:
            lc = Fn.binary_cross_entropy_with_logits(out[..., 3].reshape(-1), Cd[idx].reshape(-1),
                                                     reduction="none")
            loss = loss + CHANGE_LOSS * (lc * w).sum() / w.sum()
        if fast:
            wf = wm * MFd[fidx]
            lc = Fn.binary_cross_entropy_with_logits(oc, CFd[fidx], reduction="none")
            loss = loss + CHANGE_LOSS * (lc * wf).sum() / wf.sum().clamp(min=1.0)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if it % every == 0 or it == iters:
            net.eval()
            tot, n, totc, nc = 0.0, 0, 0.0, 0.0
            with torch.no_grad():
                for i, (x, t, c) in enumerate(zip(vx, vy, vc)):
                    if fast:
                        o, oc = net(x, vg[i])
                        cf, mf = vf[i]
                        totc += float((Fn.binary_cross_entropy_with_logits(
                            oc[0], cf, reduction="none") * mf).sum())
                        nc += float(mf.sum())
                        o = o[0]
                    else:
                        o = net(x)[0]
                    tot += float(Fn.cross_entropy(o[:, :3], t, ignore_index=IGNORE,
                                                  reduction="sum"))
                    n += int((t != IGNORE).sum())
                    if n_out == 4:
                        totc += float(Fn.binary_cross_entropy_with_logits(o[:, 3], c,
                                                                          reduction="sum"))
                        nc += len(c)
            v = tot / max(n, 1) + (CHANGE_LOSS * totc / max(nc, 1) if n_out == 4 or fast
                                   else 0.0)
            if v < best[0]:
                best = (v, {k: w_.detach().cpu().clone() for k, w_ in net.state_dict().items()}, it)
    net.load_state_dict(best[1])
    net.eval()
    del Xd, Yd, Cd, vx, vy, vc
    if fast:
        del Gd, CFd, MFd, vg, vf
    if dev == "mps":
        torch.mps.empty_cache()
    return net, (st, sg), {"val_loss": float(best[0]), "steps": int(best[2])}


def predict_net(T, net, stg, d, kind="nn"):
    """p(H), p(A), p(D) at 25 fps: both ways round, averaged, interpolated between steps; for
    'mt' and 'fr' also p(a change within 2 frames) (the same both ways round; 'mt' interpolated
    between steps, 'fr' on every frame), else None."""
    torch, _, _, dev = T
    n_out, fast = KINDS[kind], kind == "fr"
    st, sg = stg
    P = np.full((len(d["period"]), 3), 1.0 / 3.0)
    pc = np.zeros(len(d["period"])) if n_out == 4 or fast else None

    def run(x, g):
        xt = torch.from_numpy(transform(x, *st)).to(dev)[None]
        if not fast:
            return net(xt)[0], None
        o, oc = net(xt, torch.from_numpy(transform(g, *sg)).to(dev)[None])
        return o[0], torch.sigmoid(oc[0]).cpu().numpy().astype(np.float64)

    with torch.no_grad():
        for i, ((a, z), x, xs) in enumerate(zip(d["sl"], d["x"], d["xs"])):
            o, f1 = run(x, d["g"][i] if fast else None)
            os_, f2 = run(xs, d["gs"][i] if fast else None)
            p = torch.softmax(o[:, :3], -1).cpu().numpy().astype(np.float64)
            ps = torch.softmax(os_[:, :3], -1).cpu().numpy().astype(np.float64)
            p5 = (p + ps[:, [1, 0, 2]]) / 2
            idx, frames = np.arange(a, z, STEP), np.arange(a, z)
            P[a:z] = np.stack([np.interp(frames, idx, p5[:, c]) for c in range(3)], axis=1)
            if n_out == 4:
                c5 = (torch.sigmoid(o[:, 3]).cpu().numpy().astype(np.float64)
                      + torch.sigmoid(os_[:, 3]).cpu().numpy().astype(np.float64)) / 2
                pc[a:z] = np.interp(frames, idx, c5)
            if fast:
                pc[a:z] = ((f1 + f2) / 2)[:z - a]
    return P, pc


def run_train(out, kind, seeds, parts, smoke, log):
    T = torch_parts()
    iters, every = (40, 20) if smoke else (MAX_ITERS, EVAL_EVERY)
    profile, keys, index = setup()
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    data = {k: prepare(k, index[k], profile, kind == "fr") for k in keys}
    log(f"prepared {len(keys)} matches, device {T[3]}")
    tests = (met if "metrica" in parts else []) + (dfl if "dfl" in parts else [])
    info_path = out / f"training_{kind}.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    for seed in seeds:
        arrays, inf_s = {}, info.setdefault(str(seed), {})
        for k in tests:
            val = [d for d in dfl if d != k][-1]
            net, st, inf = train_net(T, data, [m for m in keys if m not in (k, val)], val, iters,
                                     every, seed, kind)
            P, pc = predict_net(T, net, st, data[k], kind)
            arrays[f"{kind}|{name(k)}|P"] = P
            if pc is not None:
                arrays[f"{kind}|{name(k)}|pc"] = pc
            inf_s[name(k)] = dict(inf, stop_on=name(val))
            log(f"{kind} seed {seed}, network for {name(k)}: lowest loss on {name(val)} "
                f"{inf['val_loss']:.4f} after {inf['steps']} steps")
        if arrays:
            path = out / f"oof_{kind}_s{seed}.npz"
            if path.exists():
                with np.load(path) as z:
                    arrays = {**{k: z[k] for k in z.files}, **arrays}
            np.savez_compressed(path, **arrays)
        if "final" in parts:
            val = dfl[-1]
            net, (st, sg), inf = train_net(T, data, [m for m in keys if m != val], val, iters,
                                           every, seed, kind)
            T[0].save(net.state_dict(), out / f"{kind}_s{seed}.pt")
            fast = {} if sg is None else {"mu_g": sg[0], "sd_g": sg[1], "flags_g": sg[2]}
            np.savez(out / f"{kind}_s{seed}_stats.npz", mu=st[0], sd=st[1], flags=st[2], **fast)
            inf_s["final"] = dict(inf, stop_on=name(val))
            log(f"{kind} seed {seed}, final network: lowest loss on {name(val)} "
                f"{inf['val_loss']:.4f} after {inf['steps']} steps")
            run_export(out, kind, seed)
        info_path.write_text(json.dumps(info, indent=1))


def run_export(out, kind, seed):
    """<kind>_s<seed>.pt + its stats -> <kind>_s<seed>.npz, the NumPy form the submission reads
    (src/possession/sequence.py)."""
    import torch
    state = torch.load(out / f"{kind}_s{seed}.pt", map_location="cpu")
    st = dict(np.load(out / f"{kind}_s{seed}_stats.npz"))
    fast = {"fast_dilations": np.array(FAST_DILATIONS)} if "mu_g" in st else {}
    np.savez(out / f"{kind}_s{seed}.npz", **{k: v.numpy() for k, v in state.items()}, **st,
             dilations=np.array(DILATIONS), step=np.array(STEP), **fast)


# ---- evaluation (no PyTorch) ----------------------------------------------------------------
def _scores(item):
    key, (P, period, y, seen, lg, configs) = item
    pm = vendor_import("possession_metrics")
    out = []
    for t in configs:
        r = pm.evaluate(y, decode_periods(P, period, tm._config(t), seen=seen, change_logit=lg))
        out.append((r["spell_f1_mean"], r["spell_f1"][0.7]["f1"]))
    return key, out


def nested(tab, label, members, configs=CONFIGS):
    res = {}
    for k in members:
        others = [m for m in members if m != k]
        mean_others = np.mean([[a for a, _ in tab[(label, m)]] for m in others], axis=0)
        b = int(np.argmax(mean_others))
        res[name(k)] = {"spell_f1": tab[(label, k)][b][0], "strict": tab[(label, k)][b][1],
                        "decoder": list(configs[b])}
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


def change_quality(ct, p):
    """How sharply p marks the change frames: average precision (1 = every change frame ranked
    above every other frame) and the mean p on change frames."""
    hits = ct[np.argsort(-p, kind="stable")] == 1
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return {"average_precision": float(prec[hits].mean()),
            "mean_p_on_changes": float(p[ct == 1].mean())}


class Members:
    """Out-of-fold predictions of the networks of one kind, averaged over the given seeds."""

    def __init__(self, out, kind, seeds):
        self.kind, self.z = kind, [np.load(out / f"oof_{kind}_s{s}.npz") for s in seeds]

    def __call__(self, k, part="P"):
        return np.mean([z[f"{self.kind}|{name(k)}|{part}"] for z in self.z], axis=0)

    def has(self, k):
        return all(f"{self.kind}|{name(k)}|P" in z.files for z in self.z)


CHANGE_KIND = {"boundary": "mt", "fullrate": "fr"}     # whose change probabilities are tested


def experiment(exp, out, seeds, frame_seeds):
    """(baseline, candidates): functions (match, LightGBM P, change p) -> (P, change p)."""
    if exp == "blend":
        nn = Members(out, "nn", [0])
        return (lambda k, P0, pc0: (P0, pc0),
                {"nn": lambda k, P0, pc0: (nn(k), pc0),
                 "blend": lambda k, P0, pc0: (BLEND * P0 + (1 - BLEND) * nn(k), pc0)})
    if exp == "avg":
        one, avg = Members(out, "nn", [0]), Members(out, "nn", seeds)
        return (lambda k, P0, pc0: (BLEND * P0 + (1 - BLEND) * one(k), pc0),
                {"avg": lambda k, P0, pc0: (BLEND * P0 + (1 - BLEND) * avg(k), pc0)})
    if exp == "fullrate":
        nn, fr = Members(out, "nn", frame_seeds), Members(out, "fr", seeds)

        def frame_p(k, P0):
            return BLEND * P0 + (1 - BLEND) * nn(k)
        return (lambda k, P0, pc0: (frame_p(k, P0), pc0),
                {"change": lambda k, P0, pc0: (
                    frame_p(k, P0), CHANGE_BLEND * pc0 + (1 - CHANGE_BLEND) * fr(k, "pc")),
                 "alone": lambda k, P0, pc0: (frame_p(k, P0), fr(k, "pc"))})
    nn, mt = Members(out, "nn", frame_seeds), Members(out, "mt", seeds)

    def pcb(k, pc0):
        return CHANGE_BLEND * pc0 + (1 - CHANGE_BLEND) * mt(k, "pc")
    return (lambda k, P0, pc0: (BLEND * P0 + (1 - BLEND) * nn(k), pc0),
            {"change": lambda k, P0, pc0: (BLEND * P0 + (1 - BLEND) * nn(k), pcb(k, pc0)),
             "both": lambda k, P0, pc0: (BLEND * P0 + (1 - BLEND) * mt(k), pcb(k, pc0))})


def run_evaluate(out, oof_dir, base_name, exp, seeds, frame_seeds, smoke, log):
    profile, keys, index = setup()
    dfl = [k for k in keys if k[0] == "dfl"]
    met = [k for k in keys if k[0] == "metrica"]
    data = {}
    for k in keys:
        ft, y, F, names, _, _ = tm.build(*k, index[k], profile)
        data[k] = (ft.period, y, F[:, names.index("ball_vis")] == 1)
    zb = np.load(oof_dir / "oof_metrica.npz")
    base, cands = experiment(exp, out, seeds, frame_seeds)
    labels = ["base"] + list(cands)
    configs = CONFIGS[:2] if smoke else CONFIGS

    def item(label, k, P, pc):
        period, y, seen = data[k]
        return (label, k), (P, period, y, seen, change_logit(pc), configs)

    its = []
    for g in met:
        P0, pc0 = zb[f"{base_name}|{name(g)}|P"], zb[f"{base_name}|{name(g)}|pc"]
        its.append(item("base", g, *base(g, P0, pc0)))
        its += [item(c, g, *f(g, P0, pc0)) for c, f in cands.items()]
    with Pool(8) as p:
        tab = dict(p.map(_scores, its))
    report = {c: nested(tab, c, met, configs) for c in labels}

    def per_game(c, field="spell_f1"):
        return [report[c][name(g)][field] for g in met]

    print(f"\nExperiment '{exp}', LightGBM '{base_name}' (decoder chosen on the other two games)")
    print("candidate   game 1  game 2  game 3 |  mean | strict (70 % overlap) mean")
    for c in labels:
        v, s = per_game(c), per_game(c, "strict")
        print(f"{c:<10} {v[0]:7.4f} {v[1]:7.4f} {v[2]:7.4f} | {np.mean(v):.4f} | {np.mean(s):.4f}")
    if exp in CHANGE_KIND:
        mt, quality = Members(out, CHANGE_KIND[exp], seeds), {}
        print("change probabilities: average precision | mean p on change frames")
        for g in met:
            period, y, _ = data[g]
            inside = period > 0
            ct = change_targets(y, period)[inside]
            pc0, pn = zb[f"{base_name}|{name(g)}|pc"][inside], mt(g, "pc")[inside]
            quality[name(g)] = q = {
                src: change_quality(ct, p) for src, p in
                (("detector", pc0), ("network", pn),
                 ("average", CHANGE_BLEND * pc0 + (1 - CHANGE_BLEND) * pn))}
            print(f"{name(g)}: " + ", ".join(f"{src} {v['average_precision']:.3f} | "
                                             f"{v['mean_p_on_changes']:.3f}" for src, v in q.items()))
        report["change_quality"] = quality
    best = max(cands, key=lambda c: np.mean(per_game(c)))
    a, b = per_game("base"), per_game(best)
    passed = all(x > y for x, y in zip(b, a)) and np.mean(b) - np.mean(a) >= MIN_GAIN
    report.update(experiment=exp, base_name=base_name, seeds=list(seeds),
                  frame_seeds=list(frame_seeds), best_candidate=best, smoke=smoke,
                  metrica_gain=float(np.mean(b) - np.mean(a)), metrica_rule_passed=bool(passed),
                  rule="fixed before the run (2026-11-11): the better candidate on the 3 Metrica "
                       "games replaces the baseline only if better on all three and the mean "
                       "rises by >= 0.005; DFL mean >= 0.572")
    log(f"best candidate {best}: {np.mean(b):.4f} vs baseline {np.mean(a):.4f}, better on "
        f"{sum(x > y for x, y in zip(b, a))}/3; rule passed: {passed}")
    if (passed or smoke) and exp in CHANGE_KIND and not all(
            Members(out, CHANGE_KIND[exp], seeds).has(d) for d in dfl):
        report["dfl_check"] = "pending: the DFL networks are not trained yet"
        log(f"train --kind {CHANGE_KIND[exp]} --parts dfl final, then evaluate again "
            "for the DFL check")
    elif passed or smoke:
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
            its += [item("base_dfl", d, *base(d, P0, pc0)),
                    item(f"{best}_dfl", d, *cands[best](d, P0, pc0))]
        with Pool(8) as p:
            tab.update(dict(p.map(_scores, its)))
        report["base_dfl"] = nested(tab, "base_dfl", dfl, configs)
        report[f"{best}_dfl"] = nested(tab, f"{best}_dfl", dfl, configs)
        base_dfl = float(np.mean([r["spell_f1"] for r in report["base_dfl"].values()]))
        dfl_mean = float(np.mean([r["spell_f1"] for r in report[f"{best}_dfl"].values()]))
        means = np.mean([[x for x, _ in tab[(best, g)]] for g in met], axis=0)
        i = int(np.argmax(means))
        cfg = tm._config(configs[i])
        save_decoder(cfg, out / f"decoder_{exp}.json")
        report.update(dfl_mean=dfl_mean, base_dfl_mean=base_dfl,
                      dfl_check_passed=bool(dfl_mean >= DFL_FLOOR),
                      adopt=bool(passed and dfl_mean >= DFL_FLOOR),
                      decoder={"min_len": list(cfg.min_len), "switch_penalty": cfg.switch_penalty,
                               "change_weight": cfg.change_weight,
                               "chosen_on": "mean spellF1 of the 3 Metrica games, out of fold",
                               "mean_on_metrica": float(means[i])})
        log(f"DFL mean {dfl_mean:.4f} (baseline {base_dfl:.4f}, must be >= {DFL_FLOOR}); "
            f"final decoder {cfg}; adopt: {report['adopt']}")
    (out / f"cv_report_{exp}.json").write_text(json.dumps(report, indent=1))
    log("done")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["train", "evaluate", "export"])
    ap.add_argument("out")
    ap.add_argument("--kind", choices=list(KINDS), default="nn",
                    help="train/export: nn = frame outputs; mt = and a change output on every "
                         "5th frame; fr = and a change output on every frame")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0],
                    help="train: one network per seed; evaluate: the networks averaged")
    ap.add_argument("--parts", nargs="+", choices=["metrica", "dfl", "final"],
                    default=["metrica", "dfl", "final"], help="train: which networks")
    ap.add_argument("--oof", help="output dir of scripts/test_ensemble.py (evaluate)")
    ap.add_argument("--base", default="single", help="LightGBM entry in its oof files")
    ap.add_argument("--experiment", choices=["blend", "avg", "boundary", "fullrate"],
                    default="blend")
    ap.add_argument("--frame-seeds", type=int, nargs="+", default=[0],
                    help="evaluate boundary: the frame networks of the baseline")
    ap.add_argument("--smoke", action="store_true",
                    help="40 steps, every phase forced: checks the code, not the idea")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def log(msg):
        print(f"{msg} ({time.time() - t0:.0f} s)", flush=True)

    if args.step == "train":
        run_train(out, args.kind, args.seeds, args.parts, args.smoke, log)
    elif args.step == "export":
        for s in args.seeds:
            run_export(out, args.kind, s)
    else:
        run_evaluate(out, Path(args.oof), args.base, args.experiment, args.seeds,
                     args.frame_seeds, args.smoke, log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
