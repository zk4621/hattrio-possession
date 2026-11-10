"""Minimum-duration Viterbi decoder over H / A / D on integer scores."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .labels import X

NEG = -(1 << 62)


@dataclass(frozen=True)
class DecoderConfig:
    min_len: tuple[int, int, int] = (25, 25, 50)   # H, A, D in frames
    switch_penalty: int = 5000                      # same unit as the scores
    # Minimum length of a possession (H or A) whose first min_len_seen frames all have the ball
    # seen. Where the ball is seen, a short possession can be told from noise; inside a blackout
    # it cannot. None: one minimum for all frames (min_len).
    min_len_seen: int | None = None


def quantize(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-4, 1.0)
    return np.round(1000.0 * np.log(p)).astype(np.int32)


def _minimums(T: int, cfg: DecoderConfig, seen: np.ndarray | None) -> list[list[int]]:
    """Per state, the minimum length of a run that starts on each frame."""
    mins = []
    for k in range(3):
        base = int(cfg.min_len[k])
        if k < 2 and cfg.min_len_seen is not None and seen is not None:
            short = int(cfg.min_len_seen)
            cs = np.concatenate(([0], np.cumsum(np.asarray(seen, dtype=np.int64))))
            start = np.arange(T)
            stop = np.minimum(start + short, T)
            full = (cs[stop] - cs[start] == short)          # all of [s, s+short) seen
            m = np.where(full, short, base)
            mins.append(m.astype(np.int64).tolist())
        else:
            mins.append([base] * T)
    return mins


def decode(scores: np.ndarray, cfg: DecoderConfig, seen: np.ndarray | None = None) -> np.ndarray:
    """Best labelling under the switch penalty with every run long enough.

    A run of state k that starts on frame s must be at least mins[k][s] long: min_len[k], or for
    H and A min_len_seen where the ball is seen on every frame of [s, s + min_len_seen).
    """
    s = np.asarray(scores, dtype=np.int64)
    T = len(s)
    P = int(cfg.switch_penalty)
    fallback = np.full(T, int(np.argmax(s.sum(axis=0))) if T else 0, dtype=np.int8)
    if T == 0:
        return fallback
    mins = _minimums(T, cfg, seen)
    options = [sorted(set(m)) for m in mins]
    if T < min(min(o) for o in options):
        return fallback

    C = [np.concatenate(([0], np.cumsum(s[:, k]))).tolist() for k in range(3)]
    col = [s[:, k].tolist() for k in range(3)]
    V = [[NEG] * T for _ in range(3)]
    BP = [bytearray(T) for _ in range(3)]   # 0 stay, 1..3 switch from j=val-1, 4 start
    BM = [[0] * T for _ in range(3)]        # length of the run entered by a switch or start
    for t in range(T):
        for k in range(3):
            Vk, Ck, mk = V[k], C[k], mins[k]
            best = NEG
            bp = 0
            run = 0
            if t >= 1 and Vk[t - 1] > NEG:
                best = Vk[t - 1] + col[k][t]
            for m in options[k]:
                start = t - m + 1
                if start < 0 or mk[start] != m:
                    continue
                if start == 0:
                    cand = Ck[t + 1]
                    if cand > best:
                        best, bp, run = cand, 4, m
                    continue
                seg = Ck[t + 1] - Ck[start] - P
                src = start - 1
                for j in range(3):
                    if j == k:
                        continue
                    vj = V[j][src]
                    if vj > NEG:
                        cand = vj + seg
                        if cand > best:
                            best, bp, run = cand, j + 1, m
            Vk[t] = best
            BP[k][t] = bp
            BM[k][t] = run
    last = [V[0][T - 1], V[1][T - 1], V[2][T - 1]]
    if max(last) <= NEG:
        return fallback
    k = int(np.argmax(last))
    out = np.empty(T, dtype=np.int8)
    t = T - 1
    while t >= 0:
        bp = BP[k][t]
        if bp == 0:
            out[t] = k
            t -= 1
        elif bp == 4:
            out[: t + 1] = k
            break
        else:
            start = t - BM[k][t] + 1
            out[start: t + 1] = k
            t = start - 1
            k = bp - 1
    return out


def decode_periods(proba: np.ndarray, period: np.ndarray, cfg: DecoderConfig,
                   live_from: dict[int, int] | None = None,
                   seen: np.ndarray | None = None) -> np.ndarray:
    proba = np.array(proba, dtype=np.float64)
    bad = ~np.isfinite(proba).all(axis=1)
    proba[bad] = 1.0 / 3.0
    scores = quantize(proba)
    period = np.asarray(period)
    out = np.full(len(period), X, dtype=np.int8)
    breaks = np.flatnonzero(period[1:] != period[:-1]) + 1
    bounds = np.concatenate(([0], breaks, [len(period)]))
    for a, z in zip(bounds[:-1], bounds[1:]):
        p = int(period[a])
        if p == 0:
            continue
        sc = scores[a:z].copy()
        if live_from and p in live_from:
            n = max(0, min(int(live_from[p]) - a, z - a))
            sc[:n] = (-10 ** 6, -10 ** 6, 0)
        out[a:z] = decode(sc, cfg, None if seen is None else np.asarray(seen)[a:z])
    return out
