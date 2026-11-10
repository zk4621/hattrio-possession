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


def quantize(proba: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=np.float64), 1e-4, 1.0)
    return np.round(1000.0 * np.log(p)).astype(np.int32)


def decode(scores: np.ndarray, cfg: DecoderConfig) -> np.ndarray:
    """Best labelling under the switch penalty with every run at least min_len[k] long."""
    s = np.asarray(scores, dtype=np.int64)
    T = len(s)
    m = [int(v) for v in cfg.min_len]
    P = int(cfg.switch_penalty)
    fallback = np.full(T, int(np.argmax(s.sum(axis=0))) if T else 0, dtype=np.int8)
    if T == 0 or T < min(m):
        return fallback

    C = [np.concatenate(([0], np.cumsum(s[:, k]))).tolist() for k in range(3)]
    col = [s[:, k].tolist() for k in range(3)]
    V = [[NEG] * T for _ in range(3)]
    BP = [bytearray(T) for _ in range(3)]  # 0 stay, 1..3 switch from j=val-1, 4 start
    V0, V1, V2 = V
    for t in range(T):
        for k in range(3):
            mk = m[k]
            Vk = V[k]
            best = NEG
            bp = 0
            if t >= 1 and Vk[t - 1] > NEG:
                best = Vk[t - 1] + col[k][t]
            if t == mk - 1:
                cand = C[k][mk]
                if cand > best:
                    best, bp = cand, 4
            elif t >= mk:
                src = t - mk
                seg = C[k][t + 1] - C[k][src + 1] - P
                for j in range(3):
                    if j == k:
                        continue
                    vj = V[j][src]
                    if vj > NEG:
                        cand = vj + seg
                        if cand > best:
                            best, bp = cand, j + 1
            Vk[t] = best
            BP[k][t] = bp
    last = [V0[T - 1], V1[T - 1], V2[T - 1]]
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
            start = t - m[k] + 1
            out[start: t + 1] = k
            t = start - 1
            k = bp - 1
    return out


def decode_periods(proba: np.ndarray, period: np.ndarray, cfg: DecoderConfig,
                   live_from: dict[int, int] | None = None) -> np.ndarray:
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
        out[a:z] = decode(sc, cfg)
    return out
