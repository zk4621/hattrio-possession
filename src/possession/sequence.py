"""The sequence network: a small convolutional network over time, next to the LightGBM frame model.

It reads the same 223 features on every 5th frame and sees +-25 s of play around each step (seven
dilated convolutions). Its p(H), p(A), p(D), averaged over the match as it is and with home and
away exchanged, are interpolated back to 25 frames per second and averaged with LightGBM's
(weight 0.5). The change model and the decoder are unchanged.

Inference is plain NumPy (float64), so the submission needs no PyTorch: the weights are trained
with PyTorch by scripts/test_sequence_model.py and exported to models/sequence_net.npz.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.special import erf

from .paths import MODELS

SEQUENCE_PATH = MODELS / "sequence_net.npz"
STEP = 5            # frames per step: 5 steps per second
BLEND = 0.5         # weight of the LightGBM probabilities in the average
META = ("mu", "sd", "flags", "dilations", "step")


def transform(x: np.ndarray, mu: np.ndarray, sd: np.ndarray, flags: np.ndarray) -> np.ndarray:
    """Standardised features; a missing value becomes 0, plus a 0/1 flag for the columns in
    `flags` (those that can be missing)."""
    z = (x - mu) / sd
    miss = np.isnan(z)
    z = np.clip(np.where(miss, 0.0, z), -10.0, 10.0)
    return np.concatenate([z, miss[:, flags]], axis=1).astype(np.float32)


def _gelu(z: np.ndarray) -> np.ndarray:
    return 0.5 * z * (1.0 + erf(z / np.sqrt(2.0)))


class SequenceNet:
    """Weights as exported from PyTorch (Conv1d weights are (out, in, kernel))."""

    def __init__(self, arrays: dict):
        self.w = {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items() if k not in META}
        self.mu = np.asarray(arrays["mu"], dtype=np.float32)
        self.sd = np.asarray(arrays["sd"], dtype=np.float32)
        self.flags = np.asarray(arrays["flags"], dtype=np.int64)
        self.dilations = [int(d) for d in np.atleast_1d(arrays["dilations"])]
        self.step = int(arrays["step"])

    def logits(self, x: np.ndarray) -> np.ndarray:
        """x: (steps, inputs) after transform -> (steps, 3)."""
        w = self.w
        h = np.asarray(x, dtype=np.float64) @ w["inp.weight"][:, :, 0].T + w["inp.bias"]
        n = len(h)
        for i, d in enumerate(self.dilations):
            W = w[f"blocks.{i}.conv.weight"]
            hp = np.pad(h, ((d, d), (0, 0)))
            z = (hp[:n] @ W[:, :, 0].T + hp[d:d + n] @ W[:, :, 1].T
                 + hp[2 * d:2 * d + n] @ W[:, :, 2].T + w[f"blocks.{i}.conv.bias"])
            z = (z - z.mean(axis=1, keepdims=True)) / np.sqrt(z.var(axis=1, keepdims=True) + 1e-5)
            z = _gelu(z * w[f"blocks.{i}.norm.weight"] + w[f"blocks.{i}.norm.bias"])
            h = h + z @ w[f"blocks.{i}.proj.weight"][:, :, 0].T + w[f"blocks.{i}.proj.bias"]
        return h @ w["out.weight"][:, :, 0].T + w["out.bias"]

    def step_proba(self, x: np.ndarray) -> np.ndarray:
        """p(H), p(A), p(D) for each step of one half (raw features of every 5th frame)."""
        lg = self.logits(transform(x, self.mu, self.sd, self.flags))
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def proba(self, F: np.ndarray, Fs: np.ndarray, slices) -> np.ndarray:
        """p(H), p(A), p(D) at 25 fps. F: features of the match, Fs: of the match with home and
        away exchanged; slices: (start, stop) of each half. 1/3 outside the halves."""
        P = np.full((len(F), 3), 1.0 / 3.0)
        for a, z in slices:
            p = self.step_proba(F[a:z:self.step])
            ps = self.step_proba(Fs[a:z:self.step])[:, [1, 0, 2]]
            p5 = (p + ps) / 2
            idx, frames = np.arange(a, z, self.step), np.arange(a, z)
            P[a:z] = np.stack([np.interp(frames, idx, p5[:, c]) for c in range(3)], axis=1)
        return P


def load_sequence(path=SEQUENCE_PATH) -> SequenceNet | None:
    """The committed network, or None if there is none (LightGBM alone)."""
    p = Path(path)
    if not p.exists():
        return None
    with np.load(p) as z:
        return SequenceNet({k: z[k] for k in z.files})
