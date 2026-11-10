"""Tactical camera view centre as a coarse ball position."""
from __future__ import annotations

import numpy as np

from .frames import FrameTable

IMAGE_CENTRE = (960.0, 540.0)
MAX_LAG = 50
MAX_ERR_CORR = 250


def camera_centre(H: np.ndarray, valid: np.ndarray, pitch: tuple[float, float]) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64)
    valid = np.asarray(valid).astype(bool)
    out = np.full((len(H), 2), np.nan)
    if valid.any():
        inv = np.linalg.inv(H[valid])
        c = np.array([IMAGE_CENTRE[0], IMAGE_CENTRE[1], 1.0])
        p = inv @ c
        xy = p[:, :2] / p[:, 2:3]
        out[valid, 0] = (xy[:, 0] - 0.5) * pitch[0]
        out[valid, 1] = (xy[:, 1] - 0.5) * pitch[1]
    return out


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _shifted(x: np.ndarray, lag: int) -> np.ndarray:
    """y[t] = x[t + lag], NaN where t + lag is out of range."""
    y = np.full_like(x, np.nan)
    if lag >= 0:
        y[:len(x) - lag] = x[lag:]
    else:
        y[-lag:] = x[:len(x) + lag]
    return y


def measure_proxy(proxy: np.ndarray, ft: FrameTable) -> dict:
    play = ft.period > 0
    p = np.where(play[:, None], proxy, np.nan).astype(np.float64)
    b = np.where(play[:, None], ft.ball, np.nan).astype(np.float64)
    std_x = float(np.nanstd(p[:, 0]))
    best_lag, best_c = 0, -np.inf
    for lag in range(-MAX_LAG, MAX_LAG + 1):
        c = _corr(_shifted(p[:, 0], lag), b[:, 0])
        if np.isfinite(c) and c > best_c:
            best_lag, best_c = lag, c
    px = _shifted(p[:, 0], best_lag)
    py = _shifted(p[:, 1], best_lag)
    ex, ey = px - b[:, 0], py - b[:, 1]
    # autocorrelation of the error within visible runs
    ok = np.isfinite(ex)
    run_id = np.cumsum(np.concatenate(([1], (ok[1:] != ok[:-1]).astype(np.int64))))
    e0 = ex - np.nanmean(ex)
    err_corr = MAX_ERR_CORR
    for k in range(1, MAX_ERR_CORR + 1):
        a, c2 = e0[:-k], e0[k:]
        same = ok[:-k] & ok[k:] & (run_id[:-k] == run_id[k:])
        if same.sum() < 10:
            break
        if np.corrcoef(a[same], c2[same])[0, 1] < 0.5:
            err_corr = k
            break
    return {
        "std_x": std_x,
        "corr_x": float(best_c),
        "med_abs_dx": float(np.nanmedian(np.abs(ex))),
        "med_abs_dy": float(np.nanmedian(np.abs(ey))),
        "best_lag": int(best_lag),
        "err_std_x": float(np.nanstd(ex)),
        "err_std_y": float(np.nanstd(ey)),
        "err_corr_frames": int(err_corr),
    }


def _interp_all(x: np.ndarray) -> np.ndarray:
    ok = np.isfinite(x)
    if not ok.any():
        return x.copy()
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def simulate_proxy(ft: FrameTable, stats: dict, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    lag = int(stats["best_lag"])
    w = max(1, int(stats["err_corr_frames"]))
    out = np.full((ft.n_frames, 2), np.nan)
    for _, a, z in ft.period_slices():
        n = z - a
        src = np.clip(np.arange(n) - lag, 0, n - 1)
        for j, key in ((0, "err_std_x"), (1, "err_std_y")):
            ball = _interp_all(ft.ball[a:z, j].astype(np.float64))
            white = rng.normal(size=n + w - 1)
            noise = np.convolve(white, np.ones(w) / w, mode="valid")
            sd = noise.std()
            noise = noise / sd * float(stats[key]) if sd > 0 else noise
            out[a:z, j] = ball[src] + noise
    return out.astype(np.float32)


def competition_proxy(ft: FrameTable) -> np.ndarray:
    """Camera view centre of the tactical video, in metres, NaN outside periods."""
    from .paths import data_dir, vendor_import
    lh = vendor_import("load_homography")
    H, _, valid = lh.load(data_dir=str(data_dir()))
    out = camera_centre(H[:ft.n_frames], valid[:ft.n_frames], ft.pitch)
    out[ft.period == 0] = np.nan
    return out.astype(np.float32)
