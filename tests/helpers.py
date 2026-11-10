"""Synthetic FrameTables shared by the tests. No competition data."""
from __future__ import annotations

import numpy as np

from possession.frames import FrameTable

PITCH = (105.0, 68.0)


def _periods_array(T, periods):
    period = np.zeros(T, np.int8)
    for p, a, b in periods or [(1, 0, T)]:
        period[a:b] = p
    return period


def make_table(T, ball, home, away, period=None, proxy=None, home_speed=None, away_speed=None):
    ball = np.broadcast_to(np.asarray(ball, np.float32), (T, 2)).copy()
    home = np.broadcast_to(np.asarray(home, np.float32), (T,) + np.shape(home)[-2:]).copy()
    away = np.broadcast_to(np.asarray(away, np.float32), (T,) + np.shape(away)[-2:]).copy()
    hs = np.zeros(home.shape[:2], np.float32) if home_speed is None else \
        np.broadcast_to(np.asarray(home_speed, np.float32), home.shape[:2]).copy()
    as_ = np.zeros(away.shape[:2], np.float32) if away_speed is None else \
        np.broadcast_to(np.asarray(away_speed, np.float32), away.shape[:2]).copy()
    if period is None:
        period = np.ones(T, np.int8)
    if proxy is not None:
        proxy = np.broadcast_to(np.asarray(proxy, np.float32), (T, 2)).copy()
    return FrameTable(pitch=PITCH, ball=ball, home=home, away=away, home_speed=hs,
                      away_speed=as_, period=np.asarray(period, np.int8), proxy=proxy)


def _blank_outside(ft):
    out = ft.period == 0
    for arr in (ft.ball, ft.home, ft.away, ft.home_speed, ft.away_speed):
        arr[out] = np.nan
    if ft.proxy is not None:
        ft.proxy[out] = np.nan
    return ft


def _sines(rng, T, n, amp, n_terms=3, fmax=0.002):
    """(T, n) sums of a few low-frequency sinusoids."""
    t = np.arange(T)[:, None]
    out = np.zeros((T, n))
    for _ in range(n_terms):
        f = rng.uniform(fmax / 4, fmax, size=n)
        ph = rng.uniform(0, 2 * np.pi, size=n)
        out += amp / n_terms * np.sin(2 * np.pi * f * t + ph)
    return out


def smooth_table(T, periods=None, seed=0):
    """11 players per team and a ball moving as low-frequency sinusoids; nothing missing in play."""
    rng = np.random.default_rng(seed)
    home = np.stack([_sines(rng, T, 11, 30.0) - 10.0, _sines(rng, T, 11, 20.0)], -1)
    away = np.stack([_sines(rng, T, 11, 30.0) + 10.0, _sines(rng, T, 11, 20.0)], -1)
    ball = np.stack([_sines(rng, T, 1, 45.0)[:, 0], _sines(rng, T, 1, 25.0)[:, 0]], -1)
    hs = np.abs(np.gradient(home[..., 0], axis=0)) * 25
    as_ = np.abs(np.gradient(away[..., 0], axis=0)) * 25
    ft = make_table(T, ball, home, away, period=_periods_array(T, periods),
                    home_speed=hs, away_speed=as_)
    return _blank_outside(ft)


def random_table(seed, T, periods=None, proxy=False):
    rng = np.random.default_rng(seed)
    ft = smooth_table(T, periods=periods, seed=seed)
    ft.ball += rng.normal(0, 0.3, ft.ball.shape).astype(np.float32)
    ft.home_speed = rng.uniform(0, 6, ft.home_speed.shape).astype(np.float32)
    ft.away_speed = rng.uniform(0, 6, ft.away_speed.shape).astype(np.float32)
    for _ in range(4):
        a = int(rng.integers(0, max(T - 60, 1)))
        ft.ball[a:a + int(rng.integers(5, 60))] = np.nan
    if proxy:
        ft.proxy = (ft.ball + rng.normal(0, 3, ft.ball.shape)).astype(np.float32)
        ft.proxy[np.isnan(ft.proxy)] = 0.0
    return _blank_outside(ft)


def perturb_period(ft, p, seed=99):
    """Copy of ft with new random positions inside period p only."""
    other = random_table(seed, ft.n_frames, proxy=ft.proxy is not None)
    m = ft.period == p
    out = make_table(ft.n_frames, ft.ball, ft.home, ft.away, period=ft.period,
                     proxy=ft.proxy, home_speed=ft.home_speed, away_speed=ft.away_speed)
    for name in ("ball", "home", "away", "home_speed", "away_speed", "proxy"):
        a = getattr(out, name)
        if a is not None:
            a[m] = getattr(other, name)[m]
    return out


def team(xs, ys=None):
    ys = [0.0] * len(xs) if ys is None else ys
    return np.array(list(zip(xs, ys)), np.float32)


def scene_home_on_ball(T=30):
    return make_table(T, [0.0, 0.0], team([0.0]), team([10.0]))


def scene_with_gaps(gaps, T=200):
    home = team([-40, -30, -30, -30, -30, -20, -20, -20, -10, -10, 0.0],
                [0, -20, -7, 7, 20, -15, 0, 15, -5, 5, 0])
    away = team([40, 30, 30, 30, 30, 20, 20, 20, 10, 10, 5],
                [0, -20, -7, 7, 20, -15, 0, 15, -5, 5, 0])
    ft = make_table(T, [0.0, 0.0], home, away, home_speed=2.0, away_speed=2.0)
    for a, b in gaps:
        ft.ball[a:b] = np.nan
    return ft


def scene_ball_at(x, y, T=20):
    return make_table(T, [x, y], team([-10.0, -20.0]), team([10.0, 20.0]))


def scene_with_duplicates(T=50):
    hx = [-40, -30, -30, -30, -30, -20, -20, -20, -10, -10, 0, 0, -5, -15]
    home = team(hx, [0, -20, -7, 7, 20, -15, 0, 15, -5, 5, 1, 1, 3, 3])
    away = team([40, 30, 30, 30, 30, 20, 20, 20, 10, 10, 5])
    return make_table(T, [0.0, 0.0], home, away)


def _two_player_scene(T, home_xy, away_xy, ball_xy):
    ft = make_table(T, ball_xy, np.zeros((1, 2)), np.zeros((1, 2)),
                    home_speed=3.0, away_speed=3.0)
    ft.home[:, 0] = home_xy
    ft.away[:, 0] = away_xy
    return ft


def scene_handover(T=400):
    """Home player on the ball, away 10 m off, for frames 0..199; the reverse for 200..399."""
    ball = np.zeros((T, 2), np.float32)
    home = np.zeros((T, 2), np.float32)
    away = np.zeros((T, 2), np.float32)
    away[:200, 0] = 10.0
    home[200:, 0] = 10.0
    return _two_player_scene(T, home, away, ball)


def scene_with_long_gap(T=400):
    ball = np.zeros((T, 2), np.float32)
    home = np.zeros((T, 2), np.float32)
    away = np.zeros((T, 2), np.float32)
    away[:, 0] = 10.0
    ft = _two_player_scene(T, home, away, ball)
    ft.ball[100:250] = np.nan
    return ft


def empty_table(T, periods):
    ft = make_table(T, [np.nan, np.nan], np.full((11, 2), np.nan), np.full((11, 2), np.nan),
                    period=_periods_array(T, periods), home_speed=np.nan, away_speed=np.nan)
    return ft


def run_lengths(mask):
    m = np.concatenate(([0], np.asarray(mask).astype(np.int8), [0]))
    d = np.diff(m)
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def broadcast_from(tactical, pieces, keep=8, noise=0.2, seed=0):
    """A broadcast-like table cut from `tactical`, plus the true tactical frame of each row."""
    rng = np.random.default_rng(seed)
    parts, truth = [], []
    for a, b in pieces:
        n = b - a
        home = np.full((n, 11, 2), np.nan, np.float32)
        away = np.full((n, 11, 2), np.nan, np.float32)
        hk = rng.choice(11, size=keep, replace=False)
        ak = rng.choice(11, size=keep, replace=False)
        home[:, hk] = tactical.home[a:b][:, hk] + rng.normal(0, noise, (n, keep, 2))
        away[:, ak] = tactical.away[a:b][:, ak] + rng.normal(0, noise, (n, keep, 2))
        ball = tactical.ball[a:b] + rng.normal(0, noise, (n, 2))
        parts.append((ball, home, away))
        truth.append(np.arange(a, b))
    ball = np.concatenate([p[0] for p in parts]).astype(np.float32)
    home = np.concatenate([p[1] for p in parts]).astype(np.float32)
    away = np.concatenate([p[2] for p in parts]).astype(np.float32)
    T = len(ball)
    ft = FrameTable(pitch=tactical.pitch, ball=ball, home=home, away=away,
                    home_speed=np.zeros(home.shape[:2], np.float32),
                    away_speed=np.zeros(away.shape[:2], np.float32),
                    period=np.zeros(T, np.int8))
    return ft, np.concatenate(truth)
